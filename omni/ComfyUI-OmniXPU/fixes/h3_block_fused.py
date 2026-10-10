"""H3 DiTBlock with producer-fused INT8 activations (2026-09-06).

Per block the stock chain materializes bf16 (M, 5376) tensors between addcmul_ (gated residual),
RMSNorm+AdaLN, Hadamard rotation and INT8 quantization -- twice -- and a bf16 (M, 14336) between
SiLU*mul and fc2's quantization. This module runs:
  norm1 -> int8              rms_norm_mod_convrot_quant(x)                     -> qkv GEMM (prequantized)
  x += gate*attn; norm2 -> int8  rms_norm_mod_convrot_quant(x, attn_out, gate) -> fc1 GEMM (prequantized)
  fc2 with the fused SiLU*mul+rotate+quantize producer (int8_linear input_act="swiglu")
Falls back to the stock forward when any weight is not a convrot INT8 QuantizedTensor on the
device or when a segment uses per-token indices. OMNIXPU_BLOCK_FUSED=0 disables.
Deployed as ComfyUI-OmniXPU/fixes/h3_block_fused.py (supersedes h3_adaln_fused.py when active).
"""
import logging
import os

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_applied = False


def apply():
    global _applied
    if _applied:
        return True, "already applied"
    if os.environ.get("OMNIXPU_BLOCK_FUSED", "1") == "0":
        return False, "disabled by OMNIXPU_BLOCK_FUSED=0"
    try:
        import comfy.ldm.minimax.model as h3
        import comfy.model_management
        import comfy.ops
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
        from omni_xpu_kernel import int8 as oint8
    except ImportError as e:
        return False, f"import failed: {e}"
    native = oint8._get_native()
    for name in ("rms_norm_mod_convrot_quant", "quantize_int8_convrot_silu_mul_split", "int8_linear_prequantized"):
        if native is None or not hasattr(native, name):
            return False, f"omni_xpu_kernel has no {name}"
    split = int(os.environ.get("OMNIXPU_BLOCK_FUSED_SPLIT", "8"))

    _orig_forward = h3.DiTBlock.forward
    _seg_cache = {}
    stats = {"fused": 0, "fallback": 0}
    reasons = {}
    debug = os.environ.get("OMNIXPU_BLOCK_FUSED_DEBUG", "0") == "1"

    def _why(linear, device):
        w = getattr(linear, "weight", None)
        if not isinstance(w, QuantizedTensor): return "not_quantized:" + type(w).__name__
        if w._layout_cls != "TensorWiseINT8Layout": return "layout:" + str(w._layout_cls)
        if not getattr(w._params, "convrot", False): return "no_convrot"
        if w.device != device: return "weight_on:" + str(w.device)
        if getattr(linear, "bias", None) is not None: return "has_bias"
        return "other"

    def _note(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    def _tick():
        n = stats["fused"] + stats["fallback"]
        if debug and n % 300 == 0:
            log.info("[OmniXPU] block_fused tally after %d calls: fused=%d fallback=%d reasons=%s", n, stats["fused"], stats["fallback"], dict(reasons))

    def _segments(mod_segments, rows, device):
        key = (tuple((a, b, int(r)) for a, b, r in mod_segments), rows, str(device))
        t = _seg_cache.get(key)
        if t is None:
            ends = [b for _, b, _ in mod_segments]
            mods = [int(r) for _, _, r in mod_segments]
            if len(ends) > 8 or ends != sorted(ends) or ends[-1] != rows or mod_segments[0][0] != 0:
                return None
            while len(ends) < 8:
                ends.append(rows)
                mods.append(mods[-1])
            t = (torch.tensor(ends, dtype=torch.int32, device=device),
                 torch.tensor(mods, dtype=torch.int32, device=device))
            if len(_seg_cache) > 64:
                _seg_cache.clear()
            _seg_cache[key] = t
        return t

    def _plain(linear, device):
        w = getattr(linear, "weight", None)
        if not (isinstance(w, QuantizedTensor) and w._layout_cls == "TensorWiseINT8Layout"
                and getattr(w._params, "convrot", False) and w.device == device
                and getattr(linear, "bias", None) is None
                and int(getattr(w._params, "convrot_groupsize", 256)) == 256):
            return None
        q, s = TensorWiseINT8Layout.get_plain_tensors(w)
        if q.dim() != 2 or not q.is_contiguous():
            return None
        return q, s.reshape(-1)

    def _norm_w(norm, x):
        return comfy.model_management.cast_to(norm.weight, dtype=x.dtype, device=x.device).contiguous()

    def _attn(attn, h_int8, h_scale, s, qkv_w, rope_freqs, transformer_options, x):
        H, D = attn.heads, attn.head_dim
        qkv = native.int8_linear_prequantized(h_int8, h_scale, qkv_w[0], qkv_w[1], None, 2)
        q, k, v = qkv.split(H * D, dim=-1)
        v = v.view(s, H, D)
        if rope_freqs is not None:
            q = q.view(1, s, H, D)
            k = k.view(1, s, H, D)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, H, D))
            k = attn.k_norm(k.view(s, H, D))
        q = h3.AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = h3.AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = h3.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = h3.optimized_attention(q, k, v, H, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return attn.out_proj(out.squeeze(0))

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        try:
            if (
                attention is None and
                x.dim() == 2 and x.dtype == torch.bfloat16 and x.device.type == "xpu" and x.is_contiguous()
                and x.shape[1] % 256 == 0
                and all(isinstance(r, int) for _, _, r in mod_segments)
                and getattr(self.norm1, "weight", None) is not None
                and not getattr(self.norm1, "weight_function", None)
                and not getattr(self.norm2, "weight_function", None)
            ):
                seg = _segments(mod_segments, x.shape[0], x.device)
                qkv_w = _plain(self.attn.qkv_proj, x.device)
                fc1_w = _plain(self.mlp.fc1, x.device)
                if seg is None: _note("segments")
                elif qkv_w is None: _note("qkv:" + _why(self.attn.qkv_proj, x.device))
                elif fc1_w is None: _note("fc1:" + _why(self.mlp.fc1, x.device))
                if seg is not None and qkv_w is not None and fc1_w is not None:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                    dt = x.dtype
                    h_i8, h_sc = native.rms_norm_mod_convrot_quant(
                        x, None, None, _norm_w(self.norm1, x),
                        scale_msa.to(dt).contiguous(), shift_msa.to(dt).contiguous(),
                        seg[0], seg[1], float(self.norm1.eps), 256, split)
                    attn_out = _attn(self.attn, h_i8, h_sc, x.shape[0], qkv_w, rope_freqs, transformer_options, x)
                    del h_i8, h_sc
                    attn_out = attn_out.contiguous()
                    h2_i8, h2_sc = native.rms_norm_mod_convrot_quant(
                        x, attn_out, gate_msa.to(dt).contiguous(), _norm_w(self.norm2, x),
                        scale_mlp.to(dt).contiguous(), shift_mlp.to(dt).contiguous(),
                        seg[0], seg[1], float(self.norm2.eps), 256, split)
                    del attn_out
                    f1 = native.int8_linear_prequantized(h2_i8, h2_sc, fc1_w[0], fc1_w[1], None, 2)
                    del h2_i8, h2_sc
                    mlp_out = comfy.ops.linear_input_act(self.mlp.fc2, f1, "swiglu")
                    del f1
                    stats["fused"] += 1
                    _tick()
                    return h3._mod_gate(x, gate_mlp, mlp_out, mod_segments)
        except Exception as e:  # noqa: BLE001
            _note("exception:" + type(e).__name__)
            if stats["fallback"] == 0:
                log.warning("[OmniXPU] fused DiTBlock path failed, using stock: %s", e)
        stats["fallback"] += 1
        _tick()
        if attention is not None:
            return _orig_forward(self, x, t_emb, mod_segments, rope_freqs,
                                 transformer_options=transformer_options, attention=attention)
        return _orig_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)

    h3.DiTBlock.forward = forward
    forward.stats = stats
    _applied = True
    log.info("[OmniXPU] H3 DiTBlock producer-fused INT8 path enabled (row_split=%d)", split)
    return True, None
