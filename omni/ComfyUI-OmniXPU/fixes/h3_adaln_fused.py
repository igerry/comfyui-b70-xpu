"""H3 DiTBlock: fuse the AdaLN scale/shift into the RMSNorm kernel (2026-09-06).

Stock: norm -> bf16, then per segment h[a:b].mul_(1+scale[row]).add_(shift[row]) = two more full
passes over (M, 5376) per norm, 3.9 ms per block for the two norms. rms_norm_mod applies the
modulation inside the normalize pass. Only the plain-int segment form is fused; per-token index
tensors (non-uniform denoise masks) fall back to the stock path. OMNIXPU_ADALN_FUSED=0 disables.
Deployed as ComfyUI-OmniXPU/fixes/h3_adaln_fused.py.
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
    if os.environ.get("OMNIXPU_ADALN_FUSED", "1") == "0":
        return False, "disabled by OMNIXPU_ADALN_FUSED=0"
    try:
        import comfy.ldm.minimax.model as h3
        import comfy.model_management
        from omni_xpu_kernel import norm as onorm
    except ImportError as e:
        return False, f"import failed: {e}"
    native = onorm._get_native()
    if native is None or not hasattr(native, "rms_norm_mod"):
        return False, "omni_xpu_kernel has no rms_norm_mod"

    _orig_forward = h3.DiTBlock.forward
    _seg_cache = {}
    stats = {"fused": 0, "fallback": 0}

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

    def _norm_mod(norm, x, scale, shift, seg):
        w = comfy.model_management.cast_to(norm.weight, dtype=x.dtype, device=x.device)
        return native.rms_norm_mod(
            w.contiguous(), x, scale.to(x.dtype).contiguous(), shift.to(x.dtype).contiguous(),
            seg[0], seg[1], float(norm.eps))

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        try:
            if (
                attention is None and
                x.dim() == 2 and x.dtype == torch.bfloat16 and x.device.type == "xpu" and x.is_contiguous()
                and x.shape[1] % 64 == 0
                and all(isinstance(r, int) for _, _, r in mod_segments)
                and getattr(self.norm1, "weight", None) is not None
                and not getattr(self.norm1, "weight_function", None)
            ):
                seg = _segments(mod_segments, x.shape[0], x.device)
                if seg is not None:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                    h = _norm_mod(self.norm1, x, scale_msa, shift_msa, seg)
                    x = h3._mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs,
                                                            transformer_options=transformer_options), mod_segments)
                    h = _norm_mod(self.norm2, x, scale_mlp, shift_mlp, seg)
                    stats["fused"] += 1
                    return h3._mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
        except Exception as e:  # noqa: BLE001
            if stats["fallback"] == 0:
                log.warning("[OmniXPU] fused AdaLN path failed, using stock DiTBlock: %s", e)
        stats["fallback"] += 1
        if attention is not None:
            return _orig_forward(self, x, t_emb, mod_segments, rope_freqs,
                                 transformer_options=transformer_options, attention=attention)
        return _orig_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)

    h3.DiTBlock.forward = forward
    forward.stats = stats
    _applied = True
    log.info("[OmniXPU] H3 DiTBlock fused RMSNorm+AdaLN enabled")
    return True, None
