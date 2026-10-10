"""MiniMax Music 3: GLU gate folded into the down-projection's INT8 quantizer (2026-10-09).

Stock FeedForward is proj -> `value, gate = chunk(2)` -> value * silu(gate) -> Linear. With INT8
ConvRot weights the down-projection then rotates and quantizes that product again. On XPU the
chunk halves are strided views, so the gate multiply alone is 6.2 ms per layer at batch 58 x 690
tokens (a third of the layer's non-GEMM time), plus the separate rotate+quantize pass.
omni_xpu_kernel's quantize_int8_convrot_silu_mul_split does SiLU*mul + Hadamard + rowwise INT8 in
one pass over the dense [M, 2K] proj output; phase-9 added gate_second=1 for Music 3's [up | gate]
row order. The result goes straight to the prequantized INT8 GEMM.

Only for INT8 ConvRot-256 down-projection weights resident on the XPU with no weight/bias
functions attached (LoRA, lowvram streaming) and outside training; everything else, including a
kernel without gate_second, takes the stock forward.
OMNIXPU_MUSIC3_GLU=0 disables. Deployed as ComfyUI-OmniXPU/fixes/music3_glu_fused.py.
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
    if os.environ.get("OMNIXPU_MUSIC3_GLU", "1") == "0":
        return False, "disabled by OMNIXPU_MUSIC3_GLU=0"
    try:
        import comfy.ldm.minimax_music.dit as m3
        import comfy.model_management
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
        from omni_xpu_kernel import int8 as oint8
    except ImportError as e:
        return False, f"import failed: {e}"
    native = oint8._get_native()
    fn = getattr(native, "quantize_int8_convrot_silu_mul_split", None)
    if fn is None or "gate_second" not in (fn.__doc__ or ""):
        return False, "omni_xpu_kernel has no gate_second (needs the phase-9 build)"
    if not hasattr(oint8, "int8_linear_prequantized"):
        return False, "omni_xpu_kernel has no int8_linear_prequantized"
    split = int(os.environ.get("OMNIXPU_MUSIC3_GLU_SPLIT", "16"))

    _orig_forward = m3.FeedForward.forward
    stats = {"fused": 0, "fallback": 0, "error": 0}

    def _plain(lin, x):
        w = getattr(lin, "weight", None)
        if not (
            isinstance(w, QuantizedTensor)
            and w._layout_cls == "TensorWiseINT8Layout"
            and getattr(w._params, "convrot", False)
            and not getattr(w._params, "transposed", False)
            and int(getattr(w._params, "convrot_groupsize", 256)) == 256
            and w.device == x.device
            and not getattr(lin, "weight_function", None)
            and not getattr(lin, "bias_function", None)
            and not getattr(lin, "_full_precision_mm", False)
            and not getattr(lin, "comfy_force_forward", False)
        ):
            return None
        q, s = TensorWiseINT8Layout.get_plain_tensors(w)
        if q.dim() != 2 or not q.is_contiguous() or q.shape[1] % 256 != 0:
            return None
        return q, s

    def forward(self, x):
        if not (
            not comfy.model_management.in_training
            and x.device.type == "xpu"
            and x.dtype in (torch.float16, torch.bfloat16)
            and len(self.ff) == 3
        ):
            stats["fallback"] += 1
            return _orig_forward(self, x)
        lin = self.ff[2]
        plain = _plain(lin, x)
        if plain is None:
            stats["fallback"] += 1
            return _orig_forward(self, x)
        p = self.ff[0].proj(x)
        try:
            if p.shape[-1] != 2 * plain[0].shape[1] or p.dtype != x.dtype:
                raise ValueError("unexpected GLU width or dtype")
            bias = getattr(lin, "bias", None)
            if bias is not None:
                bias = comfy.model_management.cast_to(bias, dtype=p.dtype, device=p.device)
            x_int8, x_scale = fn(p.reshape(-1, p.shape[-1]), 256, split, 1)
            out = oint8.int8_linear_prequantized(x_int8, x_scale, plain[0], plain[1], bias, p.dtype)
            stats["fused"] += 1
            return out.reshape(*p.shape[:-1], out.shape[-1])
        except Exception as e:  # noqa: BLE001
            if stats["error"] == 0:
                log.warning("[OmniXPU] Music 3 fused GLU failed, using stock: %s", e)
            stats["error"] += 1
            value, gate = p.chunk(2, dim=-1)
            return lin(self.ff[1](value * torch.nn.functional.silu(gate)))

    m3.FeedForward.forward = forward
    forward.stats = stats
    _applied = True
    log.info("[OmniXPU] Music 3 GLU fused into the INT8 down-projection (row_split=%d)", split)
    return True, None
