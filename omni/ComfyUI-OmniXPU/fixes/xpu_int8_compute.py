"""Tell ComfyUI that INT8 quantized matmul is native on XPU (2026-10-09).

ComfyUI 0.39 added model_management.supports_int8_compute(); it returns False for Intel XPU
because stock comfy_kitchen has no XPU backend and torch._int_mm is not implemented there.
With False, pick_operations() lists int8_tensorwise / convrot_w4a4 as "emulated" and every
INT8 Linear sets _full_precision_mm: weights are dequantized to bf16 per call and the GEMM
runs in float. On H3 that is 6.2 -> 7.3 s/step (480x864, 124 frames) and it bypasses the
whole omni_xpu_kernel INT8 path. With the grafted XPU backend registered the INT8 ops are
native, so answer True for XPU devices. OMNIXPU_INT8_COMPUTE=0 disables.
Deployed as ComfyUI-OmniXPU/fixes/xpu_int8_compute.py; must run before any model is loaded.
"""
import logging
import os

log = logging.getLogger("ComfyUI-OmniXPU")
_applied = False


def apply():
    global _applied
    if _applied:
        return True, "already applied"
    if os.environ.get("OMNIXPU_INT8_COMPUTE", "1") == "0":
        return False, "disabled by OMNIXPU_INT8_COMPUTE=0"
    try:
        import comfy.model_management as mm
        import comfy_kitchen as ck
    except ImportError as e:
        return False, f"import failed: {e}"
    if not hasattr(mm, "supports_int8_compute"):
        return False, "this ComfyUI has no supports_int8_compute (nothing to fix)"
    caps = ck.registry._capabilities.get("xpu", set())
    if not ck.registry.is_available("xpu") or "int8_linear" not in caps:
        return False, "comfy_kitchen xpu backend has no int8_linear"

    _orig = mm.supports_int8_compute

    def supports_int8_compute(device=None):
        if device is None:
            device = mm.get_torch_device()
        if getattr(device, "type", None) == "xpu":
            return True
        return _orig(device)

    mm.supports_int8_compute = supports_int8_compute
    _applied = True
    log.info("[OmniXPU] INT8 quantized matmul reported as native on XPU")
    return True, None
