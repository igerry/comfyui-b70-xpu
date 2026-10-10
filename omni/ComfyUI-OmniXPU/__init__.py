import logging
import importlib.util
import os
import sys

log = logging.getLogger("ComfyUI-OmniXPU")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Resolve package directory (works regardless of how we're loaded)
_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = "ComfyUI-OmniXPU"


def _load(rel_path, mod_name):
    """Load a .py file as a named module."""
    fpath = os.path.join(_DIR, *rel_path.split("/"))
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, "__init__.py")
    elif not fpath.endswith(".py"):
        fpath += ".py"
    search = [os.path.dirname(fpath)] if os.path.basename(fpath) == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(mod_name, fpath, submodule_search_locations=search)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        log.info("[OmniXPU] No Intel XPU detected, skipping all patches")
    else:
        # Load probe and run capability check
        probe = _load("probe.py", f"{_PKG}.probe")
        probe.probe()

        # Load config and patches
        _load("config.py", f"{_PKG}.config")
        patches = _load("patches", f"{_PKG}.patches")

        config_mod = sys.modules[f"{_PKG}.config"]
        patches.apply_all_patches(config_mod.config)

        # 2026-10-09: ComfyUI 0.39 treats INT8 as emulated on XPU unless told otherwise.
        # OMNIXPU_INT8_COMPUTE=0 disables.
        try:
            i8fix = _load("fixes/xpu_int8_compute.py", f"{_PKG}.fixes.xpu_int8_compute")
            ok, why = i8fix.apply()
            if not ok:
                log.info("[OmniXPU] XPU INT8 compute not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] XPU INT8 compute fix failed: %s", e)

        # Phase-1 (2026-09-05): pinned host memory for model offload on XPU
        # (ComfyUI only enables it for CUDA/ROCm). OMNIXPU_PIN_MEMORY=0 disables.
        try:
            pinfix = _load("fixes/xpu_pinned_memory.py", f"{_PKG}.fixes.xpu_pinned_memory")
            ok, why = pinfix.apply()
            if not ok:
                log.info("[OmniXPU] XPU pinned memory not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] XPU pinned memory fix failed: %s", e)

        # Phase-2 (2026-09-06): H3 DiTBlock RMSNorm with fused AdaLN scale/shift. OMNIXPU_ADALN_FUSED=0 disables.
        try:
            adfix = _load("fixes/h3_adaln_fused.py", f"{_PKG}.fixes.h3_adaln_fused")
            ok, why = adfix.apply()
            if not ok:
                log.info("[OmniXPU] fused AdaLN not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] fused AdaLN fix failed: %s", e)

        # Phase-2 (2026-09-06): whole-block producer-fused INT8 path. OMNIXPU_BLOCK_FUSED=0 disables.
        try:
            bfix = _load("fixes/h3_block_fused.py", f"{_PKG}.fixes.h3_block_fused")
            ok, why = bfix.apply()
            if not ok:
                log.info("[OmniXPU] block fusion not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] block fusion fix failed: %s", e)

        # 2026-10-09: Music 3 condition windows as one batch. OMNIXPU_MUSIC3_BATCH=0 disables.
        try:
            m3fix = _load("fixes/music3_window_batch.py", f"{_PKG}.fixes.music3_window_batch")
            ok, why = m3fix.apply()
            if not ok:
                log.info("[OmniXPU] Music 3 window batching not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] Music 3 window batching fix failed: %s", e)

        # 2026-10-09: Music 3 partial RoPE on dense q/k. OMNIXPU_MUSIC3_ROPE=0 disables.
        try:
            m3rope = _load("fixes/music3_rope_dense.py", f"{_PKG}.fixes.music3_rope_dense")
            ok, why = m3rope.apply()
            if not ok:
                log.info("[OmniXPU] Music 3 dense RoPE not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] Music 3 dense RoPE fix failed: %s", e)

        # 2026-10-09: Music 3 GLU gate fused into the INT8 down-projection. OMNIXPU_MUSIC3_GLU=0 disables.
        try:
            m3glu = _load("fixes/music3_glu_fused.py", f"{_PKG}.fixes.music3_glu_fused")
            ok, why = m3glu.apply()
            if not ok:
                log.info("[OmniXPU] Music 3 fused GLU not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] Music 3 fused GLU fix failed: %s", e)

        # Load diagnostics node
        _load("nodes/__init__.py", f"{_PKG}.nodes")
        diag = _load("nodes/diagnostics.py", f"{_PKG}.nodes.diagnostics")
        NODE_CLASS_MAPPINGS["OmniXPUStatus"] = diag.OmniXPUStatus
        NODE_DISPLAY_NAME_MAPPINGS["OmniXPUStatus"] = "OmniXPU Status"

except Exception as e:
    import traceback
    log.error("[OmniXPU] Initialization failed: %s\n%s", e, traceback.format_exc())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
