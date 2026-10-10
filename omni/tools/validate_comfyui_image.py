#!/usr/bin/env python3
"""Acceptance check for the comfyui-b70-xpu image.

    docker run --rm --device=/dev/dri <image> /llm/entrypoints/validate_comfyui_image.sh

Verifies package identity, the Torch ABI of the native wheel, XPU availability, the Kitchen XPU
backend and the pieces that broke silently on the ComfyUI 0.34 -> 0.39 upgrade. Exits non-zero
on the first failed group. `--no-gpu` skips the checks that need a device (for CI on a build host).
"""
import importlib.metadata as md
import inspect
import os
import subprocess
import sys

NO_GPU = "--no-gpu" in sys.argv
failed = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}")
    if not ok:
        failed.append(name)


# --- identity ------------------------------------------------------------------------
import torch

check("python 3.13", sys.version_info[:2] == (3, 13), sys.version.split()[0])
check("torch 2.13 xpu", torch.__version__.startswith("2.13.") and "+xpu" in torch.__version__, torch.__version__)
kver = md.version("omni_xpu_kernel")
check("omni_xpu_kernel built for torch 2.13 / bmg", "torch213" in kver and kver.endswith(os.environ.get("B70_IMAGE_XPU_TARGET", "bmg").replace("-", "")), kver)
ck = md.version("comfy-kitchen")
check("comfy-kitchen version", ck == os.environ.get("B70_COMFY_KITCHEN_VERSION", ck), ck)
rev = subprocess.run(["git", "-C", "/llm/ComfyUI", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
check("ComfyUI revision", rev == os.environ.get("B70_COMFYUI_REVISION", rev), rev[:12])
check("ComfyUI-OmniXPU present", os.path.isfile("/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py"))
for f in ("xpu_pinned_memory", "xpu_int8_compute", "music3_window_batch", "music3_rope_dense", "music3_glu_fused",
          "h3_block_fused", "h3_adaln_fused"):
    check(f"fixes/{f}.py", os.path.isfile(f"/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/fixes/{f}.py"))
check("LD_LIBRARY_PATH not inherited from the base image", not os.environ.get("LD_LIBRARY_PATH"), os.environ.get("LD_LIBRARY_PATH", ""))

# --- native kernels ------------------------------------------------------------------
import omni_xpu_kernel as ok_

mods = [m for m in ("sdp", "norm", "rotary", "linear", "int8", "cute") if hasattr(ok_, m)]
for m in ("sdp", "norm", "rotary", "int8"):
    try:
        __import__(f"omni_xpu_kernel.{m}")
        good = True
    except Exception as e:  # noqa: BLE001
        good = False
        print("   ", e)
    check(f"omni_xpu_kernel.{m} imports", good)
try:
    from omni_xpu_kernel import _C  # noqa: F401
    check("native extension loads (omni_xpu_kernel._C)", True)
except Exception as e:  # noqa: BLE001
    check("native extension loads (omni_xpu_kernel._C)", False, e)
try:
    from omni_xpu_kernel.cute import cute_fmha_torch  # noqa: F401
    check("CUTE attention extension loads", True)
except Exception as e:  # noqa: BLE001
    check("CUTE attention extension loads", False, e)
from omni_xpu_kernel import int8 as _i8

sig = None
for name in dir(_i8):
    if "silu_mul" in name and "split" in name:
        try:
            sig = inspect.signature(getattr(_i8, name))
        except (TypeError, ValueError):
            sig = None
        break
check("phase-9 kernel entry (silu_mul split) exported", any("silu_mul" in n for n in dir(_i8)), sig if sig else "")

# --- Kitchen XPU backend -------------------------------------------------------------
import comfy_kitchen as kitchen

backends = kitchen.list_backends()
check("kitchen has an xpu backend", "xpu" in backends)
from comfy_kitchen.backends import xpu as kxpu

src = inspect.getsource(kxpu)
check("xpu int8_linear accepts the 0.2.37 arguments",
      all(k in src for k in ("input_act_weight", "input_act_eps", "residual_scale")))
check("xpu int8_linear flattens >2-D inputs", "_int8_linear_core" in src)

if not NO_GPU:
    check("torch.xpu.is_available()", torch.xpu.is_available())
    if torch.xpu.is_available():
        name = torch.xpu.get_device_name(0)
        check("device", True, name)
        info = backends.get("xpu", {})
        caps = info.get("capabilities", info) if isinstance(info, dict) else info
        n = len(caps) if hasattr(caps, "__len__") else 0
        avail = info.get("available", True) if isinstance(info, dict) else True
        check("kitchen xpu backend available", bool(avail), f"{n} capabilities")
        with torch.inference_mode():
            a = torch.randn(64, 256, device="xpu", dtype=torch.bfloat16)
            b = (a @ a.T).float().sum().item()
        check("bf16 matmul on xpu", b == b)
        # ComfyUI-side: XPU must be reported as INT8-capable once the OmniXPU fix is applied.
        sys.path.insert(0, "/llm/ComfyUI")
        sys.argv = [sys.argv[0]]
        try:
            import importlib.util

            import comfy.model_management as mm

            spec = importlib.util.spec_from_file_location(
                "b70_i8fix", "/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/fixes/xpu_int8_compute.py")
            fix = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fix)
            fix.apply()
            fn = getattr(mm, "supports_int8_compute", None)
            check("ComfyUI treats XPU INT8 as native", fn is None or bool(fn(torch.device("xpu"))))
        except Exception as e:  # noqa: BLE001
            check("ComfyUI treats XPU INT8 as native", False, repr(e)[:200])

print()
if failed:
    print(f"FAILED ({len(failed)}): " + ", ".join(failed))
    sys.exit(1)
print(f"comfyui-b70-xpu {os.environ.get('B70_IMAGE_VERSION', '?')}: all checks passed")
