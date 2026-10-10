# -*- coding: utf-8 -*-
"""Graft the XPU backend from comfy-kitchen-xpu (0.2.28 fork) onto stock
comfy-kitchen 0.2.31, so upstream ComfyUI's newer API expectations are met
while Intel's XPU dispatch stays available.

usage: graft_kitchen.py <stock_pkg_dir> <fork_pkg_dir> <out_pkg_dir>
"""
import re, shutil, sys
from pathlib import Path
stock, fork, out = map(Path, sys.argv[1:4])
if out.exists(): shutil.rmtree(out)
shutil.copytree(stock, out)
shutil.copytree(fork / "backends" / "xpu", out / "backends" / "xpu")

# 1. registry: the fork's xpu backend / OmniXPU use get_backend_override()
reg = (out / "registry.py").read_text(encoding="utf-8")
if "def get_backend_override" not in reg:
    anchor = "    def set_priority(self, priority_list: list[str]):"
    assert anchor in reg
    reg = reg.replace(anchor, '''    def get_backend_override(self) -> str | None:
        """Return the active thread-local backend override, if any."""
        return getattr(self._thread_local, "backend_override", None)

''' + anchor)
    (out / "registry.py").write_text(reg, encoding="utf-8")

# 2. __init__: load xpu the same way stock loads hip, and prefer it
init = (out / "__init__.py").read_text(encoding="utf-8")
hip_block_end = 'registry.mark_unavailable("hip", "PyTorch ROCm/HIP runtime not available")\n'
assert hip_block_end in init
xpu_block = hip_block_end + '''
# Intel XPU backend (grafted from comfy-kitchen-xpu). Loaded only when an XPU
# is present; on any failure it is marked unavailable and dispatch falls back
# to triton/eager exactly as before.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    try:
        from .backends import xpu as _xpu_backend  # noqa: F401
        if registry.is_available("xpu"):
            registry.set_priority(["xpu", "triton", "eager"])
    except Exception as _xpu_error:  # pragma: no cover
        registry.mark_unavailable("xpu", f"comfy_kitchen xpu backend failed: {_xpu_error}")
else:
    registry.mark_unavailable("xpu", "PyTorch XPU runtime not available")
'''
init = init.replace(hip_block_end, xpu_block)
assert re.search(r"^import torch", init, re.M), "torch must be imported before the xpu block"
(out / "__init__.py").write_text(init, encoding="utf-8")
print("grafted ->", out)
