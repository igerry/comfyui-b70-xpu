#!/usr/bin/env python3
"""Phase-1g: hook fixes/xpu_pinned_memory.py into ComfyUI-OmniXPU/__init__.py (after the
other patches). Backup: __init__.py.pre-phase1"""
import shutil, sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py")
s = p.read_text()
if "xpu_pinned_memory" in s:
    print("[skip]"); sys.exit(0)
old = "        patches.apply_all_patches(config_mod.config)\n"
new = old + (
    "\n        # Phase-1 (2026-09-05): pinned host memory for model offload on XPU\n"
    "        # (ComfyUI only enables it for CUDA/ROCm). OMNIXPU_PIN_MEMORY=0 disables.\n"
    "        try:\n"
    "            pinfix = _load(\"fixes/xpu_pinned_memory.py\", f\"{_PKG}.fixes.xpu_pinned_memory\")\n"
    "            ok, why = pinfix.apply()\n"
    "            if not ok:\n"
    "                log.info(\"[OmniXPU] XPU pinned memory not enabled: %s\", why)\n"
    "        except Exception as e:\n"
    "            log.warning(\"[OmniXPU] XPU pinned memory fix failed: %s\", e)\n")
assert s.count(old) == 1
bak = p.with_name(p.name + ".pre-phase1")
if not bak.exists():
    shutil.copy2(p, bak)
p.write_text(s.replace(old, new)); print("[ok] hook added")
