#!/usr/bin/env python3
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py")
s = p.read_text()
if "h3_swiglu_epilogue" in s:
    print("[skip]"); sys.exit(0)
old = "        except Exception as e:\n            log.warning(\"[OmniXPU] XPU pinned memory fix failed: %s\", e)\n"
new = old + (
    "\n        # Phase-2 (2026-09-06): H3 MLP SwiGLU inside the fc1 GEMM epilogue. OMNIXPU_SWIGLU_EPILOGUE=0 disables.\n"
    "        try:\n"
    "            swfix = _load(\"fixes/h3_swiglu_epilogue.py\", f\"{_PKG}.fixes.h3_swiglu_epilogue\")\n"
    "            ok, why = swfix.apply()\n"
    "            if not ok:\n"
    "                log.info(\"[OmniXPU] SwiGLU epilogue not enabled: %s\", why)\n"
    "        except Exception as e:\n"
    "            log.warning(\"[OmniXPU] SwiGLU epilogue fix failed: %s\", e)\n")
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print("[ok] hook added")
