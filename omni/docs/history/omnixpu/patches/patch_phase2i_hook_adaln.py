#!/usr/bin/env python3
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py")
s = p.read_text()
if "h3_adaln_fused" in s:
    print("[skip]"); sys.exit(0)
old = "        except Exception as e:\n            log.warning(\"[OmniXPU] XPU pinned memory fix failed: %s\", e)\n"
new = old + (
    "\n        # Phase-2 (2026-09-06): H3 DiTBlock RMSNorm with fused AdaLN scale/shift. OMNIXPU_ADALN_FUSED=0 disables.\n"
    "        try:\n"
    "            adfix = _load(\"fixes/h3_adaln_fused.py\", f\"{_PKG}.fixes.h3_adaln_fused\")\n"
    "            ok, why = adfix.apply()\n"
    "            if not ok:\n"
    "                log.info(\"[OmniXPU] fused AdaLN not enabled: %s\", why)\n"
    "        except Exception as e:\n"
    "            log.warning(\"[OmniXPU] fused AdaLN fix failed: %s\", e)\n")
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print("[ok] hook added")
