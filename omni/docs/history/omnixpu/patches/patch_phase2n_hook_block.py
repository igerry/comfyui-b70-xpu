#!/usr/bin/env python3
"""Hook fixes/h3_block_fused.py after the AdaLN fix (it re-patches DiTBlock.forward; the AdaLN fix's
forward becomes the fallback when the block fix bails out)."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py")
s = p.read_text()
if "h3_block_fused" in s:
    print("[skip]"); sys.exit(0)
old = "        except Exception as e:\n            log.warning(\"[OmniXPU] fused AdaLN fix failed: %s\", e)\n"
new = old + (
    "\n        # Phase-2 (2026-09-06): whole-block producer-fused INT8 path. OMNIXPU_BLOCK_FUSED=0 disables.\n"
    "        try:\n"
    "            bfix = _load(\"fixes/h3_block_fused.py\", f\"{_PKG}.fixes.h3_block_fused\")\n"
    "            ok, why = bfix.apply()\n"
    "            if not ok:\n"
    "                log.info(\"[OmniXPU] block fusion not enabled: %s\", why)\n"
    "        except Exception as e:\n"
    "            log.warning(\"[OmniXPU] block fusion fix failed: %s\", e)\n")
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print("[ok] hook added")
