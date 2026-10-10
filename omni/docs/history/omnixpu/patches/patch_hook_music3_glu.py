"""Hook fixes/music3_glu_fused.py into ComfyUI-OmniXPU/__init__.py (idempotent)."""
import sys
from pathlib import Path
init = Path(sys.argv[1]) / "__init__.py"
s = init.read_text(encoding="utf-8")
if "music3_glu_fused" in s:
    print("skip"); sys.exit(0)
anchor = "        # Load diagnostics node\n"
assert s.count(anchor) == 1
s = s.replace(anchor, '''        # 2026-10-09: Music 3 GLU gate fused into the INT8 down-projection. OMNIXPU_MUSIC3_GLU=0 disables.
        try:
            m3glu = _load("fixes/music3_glu_fused.py", f"{_PKG}.fixes.music3_glu_fused")
            ok, why = m3glu.apply()
            if not ok:
                log.info("[OmniXPU] Music 3 fused GLU not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] Music 3 fused GLU fix failed: %s", e)

''' + anchor)
init.write_text(s, encoding="utf-8"); print("hooked", init)
