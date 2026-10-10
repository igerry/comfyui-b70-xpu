"""Hook fixes/music3_window_batch.py into ComfyUI-OmniXPU/__init__.py (idempotent)."""
import sys
from pathlib import Path
init = Path(sys.argv[1]) / "__init__.py"
s = init.read_text(encoding="utf-8")
if "music3_window_batch" in s:
    print("skip"); sys.exit(0)
anchor = "        # Load diagnostics node\n"
assert s.count(anchor) == 1
s = s.replace(anchor, '''        # 2026-10-09: Music 3 condition windows as one batch. OMNIXPU_MUSIC3_BATCH=0 disables.
        try:
            m3fix = _load("fixes/music3_window_batch.py", f"{_PKG}.fixes.music3_window_batch")
            ok, why = m3fix.apply()
            if not ok:
                log.info("[OmniXPU] Music 3 window batching not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] Music 3 window batching fix failed: %s", e)

''' + anchor)
init.write_text(s, encoding="utf-8"); print("hooked", init)
