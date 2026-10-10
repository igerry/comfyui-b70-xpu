"""ComfyUI 0.39 compatibility for ComfyUI-OmniXPU (idempotent).
 1. hook fixes/xpu_int8_compute.py first in __init__.py
 2. DiTBlock.forward gained attention=None (sparse-attention patches hand in a replacement
    attention callable); the two fused forwards take it and defer to the stock path when set.
usage: patch_omnixpu_039.py <ComfyUI-OmniXPU dir>
"""
import sys
from pathlib import Path
d = Path(sys.argv[1])

init = d / "__init__.py"
s = init.read_text(encoding="utf-8")
if "xpu_int8_compute" not in s:
    anchor = "        # Phase-1 (2026-09-05): pinned host memory for model offload on XPU\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, '''        # 2026-10-09: ComfyUI 0.39 treats INT8 as emulated on XPU unless told otherwise.
        # OMNIXPU_INT8_COMPUTE=0 disables.
        try:
            i8fix = _load("fixes/xpu_int8_compute.py", f"{_PKG}.fixes.xpu_int8_compute")
            ok, why = i8fix.apply()
            if not ok:
                log.info("[OmniXPU] XPU INT8 compute not enabled: %s", why)
        except Exception as e:
            log.warning("[OmniXPU] XPU INT8 compute fix failed: %s", e)

''' + anchor)
    init.write_text(s, encoding="utf-8"); print("hooked", init)
else:
    print("skip", init)

for name in ("h3_block_fused.py", "h3_adaln_fused.py"):
    f = d / "fixes" / name
    s = f.read_text(encoding="utf-8")
    if "attention=None" in s:
        print("skip", f); continue
    old_def = "    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):\n        try:\n            if (\n"
    new_def = "    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):\n        try:\n            if (\n                attention is None and\n"
    old_ret = "        return _orig_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)\n"
    new_ret = ("        if attention is not None:\n"
               "            return _orig_forward(self, x, t_emb, mod_segments, rope_freqs,\n"
               "                                 transformer_options=transformer_options, attention=attention)\n" + old_ret)
    assert s.count(old_def) == 1 and s.count(old_ret) == 1, name
    f.write_text(s.replace(old_def, new_def).replace(old_ret, new_ret), encoding="utf-8"); print("patched", f)
