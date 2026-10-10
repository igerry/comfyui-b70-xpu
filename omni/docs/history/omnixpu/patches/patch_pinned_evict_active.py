"""ComfyUI 0.39: model_management.pin_memory grew an evict_active argument (the Qwen-Image 2.1
KV cache and SeedVR call pin_memory(t, evict_active=False)); ensure_pin_registerable takes it
too and now reports whether the budget was met. Our XPU replacement kept the old one-argument
form and died with TypeError on the first Qwen-Image 2.1 step. Idempotent.
usage: patch_pinned_evict_active.py <path to xpu_pinned_memory.py>
"""
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
if "evict_active" in s:
    print("skip", p); sys.exit(0)
a = "    def pin_memory(tensor):\n"
b = "        mm.ensure_pin_registerable(size)\n"
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, "    def pin_memory(tensor, evict_active=True):\n")
s = s.replace(b, "        try:\n"
                 "            ok = mm.ensure_pin_registerable(size, evict_active=evict_active)\n"
                 "        except TypeError:  # ComfyUI < 0.39\n"
                 "            ok = mm.ensure_pin_registerable(size)\n"
                 "        if ok is False:\n"
                 "            return False\n")
p.write_text(s, encoding="utf-8"); print("patched", p)
