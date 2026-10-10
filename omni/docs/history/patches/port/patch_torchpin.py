# Relax ComfyUI-OmniXPU's hard-coded torch-minor gate.
# Upstream routes to CUTE only when torch is exactly 2.11 (the minor Intel
# validated). The kernels here are built against the running torch
# (omni_xpu_kernel.__torch_version__), so allow the built minor as well.
import re, sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
hits = re.findall(r"_torch_major_minor\(\) == \(2, 11\)", s)
s = s.replace("_torch_major_minor() == (2, 11)", "_torch_major_minor() in _OMNI_TORCH_MINORS")
anchor = "def _torch_major_minor():"
assert anchor in s
s = s.replace(anchor,
    "# Torch minors this adapter may route to CUTE on. Upstream pins (2, 11); the\n"
    "# kernels are built against the running torch, so its minor is allowed too.\n"
    "_OMNI_TORCH_MINORS = {(2, 11), (2, 12), (2, 13)}\n\n\n" + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("replaced", len(hits), "occurrence(s)")
