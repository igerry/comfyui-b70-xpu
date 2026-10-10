# Make the hard-coded oneDNN pin (3.9.1 / onednn==2025.3.0) overridable so the
# extension can be built against the oneDNN that ships with the 2026.0 runtime.
import sys, re
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
old = "VALIDATED_ONEDNN_VERSION = (3, 9, 1)"
assert old in s
s = s.replace(old, 'VALIDATED_ONEDNN_VERSION = tuple(int(x) for x in os.environ.get("OMNI_ONEDNN_VERSION", "3.9.1").split("."))')
old2 = "\"onednn==2025.3.0; platform_system == 'Linux' and platform_machine == 'x86_64'\","
assert old2 in s
s = s.replace(old2, "\"onednn==\" + os.environ.get(\"OMNI_ONEDNN_PIP_VERSION\", \"2025.3.0\") + \"; platform_system == 'Linux' and platform_machine == 'x86_64'\",")
# os must be imported before first use (it is, at top) - sanity
assert re.search(r"^import os", s, re.M)
open(p, "w", encoding="utf-8").write(s); print("onednn pin patched")
