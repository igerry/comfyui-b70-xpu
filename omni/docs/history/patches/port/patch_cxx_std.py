"""torch 2.14 headers require C++20 ("C++20 or later compatible compiler is required to use PyTorch");
setup.py hard-codes -std=c++17 for the Linux extension and the CUTE library. Make it OMNI_CXX_STD
(default c++17, so the torch 2.13 build is unchanged). Idempotent.
usage: patch_cxx_std.py /root/omnibuild/src/setup.py"""
import re, sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
if "OMNI_CXX_STD" in s:
    print("skip"); sys.exit(0)
n = s.count('"-std=c++17"')
assert n == 2, n
s = s.replace('"-std=c++17"', '"-std=" + os.environ.get("OMNI_CXX_STD", "c++17")')
assert re.search(r"^import os", s, re.M)
open(p, "w", encoding="utf-8").write(s); print("patched", n, "sites")
