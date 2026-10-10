#!/usr/bin/env python3
"""Phase-1f: make KV64 + 3 stages the default for the MiniMax H3 VAE d64 route on BMG."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/setup.py"
s = open(p).read()
old = '                    *os.environ.get("OMNI_CUTE_EXTRA_FLAGS", "").split(),\n'
new = (old +
       '                    # B70 (2026-09-05): the MiniMax H3 VAE d64 route is 16% faster\n'
       '                    # with a 64-wide KV tile and 3 pipeline stages; the DiT d128\n'
       '                    # route keeps the shipped KV32/2-stage config (KV64 lost 8%).\n'
       '                    *(["-DCUTE_FMHA_VAE_KV=64", "-DCUTE_FMHA_VAE_STAGES=3"]\n'
       '                      if BUILD_XPU_TARGET == "bmg" else []),\n')
if "CUTE_FMHA_VAE_KV=64" in s:
    print("[skip]")
else:
    assert s.count(old) == 1
    open(p, "w").write(s.replace(old, new)); print("[ok] setup.py VAE defaults")
