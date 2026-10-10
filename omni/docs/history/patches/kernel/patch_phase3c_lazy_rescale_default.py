#!/usr/bin/env python3
"""Phase-3c: make the lazy O-rescale (phase-3a) the default for BMG builds.
Measured: H3 DiT route 71.7 -> 70.4 ms, output bitwise identical (knowledge/PHASE3.md).
Disable with -DCUTE_FMHA_LAZY_RESCALE=0.

The define MUST come before cute_fmha_torch.cpp's own include of the FMHA mainloop (line ~38) —
the kernel header is included later, so anchoring there silently produced a wheel without the
optimisation (first attempt did exactly that: 71.35 ms, i.e. baseline). Anchor on the first
include in the file instead. Also tightens the phase-3a guards to honour a 0 value.
Idempotent, and repairs a previously misplaced block."""
import sys
from pathlib import Path

CF = Path("/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
MLS = [Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"),
       Path("/root/omnibuild/src/build/temp.linux-x86_64-cpython-313/cute_bmg_include_overlay/"
            "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")]
if len(sys.argv) > 1:
    CF = Path(sys.argv[1]); MLS = [Path(p) for p in sys.argv[2:]]

BLOCK = ("// phase-3c: lazy online-softmax rescale on by default for BMG (bitwise identical, -1.9% on H3 DiT).\n"
         "#if defined(OMNI_XPU_ARCH_BMG) && !defined(CUTE_FMHA_LAZY_RESCALE)\n"
         "#define CUTE_FMHA_LAZY_RESCALE 1\n"
         "#endif\n")
ANCHOR = "#include <ATen/ATen.h>\n"

s = CF.read_text()
had = BLOCK in s
s = s.replace(BLOCK, "")                       # drop any earlier (possibly misplaced) copy
assert s.count(ANCHOR) == 1, f"anchor {ANCHOR!r} not unique in {CF}"
s = s.replace(ANCHOR, BLOCK + ANCHOR)
# verify it now precedes the mainloop include
i_def = s.index("#define CUTE_FMHA_LAZY_RESCALE 1")
i_ml = s.index('#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"')
assert i_def < i_ml, "define still lands after the mainloop include"
CF.write_text(s)
print(f"[ok] {CF}" + (" (moved existing block)" if had else ""))

for ml in MLS:
    if not ml.exists():
        print(f"[missing] {ml}"); continue
    t = ml.read_text()
    n = t.count("#if defined(CUTE_FMHA_LAZY_RESCALE)\n") + t.count("#elif defined(CUTE_FMHA_LAZY_RESCALE)\n")
    if n == 0:
        print(f"[skip] {ml} (already value-guarded)"); continue
    t = t.replace("#if defined(CUTE_FMHA_LAZY_RESCALE)\n", "#if defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE\n")
    t = t.replace("#elif defined(CUTE_FMHA_LAZY_RESCALE)\n", "#elif defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE\n")
    ml.write_text(t); print(f"[ok] {ml} ({n} guards)")
