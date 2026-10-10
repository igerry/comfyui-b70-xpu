#!/usr/bin/env python3
"""Phase-2m: add Intel's Newton refinement of the SiLU divide to the fused fc2 producer kernel so the
bf16 rounding boundary matches the stock exact path more often."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp")
s = p.read_text()
old = """    simd<float, GroupSize> silu =
        x / (1.0f + sycl::ext::intel::esimd::exp(-x));
    simd<InputT, GroupSize> silu_rounded = silu;
"""
new = """    simd<float, GroupSize> denominator = 1.0f + sycl::ext::intel::esimd::exp(-x);
    simd<float, GroupSize> silu = x / denominator;
    // Same refinement as h3_silu_mul.cpp: one Newton step on the divide for finite positive x.
    simd<float, GroupSize> refined = silu + (x - silu * denominator) / denominator;
    silu.merge(refined, x > 0.0f);
    simd<InputT, GroupSize> silu_rounded = silu;
"""
if new in s:
    print("[skip]")
else:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new)); print("[ok] silu refine")
