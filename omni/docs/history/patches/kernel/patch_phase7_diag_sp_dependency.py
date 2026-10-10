#!/usr/bin/env python3
"""Phase-7 (diagnostic only): break the S -> P dependency between the two GEMMs.

Everything cheap has been ruled out for the 3-cycles-per-DPAS gap against a sustained GEMM
(53.8 vs 65.6 TFLOPS/GHz): softmax arithmetic, barrier period, K load format, Q residency,
tile shape, GRF mode and the AOT device target (knowledge/PHASE3.md).

What is left is the shape of the dataflow itself. Per KV block the mainloop runs
    GEMM1 (32 DPAS) -> softmax (whole-S row reductions) -> GEMM2 (32 DPAS)
and GEMM2's A operand is GEMM1's output, so the XMX pipe drains twice per block. A plain GEMM
has one dependency-free accumulation chain and never drains. DIAG_NO_SOFTMAX did not test this:
it removed the softmax *math* but kept `reorder(tSrS, tArP)`, so the dependency stayed.

  -DCUTE_FMHA_DIAG_NO_SP_DEP=1   feed GEMM2 from a register fragment that is never written from
                                 S, so the two GEMM chains are independent.

WRONG NUMBERS by construction. If this recovers most of the gap, the remaining 18% is inherent to
flash attention's dataflow on this hardware and no amount of tile tuning will reach GEMM throughput.
Idempotent; backup *.pre-phase7."""
import sys
from pathlib import Path

MLS = [Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"),
       Path("/root/omnibuild/src/build/temp.linux-x86_64-cpython-313/cute_bmg_include_overlay/"
            "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")]
if len(sys.argv) > 1:
    MLS = [Path(p) for p in sys.argv[1:]]

OLD = """      reorder(tSrS, tArP);
"""
NEW = """#if defined(CUTE_FMHA_DIAG_NO_SP_DEP) && CUTE_FMHA_DIAG_NO_SP_DEP
      /* phase-7: leave tArP alone so GEMM2 does not wait on GEMM1 + softmax. Timing only. */
#else
      reorder(tSrS, tArP);
#endif
"""
for ml in MLS:
    if not ml.exists():
        print(f"[missing] {ml}"); continue
    s = ml.read_text()
    if "CUTE_FMHA_DIAG_NO_SP_DEP" in s:
        print(f"[skip] {ml}"); continue
    assert s.count(OLD) == 1, f"reorder anchor count {s.count(OLD)} in {ml}"
    bak = ml.with_name(ml.name + ".pre-phase7")
    if not bak.exists():
        bak.write_text(s)
    ml.write_text(s.replace(OLD, NEW)); print(f"[ok] {ml}")
