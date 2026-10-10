#!/usr/bin/env python3
"""Phase-5: attack the 18% gap between the FMHA mainloop and a sustained bf16 GEMM on the same
silicon (53.8 vs 65.6 TFLOPS/GHz, both power-limited at ~230 W — bench/roofline_bench.py).

The gap is NOT softmax: DIAG_NO_SOFTMAX takes the same number of cycles as the default
(66.0 ms x 2133 MHz vs 71.6 x 1950), it only wins clock by burning less energy. So the bubbles
are in the DPAS/load pipeline. Two candidates, both macro-gated:

  -DCUTE_FMHA_BARRIER_EVERY=N   Split-barrier the workgroup every N KV blocks instead of every
                                one. Every block (N=1, upstream) drains the pipe 511 times per Q
                                tile; never (DIAG_NO_BARRIER) costs 44% because the subgroups
                                drift apart and lose L1 locality. N=2/4/8 tests the middle.
                                Numerically exact — the barrier only affects scheduling.

  -DCUTE_FMHA_DIAG_K_AS_V=1     Timing-only: describe K with V's stride pattern so the K tile is
                                fetched with the fast VNNI block load (d16v) that V uses, instead
                                of the transposed load (d32t) forced by K being contracted over the
                                contiguous head-dim. Produces WRONG NUMBERS; it measures what a
                                pre-transposed K would buy before anyone writes the transpose pass.

Idempotent; backups *.pre-phase5."""
import sys
from pathlib import Path

CF = Path("/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
MLS = [Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"),
       Path("/root/omnibuild/src/build/temp.linux-x86_64-cpython-313/cute_bmg_include_overlay/"
            "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")]
if len(sys.argv) > 1:
    CF = Path(sys.argv[1]); MLS = [Path(p) for p in sys.argv[2:]]

# ---- mainloop: parameterised barrier period -------------------------------------
ARRIVE_OLD = """      /* Split barrier to keep threads together */
#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      barrier_arrive(ScopeWorkgroup);
#endif
"""
ARRIVE_NEW = """      /* Split barrier to keep threads together */
#ifndef CUTE_FMHA_BARRIER_EVERY
#define CUTE_FMHA_BARRIER_EVERY 1
#endif
#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      /* phase-5: K is workgroup-uniform, so this predicate is uniform and the split barrier
         stays correctly paired; N>1 lets the subgroups run ahead between sync points. */
      const bool do_barrier = (CUTE_FMHA_BARRIER_EVERY == 1) || (K % CUTE_FMHA_BARRIER_EVERY == 0);
      if (do_barrier) barrier_arrive(ScopeWorkgroup);
#endif
"""
WAIT_OLD = """#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      barrier_wait(ScopeWorkgroup);
#endif
    };
"""
WAIT_NEW = """#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      if (do_barrier) barrier_wait(ScopeWorkgroup);
#endif
    };
"""
for ml in MLS:
    if not ml.exists():
        print(f"[missing] {ml}"); continue
    s = ml.read_text()
    if "CUTE_FMHA_BARRIER_EVERY" in s:
        print(f"[skip] {ml}"); continue
    assert s.count(ARRIVE_OLD) == 1, f"arrive anchor missing in {ml} (phase-3a applied?)"
    assert s.count(WAIT_OLD) == 1, f"wait anchor missing in {ml}"
    bak = ml.with_name(ml.name + ".pre-phase5")
    if not bak.exists():
        bak.write_text(s)
    ml.write_text(s.replace(ARRIVE_OLD, ARRIVE_NEW).replace(WAIT_OLD, WAIT_NEW))
    print(f"[ok] {ml}")

# ---- cute_fmha_torch.cpp: K-as-V stride diagnostic --------------------------------
s = CF.read_text()
if "CUTE_FMHA_DIAG_K_AS_V" in s:
    print("[skip] cute_fmha_torch.cpp")
else:
    old = """  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
"""
    new = """  using StrideQ = Stride<int, _1, int, int>;
#if defined(CUTE_FMHA_DIAG_K_AS_V) && CUTE_FMHA_DIAG_K_AS_V
  /* phase-5 timing probe: pretend K's sequence mode is the contiguous one, the way V's is, so
     make_block_2d_copy_B emits the VNNI load (d16v) instead of the transposing load (d32t).
     The kernel then reads the wrong elements — use only to price a pre-transposed K. */
  using StrideK = Stride<_1, int, int, int>;
#else
  using StrideK = Stride<int, _1, int, int>;
#endif
"""
    assert s.count(old) == 1, "stride type anchor not found"
    s = s.replace(old, new)
    # The stride *value* built in run_d128_tile has to match the swapped type.
    old2 = """  typename KT::StrideK stride_K =
      cute::make_stride(
          checked_int(k_stride_seq, "K sequence stride"), _1{},
          checked_int(k_stride_head, "K head stride"),
          checked_int(k_stride_batch, "K batch stride"));
"""
    new2 = """#if defined(CUTE_FMHA_DIAG_K_AS_V) && CUTE_FMHA_DIAG_K_AS_V
  typename KT::StrideK stride_K =
      cute::make_stride(
          _1{}, checked_int(k_stride_seq, "K sequence stride"),
          checked_int(k_stride_head, "K head stride"),
          checked_int(k_stride_batch, "K batch stride"));
#else
  typename KT::StrideK stride_K =
      cute::make_stride(
          checked_int(k_stride_seq, "K sequence stride"), _1{},
          checked_int(k_stride_head, "K head stride"),
          checked_int(k_stride_batch, "K batch stride"));
#endif
"""
    assert s.count(old2) == 1, "stride value anchor not found"
    s = s.replace(old2, new2)
    bak = CF.with_name(CF.name + ".pre-phase5")
    if not bak.exists():
        bak.write_text(CF.read_text())
    CF.write_text(s); print(f"[ok] {CF}")
