#!/usr/bin/env python3
"""Phase-6: hoist only PART of Q into registers.

Phase-2b hoisted all 8 head-dim chunks of Q out of the KV loop and lost 12% to a 6 KB spill
(the O accumulator alone is 128 of the 256 GRF). But Q is reloaded from L1 on every one of the
511 KV blocks, and the measured gap to a sustained GEMM is 3 cycles per DPAS of *memory* stall
(barrier and softmax are both ruled out — knowledge/PHASE3.md sections 6' and 8).

So keep the first N chunks resident and reload the rest: N=2 costs 16 GRF, N=4 costs 32, versus
64 for the full hoist. Math order is unchanged, so the output stays bitwise identical.

  -DCUTE_FMHA_HOIST_Q_PARTIAL=N     N chunks resident (1..8; 0 or unset = upstream behaviour)

Independent of CUTE_FMHA_HOIST_Q (phase-2b, all-or-nothing). Idempotent; backup *.pre-phase6."""
import sys
from pathlib import Path

MLS = [Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"),
       Path("/root/omnibuild/src/build/temp.linux-x86_64-cpython-313/cute_bmg_include_overlay/"
            "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")]
if len(sys.argv) > 1:
    MLS = [Path(p) for p in sys.argv[1:]]

DECL_OLD = """    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
"""
DECL_NEW = """#ifndef CUTE_FMHA_HOIST_Q_PARTIAL
#define CUTE_FMHA_HOIST_Q_PARTIAL 0
#endif
#if CUTE_FMHA_HOIST_Q_PARTIAL > 0
    /* phase-6: keep the first N reordered Q chunks in GRF; reload the remainder per KV block. */
    static constexpr int kQHoist = CUTE_FMHA_HOIST_Q_PARTIAL;
    decltype(tSrQ) tSrQ_part[kQHoist];
#endif

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
"""

GEMM_OLD = """      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
"""
GEMM_NEW = """#if CUTE_FMHA_HOIST_Q_PARTIAL > 0
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tKrK, tSrK);
        if (D < kQHoist) {
          cute::gemm(mma_qk, tSrQ_part[D], tSrK, tSrS);
        } else {
          copy(copy_q, tQgQ(_,_,_,D), tQrQ);
          reorder(tQrQ, tSrQ);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }
#else
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
#endif
"""

PRELOAD_ANCHOR = """    /* Main loop, blocked in k. */
"""
PRELOAD_NEW = """#if CUTE_FMHA_HOIST_Q_PARTIAL > 0
    /* Load + reorder the resident Q chunks once. */
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < kQHoist; D++) {
      if (D < size<4>(tKgK)) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        reorder(tQrQ, tSrQ_part[D]);
      }
    }
#endif
""" + PRELOAD_ANCHOR

for ml in MLS:
    if not ml.exists():
        print(f"[missing] {ml}"); continue
    s = ml.read_text()
    if "CUTE_FMHA_HOIST_Q_PARTIAL" in s:
        print(f"[skip] {ml}"); continue
    for name, old in (("decl", DECL_OLD), ("gemm", GEMM_OLD), ("preload", PRELOAD_ANCHOR)):
        assert s.count(old) == 1, f"{name} anchor count {s.count(old)} in {ml}"
    bak = ml.with_name(ml.name + ".pre-phase6")
    if not bak.exists():
        bak.write_text(s)
    s = s.replace(DECL_OLD, DECL_NEW).replace(GEMM_OLD, GEMM_NEW).replace(PRELOAD_ANCHOR, PRELOAD_NEW)
    ml.write_text(s); print(f"[ok] {ml}")
