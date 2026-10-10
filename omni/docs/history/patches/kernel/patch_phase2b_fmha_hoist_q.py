#!/usr/bin/env python3
"""Phase-2b (v2): hoist the Q fragment out of the KV loop in the sycl-tla FMHA prefill mainloop.
Upstream reloads and reorders every head-dim chunk of Q on every KV block (GEMM 1 inner loop);
for H3 (1,56,16332,128) that is 511 reloads of a 64 KB Q tile per work-group. The chunk count is a
runtime int in cute's layout, so a fixed-capacity register array (CUTE_FMHA_HOIST_Q_MAX_CHUNKS,
default 8 = 128/16) with statically unrolled, guarded indexing is used; larger chunk counts fall
back to the original path at runtime. Math order unchanged. Gated by -DCUTE_FMHA_HOIST_Q=1."""
import sys
from pathlib import Path

for path in sys.argv[1:]:
    p = Path(path)
    bak = p.with_name(p.name + ".pre-hoistq")
    s = bak.read_text() if bak.exists() else p.read_text()
    if not bak.exists():
        bak.write_text(s)
    a = """    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
"""
    b = a + """#if defined(CUTE_FMHA_HOIST_Q)
#ifndef CUTE_FMHA_HOIST_Q_MAX_CHUNKS
#define CUTE_FMHA_HOIST_Q_MAX_CHUNKS 8
#endif
    /* Q is invariant across KV blocks: keep every reordered head-dim chunk in GRF. */
    static constexpr int kQChunksMax = CUTE_FMHA_HOIST_Q_MAX_CHUNKS;
    decltype(tSrQ) tSrQ_all[kQChunksMax];
    const int q_chunks = size<4>(tKgK);
    const bool hoist_q = (q_chunks <= kQChunksMax);
#endif
"""
    assert s.count(a) == 1, path; s = s.replace(a, b)
    a = """      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
"""
    b = """#if defined(CUTE_FMHA_HOIST_Q)
      if (hoist_q) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < kQChunksMax; D++) {
          if (D < q_chunks) {
            copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
            reorder(tKrK, tSrK);
            cute::gemm(mma_qk, tSrQ_all[D], tSrK, tSrS);
          }
        }
      } else
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
"""
    assert s.count(a) == 1, path; s = s.replace(a, b)
    a = """    /* Main loop, blocked in k. */
"""
    b = """#if defined(CUTE_FMHA_HOIST_Q)
    /* Load + reorder Q once (was: per KV block, per head-dim chunk). */
    if (hoist_q) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < kQChunksMax; D++) {
        if (D < q_chunks) {
          copy(copy_q, tQgQ(_,_,_,D), tQrQ);
          reorder(tQrQ, tSrQ_all[D]);
        }
      }
    }
#endif
    /* Main loop, blocked in k. */
"""
    assert s.count(a) == 1, path; s = s.replace(a, b)
    p.write_text(s); print(f"[ok] {path}")
