#!/usr/bin/env python3
"""Phase-3a: lazy O-rescale + timing-only diagnostic switches in the sycl-tla FMHA fwd mainloop,
plus a global GRF-size override macro in cute_fmha_torch.cpp.  All macro-gated; default build unchanged.

Motivation (ISA dump of the H3 DiT route, IGC 2026.0, see knowledge/PHASE3.md): per KV block per
thread the loop is ~640 instructions of which 64 are DPAS; the online-softmax O rescale is 4x32 = 128
fmul executed on *every* block although the running row max changes on only a handful of blocks.

  -DCUTE_FMHA_LAZY_RESCALE=1   skip the O-rescale multiplies when no row max in the subgroup changed
                               (exp2(0) == 1 exactly on the EM unit -> bitwise identical output)
  -DCUTE_FMHA_DIAG_NO_SOFTMAX  timing only: no max/sum/exp, P = bf16(S)          (wrong numbers)
  -DCUTE_FMHA_DIAG_NO_RESCALE  timing only: never rescale O                       (wrong numbers)
  -DCUTE_FMHA_DIAG_NO_BARRIER  timing only: drop the per-block WG split barrier    (numerically exact)
  -DCUTE_FMHA_DIAG_NO_KPREFETCH timing only: drop the per-block K prefetch loop    (numerically exact)
  -DCUTE_FMHA_GRF=128|256      cute_fmha_torch.cpp: override PlatformConfig::GRF_SIZE for every route
"""
import sys
from pathlib import Path

ML = Path(sys.argv[1] if len(sys.argv) > 1 else
          "/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")
CF = Path(sys.argv[2] if len(sys.argv) > 2 else
          "/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")

def patch(path, edits, marker, bak_suffix):
    s = path.read_text()
    if marker in s:
        print(f"[skip] {path.name} already patched"); return
    bak = path.with_name(path.name + bak_suffix)
    if not bak.exists():
        bak.write_text(s)
    for old, new in edits:
        assert s.count(old) == 1, f"anchor not unique/found in {path}:\n{old}"
        s = s.replace(old, new)
    path.write_text(s); print(f"[ok] {path}")

# ---- mainloop ---------------------------------------------------------------
edits = []
edits.append(("""      /* Split barrier to keep threads together */
      barrier_arrive(ScopeWorkgroup);
""", """      /* Split barrier to keep threads together */
#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      barrier_arrive(ScopeWorkgroup);
#endif
"""))
edits.append(("""      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (K != blk_k0) {
""", """      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
#if defined(CUTE_FMHA_DIAG_NO_SOFTMAX)
      FragSRow rescale;
      fill(rescale, ElementS(1));
      const bool do_rescale = false;
#else
      bool max_changed = true;
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum, max_changed);
#if defined(CUTE_FMHA_DIAG_NO_RESCALE)
      const bool do_rescale = false;
#elif defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE
      /* Phase-3a: rescale == exp2(0) == 1 on every row when no row max moved -> skip the 4xVTiles fmuls. */
      const bool do_rescale = (K != blk_k0) && max_changed;
#else
      const bool do_rescale = (K != blk_k0);
#endif
#endif
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (do_rescale) {
"""))
edits.append(("""      /* K prefetch */
      int K_next = K + Stages;
""", """      /* K prefetch */
#if !defined(CUTE_FMHA_DIAG_NO_KPREFETCH)
      int K_next = K + Stages;
"""))
edits.append(("""      }
      barrier_wait(ScopeWorkgroup);
    };
""", """      }
#endif  // CUTE_FMHA_DIAG_NO_KPREFETCH
#if !defined(CUTE_FMHA_DIAG_NO_BARRIER)
      barrier_wait(ScopeWorkgroup);
#endif
    };
"""))
edits.append(("""  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
""", """  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum,      // Softmax row-wise sum accumulator
          bool     & max_changed) { // Out: did any row max in this subgroup move? (phase-3a)
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    bool changed_lane = false;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      changed_lane |= (new_max != tS_max(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
#if defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE
    max_changed = sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), changed_lane);
#else
    max_changed = true;
#endif
"""))
patch(ML, edits, "CUTE_FMHA_LAZY_RESCALE", ".pre-phase3a")

# ---- cute_fmha_torch.cpp: GRF override --------------------------------------
patch(CF, [("""  static constexpr int GrfSize = PlatformConfig::GRF_SIZE;
""", """#ifdef CUTE_FMHA_GRF
  static constexpr int GrfSize = CUTE_FMHA_GRF;   // phase-3a variant sweep
#else
  static constexpr int GrfSize = PlatformConfig::GRF_SIZE;
#endif
""")], "CUTE_FMHA_GRF", ".pre-phase3a")
