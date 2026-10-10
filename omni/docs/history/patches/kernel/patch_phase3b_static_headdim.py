#!/usr/bin/env python3
"""Phase-3b: compile-time head dimension for the H3 DiT d128 route (and any route that opts in).

Motivation (ISA dump, knowledge/PHASE3.md): head_size is a runtime int in the kernel's tensor shapes, so
the GEMM-1 head-dim chunk loop (8 chunks) and the per-block K prefetch loop are runtime loops in the
main loop (IGC unrolls the chunk loop only 4x and keeps an 8-trip scalar prefetch loop with 8 branches).
Making the head-dim mode of Q/K/V/O static lets every D loop unroll fully and lets the compiler hoist
loads earlier.

  cute_fmha_torch.cpp : D128TileKernel/run_d128_tile get a 9th template parameter `bool StaticHeadDim`;
                        when true the Tensor types carry Int<HeadDim> in the head-dim mode.  The H3 DiT
                        route passes -DCUTE_FMHA_DIT_STATIC_HD=1 (default 0 until validated); other routes keep runtime D
                        (the d120 route relies on runtime bounds for D=120 under a 128 tile).
  xe_fmha_fwd_kernel.hpp : shape_Q/K/V/O take the static extent from the mainloop's tensor types when
                        it is static (cute::is_static), runtime head_size otherwise.
  xe_fmha_fwd_mainloop.hpp : CUTLASS_PRAGMA_UNROLL on the D prefetch loops.
Idempotent; originals kept as *.pre-phase3b.
"""
import sys
from pathlib import Path

CF = Path("/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
KN = Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp")
ML = Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")
if len(sys.argv) > 3:
    CF, KN, ML = map(Path, sys.argv[1:4])

def patch(path, edits, marker, first_only=()):
    bak = path.with_name(path.name + ".pre-phase3b")
    if bak.exists():
        s = bak.read_text()          # re-apply from the pristine copy (patch may have been revised)
    else:
        s = path.read_text()
        if marker in s:
            print(f"[skip] {path.name} already patched"); return
        bak.write_text(s)
    for i, (old, new) in enumerate(edits):
        n = s.count(old)
        assert n >= 1 and (n == 1 or i in first_only), f"anchor count {n} in {path}:\n{old}"
        s = s.replace(old, new, 1)
    path.write_text(s); print(f"[ok] {path}")

# ---- cute_fmha_torch.cpp -----------------------------------------------------
cf_edits = []
# D128TileKernel template: add StaticHeadDim param
cf_edits.append(("""    int HeadDimOverride = 0,
    int KvTileOverride = 0>
struct D128TileKernel {
""", """    int HeadDimOverride = 0,
    int KvTileOverride = 0,
    bool StaticHeadDim = false>   // phase-3b: Int<HeadDim> in the tensor shapes
struct D128TileKernel {
"""))
cf_edits.append(("""  using TensorQ = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideQ>>(1), StrideQ{})));
  using TensorK = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideK>>(1), StrideK{})));
  using TensorV = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideV>>(1), StrideV{})));
  using TensorO = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideO>>(1), StrideO{})));
""", """  // phase-3b: with StaticHeadDim the head-dim mode is Int<HeadDim>, so every
  // head-dim chunk loop in the mainloop has a static trip count.
  using HD = std::conditional_t<StaticHeadDim, Int<HeadDim>, int>;
  using ShapeQKO = Shape<int, HD, int, int>;
  using ShapeVT  = Shape<HD, int, int, int>;
  using TensorQ = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(ShapeQKO{}, StrideQ{})));
  using TensorK = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(ShapeQKO{}, StrideK{})));
  using TensorV = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(ShapeVT{}, StrideV{})));
  using TensorO = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(ShapeQKO{}, StrideO{})));
"""))
# run_d128_tile template + KT instantiation
cf_edits.append(("""    int HeadDimOverride = 0,
    int KvTileOverride = 0>
void run_d128_tile(
""", """    int HeadDimOverride = 0,
    int KvTileOverride = 0,
    bool StaticHeadDim = false>
void run_d128_tile(
"""))
cf_edits.append(("""      VTileOverride,
      HeadDimOverride,
      KvTileOverride>;
  using K    = typename KT::Kernel;
""", """      VTileOverride,
      HeadDimOverride,
      KvTileOverride,
      StaticHeadDim>;
  using K    = typename KT::Kernel;
  if constexpr (StaticHeadDim) {
    TORCH_CHECK(D == KT::HeadDim, "cute_fmha: static head-dim route requires D == ", KT::HeadDim);
  }
"""))
# H3 DiT route: opt in
cf_edits.append(("""#ifndef CUTE_FMHA_DIT_KV
#define CUTE_FMHA_DIT_KV 0
#endif
      run_d128_tile<cutlass::bfloat16_t, CUTE_FMHA_DIT_STAGES, CUTE_FMHA_DIT_QTILE,
                    CUTE_FMHA_DIT_SG, CUTE_FMHA_DIT_MMAK, CUTE_FMHA_DIT_VTILE, 0,
                    CUTE_FMHA_DIT_KV>(
""", """#ifndef CUTE_FMHA_DIT_KV
#define CUTE_FMHA_DIT_KV 0
#endif
#ifndef CUTE_FMHA_DIT_STATIC_HD
#define CUTE_FMHA_DIT_STATIC_HD 0
#endif
      run_d128_tile<cutlass::bfloat16_t, CUTE_FMHA_DIT_STAGES, CUTE_FMHA_DIT_QTILE,
                    CUTE_FMHA_DIT_SG, CUTE_FMHA_DIT_MMAK, CUTE_FMHA_DIT_VTILE, 0,
                    CUTE_FMHA_DIT_KV, (CUTE_FMHA_DIT_STATIC_HD != 0)>(
"""))
patch(CF, cf_edits, "StaticHeadDim")

# ---- kernel -----------------------------------------------------------------
kn_edits = []
kn_edits.append(("""      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

      auto shape_K_cache = make_shape(seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, seq_len_kv_cache, s.num_heads_kv, batch_dim);
""", """      auto batch_dim = is_var_len ? 1 : s.batch;
      /* phase-3b: head-dim extents come from the mainloop tensor types when static (Int<D>). */
      using HDqk_t = decltype(get<1>(shape(typename CollectiveMainloop::TensorQ{})));
      using HDvo_t = decltype(get<0>(shape(typename CollectiveMainloop::TensorV{})));
      auto hd_qk = [&] { if constexpr (is_static<HDqk_t>::value) return HDqk_t{}; else return s.head_size_qk; }();
      auto hd_vo = [&] { if constexpr (is_static<HDvo_t>::value) return HDvo_t{}; else return s.head_size_vo; }();
      auto shape_Q = make_shape(seq_len_qo, hd_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, hd_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(hd_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, hd_vo, s.num_heads_q, batch_dim);

      auto shape_K_cache = make_shape(seq_len_kv_cache, hd_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(hd_vo, seq_len_kv_cache, s.num_heads_kv, batch_dim);
      /* all-int copies for make_cute_packed_stride (var-len path only, but it must still compile) */
      auto shape_Q_dyn = make_shape(seq_len_qo, int(hd_qk), s.num_heads_q, batch_dim);
      auto shape_K_dyn = make_shape(seq_len_kv, int(hd_qk), s.num_heads_kv, batch_dim);
      auto shape_V_dyn = make_shape(int(hd_vo), seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O_dyn = make_shape(seq_len_qo, int(hd_vo), s.num_heads_q, batch_dim);
      auto shape_K_cache_dyn = make_shape(seq_len_kv_cache, int(hd_qk), s.num_heads_kv, batch_dim);
      auto shape_V_cache_dyn = make_shape(int(hd_vo), seq_len_kv_cache, s.num_heads_kv, batch_dim);
"""))
kn_edits.append(("""      auto stride_q = is_var_len ? cutlass::make_cute_packed_stride(StrideQ{}, shape_Q) : p.dQ;
      auto stride_k = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K) : p.dK;
      auto stride_v = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V) : p.dV;
      auto stride_o = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_O) : p.dO;
      auto stride_k_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache) : p.dK_cache;
      auto stride_v_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache) : p.dV_cache;
""", """      auto stride_q = is_var_len ? cutlass::make_cute_packed_stride(StrideQ{}, shape_Q_dyn) : p.dQ;
      auto stride_k = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_dyn) : p.dK;
      auto stride_v = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_dyn) : p.dV;
      auto stride_o = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_O_dyn) : p.dO;
      auto stride_k_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache_dyn) : p.dK_cache;
      auto stride_v_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache_dyn) : p.dV_cache;
"""))
patch(KN, kn_edits, "phase-3b", first_only=(0,1))

# ---- mainloop: unroll the static prefetch loops -------------------------------
ml_edits = []
ml_edits.append(("""    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
""", """    CUTLASS_PRAGMA_UNROLL   // phase-3b: static with StaticHeadDim
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
"""))
ml_edits.append(("""      int K_next = K + Stages;
      for (int D = 0; D < size<4>(pKgK); D++) {
""", """      int K_next = K + Stages;
      CUTLASS_PRAGMA_UNROLL   // phase-3b
      for (int D = 0; D < size<4>(pKgK); D++) {
"""))
patch(ML, ml_edits, "phase-3b")
