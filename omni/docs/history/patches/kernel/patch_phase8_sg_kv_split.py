#!/usr/bin/env python3
"""Phase-8: let subgroups split the KV dimension, not just Q.

`D128TileKernel` hardcodes `SubgroupLayoutQK = Layout<Shape<Int<SGQ>, _1, _1>>`, so every
subgroup in the workgroup owns a slice of Q and loads the *whole* K tile. That redundancy is what
makes more subgroups catastrophic: SG32 doubles the K/V load messages and costs 79%
(knowledge/PHASE3.md section 13).

cute supports splitting the KV dimension instead. `get_sg_layout_pv` maps QK's <M,N,K> to PV's
<M,1,N>, so an N-split becomes a split of the P*V contraction, and the epilogue already implements
the reduction across those subgroups (ReduceK > 1: SharedStorageReduceK, block_2d_copy_D_subtiled,
maxima realigned before summing). None of that path has ever been instantiated here.

  -DCUTE_FMHA_DIT_SGN=2    subgroups along KV (default 1 = upstream)

With QTile 256 / SGQ 16 / SGN 2: 32 subgroups (512 work-items, the ceiling at 256 GRF), still
16 Q rows and 128 GRF of O accumulator per subgroup, but each subgroup handles half the KV tile,
so S and P halve too. Constraints: KvTile/SGN must stay >= the DPAS N of 16, and
SGQ*SGN*16 <= 512 work-items.

Numerics change: each subgroup runs its own online softmax over its KV subset and the epilogue
realigns the maxima, so the summation order differs from the single-subgroup-per-row version.
Idempotent; backup *.pre-phase8."""
import sys
from pathlib import Path

CF = Path(sys.argv[1] if len(sys.argv) > 1
          else "/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
s = CF.read_text()
if "CUTE_FMHA_DIT_SGN" in s:
    print("[skip] already patched"); sys.exit(0)

# 1. Kernel template gains a subgroup-N parameter.
old = """    int KvTileOverride = 0,
    bool StaticHeadDim = false>   // phase-3b: Int<HeadDim> in the tensor shapes
struct D128TileKernel {
"""
new = """    int KvTileOverride = 0,
    bool StaticHeadDim = false,   // phase-3b: Int<HeadDim> in the tensor shapes
    int SubgroupLayoutNOverride = 0>  // phase-8: subgroups along KV
struct D128TileKernel {
"""
assert s.count(old) == 1, "kernel template anchor"
s = s.replace(old, new)

old = """  using SubgroupLayoutQK = Layout<
      Shape<Int<SubgroupLayoutQ>, _1, _1>>;
"""
new = """  // phase-8: <Q, KV, 1>. An N > 1 splits the KV tile between subgroups, which turns into a
  // split of the P*V contraction (get_sg_layout_pv) and is reduced in the epilogue (ReduceK).
  static constexpr int SubgroupLayoutN =
      SubgroupLayoutNOverride > 0 ? SubgroupLayoutNOverride : 1;
  using SubgroupLayoutQK = Layout<
      Shape<Int<SubgroupLayoutQ>, Int<SubgroupLayoutN>, _1>>;
  static_assert(SubgroupLayoutQ * SubgroupLayoutN * 16 <= 512,
                "cute_fmha: workgroup exceeds 512 work-items at 256 GRF");
"""
assert s.count(old) == 1, "subgroup layout anchor"
s = s.replace(old, new)

# 2. run_d128_tile forwards it.
old = """    int KvTileOverride = 0,
    bool StaticHeadDim = false>
void run_d128_tile(
"""
new = """    int KvTileOverride = 0,
    bool StaticHeadDim = false,
    int SubgroupLayoutNOverride = 0>
void run_d128_tile(
"""
assert s.count(old) == 1, "run template anchor"
s = s.replace(old, new)

old = """      KvTileOverride,
      StaticHeadDim>;
"""
new = """      KvTileOverride,
      StaticHeadDim,
      SubgroupLayoutNOverride>;
"""
assert s.count(old) == 1, "KT instantiation anchor"
s = s.replace(old, new)

# 3. The H3 DiT route passes the macro.
old = """      run_d128_tile<cutlass::bfloat16_t, CUTE_FMHA_DIT_STAGES, CUTE_FMHA_DIT_QTILE,
                    CUTE_FMHA_DIT_SG, CUTE_FMHA_DIT_MMAK, CUTE_FMHA_DIT_VTILE, 0,
                    CUTE_FMHA_DIT_KV, (CUTE_FMHA_DIT_STATIC_HD != 0)>(
"""
new = """#ifndef CUTE_FMHA_DIT_SGN
#define CUTE_FMHA_DIT_SGN 0
#endif
      run_d128_tile<cutlass::bfloat16_t, CUTE_FMHA_DIT_STAGES, CUTE_FMHA_DIT_QTILE,
                    CUTE_FMHA_DIT_SG, CUTE_FMHA_DIT_MMAK, CUTE_FMHA_DIT_VTILE, 0,
                    CUTE_FMHA_DIT_KV, (CUTE_FMHA_DIT_STATIC_HD != 0), CUTE_FMHA_DIT_SGN>(
"""
assert s.count(old) == 1, "DiT route anchor"
s = s.replace(old, new)

bak = CF.with_name(CF.name + ".pre-phase8")
if not bak.exists():
    bak.write_text(CF.read_text())
CF.write_text(s)
print(f"[ok] {CF}")
