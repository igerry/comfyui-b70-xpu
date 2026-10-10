#!/usr/bin/env python3
"""Phase-1e: per-route KV tile / pipeline stages for the CUTE FMHA.
Measured on B70 (2026-09-05): KV64 is 8% slower on the DiT route (1,56,16332,128) but
KV64+Stages3 is 18% faster on the VAE route (1,32,1797,64). Add a KvTileOverride template
parameter (0 = keep the CUTE_FMHA_KV64 macro / platform default) and wire the VAE call site
to -DCUTE_FMHA_VAE_KV / -DCUTE_FMHA_VAE_STAGES (default 0 = unchanged)."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
s = p.read_text()
if "KvTileOverride" in s:
    print("[skip]"); sys.exit(0)

def rep(old, new, tag):
    global s
    assert s.count(old) == 1, f"{tag}: count={s.count(old)}"
    s = s.replace(old, new); print(f"[ok] {tag}")

rep("""    int VTileOverride = 0,
    int HeadDimOverride = 0>
struct D128TileKernel {""",
    """    int VTileOverride = 0,
    int HeadDimOverride = 0,
    int KvTileOverride = 0>
struct D128TileKernel {""", "struct template param")
rep("""#if defined(CUTE_FMHA_KV64)
  // KV tile = get<1>(ShapeQK) = 64. Per get_tiled_mma_pv, the PV tile must be
  // <TileQ, TileV, KVtile> — so ShapePV's K-dim (3rd) MUST equal 64, not 32.
  // TileV=32 -> VTiles = 128/32 = 4. (My earlier <256,32,32> broke the QK->PV
  // K-dim match and tripped the gemm.hpp static_assert.)
  static constexpr int KvTile = 64;
#else
  static constexpr int KvTile = PlatformConfig::KV_TILE;
#endif""",
    """#if defined(CUTE_FMHA_KV64)
  // KV tile = get<1>(ShapeQK) = 64. Per get_tiled_mma_pv, the PV tile must be
  // <TileQ, TileV, KVtile> — so ShapePV's K-dim (3rd) MUST equal 64, not 32.
  // TileV=32 -> VTiles = 128/32 = 4. (My earlier <256,32,32> broke the QK->PV
  // K-dim match and tripped the gemm.hpp static_assert.)
  static constexpr int KvTileDefault = 64;
#else
  static constexpr int KvTileDefault = PlatformConfig::KV_TILE;
#endif
  // Per-route override (B70: KV64 loses 8% on the DiT d128 route but wins 18%
  // together with 3 stages on the VAE d64 route).
  static constexpr int KvTile =
      KvTileOverride > 0 ? KvTileOverride : KvTileDefault;""", "KvTile override")
rep("""    int VTileOverride = 0,
    int HeadDimOverride = 0>
void run_d128_tile(""",
    """    int VTileOverride = 0,
    int HeadDimOverride = 0,
    int KvTileOverride = 0>
void run_d128_tile(""", "run template param")
rep("""      MmaKOverride,
      VTileOverride,
      HeadDimOverride>;
  using K    = typename KT::Kernel;""",
    """      MmaKOverride,
      VTileOverride,
      HeadDimOverride,
      KvTileOverride>;
  using K    = typename KT::Kernel;""", "run KT args")
rep("""#ifndef CUTE_FMHA_VAE_QTILE
#define CUTE_FMHA_VAE_QTILE 0
#endif
  run_d128_tile<cutlass::half_t, 0, CUTE_FMHA_VAE_QTILE,
                (CUTE_FMHA_VAE_QTILE > 0 ? CUTE_FMHA_VAE_QTILE / 16 : 0),
                0, 0, 64>(""",
    """#ifndef CUTE_FMHA_VAE_QTILE
#define CUTE_FMHA_VAE_QTILE 0
#endif
#ifndef CUTE_FMHA_VAE_STAGES
#define CUTE_FMHA_VAE_STAGES 0
#endif
#ifndef CUTE_FMHA_VAE_KV
#define CUTE_FMHA_VAE_KV 0
#endif
  run_d128_tile<cutlass::half_t, CUTE_FMHA_VAE_STAGES, CUTE_FMHA_VAE_QTILE,
                (CUTE_FMHA_VAE_QTILE > 0 ? CUTE_FMHA_VAE_QTILE / 16 : 0),
                0, 0, 64, CUTE_FMHA_VAE_KV>(""", "VAE call site")
p.write_text(s); print("done")
