#!/usr/bin/env python3
"""Phase-2a: expose the MiniMax H3 DiT d128 bf16 route's tile config as macros so variants can be
built for a sweep: CUTE_FMHA_DIT_STAGES / _QTILE / _SG / _MMAK (default 16, the shipped value) /
_VTILE / _KV. Defaults 0 = shipped platform config."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/cute/cute_fmha_torch.cpp")
s = p.read_text()
if "CUTE_FMHA_DIT_QTILE" in s:
    print("[skip]"); sys.exit(0)
old = """    if (use_minimax_h3_h56_mmak16(q, k, v, B, H, Lq, Lkv, D)) {
      run_d128_tile<cutlass::bfloat16_t, 0, 0, 0, 16>(
"""
new = """    if (use_minimax_h3_h56_mmak16(q, k, v, B, H, Lq, Lkv, D)) {
#ifndef CUTE_FMHA_DIT_STAGES
#define CUTE_FMHA_DIT_STAGES 0
#endif
#ifndef CUTE_FMHA_DIT_QTILE
#define CUTE_FMHA_DIT_QTILE 0
#endif
#ifndef CUTE_FMHA_DIT_SG
#define CUTE_FMHA_DIT_SG 0
#endif
#ifndef CUTE_FMHA_DIT_MMAK
#define CUTE_FMHA_DIT_MMAK 16
#endif
#ifndef CUTE_FMHA_DIT_VTILE
#define CUTE_FMHA_DIT_VTILE 0
#endif
#ifndef CUTE_FMHA_DIT_KV
#define CUTE_FMHA_DIT_KV 0
#endif
      run_d128_tile<cutlass::bfloat16_t, CUTE_FMHA_DIT_STAGES, CUTE_FMHA_DIT_QTILE,
                    CUTE_FMHA_DIT_SG, CUTE_FMHA_DIT_MMAK, CUTE_FMHA_DIT_VTILE, 0,
                    CUTE_FMHA_DIT_KV>(
"""
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print("[ok] DiT route macros")
