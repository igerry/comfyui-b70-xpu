#!/usr/bin/env python3
"""Phase-3d: with lazy rescale on, also skip *computing* the rescale factors when no row max moved.

Phase-3a already skips the 128 O-accumulator fmuls; the 16 sub + 16 exp2 that produce `rescale`
still run on every KV block because they are computed before the dynamic branch. Under the 230 W
power wall every removed instruction buys clock (knowledge/PHASE3.md), so guard them too.

Gated by -DCUTE_FMHA_LAZY_SINK_EXP=1 (requires CUTE_FMHA_LAZY_RESCALE). Numerics unchanged: the
skipped values are only consumed when max_changed is true. Idempotent; backup *.pre-phase3d."""
import sys
from pathlib import Path

MLS = [Path("/root/omnibuild/sycl-tla/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"),
       Path("/root/omnibuild/src/build/temp.linux-x86_64-cpython-313/cute_bmg_include_overlay/"
            "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp")]
if len(sys.argv) > 1:
    MLS = [Path(p) for p in sys.argv[1:]]

OLD = """    FragSRow rescale;
    bool changed_lane = false;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      changed_lane |= (new_max != tS_max(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
"""
NEW = """    FragSRow rescale;
    bool changed_lane = false;
#if defined(CUTE_FMHA_LAZY_SINK_EXP) && CUTE_FMHA_LAZY_SINK_EXP
    /* phase-3d: decide first, then only pay for the exp2s when some row actually moved. */
    FragSRow new_max_v;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      new_max_v(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      changed_lane |= (new_max_v(i) != tS_max(i));
    }
    max_changed = sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), changed_lane);
    if (max_changed) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_max(i) - new_max_v(i));
        tS_max(i) = new_max_v(i);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rescale.size(); i++) rescale(i) = ElementS(1);
    }
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      changed_lane |= (new_max != tS_max(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
#endif
"""
GUARD_OLD = """#if defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE
    max_changed = sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), changed_lane);
#else
    max_changed = true;
#endif
"""
GUARD_NEW = """#if defined(CUTE_FMHA_LAZY_SINK_EXP) && CUTE_FMHA_LAZY_SINK_EXP
    /* max_changed already set above. */
#elif defined(CUTE_FMHA_LAZY_RESCALE) && CUTE_FMHA_LAZY_RESCALE
    max_changed = sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), changed_lane);
#else
    max_changed = true;
#endif
"""

for ml in MLS:
    if not ml.exists():
        print(f"[missing] {ml}"); continue
    s = ml.read_text()
    if "CUTE_FMHA_LAZY_SINK_EXP" in s:
        print(f"[skip] {ml}"); continue
    assert s.count(OLD) == 1, f"softmax anchor not found in {ml} (phase-3a applied?)"
    assert s.count(GUARD_OLD) == 1, f"guard anchor not found in {ml}"
    bak = ml.with_name(ml.name + ".pre-phase3d")
    if not bak.exists():
        bak.write_text(s)
    ml.write_text(s.replace(OLD, NEW).replace(GUARD_OLD, GUARD_NEW))
    print(f"[ok] {ml}")
