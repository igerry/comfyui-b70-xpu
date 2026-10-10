#!/usr/bin/env python3
"""Phase-1b: gate the fused ConvRot+quantize path to K >= OMNIXPU_CONVROT_FUSED_MIN_K (8192).
B70 measured 2026-09-05: fused kernel ~205 GB/s vs ~500 GB/s for matmul-rotate + separate
quantize; it only wins for K=14336 (fc2), whose separate quantize has no tuned BMG config.
Applies to a given int8/__init__.py path (src and/or installed copy)."""
import sys
for p in sys.argv[1:]:
    s = open(p).read()
    old = ("        and x.is_contiguous()\n"
           "        and x.shape[1] % convrot_groupsize == 0\n"
           "    )\n")
    new = ("        and x.is_contiguous()\n"
           "        and x.shape[1] % convrot_groupsize == 0\n"
           "        # B70 measured (2026-09-05): the one-row-per-work-item fused kernel runs at\n"
           "        # ~205 GB/s vs ~500 GB/s for matmul-rotate + separate quantize, so it only\n"
           "        # wins where the separate quantize has no tuned BMG config (K=14336 fc2:\n"
           "        # 3.41 ms vs 1.90+2.29). OMNIXPU_CONVROT_FUSED_MIN_K overrides.\n"
           "        and x.shape[1] >= _CONVROT_FUSED_MIN_K\n"
           "    )\n")
    old2 = '_CONVROT_FUSED = os.environ.get("OMNIXPU_CONVROT_FUSED", "1") != "0"\n'
    new2 = old2 + '_CONVROT_FUSED_MIN_K = int(os.environ.get("OMNIXPU_CONVROT_FUSED_MIN_K", "8192"))\n'
    if new in s and new2 in s:
        print(f"[skip] {p}")
        continue
    assert s.count(old) == 1 and s.count(old2) == 1, p
    open(p, "w").write(s.replace(old, new).replace(old2, new2))
    print(f"[ok] {p}")
