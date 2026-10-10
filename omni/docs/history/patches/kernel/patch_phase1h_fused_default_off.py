#!/usr/bin/env python3
"""Phase-1h: fused ConvRot+quantize default OFF. Its only measurable win (fc2, 0.8 ms/block,
~0.2 s/run) is below run-to-run noise, and the 1-LSB differences it introduces flip the
6-step turbo trajectory (seed-102 PSNR 19.5 dB vs the same run without it). Opt in with
OMNIXPU_CONVROT_FUSED=1."""
import sys
for p in sys.argv[1:]:
    s = open(p).read()
    old = '_CONVROT_FUSED = os.environ.get("OMNIXPU_CONVROT_FUSED", "1") != "0"\n'
    new = ('# Default off (2026-09-05): gain is below noise and the 1-LSB quantization differences\n'
           '# change the 6-step turbo sample (seed-102 PSNR 19.5 dB). OMNIXPU_CONVROT_FUSED=1 opts in.\n'
           '_CONVROT_FUSED = os.environ.get("OMNIXPU_CONVROT_FUSED", "0") == "1"\n')
    if new in s:
        print(f"[skip] {p}"); continue
    assert s.count(old) == 1, p
    open(p, "w").write(s.replace(old, new)); print(f"[ok] {p}")
