#!/usr/bin/env python3
"""Empirical DPAS ceiling on this box, measured the same way as attn_bench.py (clock/power
sampled inside the timed loop) so attention can be compared against what the same silicon
actually sustains under the same power cap - not against a spec-sheet peak.

  roofline_bench.py [--iters N]
Prints, for a few large bf16 GEMMs and for the H3 attention shape:
  ms, TFLOPS, median clock, average watts, and TFLOPS-per-GHz (the clock-independent figure).
"""
import argparse, glob, time, threading, torch

FREQ = "/sys/class/drm/card2/device/tile0/gt0/freq0/act_freq"
_hw = glob.glob("/sys/class/drm/card2/device/hwmon/hwmon*")
ENERGY = (_hw[0] + "/energy1_input") if _hw else None


def _rd(f):
    with open(f) as fh:
        return fh.read().strip()


class Sampler:
    def __init__(self):
        self.clk = []
        self.e = []
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                self.clk.append(int(_rd(FREQ)))
                if ENERGY:
                    self.e.append((time.perf_counter(), int(_rd(ENERGY))))
            except Exception:
                pass
            time.sleep(0.02)

    def summary(self):
        nz = sorted(x for x in self.clk if x > 0)
        med = nz[len(nz) // 2] if nz else 0
        w = 0.0
        if len(self.e) > 2:
            (t0, e0), (t1, e1) = self.e[0], self.e[-1]
            w = (e1 - e0) / 1e6 / (t1 - t0)
        return med, w


def bench(name, fn, flops, iters, rounds=3):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    s = Sampler()
    th = threading.Thread(target=s.run, daemon=True)
    th.start()
    best = 1e9
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    s.stop = True
    th.join()
    med, w = s.summary()
    tf = flops / best / 1e9
    ghz = med / 1000.0
    print(f"{name:34s} {best:8.2f} ms {tf:7.1f} TFLOPS  {med:5d} MHz  {w:4.0f} W  "
          f"{tf / ghz:6.1f} TFLOPS/GHz" if ghz else f"{name}: {best:.2f} ms")
    return tf / ghz if ghz else 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
torch.manual_seed(0)
print("shape                                    ms   TFLOPS   clock  power   per-GHz")
per_ghz = {}
for n in (4096, 8192, 16384):
    x = torch.randn(n, n, device="xpu", dtype=torch.bfloat16)
    y = torch.randn(n, n, device="xpu", dtype=torch.bfloat16)
    per_ghz[f"gemm{n}"] = bench(f"bf16 matmul {n}^3", lambda: x @ y, 2.0 * n ** 3, a.iters)
    del x, y
    torch.xpu.empty_cache()

from omni_xpu_kernel import cute
M, H, D, INNER = 16332, 56, 128, 7168
qkv = torch.randn(M, 3 * INNER, device="xpu", dtype=torch.bfloat16)
q = qkv[:, :INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
k = qkv[:, INNER:2 * INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
v = qkv[:, 2 * INNER:].view(M, H, D).transpose(0, 1).unsqueeze(0)
per_ghz["attn"] = bench("H3 DiT attention", lambda: cute.sdp_bhld_d128(q, k, v),
                        4.0 * H * M * M * D, a.iters)
# Compare against the LARGEST GEMM only. Short kernels (the 4096 case runs <1 ms) finish inside
# a boost window and the 20 ms clock sampler mostly catches idle/ramp clocks, so their per-GHz
# figure is meaningless. The 16384 case runs ~70 ms like attention and is power-limited the same way.
ref = per_ghz["gemm16384"]
print(f"\nattention reaches {100 * per_ghz['attn'] / ref:.0f}% of the sustained GEMM's "
      f"clock-normalised throughput ({per_ghz['attn']:.1f} vs {ref:.1f} TFLOPS/GHz)."
      f"\nIgnore the short-GEMM rows: their clock sample is unreliable.")
