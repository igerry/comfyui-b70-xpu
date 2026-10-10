#!/usr/bin/env python3
"""Are the DiT's four INT8 GEMMs actually near the machine's ceiling?

The old "83%" came from a clock assumption. This measures achieved TOPS with the clock sampled
inside the timed loop, and compares against a sustained bf16 GEMM measured the same way (INT8
peak is 2x bf16 on Xe2), so no theoretical peak is assumed anywhere.
"""
import glob, time, threading, torch
from omni_xpu_kernel import int8

FREQ = "/sys/class/drm/card2/device/tile0/gt0/freq0/act_freq"
_hw = glob.glob("/sys/class/drm/card2/device/hwmon/hwmon*")
ENERGY = (_hw[0] + "/energy1_input") if _hw else None


def _rd(f):
    with open(f) as fh:
        return fh.read().strip()


class S:
    def __init__(self):
        self.c, self.e, self.stop = [], [], False

    def run(self):
        while not self.stop:
            try:
                self.c.append(int(_rd(FREQ)))
                if ENERGY:
                    self.e.append((time.perf_counter(), int(_rd(ENERGY))))
            except Exception:
                pass
            time.sleep(0.02)

    def out(self):
        nz = sorted(x for x in self.c if x > 0)
        med = nz[len(nz) // 2] if nz else 0
        w = 0.0
        if len(self.e) > 2:
            (t0, e0), (t1, e1) = self.e[0], self.e[-1]
            w = (e1 - e0) / 1e6 / (t1 - t0)
        return med, w


def bench(name, fn, ops, iters=20, rounds=3):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    s = S(); th = threading.Thread(target=s.run, daemon=True); th.start()
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e3)
    s.stop = True; th.join()
    med, w = s.out()
    tops = ops / best / 1e9
    per_ghz = tops / (med / 1000.0) if med else 0
    print(f"{name:34s} {best:8.3f} ms {tops:7.1f} TOPS {med:5d} MHz {w:4.0f} W  {per_ghz:6.1f} /GHz")
    return per_ghz


M, HID, INNER, FFN = 16332, 5376, 7168, 14336
dev = "xpu"
torch.manual_seed(0)
print("op                                       ms     TOPS   clock power   per-GHz")

# bf16 reference on the same silicon (the honest ceiling; INT8 peak is 2x this).
n = 16384
a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
bf16_per_ghz = bench("bf16 matmul 16384^3 (reference)", lambda: a @ b, 2.0 * n ** 3, iters=3)
del a, b
torch.xpu.empty_cache()

x = torch.randn(M, HID, device=dev, dtype=torch.bfloat16)


def mkw(nout, k):
    w = torch.randint(-127, 128, (nout, k), device=dev, dtype=torch.int8)
    s = torch.rand(nout, 1, device=dev, dtype=torch.float32) * 0.01
    return w, s


shapes = [("qkv   K5376->N21504", 3 * INNER, HID), ("out   K7168->N5376", HID, INNER),
          ("fc1   K5376->N28672", 2 * FFN, HID), ("fc2   K14336->N5376", HID, FFN)]
res = {}
for name, nout, k in shapes:
    w, sc = mkw(nout, k)
    xin = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
    ops = 2.0 * M * nout * k
    res[name] = bench("int8 " + name, lambda: int8.int8_linear(xin, w, sc), ops, iters=5)
    del w, sc, xin
    torch.xpu.empty_cache()

print(f"\nbf16 reference: {bf16_per_ghz:.1f} TFLOPS/GHz -> INT8 ceiling ~{2*bf16_per_ghz:.1f} TOPS/GHz")
for name, v in res.items():
    print(f"  {name:24s} {v:6.1f} TOPS/GHz = {100*v/(2*bf16_per_ghz):5.1f}% of the INT8 ceiling")
