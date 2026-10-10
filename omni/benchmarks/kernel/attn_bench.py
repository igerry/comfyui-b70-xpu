#!/usr/bin/env python3
"""DiT-route attention only: (1,56,16332,128) bf16 via cute.sdp_bhld_d128 with whatever
OMNI_CUTE_FMHA_SO selects. Prints ms / TFLOPS; optionally checks against a reference dump.
  usage: attn_bench.py [--ref ref.pt] [--save out.pt] [--iters N] [--M 16332]
"""
import argparse, os, time, threading, torch
import glob
FREQ = "/sys/class/drm/card2/device/tile0/gt0/freq0/act_freq"
REASONS = "/sys/class/drm/card2/device/tile0/gt0/freq0/throttle/reasons"
_hw = glob.glob("/sys/class/drm/card2/device/hwmon/hwmon*")
ENERGY = (_hw[0] + "/energy1_input") if _hw else None
_samples = []; _reasons = {}; _stop = False; _energy = []
def _rd(f):
    with open(f) as fh: return fh.read().strip()
def _sampler():
    while not _stop:
        try:
            _samples.append(int(_rd(FREQ)))
            r = _rd(REASONS); _reasons[r] = _reasons.get(r, 0) + 1
            if ENERGY: _energy.append((time.perf_counter(), int(_rd(ENERGY))))
        except Exception: pass
        time.sleep(0.02)
ap = argparse.ArgumentParser()
ap.add_argument("--ref"); ap.add_argument("--save"); ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--M", type=int, default=16332); ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
from omni_xpu_kernel import cute
torch.manual_seed(0)
M, H, D, INNER = a.M, 56, 128, 7168
qkv = torch.randn(M, 3 * INNER, device="xpu", dtype=torch.bfloat16)
q = qkv[:, :INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
k = qkv[:, INNER:2 * INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
v = qkv[:, 2 * INNER:].view(M, H, D).transpose(0, 1).unsqueeze(0)
flops = 4.0 * H * M * M * D
for _ in range(3):
    o = cute.sdp_bhld_d128(q, k, v)
torch.xpu.synchronize()
best = 1e9
th = threading.Thread(target=_sampler, daemon=True); th.start()
for r in range(a.rounds):
    t0 = time.perf_counter()
    for _ in range(a.iters):
        cute.sdp_bhld_d128(q, k, v)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    best = min(best, ms)
    print(f"round {r}: {ms:8.2f} ms  {flops/ms/1e9:6.1f} TFLOPS", flush=True)
_stop = True; th.join()
nz = sorted(x for x in _samples if x > 0)
clk = f"clock MHz n={len(nz)} min={nz[0]} med={nz[len(nz)//2]} max={nz[-1]}" if nz else "clock n/a"
if len(_energy) > 2:
    (t0, e0), (t1, e1) = _energy[0], _energy[-1]
    clk += f" avgW={(e1 - e0) / 1e6 / (t1 - t0):.0f}"
clk += " throttle=" + ",".join(f"{k}:{v}" for k, v in sorted(_reasons.items(), key=lambda x: -x[1])[:3])
print(f"{os.path.basename(os.environ.get('OMNI_CUTE_FMHA_SO','default'))}: best {best:8.2f} ms  {flops/best/1e9:6.1f} TFLOPS  [{clk}]")
o = cute.sdp_bhld_d128(q, k, v); torch.xpu.synchronize()
if a.save:
    torch.save(o.cpu(), a.save); print("saved", a.save)
if a.ref:
    ref = torch.load(a.ref).to("xpu")
    d = (o.float() - ref.float()).abs()
    nz = (o != ref).sum().item()
    print(f"vs ref: max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} differing={nz}/{o.numel()} nan={torch.isnan(o).any().item()}")
