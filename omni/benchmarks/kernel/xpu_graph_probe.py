"""torch 2.14 XPU graph capture on B70: what does replay buy, and what can be captured?
Runs in /root/comfy-xpu214. Three workloads, each timed eager vs captured-and-replayed, outputs compared:
  A. many small ops (the audio-VAE regime: ~400 small Conv1d + activations on a short signal)
  B. medium ops (the video-VAE-block regime: fp16 Linear 2048 + norm + silu, a few thousand tokens)
  C. the same with an omni_xpu_kernel call inside (does capture see custom SYCL kernels at all?)
"""
import time, torch, torch.nn as nn, torch.nn.functional as F
print("torch", torch.__version__)
dev = torch.device("xpu")
def sync(): torch.xpu.synchronize()
def T(fn, n=10):
    for _ in range(2): fn()
    sync(); best = 1e9
    for _ in range(4):
        t = time.perf_counter()
        for _ in range(n): fn()
        sync(); best = min(best, (time.perf_counter() - t) / n * 1e3)
    return best

def try_graph(name, model, x, reps=10):
    model = model.to(dev).eval()
    with torch.inference_mode(False), torch.no_grad():
        ref = model(x); sync()
        te = T(lambda: model(x), reps)
        try:
            g = torch.xpu.XPUGraph()
            static_in = x.clone()
            for _ in range(3): model(static_in)        # warm-up outside capture, as with CUDA graphs
            sync()
            with torch.xpu.graph(g):
                static_out = model(static_in)
            def replay():
                static_in.copy_(x); g.replay(); return static_out
            out = replay(); sync()
            tg = T(replay, reps)
            same = torch.equal(out, ref); rel = float((out.float() - ref.float()).norm() / ref.float().norm())
            print(f"{name:44s} eager {te:8.2f} ms   graph {tg:8.2f} ms   x{te/tg:5.2f}   bitwise={same} rel={rel:.1e}")
        except Exception as e:
            print(f"{name:44s} eager {te:8.2f} ms   graph FAILED: {type(e).__name__}: {str(e)[:220]}")

class SmallConvs(nn.Module):
    def __init__(s, n=200, c=64):
        super().__init__(); s.cs = nn.ModuleList([nn.Conv1d(c, c, 7, padding=3) for _ in range(n)])
    def forward(s, x):
        for c in s.cs: x = x + F.silu(c(x)) * 0.05
        return x
class MedBlocks(nn.Module):
    def __init__(s, n=36, d=2048):
        super().__init__(); s.a = nn.ModuleList([nn.Linear(d, d) for _ in range(n)]); s.n = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
    def forward(s, x):
        for a, n in zip(s.a, s.n): x = x + F.silu(a(n(x))) * 0.05
        return x
class WithKernel(nn.Module):
    def __init__(s, n=24, d=4096):
        super().__init__()
        s.w = nn.ParameterList([nn.Parameter(torch.randint(-127, 127, (d, d), dtype=torch.int8), requires_grad=False) for _ in range(n)])
        s.s = nn.Parameter(torch.tensor([0.0005]), requires_grad=False)
    def forward(s, x):
        import comfy_kitchen as ck
        for w in s.w: x = x + ck.int8_linear(x, w, s.s, None, x.dtype, convrot=True, convrot_groupsize=256) * 0.05
        return x

try_graph("A1 200 small Conv1d, (1,64,4096) fp32",  SmallConvs(200, 64), torch.randn(1, 64, 4096, device=dev))
try_graph("A2 200 small Conv1d, (1,64,512) fp32",   SmallConvs(200, 64), torch.randn(1, 64, 512, device=dev))
try_graph("A3 400 tiny Conv1d, (1,32,256) fp32",    SmallConvs(400, 32), torch.randn(1, 32, 256, device=dev))
try_graph("B1 36 x (LN+Linear 2048+silu), 1797 tok fp16", MedBlocks(36, 2048).half(), torch.randn(3, 1797, 2048, device=dev, dtype=torch.float16))
try_graph("B2 36 x (LN+Linear 2048+silu), 128 tok fp16",  MedBlocks(36, 2048).half(), torch.randn(1, 128, 2048, device=dev, dtype=torch.float16))
try:
    try_graph("C  24 x omni int8_linear 4096, 3650 tok bf16", WithKernel(24, 4096), torch.randn(3650, 4096, device=dev, dtype=torch.bfloat16))
except Exception as e:
    print("C setup failed:", type(e).__name__, str(e)[:200])
