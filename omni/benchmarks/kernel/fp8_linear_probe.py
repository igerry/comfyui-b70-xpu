"""What is omni_xpu_kernel.linear.onednn_w8a16_fp8 worth on B70? At LTX 2.3's real shapes
(M=4992 tokens at 768x512x97f, M=14080 at 1280x720x121f; dim 4096, ffn 16384):
  a) the native W8A16 FP8 GEMM                     (what the FP8 adapter calls)
  b) dequantize fp8 -> bf16, then F.linear          (what ComfyUI's "emulated" path does)
  c) F.linear on a bf16 weight that is already there (the bf16 ceiling: no dequantize at all)
  d) INT8 ConvRot int8_linear, same shape           (what an INT8 build of the model would cost)
Run under inference_mode, as ComfyUI does."""
import time, torch, torch.nn.functional as F
import comfy_kitchen as ck
from omni_xpu_kernel import linear as olin
dev, dt = torch.device("xpu"), torch.bfloat16
def T(fn, n=10):
    for _ in range(3): fn()
    torch.xpu.synchronize(); best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n): fn()
        torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t) / n * 1e3)
    return best
with torch.inference_mode():
    for M in (4992, 14080):
        for K, N, name in ((4096, 4096, "attn proj 4096->4096"), (4096, 16384, "ffn up 4096->16384"), (16384, 4096, "ffn down 16384->4096")):
            x = torch.randn(M, K, device=dev, dtype=dt)
            wf = (torch.randn(N, K, device=dev) * 0.02)
            w8 = wf.to(torch.float8_e4m3fn); wb = wf.to(dt)
            sc = torch.ones(N, device=dev, dtype=torch.float32)
            wi = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8); ws = torch.tensor([0.004], device=dev)
            fl = 2 * M * K * N
            r = {}
            try:
                o = olin.onednn_w8a16_fp8(x, w8, sc, None); r["a native W8A16 fp8"] = T(lambda: olin.onednn_w8a16_fp8(x, w8, sc, None))
                ref = F.linear(x, w8.to(dt))
                rel = float((o.float() - ref.float()).norm() / ref.float().norm())
            except Exception as e:
                r["a native W8A16 fp8"] = float("nan"); rel = float("nan"); print("   native failed:", str(e)[:140])
            r["b dequant + bf16 linear"] = T(lambda: F.linear(x, w8.to(dt)))
            r["c bf16 linear (no dequant)"] = T(lambda: F.linear(x, wb))
            r["d int8 convrot linear"] = T(lambda: ck.int8_linear(x, wi, ws, None, dt, convrot=True, convrot_groupsize=256))
            print(f"M={M:5d} {name:22s} " + "  ".join(f"{k[0]}={v:6.2f}ms({fl/v/1e9:5.0f}T)" for k, v in r.items()) + f"   native-vs-dequant rel {rel:.1e}")
    print("legend: a native W8A16 fp8 | b dequant+bf16 | c bf16 no dequant | d int8 convrot;  (xT) = effective TFLOPS/TOPS")
