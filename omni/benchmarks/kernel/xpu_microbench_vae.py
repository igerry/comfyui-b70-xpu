#!/usr/bin/env python3
"""Per-op microbench of one MiniMax-H3 VideoVAE decoder attention block at the
480p tile shape (1, 32 heads, 1797 tokens, 64), fp16. Run with the venv python.
Env: L (seq, default 1797), N_ITER, OMNI_CUTE_FMHA_SO (to A/B a variant .so).
"""
import os, sys, time
import torch
from omni_xpu_kernel import norm, cute

dev = "xpu"
L = int(os.environ.get("L", "1797"))
H, D = 32, 64
INNER = H * D  # 2048
N_ITER = int(os.environ.get("N_ITER", "50"))
fp16 = torch.float16
torch.manual_seed(0)


def bench(name, fn, bytes_moved=0.0, flops=0.0, warm=5):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fn()
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / N_ITER * 1e3
    extra = []
    if bytes_moved:
        extra.append(f"{bytes_moved / ms / 1e6:7.1f} GB/s")
    if flops:
        extra.append(f"{flops / ms / 1e9:7.1f} TFLOPS")
    print(f"{name:44s} {ms:9.3f} ms  {'  '.join(extra)}", flush=True)
    return ms


so = os.environ.get("OMNI_CUTE_FMHA_SO", "(default)")
print(f"L={L}  cute so={so}")
print("-" * 80)

# qkv as produced by vae.py: (1, L, H, 3D) -> chunk -> q/k/v (1, L, H, D) strided views
qkv = torch.randn(1, L, H, 3 * D, device=dev, dtype=fp16)
q_v, k_v, v_v = torch.chunk(qkv, 3, dim=-1)
w = torch.ones(D, device=dev, dtype=fp16)

# ---- q/k RMSNorm: the torch fallback the adapter currently takes vs ESIMD ---
def torch_rms(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w

qb = L * INNER * 2
bench("rms_norm torch on strided (1,L,32,64) view", lambda: torch_rms(q_v), bytes_moved=2 * qb)
bench("  .contiguous() of the view", lambda: q_v.contiguous(), bytes_moved=2 * qb)
qc = q_v.contiguous().reshape(-1, D)
bench("  ESIMD rms_norm on (L*32, 64)", lambda: norm.rms_norm(w, qc), bytes_moved=2 * qb)
bench("  contiguous + ESIMD", lambda: norm.rms_norm(w, q_v.contiguous().reshape(-1, D)), bytes_moved=4 * qb)

# ---- attention with the exact layout sdp_minimax_h3_vae_d64 expects ---------
q_n = torch_rms(q_v)  # contiguous (1,L,H,D)
k_n = torch_rms(k_v)
qb_ = q_n.transpose(1, 2)
kb_ = k_n.transpose(1, 2)
vb_ = v_v.transpose(1, 2)
print(f"  q strides={tuple(qb_.stride())} v strides={tuple(vb_.stride())}")
flops = 4.0 * L * L * D * H
ms = bench("attention cute.sdp_minimax_h3_vae_d64", lambda: cute.sdp_minimax_h3_vae_d64(qb_, kb_, vb_), flops=flops)
print(f"  x3780 calls = {ms * 3780 / 1e3:.2f} s per 480p decode")
o = cute.sdp_minimax_h3_vae_d64(qb_, kb_, vb_)
bench("  nan_to_num_ on output", lambda: o.nan_to_num_(0.0), bytes_moved=2 * qb)
if os.environ.get("REF_SDPA"):
    bench("torch sdpa (ref)", lambda: torch.nn.functional.scaled_dot_product_attention(qb_, kb_, vb_), flops=flops)

# ---- VAE FFN silu*mul eager (fp16, no fused kernel) ---------------------------
ff = torch.randn(1, L, 2 * 8192, device=dev, dtype=fp16)
g, u = ff.chunk(2, dim=-1)
fb = L * 8192 * 2
bench("ffn silu(gate)*up eager fp16", lambda: torch.nn.functional.silu(g) * u, bytes_moved=5 * fb)
print("-" * 80)
