#!/usr/bin/env python3
"""Per-op microbench of one MiniMax-H3 DiT block at the 480x864x124f shapes,
using the installed omni_xpu_kernel directly (no ComfyUI).

  OMNI_XPU_DEBUG=int8 /root/comfy-xpu/bin/python xpu_microbench.py [M]

Prints ms per op and the equivalent bandwidth / TFLOPS so each kernel can be
judged against the B70 roofline (~456 GB/s, s8 DPAS ~ 2x bf16).
"""
import os, sys, time
import torch
from omni_xpu_kernel import int8, norm, rotary, cute

dev = "xpu"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 16332
HID, H, D, FFN = 5376, 56, 128, 14336
INNER = H * D  # 7168
N_ITER = int(os.environ.get("N_ITER", "10"))
torch.manual_seed(0)
native = int8._get_native()


def bench(name, fn, bytes_moved=0.0, flops=0.0, warm=3):
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
    print(f"{name:38s} {ms:9.3f} ms  {'  '.join(extra)}", flush=True)
    return ms


def mkw(n, k):
    w = torch.randint(-127, 128, (n, k), device=dev, dtype=torch.int8)
    s = torch.rand(n, 1, device=dev, dtype=torch.float32) * 0.01
    return w, s


x = torch.randn(M, HID, device=dev, dtype=torch.bfloat16)
w_norm = torch.ones(HID, device=dev, dtype=torch.bfloat16)
w_qkv, s_qkv = mkw(3 * INNER, HID)
w_out, s_out = mkw(HID, INNER)
w_fc1, s_fc1 = mkw(2 * FFN, HID)
w_fc2, s_fc2 = mkw(HID, FFN)
bf16 = torch.bfloat16
B2 = 2  # bytes per bf16

# adaln table: 9 mod rows (3 timestep levels x 3 modalities), 3 contiguous segments
table = torch.randn(9, HID, device=dev, dtype=bf16)
segs = [(0, 14985, 0), (14985, 15399, 4), (15399, M, 8)] if M == 16332 else [(0, M, 0)]
gather_idx = torch.randint(0, 9, (segs[0][1],), device=dev)
segs_gather = [(segs[0][0], segs[0][1], gather_idx)] + segs[1:]

print(f"M={M}  torch={torch.__version__}  kernel={__import__('omni_xpu_kernel').__version__}")
print(f"xpu={torch.xpu.get_device_name(0)}")
print("-" * 80)

# ---- norm + modulation -----------------------------------------------------
xb = M * HID * B2
bench("rms_norm (M,5376) bf16", lambda: norm.rms_norm(w_norm, x), bytes_moved=2 * xb)


def modulate(segments):
    h = x.clone()
    for a, b, row in segments:
        h[a:b].mul_(1.0 + table[row].to(h.dtype)).add_(table[row].to(h.dtype))
    return h


bench("clone (M,5376) [baseline for below]", lambda: x.clone(), bytes_moved=2 * xb)
bench("modulate mul_/add_ uniform rows", lambda: modulate(segs), bytes_moved=2 * xb + 4 * xb)
bench("modulate mul_/add_ per-token gather", lambda: modulate(segs_gather), bytes_moved=2 * xb + 4 * xb)
other = torch.randn_like(x)
bench("gate addcmul_ uniform rows",
      lambda: [x.addcmul_(other[a:b], table[row].to(bf16)) if False else x[a:b].addcmul_(other[a:b], table[row].to(bf16)) for a, b, row in segs],
      bytes_moved=3 * xb)

# ---- qkv projection (K=5376 -> N=21504), decomposed ------------------------
h = x.clone()
bench("qkv int8_linear full (convrot)", lambda: int8.int8_linear(h, w_qkv, s_qkv, None, bf16, True, 256),
      flops=2 * M * HID * 3 * INNER)
rot = bench("  rotate_convrot (torch.matmul H256)", lambda: native.rotate_convrot(h, 256), bytes_moved=2 * xb)
hr = native.rotate_convrot(h, 256)
bench("  quantize_int8_rowwise", lambda: int8.quantize_int8_rowwise(hr), bytes_moved=xb + xb // 2)
bench("  quantize_int8_rowwise_fused", lambda: native.quantize_int8_rowwise_fused(hr), bytes_moved=xb + xb // 2)
xi, xs = int8.quantize_int8_rowwise(hr)
bench("  int8 GEMM prequantized qkv", lambda: int8.int8_linear_prequantized(xi, xs, w_qkv, s_qkv, None, bf16),
      flops=2 * M * HID * 3 * INNER)
if hasattr(native, "quantize_int8_convrot_fused"):
    bench("  quantize_int8_convrot_fused (radix4)", lambda: native.quantize_int8_convrot_fused(h, 256), bytes_moved=xb + xb // 2)
else:
    print("  quantize_int8_convrot_fused: NOT EXPORTED")
w_bf16 = torch.randn(3 * INNER, HID, device=dev, dtype=bf16)
bench("  torch bf16 matmul same shape (ref)", lambda: torch.nn.functional.linear(h, w_bf16), flops=2 * M * HID * 3 * INNER)
del w_bf16

# ---- rope + attention -------------------------------------------------------
qkv = int8.int8_linear(h, w_qkv, s_qkv, None, bf16, True, 256)  # (M, 21504)
q = qkv[:, :INNER].view(1, M, H, D)
k = qkv[:, INNER:2 * INNER].view(1, M, H, D)
freqs = torch.randn(1, M, 1, 48, 2, 2, device=dev, dtype=bf16).contiguous()
qw = torch.ones(D, device=dev, dtype=bf16)
kw = torch.ones(D, device=dev, dtype=bf16)
bench("rms_rope_split_half_ (q,k in place)", lambda: rotary.rms_kitchen_rope_split_half_(q, k, freqs, qw, kw, 1e-6, 96),
      bytes_moved=2 * 2 * M * INNER * B2)
qb = qkv[:, :INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
kb = qkv[:, INNER:2 * INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
vb = qkv[:, 2 * INNER:].view(M, H, D).transpose(0, 1).unsqueeze(0)
attn_flops = 4.0 * M * M * D * H
bench("attention cute.sdp_bhld_d128", lambda: cute.sdp_bhld_d128(qb, kb, vb), flops=attn_flops)
o = cute.sdp_bhld_d128(qb, kb, vb)
print(f"  attn out shape={tuple(o.shape)} stride={tuple(o.stride())}")
o2 = o.squeeze(0).transpose(0, 1).reshape(M, INNER)
print(f"  out_proj input contiguous={o2.is_contiguous()}")
bench("torch sdpa (ref, may be slow)", lambda: torch.nn.functional.scaled_dot_product_attention(qb, kb, vb), flops=attn_flops, warm=1) if os.environ.get("REF_SDPA") else None

# ---- out_proj (K=7168 -> N=5376) --------------------------------------------
bench("out_proj int8_linear full (convrot)", lambda: int8.int8_linear(o2, w_out, s_out, None, bf16, True, 256),
      flops=2 * M * INNER * HID)
bench("  rotate_convrot K=7168", lambda: native.rotate_convrot(o2, 256), bytes_moved=2 * M * INNER * B2)

# ---- fc1 (K=5376 -> N=28672) -----------------------------------------------
bench("fc1 int8_linear full (convrot)", lambda: int8.int8_linear(h, w_fc1, s_fc1, None, bf16, True, 256),
      flops=2 * M * HID * 2 * FFN)
f1 = int8.int8_linear(h, w_fc1, s_fc1, None, bf16, True, 256)  # (M, 28672)

# ---- fc2 (swiglu, K=14336 -> N=5376), decomposed ---------------------------
bench("fc2 int8_linear full (swiglu+convrot)", lambda: int8.int8_linear(f1, w_fc2, s_fc2, None, bf16, True, 256, "swiglu"),
      flops=2 * M * FFN * HID)
g, u = f1.chunk(2, dim=-1)
fb = M * FFN * B2
bench("  fused_silu_mul_exact_bf16", lambda: native.fused_silu_mul_exact_bf16(g, u), bytes_moved=3 * fb)
act = native.fused_silu_mul_exact_bf16(g, u)
bench("  rotate_convrot K=14336", lambda: native.rotate_convrot(act, 256), bytes_moved=2 * fb)
actr = native.rotate_convrot(act, 256)
bench("  quantize_int8_rowwise K=14336", lambda: int8.quantize_int8_rowwise(actr), bytes_moved=fb + fb // 2)
ai, as_ = int8.quantize_int8_rowwise(actr)
bench("  int8 GEMM prequantized fc2", lambda: int8.int8_linear_prequantized(ai, as_, w_fc2, s_fc2, None, bf16),
      flops=2 * M * FFN * HID)
if hasattr(native, "quantize_int8_convrot_fused"):
    bench("  quantize_int8_convrot_fused K=14336", lambda: native.quantize_int8_convrot_fused(act, 256), bytes_moved=fb + fb // 2)

print("-" * 80)
print("int8 cache stats:", int8.int8_cache_stats())
print(f"peak alloc {torch.xpu.max_memory_allocated() / 2**30:.2f} GB")
