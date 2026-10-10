"""Run-to-run determinism of every native kernel on the H3 fused block path: same inputs, 20 calls, bitwise compare.
Shapes are H3's at 480x864 x 124 frames (M = 16332 rows, hidden 5376, ffn 14336). Under inference_mode."""
import inspect, torch
from omni_xpu_kernel import int8 as oint8, norm as onorm
ni, nn_ = oint8._get_native(), onorm._get_native()
dev, dt = torch.device("xpu"), torch.bfloat16
M, Hd, F2 = 16332, 5376, 14336
def same(a, b):
    if isinstance(a, (tuple, list)): return all(same(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)
def ndiff(a, b):
    if isinstance(a, (tuple, list)): return sum(ndiff(x, y) for x, y in zip(a, b))
    return int((a != b).sum())
def check(name, fn, n=20):
    try:
        ref = fn(); bad = 0; worst = 0
        for _ in range(n):
            o = fn()
            if not same(o, ref): bad += 1; worst = max(worst, ndiff(o, ref))
        torch.xpu.synchronize()
        print(f"  {'DETERMINISTIC' if bad == 0 else 'NON-DETERMINISTIC':18s} {name:58s} differing calls {bad}/{n}" + (f", up to {worst} elements" if bad else ""))
    except Exception as e:
        print(f"  {'ERROR':18s} {name:58s} {type(e).__name__}: {str(e)[:120]}")
with torch.inference_mode():
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(M, Hd, generator=g).to(dev, dt)
    other = torch.randn(M, Hd, generator=g).to(dev, dt)
    w = torch.rand(Hd, generator=g).to(dev, dt) + 0.5
    nmod = 6
    scale = (torch.randn(nmod, Hd, generator=g) * 0.1).to(dev, dt); shift = (torch.randn(nmod, Hd, generator=g) * 0.1).to(dev, dt); gate = (torch.randn(nmod, Hd, generator=g) * 0.1).to(dev, dt)
    ends = torch.tensor([220, 4000, 4300, M, M, M, M, M], dtype=torch.int32, device=dev); mods = torch.tensor([0, 1, 2, 3, 3, 3, 3, 3], dtype=torch.int32, device=dev)
    f1 = torch.randn(M, 2 * F2, generator=g).to(dev, dt)
    wi = torch.randint(-127, 127, (3 * 7168, Hd), generator=g, dtype=torch.int8).to(dev); ws = torch.tensor([0.004], device=dev)
    print("signature:", (ni.rms_norm_mod_convrot_quant.__doc__ or "").split("\n")[0][:230])
    for split in (8, 16):
        check(f"rms_norm_mod_convrot_quant (no residual), split={split}", lambda: ni.rms_norm_mod_convrot_quant(x, None, None, w, scale, shift, ends, mods, 1e-5, 256, split))
        def resid():
            xc = x.clone()
            o = ni.rms_norm_mod_convrot_quant(xc, other, gate, w, scale, shift, ends, mods, 1e-5, 256, split)
            return (o[0], o[1], xc)
        check(f"rms_norm_mod_convrot_quant (+gated residual), split={split}", resid)
        check(f"quantize_int8_convrot_silu_mul_split, split={split}", lambda: ni.quantize_int8_convrot_silu_mul_split(f1, 256, split))
        check(f"quantize_int8_convrot_fused_split, split={split}", lambda: ni.quantize_int8_convrot_fused_split(x, 256, split))
    qi, qs = ni.rms_norm_mod_convrot_quant(x, None, None, w, scale, shift, ends, mods, 1e-5, 256, 8)
    check("int8_linear_prequantized (qkv 5376 -> 21504)", lambda: ni.int8_linear_prequantized(qi, qs, wi, ws, None, 2))
    check("rms_norm_mod (adaln-fused norm)", lambda: nn_.rms_norm_mod(w, x, scale, shift, ends, mods, 1e-5))
    import comfy_kitchen as ck
    H, D = 56, 128
    qq = torch.randn(1, M, H, D, generator=g).to(dev, dt); kk = torch.randn(1, M, H, D, generator=g).to(dev, dt)
    rot = torch.randn(1, M, 1, 48, 2, 2, generator=g).to(dev, torch.float32); one = torch.ones(D, device=dev, dtype=dt)
    def rr():
        a, b = qq.clone(), kk.clone(); ck.rms_rope_split_half_(a, b, rot, one, one, epsilon=1e-5, rot_dim=96); return a, b
    check("rms_rope_split_half_ (kitchen xpu)", rr, n=8)
    wd = torch.randint(-127, 127, (Hd, F2), generator=g, dtype=torch.int8).to(dev)
    check("int8_linear swiglu fc2 via kitchen (14336x2 -> 5376)", lambda: ck.int8_linear(f1, wd, ws, None, dt, convrot=True, convrot_groupsize=256, input_act="swiglu"), n=10)
