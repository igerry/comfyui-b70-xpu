"""Clock under the 230 W limit versus data content (same kernel, same shape). It uses softmax_scale 1.0 on random inputs, i.e. logits ~11x wider than
real attention (scale 1/sqrt(128)): P is nearly one-hot, the P*V multiplies see mostly zeros, the card draws less
power per instruction and - sitting on its 230 W limit - clocks higher. Same kernel, same shape, q pre-multiplied."""
import glob, os, time, threading, torch
import omni_xpu_kernel
so = os.environ.get("OMNI_CUTE_FMHA_SO") or glob.glob(os.path.join(os.path.dirname(omni_xpu_kernel.__file__), "cute", "*cute_fmha*.so"))[0]
torch.ops.load_library(so); op = torch.ops.cute_fmha.sdp_bhld_d128
FREQ = glob.glob("/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq")[0]
HW = glob.glob(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FREQ)))) + "/hwmon/hwmon*/energy1_input")
M, H, D = 16332, 56, 128; INNER = H * D
dev, dt = torch.device("xpu"), torch.bfloat16
def run(q, k, v, n=10, rounds=3):
    for _ in range(3): op(q, k, v)
    torch.xpu.synchronize(); best = 1e9; fs = []; stop = [False]; es = []
    def samp():
        while not stop[0]:
            try:
                fs.append(int(open(FREQ).read()))
                if HW: es.append((time.perf_counter(), int(open(HW[0]).read())))
            except Exception: pass
            time.sleep(0.02)
    th = threading.Thread(target=samp, daemon=True); th.start()
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(n): op(q, k, v)
        torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t) / n * 1e3)
    stop[0] = True; th.join(); fs.sort(); mhz = fs[len(fs) // 2]
    w = (es[-1][1] - es[0][1]) / 1e6 / (es[-1][0] - es[0][0]) if len(es) > 1 else 0
    return best, mhz, w
with torch.inference_mode():
    torch.manual_seed(0)
    qkv = torch.randn(M, 3 * INNER, device=dev, dtype=dt)
    q = qkv[:, :INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
    k = qkv[:, INNER:2 * INNER].view(M, H, D).transpose(0, 1).unsqueeze(0)
    v = qkv[:, 2 * INNER:].view(M, H, D).transpose(0, 1).unsqueeze(0)
    def mk(t):
        t = t.to(dt)
        return (t[:, :INNER].view(M, H, D).transpose(0, 1).unsqueeze(0), t[:, INNER:2 * INNER].view(M, H, D).transpose(0, 1).unsqueeze(0),
                t[:, 2 * INNER:].view(M, H, D).transpose(0, 1).unsqueeze(0))
    g = torch.Generator(device="cpu").manual_seed(0)
    cases = (
        ("randn (dense mantissas, like real activations)", torch.randn(M, 3 * INNER, generator=g)),
        ("small integers -4..4 (few mantissa bits set)",  torch.randint(-4, 5, (M, 3 * INNER), generator=g).float()),
        ("values in {-1, 0, 1}",                          torch.randint(-1, 2, (M, 3 * INNER), generator=g).float()),
        ("randn rounded to 1/4 steps",                    (torch.randn(M, 3 * INNER, generator=g) * 4).round() / 4),
        ("randn, 50% of entries zeroed",                  torch.randn(M, 3 * INNER, generator=g) * (torch.rand(M, 3 * INNER, generator=g) > 0.5)),
        ("all zeros",                                     torch.zeros(M, 3 * INNER)),
    )
    for name, t in cases:
        qq, kk, vv = mk(t.to(dev))
        ms, mhz, w = run(qq, kk, vv)
        print(f"  {name:48s} {ms:7.2f} ms  clock {mhz} MHz  {w:5.0f} W  cycles {ms*mhz/1e3:6.1f}k")
