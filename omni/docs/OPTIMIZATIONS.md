# Optimizations

Everything this repository changes relative to Intel's `0.2.0-b1` image, with what each change
is worth. Summary and quick start: [README](../../README.md). Full benchmark tables: [BENCHMARKS.md](BENCHMARKS.md).

Three upstream layers are involved. Each row says where the change lives.

| Layer | Upstream | In this repo |
|---|---|---|
| Native kernels (SYCL / ESIMD / CUTLASS-SYCL) | `omni_xpu_kernel`, `sycl-tla` | `omni_xpu_kernel/` (full source), `patches/sycl-tla-fmha-b70.patch` |
| Operator dispatch | `comfy-kitchen` + Intel's XPU backend fork | `patches/comfy_kitchen-0.2.37-xpu.patch` |
| ComfyUI integration | `ComfyUI-OmniXPU` custom node | `ComfyUI-OmniXPU/` (full source); our modules are in its `fixes/` |

`patches/vs-intel-0.2.0-b1/` holds the exact diff of the two source directories against the
sources inside Intel's `0.2.0-b1` image, for review.

## 1. Compatibility fixes for ComfyUI 0.39

All three had a startup log that reported every hook as applied.

| What upstream changed | Effect on XPU | Fix |
|---|---|---|
| New `model_management.supports_int8_compute()` returns `False` for Intel XPU | Every INT8 linear layer is flagged "full precision": weights are de-quantised to bf16 on each call. H3 sampling went from 6.2 to 7.3 s/step, with no error | `ComfyUI-OmniXPU/fixes/xpu_int8_compute.py` |
| comfy-kitchen's `int8_linear` gained four parameters that the dispatcher always passes | `TypeError` in the XPU backend the first time an INT8 layer runs | XPU backend wrapper accepts and applies them |
| `pin_memory()` gained an `evict_active` parameter | Qwen-Image 2.1 crashes on its first step (its KV cache is the only caller; H3 and Krea-2 never reach it) | `ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py` |

Also part of the port: building `omni_xpu_kernel` for torch 2.13 / Python 3.13 (compiler and
oneDNN moved to the 2026.0 generation, parallel build, torch-minor whitelist), re-applying
Intel's XPU backend onto comfy-kitchen 0.2.37, and widening the node's torch version checks.

## 2. Pinned host memory (any Intel GPU, any model)

ComfyUI sets up pinned memory only when `is_nvidia() or is_amd()`, and pins through CUDA's
host-register call. On XPU every model load and unload is a pageable copy. Measured on this
machine: host→device 1.87 GB/s pageable vs. 26.75 GB/s pinned; device→host 2.14 vs. 9.56 GB/s.

`ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py` sets the same budget ComfyUI uses elsewhere, pins by copy-and-swap
(torch-xpu has no register API), and makes whole-model unloads land in pinned memory.

- H3, every new prompt (text encoder in, text encoder out, main model back in): **81 – 83 s → 69 – 74 s**.
- Same-seed reruns are unchanged (nothing is swapped).
- Under rotation of all four models on a 96 GB host the benefit is mixed: some switches get 22 – 24 s
  faster, one gets 14 – 21 s slower. Our reading is host-memory pressure (pinned copies plus page
  cache exceed RAM and a model is re-read from disk), inferred from two sequences, not directly sampled.

Intel's image has no pinned memory in either of its start modes.

## 3. INT8 fast path for inputs that are not 2-D (Krea-2, Qwen-Image 2.1, Music 3)

The kernel's fused "rotate + quantise + GEMM" path only admitted 2-D inputs. H3 passes `(M, K)`;
every other model passes `(B, N, K)` and silently took the generic path (materialise bf16 →
rotate → quantise → GEMM). The XPU backend now flattens contiguous inputs to `(rows, K)` first.

| Full run, INT8 weights | Before | After |
|---|---|---|
| Qwen-Image 2.1, 25 steps, 1280×720 | 11.6 s | 10.3 s |
| Krea-2 Turbo, 1024² | 7.4 s | 6.9 s |
| Music 3 | 66.8 s | 61.8 s |

Qwen-Image 2.1 stays pixel-identical between reruns of the same seed.

## 4. Music 3

| Change | Switch | Full run | Output |
|---|---|---|---|
| Batch the equal-length windows of each step (a 2-minute song is 30 overlapping windows per step, evaluated one by one upstream) | `OMNIXPU_MUSIC3_BATCH` | 87.7 → 83.5 s | correlation 1.0000 |
| Partial RoPE on dense copies. Element-wise ops on strided XPU tensors run at ~80 GB/s, dense ones at ~450 GB/s | `OMNIXPU_MUSIC3_ROPE` | fp16 83.5 → 79.0 s; INT8 61.8 → 58.2 s | sample-identical |
| GLU gate folded into the INT8 down-projection: SiLU × mul + rotation + quantisation in one pass, straight to int8 (needs kernel phase 9) | `OMNIXPU_MUSIC3_GLU` | INT8 58.2 → 53.2 s | correlation 0.99998 |

Altogether: 88.4 s before the upgrade → 79.0 s (fp16) → 53.2 s (INT8 DiT).

## 5. Qwen-Image 2.1: large attention on the CUTE kernel

Qwen-Image 2.1's attention has more keys than queries (prefix + image rows), which the adapter
did not route to the CUTLASS kernel. Measured per call, the kernel loses on small shapes and wins
on large ones (q 3600 / kv 3650: ×0.93; + 3 reference images: ×1.09; + 10 images: ×1.36;
2048² text-to-image: ×1.43), so it is admitted only from `kv_len ≥ 7168`
(`OMNIXPU_QWEN21_CUTE_MIN_KV`). 2048², INT8, 25 steps: **76.7 → 67.8 s**; 1280×720 is unchanged
and pixel-identical. Multi-reference full runs were not measured, only the per-call numbers.

## 6. Native kernels for H3 (September work, carried over and re-verified)

Per-layer microbenchmarks at real shapes; the sum of the per-block microbenchmark (140.3 ms)
matches the profiled layer time (141.6 ms).

| Kernel | What changed | Microbench | Numerics |
|---|---|---|---|
| Fused ConvRot + INT8 quantise, v2 | Intel's kernel was slow because the first Hadamard stage ran 64 scalar butterflies. Rewritten as 4 vector ops on a 2-D view; one row per 16 threads; rotated values stay in registers for the second pass | K=14336: 4.13 → 1.43 ms (490 GB/s) | scale identical; 1–3 int8 values per 10⁸ differ by 1 LSB |
| Attention: lazy rescale | Skip the 128 multiplies of the output accumulator when no row maximum in the subgroup changed | 71.7 → 70.4 ms | bit-identical |
| RMSNorm with AdaLN modulation folded in | scale/shift applied in the normalisation pass | 2.03 → 0.67 ms | 1 LSB |
| RoPE subgroup kernel | 16-lane subgroup per row, no shared local memory, no barrier | 2.75 → 1.90 ms (492 GB/s) | max 1 LSB |
| "Producer" fusions: SiLU×mul → int8, and (residual +) norm + modulation → int8 | The op in front of each INT8 GEMM writes int8 directly instead of bf16 that is re-read and quantised | 4.03 → 2.20 ms; 2.23 → 1.18 ms | bit-identical / ~1500 per 10⁸ at 1 LSB |
| H3 video-VAE attention tile | KV tile 64 + 3 pipeline stages, for this route only | 0.346 → 0.292 ms | 40.9 dB |
| Adapter: norm on VAE q/k views | 7560 calls per video no longer fall back to torch | 0.33 → 0.21 ms | |

On a full 480p run these took the same-seed time from 58.0 to 56.3 – 56.6 s. Two of them (the block
fusion and the AdaLN fusion) are **off by default** in this repo; see the next section.

## 7. Reproducibility: same seed, same video

With everything enabled, the same H3 seed produced visibly different videos from run to run
(30.8 – 37.7 dB between runs). We ran each configuration six times and compared latents bit by bit:

| Configuration | Distinct latents | Relative latent difference | Video |
|---|---|---|---|
| Upstream only (all of our code and the XPU backend off) | 4 in 4 runs | 5e-6 or 5e-2 | 2 runs differ (36.4 dB) |
| XPU backend on, both H3 fusions off | 3 in 6 runs | ~1e-6 | **6 of 6 identical** |
| Everything on | 4 in 6 runs | 4e-2 … 1.2e-1 | 4 runs differ |

The source is upstream: a ~1e-6 perturbation that appears whenever the model has been reloaded
and lands on one of a few fixed results. It is there with all of our code disabled. Our two H3
in-layer fusions amplify it to 4 – 12%; without them it stays at 1e-6 and the frames are identical.
Decoding is deterministic; the fork happens in sampling.

So `OMNIXPU_BLOCK_FUSED` and `OMNIXPU_ADALN_FUSED` default to `0`. Cost: under 1 s per video
(55.7 – 56.0 s vs. 54.9 – 55.2 s). Qwen-Image 2.1 is pixel-identical between reruns either way.
Krea-2 is not, with or without our code. Where the upstream perturbation comes from is not known.

## What did not help

Recorded so that nobody has to repeat it.

- **torch 2.14.1.** A complete side-by-side environment was built (needs C++20, compiler 2026.1,
  oneDNN 2026.0.2). Steady-state times for all four models are the same as on 2.13. No model needs
  it. XPU graph capture (new in 2.14) gives 4 – 13× on workloads made of hundreds of tiny ops and
  1.0× on compute-bound ones; none of these four models has a stage where it pays.
- **FP8 weights.** The card has no FP8 matrix multiply. The native FP8 path runs at bf16 speed
  (130 vs. 129 TFLOPS); INT8 runs at 207 – 238. For speed on this card, use INT8.
- **A faster attention kernel.** The H3 attention kernel is at 82% of a sustained bf16 GEMM on the
  same silicon. Tile sizes, pipeline stages, keeping Q in registers, splitting the dependency
  chain, different subgroup layouts: all flat or slower. The card sits at its 230 W limit, so
  time = cycles ÷ clock, and the clock depends on how many bits toggle (random data 70.8 ms at
  1983 MHz; all-zero data 52.5 ms at 2633 MHz, same instruction count). Benchmarks that feed an
  attention kernel constant or low-entropy data overstate it by about 11%. Upstream `sycl-tla`
  main is slower than the pinned revision on this card (147 vs. 63 ms on its own example).
- **Sharing one activation quantisation across q/k/v.** 13% in a microbenchmark, but it needs
  tensor version counters and ComfyUI runs under `torch.inference_mode()`, where they do not exist.
  Reverted. Microbenchmarks of anything touching tensor metadata must run under `inference_mode`.
- **INT8 text encoder for Music 3**, **CUTE attention for Music 3** (sequences too short),
  **SwiGLU folded into the GEMM post-op** (slower), **Intel's `--enable-dynamic-vram` entrypoint**
  (the slowest way to start on XPU: 90.6 s where a direct start takes 59 s).
