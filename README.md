# comfyui-b70-xpu

**Current ComfyUI on an Intel Arc Pro B70, with the fast paths actually in use.**

[中文说明](README.zh-CN.md)

Intel publishes a ComfyUI image for Arc cards (`intel/llm-scaler-omni`). It is fast, but the
published image (0.2.0-b1) is frozen at ComfyUI 0.31 and torch 2.11, and newer models need a newer
ComfyUI. This repository is what we run instead on one Arc Pro B70 (32 GB): upstream ComfyUI
master on torch 2.13 with Intel's XPU acceleration layer carried over, repaired where the upgrade
broke it, and extended with kernels and fixes of our own.

The repository follows the layout of [intel/llm-scaler](https://github.com/intel/llm-scaler)'s
`omni/` directory, and the image is built **from Intel's published image**, so deployment is
one `bash build.sh` and the run commands are the ones Intel documents.

> Intel's `main` branch now carries an unpublished `0.2.0-b2` source build for ComfyUI 0.39 and
> torch 2.14. We have not built or measured it. Everything here is compared against the published
> `0.2.0-b1` image; when `0.2.0-b2` ships, the comparison should be redone.

Four models are supported and measured: **MiniMax H3** (video with audio), **Krea-2 Turbo**,
**Qwen-Image 2.1** (with Viggle Turbo) and **MiniMax Music 3**.

## Results at a glance

One Arc Pro B70, 230 W. Seconds per job, measured by the server (`execution_start` →
`execution_success`). "New prompt" is what a real job costs; see [How the numbers were measured](#how-the-numbers-were-measured).

**Against Intel's image, same card, same day, same workflow (H3 image-to-video, 6 steps, 124 frames):**

| | Intel image 0.2.0-b1 | This repo | |
|---|---|---|---|
| 480×864, new seed | 59.2 – 59.4 s | 55.7 – 56.0 s | −6% |
| 480×864, **new prompt** | 87.5 – 87.8 s | 69.9 – 72.4 s | **−17 to −20%** |
| 704×1280, new seed | 181.5 – 181.7 s | 169.5 – 169.8 s | −7% |
| 704×1280, **new prompt** | 210.9 – 213.3 s | 185.1 – 185.2 s | **−12%** |

**The other three models**, float weights vs. the official INT8 ConvRot weights on this stack.
(On Intel's image we measured only H3. Krea-2 runs there too but we have not benchmarked it there; Qwen-Image 2.1 cannot load on its ComfyUI 0.31.)

| | Float weights | INT8 ConvRot weights | |
|---|---|---|---|
| Krea-2 Turbo, 8 steps, 1024², new seed | 9.9 s | 6.9 – 7.3 s | −28% |
| Krea-2 Turbo, new prompt | 14 – 16 s | 7.6 s | −50% |
| Qwen-Image 2.1, 25 steps, 1280×720 | 15.4 – 15.8 s | 10.3 – 10.6 s | −33% |
| Qwen-Image 2.1, 25 steps, 2048² | 89.7 – 89.8 s | 67 – 68 s | −25% |
| Qwen-Image 2.1 + Viggle Turbo, 6 steps, 1280×720 | 5.1 s | (not recommended) | |
| Music 3, 2 minutes, 30 steps, new seed | 79.0 s | 53.2 s | −33% |
| Music 3, new song | 110 – 113 s | 87 – 90 s | −21% |

Full tables, including the pre-upgrade stack and every caveat, are in [Benchmarks](#benchmarks).

## Why this exists

1. **Run current models.** Qwen-Image 2.1 needs ComfyUI ≥ 0.37. Intel's published image ships 0.31.0 with
   an acceleration layer pinned to torch 2.11 (`omni_xpu_kernel` refuses other torch minors, the
   comfy-kitchen XPU fork is based on 0.2.28, the node checks `torch == 2.11` in five places).
   Moving to ComfyUI master means rebuilding all three layers.
2. **Keep the speed.** Plain ComfyUI + torch XPU without that layer is about 1.6× slower on H3.
   And after an upgrade, the layer can *look* healthy while being bypassed: our 0.34 → 0.39
   upgrade produced three silent regressions with an all-green startup log.
3. **Go further where the card allows.** ComfyUI never enables pinned host memory on Intel GPUs,
   several kernels had headroom, and three of the four models were falling off the fast path for
   reasons unrelated to the models themselves.

## Quick start (Docker)

Requirements: a Linux host with an Arc B-series GPU and working drivers, Docker with BuildKit,
about 45 GB of free disk during the build (Intel's base image is 21 GB; the result is 10.8 GB), internet access during the build.

```bash
git clone https://github.com/igerry/comfyui-b70-xpu.git
cd comfyui-b70-xpu/omni
bash build.sh                 # -> comfyui-b70-xpu:0.1.0-b1-comfyui-bmg
```

The build pulls `intel/llm-scaler-omni:0.2.0-b1`, adds Python 3.13 with `torch 2.13.0+xpu`,
installs the 2026.0 DPC++ compiler in a throw-away stage, compiles `omni_xpu_kernel` from this
repository (about 3 minutes on 16 threads), checks out ComfyUI at a pinned commit, and installs
the Kitchen XPU backend and the `ComfyUI-OmniXPU` node. Intel's original ComfyUI 0.31 / torch 2.11 /
oneAPI stack is then removed and the result is written as a single-layer image; what is kept from
Intel's image is the OS and the GPU user-space drivers.

Validate, then run. Mount points and flags are the same as for Intel's image:

```bash
IMAGE=comfyui-b70-xpu:0.1.0-b1-comfyui-bmg

docker run --rm --device=/dev/dri "$IMAGE" /llm/entrypoints/validate_comfyui_image.sh

docker run -itd --name comfyui \
    --device=/dev/dri --network=host --shm-size=16g --memory=96g \
    -v /path/to/comfyui_models:/models/host:ro \
    -v /path/to/comfyui_input:/data/input \
    -v /path/to/comfyui_output:/data/output \
    -v /path/to/comfyui_user:/data/user \
    "$IMAGE"
```

`--memory` is the RAM ComfyUI may count on: the machine's RAM, or the LXC container's memory size
when Docker runs inside LXC (96 GB on the test machine).

ComfyUI is then on `http://<host>:8188`. The default command is
`/llm/entrypoints/start_comfyui.sh` (a direct start; arguments after it go to `main.py`). You can
also run `python main.py ...` yourself, exactly as with Intel's image.

- We ran it with `--privileged` added (our host is an LXC container and needs it; it is also what Intel's
  older instructions use). Without `--privileged` the image is untested, pinned memory in particular.
- **Inside LXC, pass the LXC's memory size: `--memory=96g`** (use your container's value). Docker
  nested in an LXC container reports the physical host's RAM, not the LXC limit: on our machine
  the pinned-memory budget came out as 917 GB instead of 86 GB. ComfyUI honours the cgroup limit
  that `--memory` sets, so the budget becomes 90% of that value. On a plain host this is not needed.
- **Host RAM.** Pinned memory keeps a page-locked copy of every model that has been loaded, up to
  90% of RAM. H3 alone needs about 52 GB. The test machine has 96 GB. With less, add
  `-e OMNIXPU_PIN_MEMORY=0`.
- **Custom nodes.** Mount single nodes into `/llm/ComfyUI/custom_nodes/<name>`; do not mount over
  the whole directory, it holds `ComfyUI-OmniXPU`. Their Python dependencies go into the image's
  venv: `docker exec comfyui pip install ...`.

More: [omni/README.md](omni/README.md), [build details and bare-metal install](omni/docs/IMAGE_BUILD.md),
[models and switches](omni/docs/COMFYUI.md).

### Check that it is working

A green startup log is not proof (see [Compatibility fixes](#1-compatibility-fixes-for-comfyui-039)).
`validate_comfyui_image.sh` covers the static part. On a running server check these three things:

```bash
docker logs comfyui 2>&1 | grep -E "OmniXPU|Native ops"
```

1. `[OmniXPU] omni_xpu_kernel 0.2.0b1+torch213.bmg - available: sdp, norm, rotary, linear_fp8, int8`
2. The line `Native ops: ...` contains `int8_tensorwise` and `convrot_w4a4`. If they appear under
   `emulated ops`, INT8 weights are being de-quantised and computed in floating point.
3. After running a workflow, `python3 omni/benchmarks/xpu_status.py` prints the routing counters;
   attention calls should be going to `cute` / `esimd`, not falling back.

### Model files

Nothing is downloaded for you. These are the file names the benchmark scripts use; all are the
official repackaged files for ComfyUI.

| Model | `diffusion_models/` | `text_encoders/` | `vae/` | `loras/` |
|---|---|---|---|---|
| MiniMax H3 | `minimax_h3_fl2va_pruned_int8_convrot` | `MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot` | `minimax_h3_video_vae_fp16` (or `_int8_convrot`), `minimax_h3_audio_vae_fp32` | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui` (community Turbo LoRA) |
| Krea-2 Turbo | `krea2_turbo_bf16` or `krea2_turbo_int8_convrot` | `qwen3vl_4b_bf16` | `qwen_image_vae` | |
| Qwen-Image 2.1 | `qwen_image_2.1_bf16` or `_int8_convrot` | `qwen3vl_8b_bf16` or `_int8_convrot` | `qwen_image_2.1_vae_bf16` | `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256` + Viggle's `viggle_turbo.py` in `custom_nodes/` |
| Music 3 | `minimax_music3_dit_fp16` or `_int8_convrot` | `minimax_music3_text_encoder_pruned_bf16` | `minimax_music3_dav` | |

Which weights to pick on this card:

| Model | Use | Why |
|---|---|---|
| H3 | INT8 main model (the only practical one in 32 GB). Video VAE: fp16 at 480p, INT8 at 704×1280 | at 480p the INT8 VAE gains nothing once the prompt changes; see the H3 table below |
| Krea-2 Turbo | **INT8** | 28% faster; composition nearly identical; model and text encoder fit in VRAM together, so a new prompt no longer swaps |
| Qwen-Image 2.1, original 25 steps | **INT8** | 33% faster; same quality tier (PSNR 35 dB vs. float) |
| Qwen-Image 2.1 + Viggle Turbo | **bf16** | on INT8 faces turn grey and soft |
| Music 3 | **INT8 DiT, bf16 text encoder** | DiT: 33% faster, audio correlation 1.000. INT8 text encoder: not faster, and changes the audio (correlation 0.854) |

## What we changed

Three upstream layers are involved. Each row says where the change lives.

| Layer | Upstream | In this repo |
|---|---|---|
| Native kernels (SYCL / ESIMD / CUTLASS-SYCL) | `omni_xpu_kernel`, `sycl-tla` | `omni/omni_xpu_kernel/` (full source), `omni/patches/sycl-tla-fmha-b70.patch` |
| Operator dispatch | `comfy-kitchen` + Intel's XPU backend fork | `omni/patches/comfy_kitchen-0.2.37-xpu.patch` |
| ComfyUI integration | `ComfyUI-OmniXPU` custom node | `omni/ComfyUI-OmniXPU/` (full source); our modules are in its `fixes/` |

`omni/patches/vs-intel-0.2.0-b1/` holds the exact diff of the two source directories against the
sources inside Intel's `0.2.0-b1` image, for review.

### 1. Compatibility fixes for ComfyUI 0.39

All three had a startup log that reported every hook as applied.

| What upstream changed | Effect on XPU | Fix |
|---|---|---|
| New `model_management.supports_int8_compute()` returns `False` for Intel XPU | Every INT8 linear layer is flagged "full precision": weights are de-quantised to bf16 on each call. H3 sampling went from 6.2 to 7.3 s/step, with no error | `ComfyUI-OmniXPU/fixes/xpu_int8_compute.py` |
| comfy-kitchen's `int8_linear` gained four parameters that the dispatcher always passes | `TypeError` in the XPU backend the first time an INT8 layer runs | XPU backend wrapper accepts and applies them |
| `pin_memory()` gained an `evict_active` parameter | Qwen-Image 2.1 crashes on its first step (its KV cache is the only caller; H3 and Krea-2 never reach it) | `ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py` |

Also part of the port: building `omni_xpu_kernel` for torch 2.13 / Python 3.13 (compiler and
oneDNN moved to the 2026.0 generation, parallel build, torch-minor whitelist), re-applying
Intel's XPU backend onto comfy-kitchen 0.2.37, and widening the node's torch version checks.

### 2. Pinned host memory (any Intel GPU, any model)

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

### 3. INT8 fast path for inputs that are not 2-D (Krea-2, Qwen-Image 2.1, Music 3)

The kernel's fused "rotate + quantise + GEMM" path only admitted 2-D inputs. H3 passes `(M, K)`;
every other model passes `(B, N, K)` and silently took the generic path (materialise bf16 →
rotate → quantise → GEMM). The XPU backend now flattens contiguous inputs to `(rows, K)` first.

| Full run, INT8 weights | Before | After |
|---|---|---|
| Qwen-Image 2.1, 25 steps, 1280×720 | 11.6 s | 10.3 s |
| Krea-2 Turbo, 1024² | 7.4 s | 6.9 s |
| Music 3 | 66.8 s | 61.8 s |

Qwen-Image 2.1 stays pixel-identical between reruns of the same seed.

### 4. Music 3

| Change | Switch | Full run | Output |
|---|---|---|---|
| Batch the equal-length windows of each step (a 2-minute song is 30 overlapping windows per step, evaluated one by one upstream) | `OMNIXPU_MUSIC3_BATCH` | 87.7 → 83.5 s | correlation 1.0000 |
| Partial RoPE on dense copies. Element-wise ops on strided XPU tensors run at ~80 GB/s, dense ones at ~450 GB/s | `OMNIXPU_MUSIC3_ROPE` | fp16 83.5 → 79.0 s; INT8 61.8 → 58.2 s | sample-identical |
| GLU gate folded into the INT8 down-projection: SiLU × mul + rotation + quantisation in one pass, straight to int8 (needs kernel phase 9) | `OMNIXPU_MUSIC3_GLU` | INT8 58.2 → 53.2 s | correlation 0.99998 |

Altogether: 88.4 s before the upgrade → 79.0 s (fp16) → 53.2 s (INT8 DiT).

### 5. Qwen-Image 2.1: large attention on the CUTE kernel

Qwen-Image 2.1's attention has more keys than queries (prefix + image rows), which the adapter
did not route to the CUTLASS kernel. Measured per call, the kernel loses on small shapes and wins
on large ones (q 3600 / kv 3650: ×0.93; + 3 reference images: ×1.09; + 10 images: ×1.36;
2048² text-to-image: ×1.43), so it is admitted only from `kv_len ≥ 7168`
(`OMNIXPU_QWEN21_CUTE_MIN_KV`). 2048², INT8, 25 steps: **76.7 → 67.8 s**; 1280×720 is unchanged
and pixel-identical. Multi-reference full runs were not measured, only the per-call numbers.

### 6. Native kernels for H3 (September work, carried over and re-verified)

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

### 7. Reproducibility: same seed, same video

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

## Benchmarks

### Test machine

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB, 230 W power limit (stock) |
| Host | AMD EPYC 7C13, 96 GB RAM, Debian 13 in an unprivileged LXC container on Proxmox (kernel 7.0.14) |
| Driver | compute-runtime 26.22.38646.4, Level Zero 1.31.0 |
| Stack | ComfyUI master `a4b5a045` (0.39.0), torch 2.13.0+xpu, Python 3.13, comfy-kitchen 0.2.37 + XPU backend, `omni_xpu_kernel 0.2.0b1+torch213.bmg` built from this repo |
| Dates | 2026-10-09 and 10-10. Intel's image was re-run on the same day |

### How the numbers were measured

- **ComfyUI caches node outputs by input.** Re-running with only a new seed skips the text
  encoder and the model swaps around it. That is the "new seed" (resident) number and it mostly
  measures the kernels. A real job has a new prompt: text encoder loaded, run, unloaded, main
  model reloaded. On H3 the difference is 14 – 30 s. **Compare "new prompt" numbers.**
- Times are server-side, from `/history`: `execution_start` → `execution_success`. No queueing, no polling.
- Run-to-run noise is about ±1 s on H3. Ranges are min – max over 2 – 3 runs. Rows marked
  *single run* are exactly that.
- "First" is the first run after another model was on the card (includes loading from page cache).
- Scripts: `omni/benchmarks/suite.sh` runs everything in the two summary tables against a running server.

### MiniMax H3 (INT8 main model + Turbo LoRA, 6 steps, 124 frames with audio)

| | Intel image | Before upgrade (ComfyUI 0.34) | **This repo, fp16 video VAE** | This repo, INT8 video VAE |
|---|---|---|---|---|
| 480×864, new seed | 59.2 – 59.4 | 55.9 – 56.1 | **55.7 – 56.0** | 52.3 – 52.4 |
| 480×864, new prompt | 87.5 – 87.8 | 69.7 – 74.7 | **69.9 – 72.4** | 77.3 – 78.7 |
| 704×1280, new seed | 181.5 – 181.7 | 169.9 – 170.0 | **169.5 – 169.8** | 163.0 – 163.2 |
| 704×1280, new prompt | 210.9 – 213.3 | 185.4 – 185.5 | **185.1 – 185.2** | 176.8 – 177.2 |

- "Before upgrade" already contained the September kernels and pinned memory, which is why it
  matches the current column: for H3 the October work kept the speed through the ComfyUI upgrade,
  it did not add to it.
- The INT8 video VAE is faster whenever only the seed changes, and at 704×1280 also with a new
  prompt. At 480×864 with a new prompt it shows no gain: 77 – 79 s on bare metal (two runs, slower
  than fp16's 70 – 72 s) and 73.9 s in the container (one run). We have not found out why the
  seed-only gain does not carry over; for 480p production, stay on the fp16 VAE. Same-seed PSNR
  between the two VAEs is 37.8 dB.

Other H3 workflows (seconds; *single runs* unless a range is given):

| | Before upgrade | This repo, fp16 VAE | This repo, INT8 VAE |
|---|---|---|---|
| Third-party Turbo sampler node, right after a model swap / repeated | 69.4 / 57.0 | 73.7 / 61.4 | 93.3 / 57.7 |
| MultiRef, two shots, 2 × 362 frames, euler chain | 562.3 | 556.2 (510 in an earlier run) | 496.0 |

### Krea-2 Turbo (8 steps)

| | Before upgrade, bf16 | This repo, bf16 | This repo, INT8 |
|---|---|---|---|
| 1024², new seed | 9.9 – 10.2 | 9.9 | **6.9 – 7.3** |
| 1024², new prompt | 13.3 – 13.4 | 14 – 16 | **7.6** |
| 1536×1024 | 19.8 | 23.2 | **11.8** |

With bf16 weights the 26 GB model and its text encoder do not fit together, so every new prompt
swaps them. The INT8 model is 13.5 GB and both stay resident.

### Qwen-Image 2.1

ComfyUI 0.34 and Intel's image cannot load this model, so there is no "before" column.

| | bf16 | INT8 |
|---|---|---|
| Original, 25 steps, 1280×720 | 15.4 – 15.8 | **10.3 – 10.6** |
| Original, 25 steps, 2048×2048 | 89.7 – 89.8 (first: 126.6) | **67 – 68** |
| Viggle Turbo, 6 steps, 1280×720 | **5.1** | 4.2 – 4.4, faces degrade |
| Viggle Turbo, 6 steps, 2048×2048 | **27.5 – 29.4** | not measured |

### MiniMax Music 3 (2 minutes, 30 steps)

| | Before upgrade, fp16 | This repo, fp16 DiT | This repo, INT8 DiT |
|---|---|---|---|
| Same lyrics, new seed | 87.9 – 88.0 | 79.0 | **53.2** |
| New song (new lyrics / new text-encoder seed) | 113 – 121 | 110 – 113 | **87 – 90** |

A new song costs about 26 s of autoregressive token generation, then DiT sampling (80 s fp16 /
57 s INT8), then about 6 s of decoding and saving. The first song of each process is slower
(186.6 s measured once on fp16).

### The Docker image, verified

The numbers above are from the bare-metal install. The image built from this repository
(`comfyui-b70-xpu:0.1.0-b1-comfyui-bmg`, **10.8 GB**) was then run on the same machine with
`omni/benchmarks/suite.sh`, started with `--privileged --device=/dev/dri --network=host
--shm-size=16g --memory=96g`. `validate_comfyui_image.sh`: all checks passed. Errors in the
container log over the whole run: 0.

| Seconds | Bare metal | In the container |
|---|---|---|
| H3 480×864, new seed | 55.7 – 56.0 | 55.1 – 55.7 |
| H3 480×864, new prompt | 69.9 – 72.4 | 73.8 |
| H3 480×864, INT8 video VAE, new seed / new prompt | 52.3 – 52.4 / 77.3 – 78.7 | 52.8 / 73.9 |
| H3 704×1280, new seed / new prompt | 169.5 – 169.8 / 185.1 – 185.2 | 171.4 / 187.4 |
| Krea-2 Turbo bf16, new seed / new prompt | 9.9 / 14 – 16 | 9.9 – 10.0 / 12.7 |
| Krea-2 Turbo INT8, new seed | 6.9 – 7.3 | 6.9 |
| Qwen-Image 2.1 bf16, 25 steps, 1280×720 | 15.4 – 15.8 | 15.8 |
| Qwen-Image 2.1 + Viggle Turbo, 6 steps | 5.1 | 5.1 |
| Qwen-Image 2.1 INT8, 1280×720 / 2048² | 10.3 – 10.6 / 67 – 68 | 10.3 / 67.5 – 69.8 |
| Music 3 fp16 DiT, same lyrics | 79.0 | 78.0 |
| Music 3 INT8 DiT, same lyrics / new song | 53.2 / 87 – 90 | 53.1 – 55.0 / 75.2 – 86.3 |

Container values are one or two runs each. Within run-to-run noise the image reproduces the
bare-metal install. With `--memory=96g` the pinned-memory budget inside the container is 80 GB;
without it, on this LXC host, it came out as 917 GB. The startup log lists `convrot_w4a4` and
`int8_tensorwise` under `Native ops`. The GPU drivers in the image are the ones from Intel's image,
older than the host's (the card is reported as `Intel(R) Graphics [0xe223]`); no speed difference
showed up. Not run in the container: the Turbo sampler node and MultiRef workflows (third-party
nodes), and anything without `--privileged`.

## Switches

Environment variables, read at startup. `0` disables. In the image the two H3 fusions are set to 0 through `ENV`.

| Variable | Default | What it does |
|---|---|---|
| `OMNIXPU_PIN_MEMORY` | 1 | Pinned host memory |
| `OMNIXPU_INT8_COMPUTE` | 1 | Tell ComfyUI that XPU computes INT8 natively |
| `OMNIXPU_BLOCK_FUSED` | **0** | H3: producer fusion of a whole DiT block |
| `OMNIXPU_ADALN_FUSED` | **0** | H3: RMSNorm + AdaLN fusion |
| `OMNIXPU_MUSIC3_BATCH` | 1 | Music 3: batch equal-length windows |
| `OMNIXPU_MUSIC3_ROPE` | 1 | Music 3: dense partial RoPE |
| `OMNIXPU_MUSIC3_GLU` | 1 | Music 3: GLU folded into the INT8 down-projection |
| `OMNIXPU_QWEN21_CUTE` / `OMNIXPU_QWEN21_CUTE_MIN_KV` | 1 / 7168 | Qwen-Image 2.1: CUTE attention above this key length |
| `OMNIXPU_CONVROT_FUSED` / `OMNIXPU_CONVROT_SPLIT` | 1 / 16 | Fused rotate + quantise kernel / threads per row |
| `OMNI_H3_RMS_ROPE_SG` | 1 | RoPE subgroup kernel (0 = Intel's original) |
| `OMNIXPU_DEBUG` | 0 | Trace every kernel route and fallback |

Changing any numeric path changes what a given seed produces (6-step Turbo sampling is chaotic
with respect to 1-LSB differences). To reproduce an old output, restore the switches it was made with.

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

## Scope and limits

- Tested on **one** Arc Pro B70. The kernels are compiled for `bmg`, which also covers B580 and
  B60, but nothing here was run on them. Pinned memory and the compatibility fixes do not depend
  on the GPU model.
- Tested with the four models above. Other models run through the same layers but were not measured.
- The stack is pinned (ComfyUI commit, comfy-kitchen 0.2.37, torch 2.13.0). A newer ComfyUI will
  need the checks in [omni/docs/IMAGE_BUILD.md](omni/docs/IMAGE_BUILD.md#upgrading-comfyui-later).
- The power limit was left at the rated 230 W.
- This is a working setup published as-is, not a maintained product.

## Repository layout

Same shape as `intel/llm-scaler`:

```
README.md, README.zh-CN.md   this page
Releases.md                  list of releases
omni/
  build.sh                   builds the image (reads omni/VERSION)
  docker/Dockerfile          FROM intel/llm-scaler-omni:0.2.0-b1
  entrypoints/               start_comfyui.sh, validate_comfyui_image.sh
  configs/                   comfyui_host_models.yaml (models mounted at /models/host)
  tools/                     validate_comfyui_image.py
  omni_xpu_kernel/           native kernels, full source with our changes
  ComfyUI-OmniXPU/           ComfyUI node, full source with our adapters and fixes/
  patches/                   comfy-kitchen XPU backend, sycl-tla; vs-intel-0.2.0-b1/ = our delta
  benchmarks/                the scripts behind the tables; kernel/ has microbenchmarks and probes
  docs/                      IMAGE_BUILD.md, COMFYUI.md, history/ (step-by-step patch scripts)
```

## Releases

Releases follow Intel's convention: a version, a git tag `omni-<version>`, and a Docker image.
Version numbers are our own and independent of Intel's; each release states which Intel image it
is built on (`0.1.0-b1` is built on `intel/llm-scaler-omni:0.2.0-b1`). See [Releases.md](Releases.md).

## Licence and credits

Apache-2.0, the licence of `intel/llm-scaler` (`LICENSE`). What we changed in the upstream
sources, and the BSD-3-Clause part, are listed in [NOTICE.md](NOTICE.md).

This work stands on [intel/llm-scaler](https://github.com/intel/llm-scaler) (`omni_xpu_kernel`,
`ComfyUI-OmniXPU`, the Docker image), [intel/sycl-tla](https://github.com/intel/sycl-tla),
[xiangyuT/comfy-kitchen-xpu](https://github.com/xiangyuT/comfy-kitchen-xpu),
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) and
[ComfyUI](https://github.com/comfyanonymous/ComfyUI). It is an independent project, not
affiliated with or endorsed by any of them.
