# Benchmarks

All measurements behind the README. Seconds per job unless stated. What each optimization
contributes: [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## Test machine

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB, 230 W power limit (stock) |
| Host | AMD EPYC 7C13, 96 GB RAM, Debian 13 in an unprivileged LXC container on Proxmox (kernel 7.0.14) |
| Driver | compute-runtime 26.22.38646.4, Level Zero 1.31.0 |
| Stack | ComfyUI master `a4b5a045` (0.39.0), torch 2.13.0+xpu, Python 3.13, comfy-kitchen 0.2.37 + XPU backend, `omni_xpu_kernel 0.2.0b1+torch213.bmg` built from this repo |
| Dates | 2026-10-09 and 10-10. Intel's image was re-run on the same day |

## How the numbers were measured

- **ComfyUI caches node outputs by input.** Re-running with only a new seed skips the text
  encoder and the model swaps around it. That is the "new seed" (resident) number and it mostly
  measures the kernels. A real job has a new prompt: text encoder loaded, run, unloaded, main
  model reloaded. On H3 the difference is 14 – 30 s. **Compare "new prompt" numbers.**
- Times are server-side, from `/history`: `execution_start` → `execution_success`. No queueing, no polling.
- Run-to-run noise is about ±1 s on H3. Ranges are min – max over 2 – 3 runs. Rows marked
  *single run* are exactly that.
- "First" is the first run after another model was on the card (includes loading from page cache).
- Scripts: `../benchmarks/suite.sh` runs everything in the two summary tables against a running server.

## MiniMax H3 (INT8 main model + Turbo LoRA, 6 steps, 124 frames with audio)

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

## Krea-2 Turbo (8 steps)

| | Before upgrade, bf16 | This repo, bf16 | This repo, INT8 |
|---|---|---|---|
| 1024², new seed | 9.9 – 10.2 | 9.9 | **6.9 – 7.3** |
| 1024², new prompt | 13.3 – 13.4 | 14 – 16 | **7.6** |
| 1536×1024 | 19.8 | 23.2 | **11.8** |

With bf16 weights the 26 GB model and its text encoder do not fit together, so every new prompt
swaps them. The INT8 model is 13.5 GB and both stay resident.

## Qwen-Image 2.1

ComfyUI 0.34 and Intel's image cannot load this model, so there is no "before" column.

| | bf16 | INT8 |
|---|---|---|
| Original, 25 steps, 1280×720 | 15.4 – 15.8 | **10.3 – 10.6** |
| Original, 25 steps, 2048×2048 | 89.7 – 89.8 (first: 126.6) | **67 – 68** |
| Viggle Turbo, 6 steps, 1280×720 | **5.1** | 4.2 – 4.4, faces degrade |
| Viggle Turbo, 6 steps, 2048×2048 | **27.5 – 29.4** | not measured |

## MiniMax Music 3 (2 minutes, 30 steps)

| | Before upgrade, fp16 | This repo, fp16 DiT | This repo, INT8 DiT |
|---|---|---|---|
| Same lyrics, new seed | 87.9 – 88.0 | 79.0 | **53.2** |
| New song (new lyrics / new text-encoder seed) | 113 – 121 | 110 – 113 | **87 – 90** |

A new song costs about 26 s of autoregressive token generation, then DiT sampling (80 s fp16 /
57 s INT8), then about 6 s of decoding and saving. The first song of each process is slower
(186.6 s measured once on fp16).

## The Docker image, verified

The numbers above are from the bare-metal install. The image built from this repository
(`comfyui-b70-xpu:0.1.0-b1-comfyui-bmg`, **10.8 GB**) was then run on the same machine with
`../benchmarks/suite.sh`, started with `--privileged --device=/dev/dri --network=host
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
showed up. The image published on Docker Hub is a later build of the same content split into 17
layers; it passed the validator and one job per model (H3 55.8 s new seed) before it was pushed.
Not run in the container: the Turbo sampler node and MultiRef workflows (third-party
nodes), and anything without `--privileged`.
