# comfyui-b70-xpu

**Current ComfyUI on an Intel Arc Pro B70, with the fast paths actually in use.**

Upstream ComfyUI master (0.39.0) on `torch 2.13.0+xpu`, with Intel's XPU acceleration stack carried
over from [`intel/llm-scaler-omni`](https://hub.docker.com/r/intel/llm-scaler-omni), repaired where
the upgrade silently broke it, and extended with kernels and fixes of our own. Same mount points
and run commands as Intel's image.

Source, documentation and all measurements: **[github.com/igerry/comfyui-b70-xpu](https://github.com/igerry/comfyui-b70-xpu)**
· [中文说明](https://github.com/igerry/comfyui-b70-xpu/blob/main/README.zh-CN.md)

## Quick start

Requirements: Linux, an Intel Arc B-series GPU with working drivers, Docker.

```bash
# 1. Pull (10.8 GB in 17 layers, 4.05 GB to download)
docker pull gerrywang/comfyui-b70-xpu:0.1.0-b1

# 2. Validate: kernels load, GPU is visible, INT8 is native
docker run --rm --device=/dev/dri gerrywang/comfyui-b70-xpu:0.1.0-b1 \
    /llm/entrypoints/validate_comfyui_image.sh

# 3. Run
docker run -itd --name comfyui \
    --device=/dev/dri --network=host --shm-size=16g --memory=96g \
    -v /path/to/comfyui_models:/models/host:ro \
    -v /path/to/comfyui_input:/data/input \
    -v /path/to/comfyui_output:/data/output \
    -v /path/to/comfyui_user:/data/user \
    gerrywang/comfyui-b70-xpu:0.1.0-b1
```

Open `http://<host>:8188`.

**Set `--memory` to the RAM ComfyUI may count on**: the machine's RAM, or the LXC container's
memory size when Docker runs inside LXC. Pinned memory uses up to 90% of it, and Docker nested in
LXC reports the physical host's RAM. With less than about 64 GB, add `-e OMNIXPU_PIN_MEMORY=0`.

All of our runs added `--privileged` (the test host is an LXC container and needs it). Without it
the image is untested.

Model files are not included. Mount your ComfyUI model directory at `/models/host`.

## Results

One Arc Pro B70 (32 GB, 230 W). Seconds per job, server-side. "New prompt" is what a real job
costs: with only a new seed, ComfyUI's node cache skips the text encoder and the model swaps.

![MiniMax H3: Intel's image vs. this image](https://raw.githubusercontent.com/igerry/comfyui-b70-xpu/main/omni/assets/results-h3.svg)

![Float weights vs. INT8 ConvRot weights](https://raw.githubusercontent.com/igerry/comfyui-b70-xpu/main/omni/assets/results-int8.svg)

| MiniMax H3, 6 steps, 124 frames | `intel/llm-scaler-omni:0.2.0-b1` | This image |
|---|---|---|
| 480×864, new prompt | 87.5 – 87.8 s | **69.9 – 72.4 s** |
| 480×864, new seed | 59.2 – 59.4 s | **55.7 – 56.0 s** |
| 704×1280, new prompt | 210.9 – 213.3 s | **185.1 – 185.2 s** |
| 704×1280, new seed | 181.5 – 181.7 s | **169.5 – 169.8 s** |

| On this image | Float weights | INT8 ConvRot weights |
|---|---|---|
| Krea-2 Turbo, 8 steps, 1024², new prompt | 14 – 16 s | **7.6 s** |
| Qwen-Image 2.1, 25 steps, 1280×720 | 15.4 – 15.8 s | **10.3 – 10.6 s** |
| Qwen-Image 2.1, 25 steps, 2048² | 89.7 – 89.8 s | **67 – 68 s** |
| MiniMax Music 3, 2 minutes, new song | 110 – 113 s | **87 – 90 s** |

This image was verified as pulled: fetched anonymously from Docker Hub onto a machine with no local
copy, then the full suite run inside it (29 jobs, all successful, numbers matching the bare-metal
install).

## Which weights to use on a B70

| Model | Use |
|---|---|
| MiniMax H3 | INT8 main model; fp16 video VAE at 480p, INT8 VAE at 704×1280 |
| Krea-2 Turbo | INT8 |
| Qwen-Image 2.1, original 25 steps | INT8 |
| Qwen-Image 2.1 + Viggle Turbo | bf16 (on INT8 faces turn grey and soft) |
| MiniMax Music 3 | INT8 DiT, bf16 text encoder |

## What is in the image

- ComfyUI pinned to `a4b5a045` (0.39.0), in `/llm/ComfyUI`
- Python 3.13 venv with `torch 2.13.0+xpu` (stock wheel, Intel runtime 2026.0)
- `omni_xpu_kernel 0.2.0b1+torch213.bmg`, built from the repository above
- `comfy-kitchen 0.2.37` with the XPU backend
- `ComfyUI-OmniXPU` with pinned host memory, the INT8 capability fix, and the Music 3 and
  Qwen-Image 2.1 optimizations
- From Intel's image: Ubuntu 24.04 and the Intel GPU user-space drivers. Intel's ComfyUI 0.31,
  its torch 2.11 environment and the oneAPI toolkit are removed.

No custom nodes other than `ComfyUI-OmniXPU` are installed. Mount your own into
`/llm/ComfyUI/custom_nodes/<name>`.

## Switches

| Variable | Default | What it does |
|---|---|---|
| `OMNIXPU_PIN_MEMORY` | 1 | Pinned host memory |
| `OMNIXPU_BLOCK_FUSED`, `OMNIXPU_ADALN_FUSED` | 0 | H3 in-layer fusions: under 1 s faster per video, but the same seed no longer reproduces |
| `OMNIXPU_DEBUG` | 0 | Trace every kernel route and fallback |
| `COMFYUI_PORT` | 8188 | Port |

## Scope and limits

- Tested on one Arc Pro B70. The kernels are compiled for `bmg` (B580 and B60 share the target but
  were not run).
- Tested with four models: MiniMax H3, Krea-2 Turbo, Qwen-Image 2.1, MiniMax Music 3.
- Compared against Intel's published `0.2.0-b1` image. Intel's unpublished `0.2.0-b2` source build
  was not measured.

## Tags

- `0.1.0-b1`: built from `intel/llm-scaler-omni:0.2.0-b1`, git tag
  [`omni-0.1.0-b1`](https://github.com/igerry/comfyui-b70-xpu/releases/tag/omni-0.1.0-b1)

## License

Apache-2.0. Independent project, not affiliated with or endorsed by Intel, Comfy Org or the model
vendors. The image keeps the operating system and Intel GPU drivers of Intel's image under the
terms that apply to them.
