# comfyui-b70-xpu: Omni image for Intel Arc Pro B70

A single-XPU ComfyUI image for Intel Arc B-series (Battlemage) cards, built from
[`intel/llm-scaler-omni:0.2.0-b1`](https://github.com/intel/llm-scaler/releases/tag/omni-0.2.0-b1).
It replaces the ComfyUI 0.31 / torch 2.11 stack of that image with upstream ComfyUI master on
torch 2.13, a rebuilt `omni_xpu_kernel`, the XPU-enabled Comfy Kitchen backend on stock
comfy-kitchen 0.2.37, and an extended `ComfyUI-OmniXPU` integration layer.

What changed and what it is worth, with all measurements: [../README.md](../README.md).

> [!IMPORTANT]
> `0.1.0-b1` is available as a source build. It is tested on one Arc Pro B70 with four models
> (MiniMax H3, Krea-2 Turbo, Qwen-Image 2.1, MiniMax Music 3). It does not include the SGLang
> Diffusion, multi-XPU, GGUF or Nunchaku pieces of Intel's image; for those, use Intel's image directly.

## Getting Started with the Docker Image

Build from the `omni` directory:

```bash
cd omni

# Intel Arc B-series / Battlemage
bash build.sh
```

The source-build command assigns this tag to the local image:

```text
comfyui-b70-xpu:<version>-comfyui-bmg
```

The version is defined in [`VERSION`](VERSION). Released versions are listed in
[Releases](../Releases.md).

### Validate the image

```bash
IMAGE=comfyui-b70-xpu:0.1.0-b1-comfyui-bmg

sudo docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    /llm/entrypoints/validate_comfyui_image.sh
```

The check verifies package identity, the Torch ABI of the native wheel, the Kitchen XPU backend,
XPU availability, and that ComfyUI treats INT8 as a native XPU format.

### Run ComfyUI

Mount the existing ComfyUI model directory rather than copying models into the image:

```bash
IMAGE=comfyui-b70-xpu:0.1.0-b1-comfyui-bmg
CONTAINER_NAME=comfyui
COMFYUI_MODEL_DIR=/path/to/comfyui_models
COMFYUI_INPUT_DIR=/path/to/comfyui_input
COMFYUI_OUTPUT_DIR=/path/to/comfyui_output
COMFYUI_USER_DIR=/path/to/comfyui_user

sudo docker run -itd \
    --device=/dev/dri \
    --network=host \
    --shm-size=16g \
    --memory=96g \
    \
    --name="$CONTAINER_NAME" \
    -v "$COMFYUI_MODEL_DIR":/models/host:ro \
    -v "$COMFYUI_INPUT_DIR":/data/input \
    -v "$COMFYUI_OUTPUT_DIR":/data/output \
    -v "$COMFYUI_USER_DIR":/data/user \
    "$IMAGE"
```

Open `http://<host>:8188`. The default command, `/llm/entrypoints/start_comfyui.sh`, is a direct
ComfyUI start on `0.0.0.0:8188` with `/llm/configs/comfyui_host_models.yaml` and the `/data`
directories. It is equivalent to:

```bash
sudo docker run -itd ... "$IMAGE" \
    python main.py --listen 0.0.0.0 \
        --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
        --input-directory /data/input \
        --output-directory /data/output \
        --user-directory /data/user
```

Notes:

- All our runs added `--privileged` (the test host is an LXC container and needs it). Without it the
  image is untested, pinned memory in particular.
- **Inside an LXC container add `--memory=<LXC memory size>`, for example `--memory=96g`.** Docker
  nested in LXC sees the physical host's RAM, so the pinned-memory budget (90% of RAM) is computed
  from the wrong number. ComfyUI honours the cgroup limit set by `--memory`.
- Pinned host memory is on by default and keeps a page-locked copy of every loaded model (up to
  90% of host RAM). Disable it with `-e OMNIXPU_PIN_MEMORY=0` on hosts with less than about 64 GB.
- DynamicVRAM (`--enable-dynamic-vram`) is not used: on XPU it is the slowest way to run a
  workflow that fits in memory.
- Keep host models and mutable data outside `/llm/ComfyUI`, as with Intel's image.
- If your model directory contains symlinks, also mount their targets at the same absolute path.

## Image contents

- upstream [ComfyUI](https://github.com/comfyanonymous/ComfyUI) pinned to
  `a4b5a045e56fc334903db8457b728b64e006119c` (0.39.0), in `/llm/ComfyUI`;
- Python 3.13 venv `/opt/venv-b70` with `torch 2.13.0+xpu` (stock wheel, Intel runtime 2026.0);
- [`omni_xpu_kernel`](omni_xpu_kernel/README.md) `0.2.0b1+torch213.bmg`, built from this repository;
- official `comfy-kitchen==0.2.37` plus the XPU backend
  ([patch](patches/comfy_kitchen-0.2.37-xpu.patch));
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md) with the modules in `fixes/`;
- the benchmark scripts in `/llm/benchmarks`;
- an installed-package snapshot at `/llm/manifests/comfyui-b70-python-freeze.txt`;
- from the base image: Ubuntu 24.04 and the Intel GPU user-space drivers. Intel's ComfyUI 0.31,
  its torch 2.11 venv and the oneAPI 2025.3 toolkit are removed.

No custom nodes other than `ComfyUI-OmniXPU` are installed in the new stack.

## Build and component documentation

- [Image build, acceptance, and bare-metal install](docs/IMAGE_BUILD.md)
- [ComfyUI usage: models, switches, custom nodes](docs/COMFYUI.md)
- [Omni XPU kernel](omni_xpu_kernel/README.md)
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md)
- [Benchmarks](benchmarks/)
- [Patch history with per-change reasoning](docs/history/README.md)
