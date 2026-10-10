# Image build, acceptance, and installing without Docker

## Building the image

```bash
cd omni
bash build.sh        # comfyui-b70-xpu:<omni/VERSION>-comfyui-bmg
```

`build.sh` mirrors Intel's: every pinned component is an environment variable.

| Variable | Default |
|---|---|
| `B70_BASE_IMAGE` | `intel/llm-scaler-omni:0.2.0-b1` |
| `B70_IMAGE_REPOSITORY` | `comfyui-b70-xpu` |
| `OMNI_TORCH_VERSION` / `OMNI_TORCHVISION_VERSION` / `OMNI_TORCHAUDIO_VERSION` | `2.13.0+xpu` / `0.28.0+xpu` / `2.11.0+xpu` |
| `OMNI_ONEDNN_VERSION` | `2026.0.0` (must match torch's bundled Intel runtime) |
| `COMFYUI_REPOSITORY` / `COMFYUI_COMMIT` | upstream / `a4b5a045e56fc334903db8457b728b64e006119c` |
| `COMFY_KITCHEN_VERSION` | `0.2.37` (the XPU patch is written against exactly this version) |
| `MAX_JOBS` | 16 |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | passed through |

Stages of `docker/Dockerfile`:

1. `pybase` — the base image plus Python 3.13 (deadsnakes) and a venv at `/opt/venv-b70` with torch.
2. `kernel` — adds the 2026.0 DPC++ compiler and `onednn-devel`, patches `/llm/sycl-tla`, builds the
   `omni_xpu_kernel` wheel. Not part of the final image.
3. `assemble` — ComfyUI at the pinned commit in `/llm/ComfyUI`, requirements under `docker/constraints.txt` (the package list of the
   measured host), the Kitchen XPU patch, the wheel, the node, entrypoints, configs, tools, benchmarks.

4. `runtime-comfyui` — `FROM scratch` plus one `COPY --from=assemble / /`. The `assemble` stage ends by
   deleting Intel's original stack (`/opt/venv`, `/opt/intel`, old source checkouts and wheels);
   copying into a fresh image is what actually frees that space, since deleting in a later layer
   does not shrink an image.

## Acceptance

```bash
docker run --rm --device=/dev/dri <image> /llm/entrypoints/validate_comfyui_image.sh
```

Checks package identity, the Torch ABI of the native wheel, the Kitchen XPU backend signature,
XPU availability, and that ComfyUI treats XPU INT8 as native. `--no-gpu` skips the device checks.
Then run `benchmarks/suite.sh` against the running server and compare with the README tables.

# Installing without Docker

This is how the machine behind the README numbers is set up: a Debian host (an unprivileged LXC
container, in fact) with the Intel GPU driver stack, a Python 3.13 venv and ComfyUI as a systemd
service. The Docker image performs the same steps; read `../docker/Dockerfile` next to this page.

You still need Intel's image once, for the pinned `sycl-tla` checkout (or clone intel/sycl-tla at `2fc09973`).

## 0. Prerequisites

- Intel Arc B-series GPU with working `level-zero` / `intel-opencl` drivers (`clinfo`, `sycl-ls` see the card).
- Python 3.13, git, `patch`, Docker (only to copy sources out of Intel's image).
- About 25 GB of disk for the image and 10 GB for the venvs.

## 1. Runtime venv and ComfyUI

```bash
python3.13 -m venv /opt/b70/venv
/opt/b70/venv/bin/pip install torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
/opt/b70/venv/bin/pip install onednn==2026.0.0        # libdnnl.so.3 for the native kernels; torch does not pull it in
git clone https://github.com/comfyanonymous/ComfyUI.git /opt/b70/ComfyUI
git -C /opt/b70/ComfyUI checkout a4b5a045e56fc334903db8457b728b64e006119c
/opt/b70/venv/bin/pip install -r /opt/b70/ComfyUI/requirements.txt -c <repo>/omni/docker/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/xpu
```

torch is the stock wheel from the PyTorch XPU index. Nothing in torch is rebuilt.

## 2. Sources

```bash
mkdir -p /opt/b70/src
docker run --rm --entrypoint bash -v /opt/b70/src:/out intel/llm-scaler-omni:0.2.0-b1 -c \
  "cp -r /llm/sycl-tla /out/"
cp -r <repo>/omni/omni_xpu_kernel <repo>/omni/ComfyUI-OmniXPU /opt/b70/src/
cd /opt/b70/src
(cd sycl-tla && patch -p1 < <repo>/omni/patches/sycl-tla-fmha-b70.patch)
```

## 3. Compiler: same generation as torch's runtime

torch 2.13+xpu installs Intel runtime **2026.0** into the venv (`libsycl.so.9`, `libdnnl.so.3.11`).
The DPC++ compiler and the oneDNN headers used to build the kernels must be 2026.0 as well.
A mismatch shows up as a link error or as `undefined symbol: urDeviceWaitExp` at `import torch`.

```bash
curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor --yes -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  > /etc/apt/sources.list.d/oneAPI.list
apt-get update && apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2026.0
```

If a newer compiler is installed later, `/opt/intel/oneapi/compiler/latest` moves to it. Always
source the versioned path, as below.

## 4. Build the kernel wheel

Use a separate build venv (same Python, same torch) so that `onednn-devel` does not end up in the
runtime venv.

```bash
python3.13 -m venv /opt/b70/buildvenv
/opt/b70/buildvenv/bin/pip install torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
/opt/b70/buildvenv/bin/pip install onednn==2026.0.0 onednn-devel==2026.0.0 setuptools wheel

# no `set -u` in this shell: Intel's env scripts reference unset variables
source /opt/intel/oneapi/compiler/2026.0/env/vars.sh
export LD_LIBRARY_PATH=/opt/b70/buildvenv/lib:$LD_LIBRARY_PATH      # torch's runtime first
export OMNI_BUILD_JOBS=16 OMNI_ONEDNN_VERSION=3.11.2 OMNI_ONEDNN_PIP_VERSION=2026.0.0
export CUTLASS_SYCL_ROOT=/opt/b70/src/sycl-tla OMNI_XPU_REQUIRE_CUTE=1 OMNI_XPU_DEVICE=bmg
cd /opt/b70/src/omni_xpu_kernel && rm -rf build
/opt/b70/buildvenv/bin/python -m pip wheel . --no-build-isolation --no-deps --wheel-dir /opt/b70/dist
/opt/b70/venv/bin/pip install --no-deps --force-reinstall /opt/b70/dist/omni_xpu_kernel-*.whl
```

About two and a half minutes on 16 threads. `OMNI_XPU_DEVICE=bmg` covers B580, B60 and B70 (same
device code). Other Intel GPUs need a different target and are untested here.

## 5. comfy-kitchen XPU backend

`requirements.txt` already installed stock `comfy-kitchen==0.2.37`. Patch it in place:

```bash
cd $(/opt/b70/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
patch -p1 < <repo>/omni/patches/comfy_kitchen-0.2.37-xpu.patch
```

## 6. The node

```bash
cp -r /opt/b70/src/ComfyUI-OmniXPU /opt/b70/ComfyUI/custom_nodes/
```

## 7. Run

```bash
cd /opt/b70/ComfyUI
OMNIXPU_BLOCK_FUSED=0 OMNIXPU_ADALN_FUSED=0 /opt/b70/venv/bin/python main.py --listen 0.0.0.0 --port 8188
```

Do not source oneAPI's `setvars.sh` in the shell that runs ComfyUI, and leave `LD_LIBRARY_PATH`
unset: the venv carries the matching runtime.

Then follow "Check that it is working" in the README.

## Upgrading ComfyUI later

Upstream moves quickly and the XPU layer hooks into private call sites. After any ComfyUI or
comfy-kitchen upgrade, a startup log full of `applied` lines does **not** mean the fast paths are
in use. On the 0.34 → 0.39 upgrade three things broke silently (README, "Compatibility fixes").
Check, in this order:

1. The log line `Native ops: ...` must contain `int8_tensorwise` and `convrot_w4a4`. If they are
   listed under `emulated ops`, every INT8 layer is being de-quantised and run in floating point.
2. Compare the signature of every function in `comfy_kitchen/backends/xpu/__init__.py` with its
   counterpart in `backends/eager/`. A missing parameter is the next `TypeError`.
3. Run **all four** models once (`omni/benchmarks/suite.sh`). One of the three regressions only fired on
   Qwen-Image 2.1.
4. `python omni/benchmarks/xpu_status.py` prints the attention routing counters of the running server.
