# NOTICE

comfyui-b70-xpu
Copyright 2026 Gerry Wang

This repository is licensed under the Apache License 2.0 (`LICENSE`). It is an independent
project and is not affiliated with or endorsed by Intel, Comfy Org, MiniMax, Krea, Alibaba or
Viggle. Model weights are not included and have their own licences.

## Upstream sources included in modified form

| Path | Upstream | Licence |
|---|---|---|
| `omni/omni_xpu_kernel/` | `omni/omni_xpu_kernel` of [intel/llm-scaler](https://github.com/intel/llm-scaler), as shipped in the image `intel/llm-scaler-omni:0.2.0-b1` (source revision `c07d93a2a6106e0ef8d37a7b5247fa2c1f281b47`). Copyright Intel Corporation | Apache-2.0 |
| `omni/ComfyUI-OmniXPU/` | `omni/ComfyUI-OmniXPU` of the same repository and revision. Copyright Intel Corporation | Apache-2.0 |
| `omni/configs/comfyui_host_models.yaml`, `omni/.dockerignore`, the structure of `omni/build.sh` | intel/llm-scaler `main` | Apache-2.0 |
| `omni/patches/comfy_kitchen-0.2.37-xpu.patch` | A patch against [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) 0.2.37. The `backends/xpu/` files and related hunks originate from [xiangyuT/comfy-kitchen-xpu](https://github.com/xiangyuT/comfy-kitchen-xpu) (`575741da`, based on 0.2.28) | Apache-2.0 |
| `omni/patches/sycl-tla-fmha-b70.patch` | A patch against [intel/sycl-tla](https://github.com/intel/sycl-tla) at `2fc09973bfdf15755090fcb0e3b6ad236408a992`. Copyright NVIDIA Corporation and Intel Corporation | BSD-3-Clause |

## Statement of changes (Apache-2.0, section 4b)

The files under `omni/omni_xpu_kernel/` and `omni/ComfyUI-OmniXPU/` were changed by this project
in September and October 2026. The complete, file-by-file delta against the sources in Intel's
image is in `omni/patches/vs-intel-0.2.0-b1/`:

- `omni_xpu_kernel.diff`: 10 files. Torch 2.13/2.14 support and parallel build in `setup.py`;
  new and rewritten kernels in `csrc/` (fused ConvRot quantisation, RMSNorm with modulation,
  RoPE subgroup kernel, producer fusions, oneDNN INT8 post-op entry); attention route and tile
  changes in `cute/`; Python entry points in `int8/` and `norm/`.
- `ComfyUI-OmniXPU.diff`: 10 files. Adapter changes in `adapters/attention.py` and
  `adapters/norm.py`, hooks in `__init__.py`, and seven new modules under `fixes/`.

The Docker image built from `omni/docker/Dockerfile` starts from Intel's published image and keeps
its operating system and Intel GPU user-space drivers, under the terms that apply to them.
