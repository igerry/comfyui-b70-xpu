# syntax=docker/dockerfile:1
#
# comfyui-b70-xpu: ComfyUI master for Intel Arc Pro B70, built from Intel's published image.
#
# Intel's image supplies the OS, the GPU user-space drivers (Level Zero, compute-runtime, ocloc)
# and the pinned sycl-tla checkout. Its ComfyUI 0.31 / torch 2.11 / oneAPI 2025.3 stack is
# removed and replaced by:
#
#   /opt/venv-b70   Python 3.13, torch 2.13.0+xpu (brings Intel runtime 2026.0)
#   /llm/ComfyUI    upstream master, pinned commit
#   omni_xpu_kernel +torch213, built from ../omni_xpu_kernel
#   comfy-kitchen 0.2.37 + XPU backend (../patches)
#   ../ComfyUI-OmniXPU
#
# The last stage copies the cleaned filesystem into a fresh image, so the removed files do not
# stay behind in lower layers.
#
# Build with ../build.sh (context is the omni/ directory).
#
ARG BASE_IMAGE=intel/llm-scaler-omni:0.2.0-b1

# --------------------------------------------------------------------------------------
# pybase: Python 3.13 + torch (shared by the kernel build stage and the runtime stage)
# --------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS pybase
SHELL ["/bin/bash", "-c"]
ARG TORCH_VERSION=2.13.0+xpu
ARG TORCHVISION_VERSION=0.28.0+xpu
ARG TORCHAUDIO_VERSION=2.11.0+xpu
ARG ONEDNN_VERSION=2026.0.0
ENV B70_VENV=/opt/venv-b70
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends software-properties-common gpg-agent curl ca-certificates git patch \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends python3.13 python3.13-venv python3.13-dev \
 && rm -rf /var/lib/apt/lists/*
RUN python3.13 -m venv ${B70_VENV} \
 && ${B70_VENV}/bin/pip install --no-cache-dir -U pip setuptools wheel \
 && ${B70_VENV}/bin/pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/xpu
# oneDNN runtime (libdnnl.so.3) for omni_xpu_kernel's INT8 GEMM. torch does not depend on it.
RUN ${B70_VENV}/bin/pip install --no-cache-dir onednn==${ONEDNN_VERSION}

# --------------------------------------------------------------------------------------
# kernel: build omni_xpu_kernel against torch 2.13
# --------------------------------------------------------------------------------------
FROM pybase AS kernel
# torch 2.13+xpu ships Intel runtime 2026.0. Compiler and oneDNN headers must be the same
# generation, or the extension fails at link time or at `import torch` (undefined symbol).
# The base image only has the 2025.3 compiler.
ARG ONEDNN_VERSION=2026.0.0
ARG ONEDNN_API_VERSION=3.11.2
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor --yes -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update -qq \
 && apt-get install -y -qq intel-oneapi-compiler-dpcpp-cpp-2026.0 \
 && rm -rf /var/lib/apt/lists/*
RUN ${B70_VENV}/bin/pip install --no-cache-dir onednn-devel==${ONEDNN_VERSION}
# sycl-tla (CUTLASS-SYCL headers) is already in the base image at the pinned revision.
COPY patches/sycl-tla-fmha-b70.patch /tmp/
RUN cd /llm/sycl-tla && patch -p1 --no-backup-if-mismatch < /tmp/sycl-tla-fmha-b70.patch
RUN rm -rf /llm/omni_xpu_kernel
COPY omni_xpu_kernel /llm/omni_xpu_kernel
ARG XPU_TARGET=bmg
ARG MAX_JOBS=16
# No `set -u`: Intel's env scripts reference unset variables.
# LD_LIBRARY_PATH is rebuilt so that torch's bundled 2026.0 runtime comes first and the
# base image's 2025.3 paths are gone.
RUN set -e; unset LD_LIBRARY_PATH; \
    source /opt/intel/oneapi/compiler/2026.0/env/vars.sh >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=${B70_VENV}/lib:${LD_LIBRARY_PATH:-}; \
    export OMNI_BUILD_JOBS=${MAX_JOBS} OMNI_ONEDNN_VERSION=${ONEDNN_API_VERSION} OMNI_ONEDNN_PIP_VERSION=${ONEDNN_VERSION}; \
    export CUTLASS_SYCL_ROOT=/llm/sycl-tla OMNI_XPU_REQUIRE_CUTE=1 OMNI_XPU_DEVICE=${XPU_TARGET}; \
    icpx --version | head -1; \
    cd /llm/omni_xpu_kernel; \
    ${B70_VENV}/bin/python -m pip wheel . --no-build-isolation --no-deps --wheel-dir /wheels-b70; \
    ls -l /wheels-b70

# --------------------------------------------------------------------------------------
# runtime-comfyui
# --------------------------------------------------------------------------------------
FROM pybase AS assemble
ARG COMFYUI_REPOSITORY=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_COMMIT=a4b5a045e56fc334903db8457b728b64e006119c
ARG COMFYUI_VERSION=0.39.0
ARG COMFY_KITCHEN_VERSION=0.2.37
ARG IMAGE_TAG=dev
ARG XPU_TARGET=bmg
ARG SOURCE_REVISION=unknown
ARG SOURCE_DIRTY=unknown

RUN rm -rf /llm/ComfyUI /llm/entrypoints /llm/tools /llm/manifests /llm/patches \
 && git clone ${COMFYUI_REPOSITORY} /llm/ComfyUI \
 && cd /llm/ComfyUI && git checkout -q ${COMFYUI_COMMIT}

# constraints.txt is the package list of the host that the published numbers were measured on.
COPY docker/constraints.txt /tmp/constraints.txt
RUN ${B70_VENV}/bin/pip install --no-cache-dir -r /llm/ComfyUI/requirements.txt \
      -c /tmp/constraints.txt --extra-index-url https://download.pytorch.org/whl/xpu \
 && ${B70_VENV}/bin/pip freeze | grep -i "^comfy-kitchen==${COMFY_KITCHEN_VERSION}$"

# comfy-kitchen: stock wheel + XPU backend.
COPY patches/comfy_kitchen-0.2.37-xpu.patch /tmp/
RUN cd $(${B70_VENV}/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && patch -p1 --no-backup-if-mismatch < /tmp/comfy_kitchen-0.2.37-xpu.patch

# Native kernels.
COPY --from=kernel /wheels-b70 /tmp/wheels
RUN ${B70_VENV}/bin/pip install --no-cache-dir --no-deps /tmp/wheels/omni_xpu_kernel-*.whl && rm -rf /tmp/wheels

# Sources shipped in the image, like the base image does.
RUN rm -rf /llm/omni_xpu_kernel
COPY omni_xpu_kernel /llm/omni_xpu_kernel
COPY ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
COPY entrypoints/ /llm/entrypoints/
COPY configs/ /llm/configs/
COPY tools/ /llm/tools/
COPY benchmarks/ /llm/benchmarks/
RUN chmod +x /llm/entrypoints/*.sh \
 && mkdir -p /llm/manifests /data/input /data/output /data/user /models/host \
 && ${B70_VENV}/bin/pip freeze > /llm/manifests/comfyui-b70-python-freeze.txt \
 && rm -f /tmp/constraints.txt /tmp/*.patch

# Remove Intel's original stack: its venv (torch 2.11), the oneAPI 2025.3 toolkit, the old
# source checkouts and wheels. At run time the GPU drivers under /usr and the runtime inside
# /opt/venv-b70 are all that is needed.
RUN rm -rf /opt/venv /opt/intel /wheels /llm/sycl-tla /llm/comfy-kitchen-xpu /llm/comfy-aimdo-xpu \
           /root/.cache /tmp/* /var/tmp/* /var/lib/apt/lists/* \
 && apt-get clean

# --------------------------------------------------------------------------------------
# runtime-comfyui: the cleaned filesystem as a single layer
# --------------------------------------------------------------------------------------
FROM scratch AS runtime-comfyui
ARG COMFYUI_COMMIT=a4b5a045e56fc334903db8457b728b64e006119c
ARG COMFYUI_VERSION=0.39.0
ARG COMFY_KITCHEN_VERSION=0.2.37
ARG IMAGE_TAG=dev
ARG XPU_TARGET=bmg
ARG SOURCE_REVISION=unknown
ARG SOURCE_DIRTY=unknown
ARG BASE_IMAGE
COPY --from=assemble / /

ENV PATH=/opt/venv-b70/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv-b70 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    B70_BASE_IMAGE=${BASE_IMAGE} \
    OMNIXPU_BLOCK_FUSED=0 \
    OMNIXPU_ADALN_FUSED=0 \
    B70_IMAGE_VERSION=${IMAGE_TAG} \
    B70_IMAGE_XPU_TARGET=${XPU_TARGET} \
    B70_COMFYUI_VERSION=${COMFYUI_VERSION} \
    B70_COMFYUI_REVISION=${COMFYUI_COMMIT} \
    B70_COMFY_KITCHEN_VERSION=${COMFY_KITCHEN_VERSION} \
    B70_SOURCE_REVISION=${SOURCE_REVISION} \
    B70_SOURCE_DIRTY=${SOURCE_DIRTY}
LABEL org.opencontainers.image.title="comfyui-b70-xpu" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/igerry/comfyui-b70-xpu" \
      org.opencontainers.image.description="ComfyUI master + torch 2.13 XPU with tuned kernels for Intel Arc Pro B70, built from intel/llm-scaler-omni"
WORKDIR /llm/ComfyUI
EXPOSE 8188
CMD ["/llm/entrypoints/start_comfyui.sh"]
