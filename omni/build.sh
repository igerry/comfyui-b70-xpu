#!/usr/bin/env bash
# Build the comfyui-b70-xpu image. Run from anywhere; the Docker context is this directory.
#
#   bash build.sh                                   -> comfyui-b70-xpu:<version>-comfyui-bmg
#   B70_IMAGE_REPOSITORY=igerry/comfyui-b70-xpu bash build.sh
#
# Every pinned component can be overridden through the environment, like Intel's build.sh.
set -euo pipefail

HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY}}}"
NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export HTTP_PROXY HTTPS_PROXY NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
TAG="$(tr -d '[:space:]' < "${SCRIPT_DIR}/VERSION")"
if [ -z "${TAG}" ]; then
    echo "Unable to read the image version from ${SCRIPT_DIR}/VERSION" >&2
    exit 1
fi

DETECTED_SOURCE_REVISION=unknown
DETECTED_SOURCE_DIRTY=unknown
if [ -n "${REPOSITORY_ROOT}" ] && git -C "${REPOSITORY_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    DETECTED_SOURCE_REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
    if [ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain --untracked-files=normal -- omni)" ]; then
        DETECTED_SOURCE_DIRTY=true
    else
        DETECTED_SOURCE_DIRTY=false
    fi
fi
SOURCE_REVISION="${B70_SOURCE_REVISION:-${DETECTED_SOURCE_REVISION}}"
SOURCE_DIRTY="${B70_SOURCE_DIRTY:-${DETECTED_SOURCE_DIRTY}}"

DEVICE_TARGET="${XPU_TARGET:-${OMNI_XPU_DEVICE:-bmg}}"
case "${DEVICE_TARGET}" in
    bmg) ;;
    *)
        echo "Unsupported XPU target '${DEVICE_TARGET}'; only bmg (Arc B-series) is built and tested here" >&2
        exit 1
        ;;
esac

BASE_IMAGE="${B70_BASE_IMAGE:-intel/llm-scaler-omni:0.2.0-b1}"
TORCH_VERSION="${OMNI_TORCH_VERSION:-2.13.0+xpu}"
TORCHVISION_VERSION="${OMNI_TORCHVISION_VERSION:-0.28.0+xpu}"
TORCHAUDIO_VERSION="${OMNI_TORCHAUDIO_VERSION:-2.11.0+xpu}"
ONEDNN_VERSION="${OMNI_ONEDNN_VERSION:-2026.0.0}"
BUILD_MAX_JOBS="${MAX_JOBS:-16}"
IMAGE_REPOSITORY="${B70_IMAGE_REPOSITORY:-comfyui-b70-xpu}"
COMFYUI_REPOSITORY="${COMFYUI_REPOSITORY:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_COMMIT="${COMFYUI_COMMIT:-a4b5a045e56fc334903db8457b728b64e006119c}"
COMFYUI_VERSION="${COMFYUI_VERSION:-0.39.0}"
KITCHEN_VERSION="${COMFY_KITCHEN_VERSION:-0.2.37}"

DOCKERFILE_PATH="${SCRIPT_DIR}/docker/Dockerfile"
DOCKER_TARGET=runtime-comfyui
IMAGE_NAME="${IMAGE_REPOSITORY}:${TAG}-comfyui-${DEVICE_TARGET}"

cd "${SCRIPT_DIR}"

DOCKER_ARGS=(
    -f "${DOCKERFILE_PATH}"
    --target "${DOCKER_TARGET}"
    -t "${IMAGE_NAME}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "TORCH_VERSION=${TORCH_VERSION}"
    --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}"
    --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}"
    --build-arg "ONEDNN_VERSION=${ONEDNN_VERSION}"
    --build-arg "IMAGE_TAG=${TAG}"
    --build-arg "XPU_TARGET=${DEVICE_TARGET}"
    --build-arg "MAX_JOBS=${BUILD_MAX_JOBS}"
    --build-arg "COMFYUI_REPOSITORY=${COMFYUI_REPOSITORY}"
    --build-arg "COMFYUI_COMMIT=${COMFYUI_COMMIT}"
    --build-arg "COMFYUI_VERSION=${COMFYUI_VERSION}"
    --build-arg "COMFY_KITCHEN_VERSION=${KITCHEN_VERSION}"
    --build-arg "SOURCE_REVISION=${SOURCE_REVISION}"
    --build-arg "SOURCE_DIRTY=${SOURCE_DIRTY}"
    --build-arg "https_proxy=${HTTPS_PROXY}"
    --build-arg "http_proxy=${HTTP_PROXY}"
    --build-arg "no_proxy=${NO_PROXY}"
)

set -x
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" .
