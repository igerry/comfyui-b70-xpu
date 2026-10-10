#!/usr/bin/env bash
# Default command: direct ComfyUI start (no DynamicVRAM), listening on all interfaces, with
# host models / input / output / user data under the same mount points as Intel's image.
set -euo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
cd /llm/ComfyUI
exec python main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --input-directory /data/input \
    --output-directory /data/output \
    --user-directory /data/user \
    "$@"
