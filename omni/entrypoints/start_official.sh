#!/usr/bin/env bash
# A/B helper: start Intel's original stack from the base image (ComfyUI 0.31, torch 2.11) the
# way Intel's README recommends (direct start). Mount models the same way as for the B70 stack.
set -eo pipefail
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:/opt/intel/oneapi/compiler/latest/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/dnnl/latest/lib:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/mpi/latest/lib
export OMNIXPU_INTERPOLATE_FIX=0
unset OMNIXPU_BLOCK_FUSED OMNIXPU_ADALN_FUSED
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
cd /llm/ComfyUI-official
exec python main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --input-directory /data/input \
    --output-directory /data/output \
    --user-directory /data/user \
    "$@"
