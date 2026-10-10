#!/usr/bin/env bash
set -euo pipefail
exec python /llm/tools/validate_comfyui_image.py "$@"
