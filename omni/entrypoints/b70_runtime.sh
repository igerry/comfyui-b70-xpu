#!/usr/bin/env bash
# Image entrypoint. The base image exports LD_LIBRARY_PATH pointing at its own venv and the
# oneAPI 2025.3 runtime. torch 2.13+xpu carries its 2026.0 runtime inside /opt/venv-b70 and
# nothing may shadow it, so the variable is dropped before any command runs.
# OMNIXPU_INTERPOLATE_FIX is a base-image default that the host these numbers come from does not set.
unset LD_LIBRARY_PATH OMNIXPU_INTERPOLATE_FIX
exec "$@"
