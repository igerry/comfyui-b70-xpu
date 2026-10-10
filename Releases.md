# List of comfyui-b70-xpu Releases

Version numbers are independent of Intel's `llm-scaler-omni` numbering. Each entry names the Intel
image it is built from.
Each release is a git tag `omni-<version>` and a Docker image `comfyui-b70-xpu:<version>`.

## comfyui-b70-xpu

### Latest Release
* `comfyui-b70-xpu:0.1.0-b1` [10/2026] — git tag [`omni-0.1.0-b1`](https://github.com/igerry/comfyui-b70-xpu/tree/omni-0.1.0-b1),
  [release notes](omni/docs/RELEASE_NOTES_0.1.0-b1.md).
  Source build: `cd omni && bash build.sh` produces `comfyui-b70-xpu:0.1.0-b1-comfyui-bmg` (10.8 GB).
  No prebuilt image is published.
  Built from [`intel/llm-scaler-omni:0.2.0-b1`](https://hub.docker.com/layers/intel/llm-scaler-omni/0.2.0-b1/images/sha256-b692be544940c3896bdf11fb3d66bbc089f1214a9e4f6749c34de43db8fddf23).

### Previous Releases
* none
