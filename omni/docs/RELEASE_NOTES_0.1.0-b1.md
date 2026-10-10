## Highlights

### Resources

- **Docker Image**: [gerrywang/comfyui-b70-xpu:0.1.0-b1](https://hub.docker.com/r/gerrywang/comfyui-b70-xpu/tags) (`sha256:7a2c1aff2843f16190236d89aa84ec2f4868ab1b353770ca8e231a36f73cc76b`)
- **Source build**: `cd omni && bash build.sh` → `comfyui-b70-xpu:0.1.0-b1-comfyui-bmg`
- **Base image**: [intel/llm-scaler-omni:0.2.0-b1](https://hub.docker.com/layers/intel/llm-scaler-omni/0.2.0-b1/images/sha256-b692be544940c3896bdf11fb3d66bbc089f1214a9e4f6749c34de43db8fddf23)
- **Image size**: 10.8 GB in 17 layers
- **Tested on**: one Intel Arc Pro B70 (32 GB, 230 W); the benchmark suite was run both bare-metal and inside the image

### What’s new

* **ComfyUI 0.39 XPU stack built from the 0.2.0-b1 image**
  * Upstream ComfyUI master (`a4b5a045`, 0.39.0), Python 3.13, `torch 2.13.0+xpu`.
  * `omni_xpu_kernel` rebuilt for torch 2.13 with the 2026.0 compiler and oneDNN.
  * Stock `comfy-kitchen 0.2.37` with the XPU backend re-applied.
  * Intel's original ComfyUI 0.31 / torch 2.11 / oneAPI stack is removed; the OS and GPU drivers are kept.

* **Compatibility fixes for ComfyUI 0.39** (all three were silent: the startup log was green)
  * XPU is reported as INT8-capable again (H3 sampling 7.3 → 6.2 s/step).
  * XPU `int8_linear` accepts the four arguments added in comfy-kitchen 0.2.37.
  * Pinned-memory hook accepts `evict_active` (Qwen-Image 2.1 crashed on its first step).

* **Pinned host memory on XPU**
  * H3 with a new prompt: 81–83 s → 69–74 s. Applies to any Intel GPU and any model.

* **New model coverage**
  * **Qwen-Image 2.1** (with Viggle Turbo): 25 steps 1280×720 in 15.4–15.8 s (bf16) / 10.3–10.6 s
    (INT8); 2048² in 89.7 s / 67–68 s with large attention routed to CUTE.
  * **MiniMax Music 3**: window batching, dense partial RoPE, GLU folded into the INT8
    down-projection. 2 minutes, 30 steps: 88.4 s → 79.0 s (fp16) → 53.2 s (INT8 DiT).
  * **Krea-2 Turbo**: INT8 fast path for 3-D inputs. 1024², 8 steps: 9.9 s (bf16) / 6.9–7.3 s (INT8).

* **MiniMax H3 against the 0.2.0-b1 image** (6 steps, 124 frames)
  * 480×864: 59.2–59.4 s → 55.7–56.0 s (new seed); 87.5–87.8 s → 69.9–72.4 s (new prompt).
  * 704×1280: 181.5–181.7 s → 169.5–169.8 s; 210.9–213.3 s → 185.1–185.2 s.
  * Fused ConvRot quantisation v2, RMSNorm + AdaLN, RoPE subgroup kernel, producer fusions,
    VideoVAE attention tile, lazy attention rescale.

* **Reproducibility**
  * The two H3 in-layer fusions are off by default: the same seed now gives the same video
    (6 of 6 runs) at a cost of under 1 s.

### Known limits

* Tested on one Arc Pro B70 only; kernels are compiled for `bmg`.
* Not compared against Intel's unpublished `0.2.0-b2` source build (ComfyUI 0.39 / torch 2.14).
* H3 with the INT8 video VAE shows no gain at 480×864 once the prompt changes (74–79 s vs. 70–72 s); cause unknown.
* All container runs used `--privileged`; inside LXC pass `--memory=<LXC memory size>`.
