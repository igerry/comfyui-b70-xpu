# ComfyUI usage

## Model placement

The image reads models from `/models/host` through `/llm/configs/comfyui_host_models.yaml`, using
the standard ComfyUI sub-directories (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`, …).
Nothing is downloaded for you. File names used by the benchmark scripts:

| Model | `diffusion_models/` | `text_encoders/` | `vae/` | `loras/` |
|---|---|---|---|---|
| MiniMax H3 | `minimax_h3_fl2va_pruned_int8_convrot` | `MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot` | `minimax_h3_video_vae_fp16` (or `_int8_convrot`), `minimax_h3_audio_vae_fp32` | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui` |
| Krea-2 Turbo | `krea2_turbo_bf16` or `krea2_turbo_int8_convrot` | `qwen3vl_4b_bf16` | `qwen_image_vae` | |
| Qwen-Image 2.1 | `qwen_image_2.1_bf16` or `_int8_convrot` | `qwen3vl_8b_bf16` or `_int8_convrot` | `qwen_image_2.1_vae_bf16` | `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256` |
| Music 3 | `minimax_music3_dit_fp16` or `_int8_convrot` | `minimax_music3_text_encoder_pruned_bf16` | `minimax_music3_dav` | |

Which weights to use on a B70 is in the top-level README ("Model files").

## Workflows

Use ComfyUI's Template Browser for the official workflows of all four models. The scripts in
`benchmarks/` submit the same graphs through the API and are a compact reference for the wiring:

| Script | Graph |
|---|---|
| `h3_bench.py` | H3 image-to-video with the Turbo LoRA, core nodes only |
| `img_bench.py` | Krea-2 Turbo text-to-image |
| `q21_bench.py` | Qwen-Image 2.1, original 25 steps and Viggle Turbo 6 steps |
| `music3_bench.py` | Music 3, the official template flattened |

## Optional nodes

Only `ComfyUI-OmniXPU` is installed. Mount other nodes one by one:

```bash
-v /path/to/viggle_turbo.py:/llm/ComfyUI/custom_nodes/viggle_turbo.py:ro
-v /path/to/ComfyUI-VideoHelperSuite:/llm/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
```

and install their requirements into the image's venv (`docker exec <container> pip install -r ...`).
Viggle Turbo for Qwen-Image 2.1 needs Viggle's single-file node `viggle_turbo.py`.
The third-party H3 Turbo sampler node works for single clips; with the MultiRef two-shot node it
fails (`tensor (11) vs (2)`), use the `LoraLoaderModelOnly` + euler chain there.

## Runtime switches

Environment variables, read at startup. `0` disables.

| Variable | Default | What it does |
|---|---|---|
| `OMNIXPU_PIN_MEMORY` | 1 | Pinned host memory |
| `OMNIXPU_INT8_COMPUTE` | 1 | Tell ComfyUI that XPU computes INT8 natively |
| `OMNIXPU_BLOCK_FUSED` | **0** (image default) | H3: producer fusion of a whole DiT block |
| `OMNIXPU_ADALN_FUSED` | **0** (image default) | H3: RMSNorm + AdaLN fusion |
| `OMNIXPU_MUSIC3_BATCH` | 1 | Music 3: batch equal-length windows |
| `OMNIXPU_MUSIC3_ROPE` | 1 | Music 3: dense partial RoPE |
| `OMNIXPU_MUSIC3_GLU` | 1 | Music 3: GLU folded into the INT8 down-projection |
| `OMNIXPU_QWEN21_CUTE` / `OMNIXPU_QWEN21_CUTE_MIN_KV` | 1 / 7168 | Qwen-Image 2.1: CUTE attention above this key length |
| `OMNIXPU_CONVROT_FUSED` / `OMNIXPU_CONVROT_SPLIT` | 1 / 16 | Fused rotate + quantise kernel / threads per row |
| `OMNI_H3_RMS_ROPE_SG` | 1 | RoPE subgroup kernel (0 = Intel's original) |
| `OMNIXPU_DEBUG` | 0 | Trace every kernel route and fallback |
| `COMFYUI_PORT` | 8188 | Port used by `start_comfyui.sh` |

The two H3 fusions are set to 0 by the image (`ENV`), not by the node: with a bare-metal install
export them yourself. With both on, H3 is under 1 s faster per video and the same seed no longer
reproduces the same video.

## Checking the fast paths on a running server

```bash
docker logs <container> 2>&1 | grep -E "OmniXPU|Native ops"
python3 benchmarks/xpu_status.py          # attention / INT8 routing counters
```

`Native ops:` must list `int8_tensorwise` and `convrot_w4a4`.
