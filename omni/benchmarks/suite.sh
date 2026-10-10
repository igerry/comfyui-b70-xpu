#!/bin/bash
# Runs the four models against a ComfyUI server and prints the server-side seconds of every run.
# usage: suite.sh [tag=run] [host=http://127.0.0.1:8188]
# Needs python3 (standard library only), the model files listed in the README, and a first-frame
# image named cat.jpeg (or H3_IMAGE=<name>) in ComfyUI/input.
# Each block prints: first run after a model swap, then resident runs, then a run with a new prompt.
T=${1:-run}; HOST=${2:-http://127.0.0.1:8188}
cd "$(dirname "$0")"; P=${PYTHON:-python3}; c() { tail -1 | cut -c1-175; }
NP="A red fox reading a newspaper on a park bench in autumn, the headline reads \"B70\", soft morning light, film photo."
h3() { timeout 900 $P h3_bench.py "$1" "$2" 6 124 ${3:-480} ${4:-864} $HOST 2>&1 | c; }
k2() { timeout 400 $P img_bench.py krea2 "$1" "$2" 1024 1024 8 $HOST | c; }
q()  { timeout 900 $P q21_bench.py "$1" "$2" "$3" ${4:-1280} ${5:-720} 25 $HOST | c; }
m3() { timeout 900 $P music3_bench.py "$1" "$2" 30 120 $HOST 2>&1 | c; }

echo "== H3 i2v, 6 steps, 480x864, 124 frames (first, resident x2, new prompt)"
h3 ${T}_h3_a 311; h3 ${T}_h3_b 312; h3 ${T}_h3_c 313
H3_PROMPT_SUFFIX="门口的风铃轻轻晃动。" h3 ${T}_h3_np 314
echo "== H3 with the INT8 video VAE (first, resident, new prompt)"
export H3_VIDEO_VAE=int8_convrot; h3 ${T}_h3i_a 311; h3 ${T}_h3i_b 312
H3_PROMPT_SUFFIX="柜台上放着一杯热茶。" h3 ${T}_h3i_np 315; unset H3_VIDEO_VAE
echo "== Krea-2 Turbo bf16 (first, resident x2, new prompt)"
k2 ${T}_k2_a 401; k2 ${T}_k2_b 402; k2 ${T}_k2_c 403; IMG_PROMPT="$NP" k2 ${T}_k2_np 404
echo "== Krea-2 Turbo INT8 (first, resident x2)"
export K2_QUANT=int8_convrot; k2 ${T}_k2i_a 401; k2 ${T}_k2i_b 402; k2 ${T}_k2i_c 403; unset K2_QUANT
echo "== Qwen-Image 2.1 bf16: original 25 steps x2, Viggle Turbo 6 steps x3 (needs viggle_turbo.py)"
q orig ${T}_q_a 701; q orig ${T}_q_b 702; q viggle ${T}_qv_a 701; q viggle ${T}_qv_b 702; q viggle ${T}_qv_c 704
echo "== Qwen-Image 2.1 INT8: 1280x720 x2, 2048x2048 x2"
export Q21_QUANT=int8_convrot; q orig ${T}_qi_a 701; q orig ${T}_qi_b 702; q orig ${T}_qi2k_a 705 2048 2048; q orig ${T}_qi2k_b 706 2048 2048; unset Q21_QUANT
echo "== Music 3, 120 s, 30 steps: fp16 DiT x2, INT8 DiT x2 (same lyrics, new seed)"
m3 ${T}_m3_a 601; m3 ${T}_m3_b 602; export M3_DIT=int8_convrot; m3 ${T}_m3i_a 601; m3 ${T}_m3i_b 602
echo "== Music 3 INT8 DiT, new song (text-encoder seed changes, so the token stage runs again)"
M3_TE_SEED=3001 m3 ${T}_m3n_a 3001; M3_TE_SEED=3002 m3 ${T}_m3n_b 3002; unset M3_DIT
echo "== routing counters"; COMFY_HOST=$HOST $P xpu_status.py | sed -n "/Attention calls/,\$p" | cut -c1-120 | head -12
echo "=== SUITE DONE ==="
