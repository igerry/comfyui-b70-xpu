#!/usr/bin/env python3
"""H3 turbo 6-step bench, core-nodes only so it runs on both stacks.

The production workflow uses ComfyUI-MiniMax-H3-Turbo's MiniMaxH3TurboLoRA /
MiniMaxH3TurboSampler, which the official omni image does not ship. This graph
uses only core ComfyUI nodes (LoraLoaderModelOnly + KSamplerSelect), so the
identical prompt JSON can be submitted to either server and the only variable
left is the runtime.

usage: h3_bench.py <label> <seed> [steps] [length] [w] [h] [host]
env:   H3_PROMPT_SUFFIX=...  appended to the prompt. Change it between runs: ComfyUI caches nodes by
                             input, so a new seed alone skips the text encoder and the model swap.
       H3_VIDEO_VAE=fp16|int8_convrot   H3_IMAGE=<file in ComfyUI/input>
"""
import json, os, sys, time, urllib.request, urllib.error

label  = sys.argv[1]
seed   = int(sys.argv[2])
steps  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
length = int(sys.argv[4]) if len(sys.argv) > 4 else 124
W      = int(sys.argv[5]) if len(sys.argv) > 5 else 480
H      = int(sys.argv[6]) if len(sys.argv) > 6 else 864
HOST   = sys.argv[7] if len(sys.argv) > 7 else "http://127.0.0.1:8188"

VAE   = os.environ.get("H3_VIDEO_VAE", "fp16")          # fp16 | int8_convrot
IMAGE = os.environ.get("H3_IMAGE", "cat.jpeg")          # first frame, a file in ComfyUI/input
BASE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
PROMPT = (
    "寿司店柜台后，一只健壮的虎斑猫师傅站在高台后制作寿司，"
    "前爪是闭合的肉垫猫爪，四趾并排朝前，爪甲完全收进肉垫里，"
    "手腕到脚趾是一条完整连贯的毛茸茸弧线。"
    "猫爪用肉垫平面向下压实米饭，动作缓慢从容。"
    "固定机位，低视角微仰角，全景深清晰。"
    "环境音：案板轻响、食材摩擦声、店内细微环境声，没有配乐。"
) + os.environ.get("H3_PROMPT_SUFFIX", "")

wf = {
 "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": BASE, "weight_dtype": "default"}},
 "5":  {"class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}},
 "2":  {"class_type": "CLIPLoader",
        "inputs": {"clip_name": "MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                   "type": "minimax", "device": "default"}},
 "3":  {"class_type": "VAELoader", "inputs": {"vae_name": f"minimax_h3_video_vae_{VAE}.safetensors"}},
 "4":  {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "16": {"class_type": "LoadImage", "inputs": {"image": IMAGE, "upload": "image"}},
 "6":  {"class_type": "MiniMaxH3ImageToVideo",
        "inputs": {"clip": ["2", 0], "vae": ["3", 0], "first_frame": ["16", 0],
                   "prompt": PROMPT, "width": W, "height": H, "length": length}},
 "7":  {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["6", 0]}},
 "8":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
 "9":  {"class_type": "BasicScheduler",
        "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
 "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
 "11": {"class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0],
                   "sigmas": ["9", 0], "latent_image": ["6", 1]}},
 "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
 "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
 "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24.0}},
 "15": {"class_type": "SaveVideo",
        "inputs": {"video": ["14", 0], "filename_prefix": "bench/" + label,
                   "format": "auto", "codec": "auto"}},
}

def post(path, body):
    r = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=120).read()

t_submit = time.time()
try:
    pid = json.loads(post("/prompt", {"prompt": wf, "client_id": "h3bench"}))["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED", e.code, e.read().decode()[:2000]); sys.exit(1)

while True:
    h = json.loads(urllib.request.urlopen(HOST + "/history/" + pid, timeout=60).read()).get(pid)
    if h and h["status"]["status_str"] in ("success", "error"):
        break
    if time.time() - t_submit > 7200:
        print("TIMEOUT"); sys.exit(4)
    time.sleep(3)

st = h["status"]
ts = {m[0]: m[1].get("timestamp") for m in st["messages"]}
end = ts.get("execution_success") or ts.get("execution_error")
# server-side timestamps: excludes client queue/poll jitter
srv = round((end - ts["execution_start"]) / 1000, 1) if end and ts.get("execution_start") else None
err = [m[1].get("exception_message", "")[:300] for m in st["messages"] if m[0] == "execution_error"]
print(json.dumps({"label": label, "seed": seed, "status": st["status_str"],
                  "server_s": srv, "wall_s": round(time.time() - t_submit, 1),
                  "steps": steps, "len": length, "res": f"{W}x{H}", "err": err},
                 ensure_ascii=False), flush=True)
