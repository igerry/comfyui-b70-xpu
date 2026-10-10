#!/usr/bin/env python3
"""Qwen-Image 2.1 text-to-image bench, wired like the official template (orig) and like
Viggle's README (viggle: unmerged LoRA -> QwenImage21Cache -> BasicGuider, ViggleTurboSigmas).

usage: q21_bench.py <orig|viggle> <label> <seed> [w=1280] [h=720] [steps] [host]
env:   Q21_QUANT=bf16|int8_convrot (default bf16)   Q21_CACHE=auto|gpu|cpu|off (KV cache device)   IMG_PROMPT=... (change to defeat the node cache)
"""
import json, os, sys, time, urllib.request, urllib.error
mode, label, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
H = int(sys.argv[5]) if len(sys.argv) > 5 else 720
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 25
HOST = sys.argv[7] if len(sys.argv) > 7 else "http://127.0.0.1:8188"
Q = os.environ.get("Q21_QUANT", "bf16")
CACHE = os.environ.get("Q21_CACHE", "auto")   # auto|gpu|cpu|off
PROMPT = os.environ.get("IMG_PROMPT",
    "A tabby cat sushi chef behind a wooden counter, pressing rice with a paw, warm lantern light, "
    "shallow depth of field, a small wooden sign on the wall that reads \"OPEN\", photorealistic.")
wf = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f"qwen_image_2.1_{Q}.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": f"qwen3vl_8b_{Q}.safetensors", "type": "qwen_image", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": PROMPT, "negative_prompt": "", "resolution": 1024}},
 "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
 "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "bench/" + label}},
}
if mode == "orig":
    wf["5"] = {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": CACHE, "dtype": "default"}}
    wf["7"] = {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1],
               "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
else:
    wf["10"] = {"class_type": "ViggleTurboLora", "inputs": {"model": ["1", 0],
                "lora_name": "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors", "strength": 1.0}}
    wf["5"] = {"class_type": "QwenImage21Cache", "inputs": {"model": ["10", 0], "device": CACHE, "dtype": "default"}}
    wf["11"] = {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["4", 0]}}
    wf["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["13"] = {"class_type": "ViggleTurboSigmas", "inputs": {"latent": ["6", 0], "nodes": "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"}}
    wf["14"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["7"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0],
               "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["6", 0]}}
    steps = 6

def call(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        print(json.dumps({"label": label, "status": "http_error", "body": e.read().decode()[:1500]})); sys.exit(1)
t0 = time.time()
pid = call("/prompt", {"prompt": wf})["prompt_id"]
while True:
    time.sleep(1)
    h = call("/history/" + pid)
    if pid in h and h[pid].get("status", {}).get("completed") is not None:
        st = h[pid]["status"]; break
msgs = {m[0]: m[1] for m in st.get("messages", [])}
ts = {k: v.get("timestamp") for k, v in msgs.items()}
end = ts.get("execution_success") or ts.get("execution_error")
srv = round((end - ts["execution_start"]) / 1000, 1) if end and ts.get("execution_start") else None
err = msgs.get("execution_error", {})
print(json.dumps({"label": label, "mode": mode, "quant": Q, "seed": seed, "status": st.get("status_str"), "server_s": srv,
                  "steps": steps, "res": f"{W}x{H}",
                  "err": (str(err.get("exception_type", "")) + ": " + str(err.get("exception_message", ""))[:700]) if err else ""},
                 ensure_ascii=False))
