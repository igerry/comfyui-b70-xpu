#!/usr/bin/env python3
"""Text-to-image bench with core nodes only, wired like the official blueprints.

Krea-2 Turbo, 8 steps. (Qwen-Image 2.1 has its own script, q21_bench.py.)

usage: img_bench.py krea2 <label> <seed> [w=1024] [h=1024] [steps=8] [host]
env:   K2_QUANT=bf16|int8_convrot   IMG_PROMPT=... (change it to defeat ComfyUI's node cache)
Prints one JSON line with the server-side execution time (execution_start -> success).
"""
import json, os, sys, time, urllib.request

kind, label, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
H = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
HOST = sys.argv[7] if len(sys.argv) > 7 else "http://127.0.0.1:8188"
Q = os.environ.get("K2_QUANT", "bf16")
PROMPT = os.environ.get("IMG_PROMPT",
    "A tabby cat sushi chef behind a wooden counter, pressing rice with a paw, warm lantern light, "
    "shallow depth of field, a small sign on the wall that reads \"OPEN\", photorealistic.")

steps = int(sys.argv[6]) if len(sys.argv) > 6 else 8
wf = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f"krea2_turbo_{Q}.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": PROMPT}},
 "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
 "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
 "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
        "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
 "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "bench/" + label}},
}
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
print(json.dumps({"label": label, "kind": kind, "seed": seed, "status": st.get("status_str"), "server_s": srv,
                  "wall_s": round(time.time() - t0, 1), "steps": steps, "res": f"{W}x{H}",
                  "err": (str(err.get("exception_type", "")) + ": " + str(err.get("exception_message", ""))[:600]) if err else ""},
                 ensure_ascii=False))
