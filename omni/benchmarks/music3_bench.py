#!/usr/bin/env python3
"""MiniMax Music 3 bench, derived from ComfyUI's own audio_minimax_music_3 template.

The template's subgraph is expanded into a flat API graph. The text encoder stays on the
pruned bf16 checkpoint: its INT8 build is not faster on this card and changes the audio.

usage: music3_bench.py <label> <seed> [steps=30] [seconds=120] [host]
env:   M3_DIT=fp16|int8_convrot   M3_TE_SEED=<int>
"""
import json, os, sys, time, urllib.request, urllib.error

label = sys.argv[1]
seed = int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
HOST = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:8188"
DIT = os.environ.get("M3_DIT", "fp16")              # fp16 | int8_convrot
TE_SEED = int(os.environ.get("M3_TE_SEED", "222"))  # change it to get a "new song": reruns the autoregressive token stage

GLOBAL_META = ("Global Metadata: Lo-fi hip-hop, chillhop. 78 BPM, D flat major, major scale "
               "with jazzy extensions. Laid-back and dreamy throughout, a gentle warm drift "
               "with a subtle late-night glow.")
LYRICS = ""

wf = {
 "1": {"class_type": "UNETLoader",
       "inputs": {"unet_name": f"minimax_music3_dit_{DIT}.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader",
       "inputs": {"clip_name": "minimax_music3_text_encoder_pruned_bf16.safetensors",
                  "type": "minimax", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_music3_dav.safetensors"}},
 "4": {"class_type": "MiniMaxMusic3TextEncode",
       "inputs": {"clip": ["2", 0], "caption": GLOBAL_META, "lyrics": LYRICS,
                  "seed": TE_SEED, "max_duration": seconds, "cfg_scale": 1.7, "top_k": 50}},
 "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
 "6": {"class_type": "EmptyMiniMaxMusic3LatentAudio",
       "inputs": {"seconds": seconds, "batch_size": 1}},
 "7": {"class_type": "KSampler",
       "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                  "latent_image": ["6", 0], "seed": seed,
                  "steps": steps, "cfg": 1.7, "sampler_name": "euler",
                  "scheduler": "simple", "denoise": 1.0}},
 "8": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
 "9": {"class_type": "SaveAudio",
       "inputs": {"audio": ["8", 0], "filename_prefix": "bench/" + label}},
}

def post(path, body):
    r = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=120).read()

t0 = time.time()
try:
    pid = json.loads(post("/prompt", {"prompt": wf, "client_id": "music3bench"}))["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED", e.code, e.read().decode()[:2500]); sys.exit(1)

while True:
    h = json.loads(urllib.request.urlopen(HOST + "/history/" + pid, timeout=60).read()).get(pid)
    if h and h["status"]["status_str"] in ("success", "error"):
        break
    if time.time() - t0 > 7200:
        print("TIMEOUT"); sys.exit(4)
    time.sleep(5)

st = h["status"]
ts = {m[0]: m[1].get("timestamp") for m in st["messages"]}
end = ts.get("execution_success") or ts.get("execution_error")
srv = round((end - ts["execution_start"]) / 1000, 1) if end and ts.get("execution_start") else None
err = [m[1].get("exception_message", "")[:400] for m in st["messages"] if m[0] == "execution_error"]
out = [f.get("filename") for o in (h.get("outputs") or {}).values() for f in (o.get("audio") or [])]
print(json.dumps({"label": label, "seed": seed, "status": st["status_str"], "server_s": srv,
                  "steps": steps, "seconds": seconds, "files": out, "err": err}, ensure_ascii=False))
