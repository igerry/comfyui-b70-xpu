#!/usr/bin/env python3
"""Print the OmniXPUStatus node's report (attention / INT8 routing counters of the running server).
The node has no inputs, so ComfyUI caches it: pass a different integer each call to force a rerun."""
import json, sys, time, urllib.request
import os
HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
n = sys.argv[1] if len(sys.argv) > 1 else str(int(time.time()))
wf = {"1": {"class_type": "OmniXPUStatus", "inputs": {}},
      "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
      "3": {"class_type": "PrimitiveInt", "inputs": {"value": int(n)}},
      "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}}
def call(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data else None, headers={"Content-Type": "application/json"})
    try: return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e: print(e.read().decode()[:1500]); sys.exit(1)
pid = call("/prompt", {"prompt": wf})["prompt_id"]
for _ in range(30):
    time.sleep(0.5); h = call("/history/" + pid)
    if pid in h and h[pid].get("status", {}).get("completed") is not None: break
out = h[pid].get("outputs", {}).get("2", {})
for v in out.values():
    for s in v: print(s)
