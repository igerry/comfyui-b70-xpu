#!/usr/bin/env python3
"""Server-side seconds (execution_start -> execution_success/error) of the most recent prompt in /history."""
import json, sys, urllib.request
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/history?max_items=1", timeout=30).read())
for pid, v in h.items():
    m = {x[0]: x[1].get("timestamp") for x in v.get("status", {}).get("messages", [])}
    end = m.get("execution_success") or m.get("execution_error")
    print(f"{(end - m['execution_start']) / 1000:.1f} s ({v['status'].get('status_str')})" if end and m.get("execution_start") else "no timing")
