"""Attention adapter: admit Qwen-Image 2.1's rectangular self-attention to the BMG D128 CUTE kernel
when it is large enough to win (2026-10-09).

Qwen-Image 2.1 is block-causal: image rows query the prefix (text + reference latents) plus
themselves, so q_len < kv_len at b=1, heads=32, d=128, bf16. The adapter only admitted q == kv
(or the kv=1024 cross shape) at heads=32, so every call went to torch SDPA. Measured on B70
(bench/q21_attn_probe.py, same relative error as torch in every row):
    q 3600 kv  3650 (1280x720 t2i)        cute  2.59 ms  torch  2.41 ms  x0.93
    q 3600 kv  7746 (+1 reference image)  cute  4.70     torch  4.87     x1.04
    q 3600 kv 15938 (+3 references)       cute  9.31     torch 10.17     x1.09
    q 3600 kv 44610 (+10 references)      cute 23.82     torch 32.47     x1.36
    q16384 kv 16434 (2048x2048 t2i)       cute 35.91     torch 51.48     x1.43
So the route is admitted only from kv_len >= OMNIXPU_QWEN21_CUTE_MIN_KV (default 7168); small
images stay on torch, where CUTE is level or slower. OMNIXPU_QWEN21_CUTE=0 disables. Idempotent.
usage: patch_attn_qwen21_cute.py <ComfyUI-OmniXPU dir>
"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "adapters" / "attention.py"
s = p.read_text(encoding="utf-8")
if "_is_qwen_image21_cute_shape" in s:
    print("skip"); sys.exit(0)

a1 = "def _prepare_bmg_d128_bhld_cute(\n"
assert s.count(a1) == 1
s = s.replace(a1, '''def _is_qwen_image21_cute_shape(b, heads, dim_head, q_len, kv_len, dtype):
    """Qwen-Image 2.1 image rows attending to prefix + image rows; see patch_attn_qwen21_cute.py."""
    if os.environ.get("OMNIXPU_QWEN21_CUTE", "1").strip().lower() in ("", "0", "false", "no", "off"):
        return False
    return (
        b == 1
        and heads == 32
        and dim_head == 128
        and dtype == torch.bfloat16
        and q_len < kv_len
        and q_len >= 1024
        and kv_len >= int(os.environ.get("OMNIXPU_QWEN21_CUTE_MIN_KV", "7168"))
    )


''' + a1)

a2 = ("    supported_batch_kind = (\n"
      "        (b == 2 and (self_attention or cross_attention))\n"
      "        or (b == 1 and cross_attention)\n"
      "        or minimax_h3_attention\n"
      "        or animate2_attention\n"
      "    )\n")
assert s.count(a2) == 1
s = s.replace(a2, a2.replace("        or animate2_attention\n",
                             "        or animate2_attention\n"
                             "        or _is_qwen_image21_cute_shape(b, heads, dim_head, q_len, kv_len, q.dtype)\n"))
p.write_text(s, encoding="utf-8"); print("patched", p)
