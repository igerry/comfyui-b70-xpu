"""MiniMax Music 3: run the overlapping condition windows as one batch (2026-10-09).

Stock MiniMaxMusic3DiT.forward walks a long latent in overlapping windows and calls the
36-layer transformer once per window. A 120 s song is 30 windows per step, each a batch of 2
(cond + uncond) over a few hundred tokens of width 2048: 1080 layer calls per step at about
2.4 ms each, where the arithmetic alone is worth roughly 0.7 s of the 2.6 s step. On XPU the
rest is per-call launch overhead on small matrices. Windows of equal length are independent,
so they are stacked on the batch axis and sent through the transformer together; shorter
tail windows go through one at a time exactly as before. Same math, different batch size.

OMNIXPU_MUSIC3_BATCH=0 disables; OMNIXPU_MUSIC3_BATCH_MAX caps windows per call (default 32).
Deployed as ComfyUI-OmniXPU/fixes/music3_window_batch.py.
"""
import logging
import os

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_applied = False


def apply():
    global _applied
    if _applied:
        return True, "already applied"
    if os.environ.get("OMNIXPU_MUSIC3_BATCH", "1") == "0":
        return False, "disabled by OMNIXPU_MUSIC3_BATCH=0"
    try:
        import comfy.ldm.minimax_music.dit as m3
    except ImportError as e:
        return False, f"import failed: {e}"
    for name in ("latent_length", "MAX_CONDITION_FRAMES", "CONDITION_HOP_FRAMES"):
        if not hasattr(m3, name):
            return False, f"minimax_music.dit has no {name}"
    max_windows = max(1, int(os.environ.get("OMNIXPU_MUSIC3_BATCH_MAX", "32")))
    _orig_forward = m3.MiniMaxMusic3DiT.forward
    stats = {"batched": 0, "fallback": 0}

    def forward(self, x, timestep, context, conditioning_scale, **kwargs):
        window = m3.latent_length(m3.MAX_CONDITION_FRAMES)
        if x.device.type != "xpu" or x.shape[-1] <= window or timestep.shape[0] != x.shape[0]:
            stats["fallback"] += 1
            return _orig_forward(self, x, timestep, context, conditioning_scale, **kwargs)

        condition = self.aligned_condition(context)
        condition = condition * conditioning_scale[:, :1, :1]
        if condition.shape[-1] < x.shape[-1]:
            condition = torch.nn.functional.pad(condition, (0, x.shape[-1] - condition.shape[-1]))
        else:
            condition = condition[..., :x.shape[-1]]

        hop = m3.latent_length(m3.CONDITION_HOP_FRAMES)
        total, b = x.shape[-1], x.shape[0]
        spans, start = [], 0
        while start < total:
            end = min(start + window, total)
            spans.append((start, end))
            if end == total:
                break
            start += hop
        full = [s for s in spans if s[1] - s[0] == window]
        tail = [s for s in spans if s[1] - s[0] != window]

        output = torch.zeros_like(x)
        count = torch.zeros((1, 1, total), device=x.device, dtype=x.dtype)
        for i in range(0, len(full), max_windows):
            group = full[i:i + max_windows]
            out = self.diffusion_transformer(
                torch.cat([x[..., s:e] for s, e in group], dim=0),
                timestep.repeat(len(group)),
                torch.cat([condition[..., s:e] for s, e in group], dim=0),
            )
            for j, (s, e) in enumerate(group):
                output[..., s:e] -= out[j * b:(j + 1) * b]
                count[..., s:e] += 1
            del out
        for s, e in tail:
            output[..., s:e] -= self.diffusion_transformer(x[..., s:e], timestep, condition[..., s:e])
            count[..., s:e] += 1
        stats["batched"] += 1
        return output / count

    m3.MiniMaxMusic3DiT.forward = forward
    forward.stats = stats
    _applied = True
    log.info("[OmniXPU] Music 3 condition windows batched (max %d per call)", max_windows)
    return True, None
