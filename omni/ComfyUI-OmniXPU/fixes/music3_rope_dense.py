"""MiniMax Music 3: partial RoPE without strided kernels or concatenations (2026-10-09).

Stock Attention.forward rotates only the first `rotary_dims` of each 64-wide head, feeding the
kernel last-dim slices of transposed views of the qkv buffer, then rebuilds q and k with two
torch.cat. On XPU every one of those is a strided pass: at batch 58 x 690 tokens the step costs
8.0 ms of a 46 ms layer (rope on strided slices 4.1 ms, the two cats 3.8 ms) where the rotation
itself is 0.95 ms on dense input.
Here q and k get one dense copy each, the rotary half is rotated from a dense slice and written
back through a (..., 2, 32) view. Same kernel, same values: bitwise equal to the stock result.
Measured 8.0 -> 4.9 ms for the step, 12.4 -> 9.1 ms including the attention that follows.

Applies when not training, on XPU, and rotary_dims is exactly half the head width (the only
configuration the model ships with); anything else takes the stock forward.
OMNIXPU_MUSIC3_ROPE=0 disables. Deployed as ComfyUI-OmniXPU/fixes/music3_rope_dense.py.
"""
import logging
import os

log = logging.getLogger("ComfyUI-OmniXPU")
_applied = False


def apply():
    global _applied
    if _applied:
        return True, "already applied"
    if os.environ.get("OMNIXPU_MUSIC3_ROPE", "1") == "0":
        return False, "disabled by OMNIXPU_MUSIC3_ROPE=0"
    try:
        import comfy.ldm.minimax_music.dit as m3
        import comfy.model_management
        import comfy.quant_ops
    except ImportError as e:
        return False, f"import failed: {e}"
    for name in ("AttentionTensorContainer", "optimized_attention_for_device"):
        if not hasattr(m3, name):
            return False, f"minimax_music.dit has no {name}"

    _orig_forward = m3.Attention.forward
    stats = {"dense": 0, "fallback": 0}

    def forward(self, x, rotation_matrix):
        batch, length, _ = x.shape
        heads, dim = self.num_heads, self.dim_heads
        rotary_dims = rotation_matrix.shape[-3] * 2
        if comfy.model_management.in_training or x.device.type != "xpu" or rotary_dims * 2 != dim:
            stats["fallback"] += 1
            return _orig_forward(self, x, rotation_matrix)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, heads, dim).transpose(1, 2).contiguous().view(batch, heads, length, 2, rotary_dims)
        k = k.reshape(batch, length, heads, dim).transpose(1, 2).contiguous().view(batch, heads, length, 2, rotary_dims)
        v = v.reshape(batch, length, heads, dim).transpose(1, 2)
        rotated_q, rotated_k = comfy.quant_ops.ck.apply_rope_split_half(
            q[..., 0, :].contiguous(), k[..., 0, :].contiguous(), rotation_matrix)
        q[..., 0, :] = rotated_q
        k[..., 0, :] = rotated_k
        del rotated_q, rotated_k
        q, k = q.view(batch, heads, length, dim), k.view(batch, heads, length, dim)
        attention = m3.optimized_attention_for_device(q.device)
        q, k, v = m3.AttentionTensorContainer(q), m3.AttentionTensorContainer(k), m3.AttentionTensorContainer(v)
        out = attention(q, k, v, heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        stats["dense"] += 1
        return self.to_out(out)

    m3.Attention.forward = forward
    forward.stats = stats
    _applied = True
    log.info("[OmniXPU] Music 3 partial RoPE on dense q/k enabled")
    return True, None
