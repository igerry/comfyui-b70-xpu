"""XPU pinned host memory for ComfyUI model offload (found 2026-09-05 on Arc Pro B70).

ComfyUI enables pinned memory only for CUDA/ROCm (``is_nvidia() or is_amd()`` in
comfy/model_management.py) and pins with ``cudaHostRegister``. On XPU ``MAX_PINNED_MEMORY``
stays -1, so every weight move between host and device is a pageable copy. Measured on
B70 in an LXC: 1.9 GB/s H2D and 2.1 GB/s D2H pageable vs 26.8 / 9.6 GB/s pinned. A new
prompt swaps the 26 GB text encoder in and out and reloads the 20 GB DiT: ~28 s, of which
~18 s is copy time.

This module
  (a) enables MAX_PINNED_MEMORY with ComfyUI's own formula,
  (b) replaces pin_memory / unpin_memory with a storage-swap implementation (there is no
      host-register API in torch-xpu; the tensor's storage is replaced by a pinned copy),
  (c) wraps ModelPatcher.unpatch_model so a full offload lands in pinned host memory
      (``nn.Module.to("cpu")`` would allocate fresh pageable tensors).

OMNIXPU_PIN_MEMORY=0 disables it. Deployed as ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py.
"""
import logging
import os

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

_applied = False


def _inner(tensor):
    """The plain tensor whose storage is the payload (QuantizedTensor keeps it in _qdata)."""
    q = getattr(tensor, "_qdata", None)
    return q if isinstance(q, torch.Tensor) else tensor


def _pinned_copy(t):
    out = torch.empty_like(t, device="cpu", pin_memory=True)
    out.copy_(t, non_blocking=True)
    return out


def _to_pinned_cpu(t):
    """Pinned-CPU copy of a device tensor; QuantizedTensor keeps its layout/params."""
    q = getattr(t, "_qdata", None)
    if isinstance(q, torch.Tensor) and hasattr(t, "_copy_with"):
        return t._copy_with(qdata=_pinned_copy(q), params=t._params.to_device("cpu"), clone_params=False)
    return _pinned_copy(t)


def apply():
    global _applied
    if _applied:
        return True, "already applied"
    if os.environ.get("OMNIXPU_PIN_MEMORY", "1") == "0":
        return False, "disabled by OMNIXPU_PIN_MEMORY=0"
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.utils
    from comfy.cli_args import args

    if args.disable_pinned_memory:
        return False, "--disable-pinned-memory"
    if mm.MAX_PINNED_MEMORY > 0:
        return False, "ComfyUI already enabled pinned memory"
    try:
        probe = torch.empty(1, pin_memory=True)
        if not probe.is_pinned():
            return False, "torch cannot allocate pinned memory here"
    except Exception as e:  # noqa: BLE001
        return False, f"pinned allocation failed: {e}"

    ram = mm.get_total_memory(torch.device("cpu"))
    swap = 0 if comfy.system_memory.cgroup_memory_limit() is not None else mm.get_disk_swap_total()
    mm.MAX_PINNED_MEMORY = max(ram * 0.40, min(ram * 0.90, ram - 4 * 1024 ** 3, ram + swap - 16 * 1024 ** 3))

    def pin_memory(tensor, evict_active=True):
        if mm.MAX_PINNED_MEMORY <= 0:
            return False
        if type(tensor).__name__ not in mm.PINNING_ALLOWED_TYPES:
            return False
        if not mm.is_device_cpu(tensor.device):
            return False
        inner = _inner(tensor)
        if not inner.is_contiguous():
            return False
        size = inner.nbytes
        if inner.is_pinned():
            # already pinned by an earlier swap or by our unpatch wrapper: keep the books
            if inner.data_ptr() not in mm.PINNED_MEMORY:
                mm.PINNED_MEMORY[inner.data_ptr()] = size
                mm.TOTAL_PINNED_MEMORY += size
                return True
            return False
        comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
        try:
            ok = mm.ensure_pin_registerable(size, evict_active=evict_active)
        except TypeError:  # ComfyUI < 0.39
            ok = mm.ensure_pin_registerable(size)
        if ok is False:
            return False
        try:
            pinned = inner.pin_memory()
            inner.set_(pinned.untyped_storage(), pinned.storage_offset(), pinned.size(), pinned.stride())
        except Exception as e:  # noqa: BLE001
            log.warning("[OmniXPU] XPU pin failed (%s), leaving tensor pageable", e)
            return False
        mm.PINNED_MEMORY[inner.data_ptr()] = size
        mm.TOTAL_PINNED_MEMORY += size
        return True

    def unpin_memory(tensor):
        # The pinned storage is released with the tensor; only the accounting is undone.
        if mm.MAX_PINNED_MEMORY <= 0:
            return False
        if not mm.is_device_cpu(tensor.device):
            return False
        size = mm.PINNED_MEMORY.pop(_inner(tensor).data_ptr(), None)
        if size is None:
            return False
        mm.TOTAL_PINNED_MEMORY -= size
        return True

    mm.pin_memory = pin_memory
    mm.unpin_memory = unpin_memory

    _orig_unpatch_model = mp.ModelPatcher.unpatch_model

    def _offload_to_pinned(model):
        """Copy every XPU parameter/buffer into pinned host memory before Module.to(cpu)."""
        moved = []
        for name, p in list(model.named_parameters()):
            if p.device.type == "xpu":
                moved.append((name, True, _to_pinned_cpu(p)))
        for name, b in list(model.named_buffers()):
            if b.device.type == "xpu":
                moved.append((name, False, _to_pinned_cpu(b)))
        if not moved:
            return 0
        torch.xpu.synchronize()  # non_blocking D2H copies must land before the sources die
        for name, is_param, t in moved:
            if is_param:
                comfy.utils.set_attr_param(model, name, t)
            else:
                comfy.utils.set_attr(model, name, t)
        return len(moved)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights and device_to is not None and mm.is_device_cpu(device_to):
            try:
                n = _offload_to_pinned(self.model)
                if n:
                    log.debug("[OmniXPU] offloaded %d tensors of %s into pinned host memory", n, type(self.model).__name__)
            except Exception as e:  # noqa: BLE001
                log.warning("[OmniXPU] pinned offload failed, falling back to Module.to(cpu): %s", e)
        return _orig_unpatch_model(self, device_to=device_to, unpatch_weights=unpatch_weights)

    mp.ModelPatcher.unpatch_model = unpatch_model
    _applied = True
    log.info("[OmniXPU] XPU pinned host memory enabled (limit %d MB)", mm.MAX_PINNED_MEMORY // (1024 * 1024))
    return True, None
