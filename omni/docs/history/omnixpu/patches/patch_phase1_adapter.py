#!/usr/bin/env python3
"""Phase-1 patches for ComfyUI-OmniXPU adapters (pure Python, no rebuild).
  #4 norm.py: let any last-dim-dense strided view (e.g. H3 VideoVAE q/k (1,L,32,64) fp16 chunks
     of the packed qkv) take one .contiguous() copy and the ESIMD rms_norm instead of torch's
     decomposed fallback (0.33 ms -> 0.21 ms per call, 7560 calls per 480p decode). Cache the
     all-ones weight for elementwise_affine=False norms.
  #5 attention.py: memoize _torch_major_minor() / _omni_xpu_target() (string parse + import per
     call, 4-5x per attention call) and the trace-contract env check.
Backups: <file>.pre-phase1
"""
import shutil, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/custom_nodes/ComfyUI-OmniXPU/adapters")


def patch(path, old, new, tag):
    p = ROOT / path
    s = p.read_text()
    if new in s:
        print(f"  [skip] {tag}")
        return
    assert s.count(old) == 1, f"{tag}: anchor count={s.count(old)}"
    bak = p.with_name(p.name + ".pre-phase1")
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(s.replace(old, new))
    print(f"  [ok]   {tag}")


# ---- #4 norm.py ------------------------------------------------------------------------
patch("norm.py",
      "        if (\n"
      "            not _allow_noncontiguous_rms\n"
      "            or x.ndim != 4\n"
      "            or x.shape[0] != 1\n"
      "            or x.shape[2] != 30\n"
      "            or h != 128\n"
      "            or x.stride(-1) != 1\n"
      "            or x.stride(2) != h\n"
      "            or x.stride(1) != 3 * x.shape[2] * h\n"
      "            or x.stride(0) != x.shape[1] * x.stride(1)\n"
      "        ):\n"
      "            return None\n"
      "        x = x.contiguous()\n",
      "        # Phase-1 (2026-09-05): generalized. The H3 VideoVAE q/k are (1,L,32,64)\n"
      "        # fp16 chunks of a packed (1,L,32,192) qkv (gap per head, not per token)\n"
      "        # and were rejected by the Lumina-only shape test, sending 7560 calls per\n"
      "        # 480p decode to torch's decomposed RMSNorm. One dense copy (~7 MB) plus\n"
      "        # the ESIMD kernel measured 0.21 ms vs 0.33 ms for the fallback.\n"
      "        # OMNIXPU_NORM_STRICT_VIEWS=1 restores the original Lumina-only rule.\n"
      "        if not _allow_noncontiguous_rms or x.stride(-1) != 1:\n"
      "            return None\n"
      "        if _strict_noncontiguous_views and (\n"
      "            x.ndim != 4\n"
      "            or x.shape[0] != 1\n"
      "            or x.shape[2] != 30\n"
      "            or h != 128\n"
      "            or x.stride(2) != h\n"
      "            or x.stride(1) != 3 * x.shape[2] * h\n"
      "            or x.stride(0) != x.shape[1] * x.stride(1)\n"
      "        ):\n"
      "            return None\n"
      "        x = x.contiguous()\n",
      "norm relax views")
patch("norm.py",
      "\n_allow_noncontiguous_rms = False\n",
      "\n_allow_noncontiguous_rms = False\n"
      "_strict_noncontiguous_views = os.environ.get(\"OMNIXPU_NORM_STRICT_VIEWS\", \"0\") == \"1\"\n"
      "_ones_weight_cache = {}\n",
      "norm globals")
patch("norm.py",
      "                else:\n"
      "                    w = torch.ones(orig[-1], dtype=x.dtype, device=x.device)\n",
      "                else:\n"
      "                    key = (orig[-1], x.dtype, x.device)\n"
      "                    w = _ones_weight_cache.get(key)\n"
      "                    if w is None:\n"
      "                        w = torch.ones(orig[-1], dtype=x.dtype, device=x.device)\n"
      "                        _ones_weight_cache[key] = w\n",
      "norm ones cache")
s = (ROOT / "norm.py").read_text()
assert "import os" in s.split("\n\n")[0] or "\nimport os\n" in s, "norm.py needs `import os`"

# ---- #5 attention.py -------------------------------------------------------------------
patch("attention.py",
      "def _torch_major_minor():\n    try:\n",
      "_torch_major_minor_cache = None\n\n\ndef _torch_major_minor():\n"
      "    # Phase-1: memoized; was re-parsing torch.__version__ 4-5x per attention call.\n"
      "    global _torch_major_minor_cache\n"
      "    if _torch_major_minor_cache is None:\n"
      "        _torch_major_minor_cache = _torch_major_minor_uncached()\n"
      "    return _torch_major_minor_cache\n\n\ndef _torch_major_minor_uncached():\n    try:\n",
      "attention memo torch minor")
patch("attention.py",
      "def _omni_xpu_target():\n    try:\n        import omni_xpu_kernel as pkg\n",
      "_omni_xpu_target_cache = (False, None)\n\n\ndef _omni_xpu_target():\n"
      "    # Phase-1: memoized; was importing omni_xpu_kernel + getattr up to 5x per call.\n"
      "    global _omni_xpu_target_cache\n"
      "    if not _omni_xpu_target_cache[0]:\n"
      "        _omni_xpu_target_cache = (True, _omni_xpu_target_uncached())\n"
      "    return _omni_xpu_target_cache[1]\n\n\ndef _omni_xpu_target_uncached():\n    try:\n        import omni_xpu_kernel as pkg\n",
      "attention memo target")
patch("attention.py",
      "    if os.environ.get(\"OMNI_ATTN_TRACE_CONTRACTS\", \"0\") not in {\n"
      "        \"1\",\n        \"true\",\n        \"yes\",\n    }:\n        return\n",
      "    if not _TRACE_CONTRACTS:\n        return\n",
      "attention trace env")
patch("attention.py",
      "def _trace_attention_contract(\n",
      "_TRACE_CONTRACTS = os.environ.get(\"OMNI_ATTN_TRACE_CONTRACTS\", \"0\") in {\"1\", \"true\", \"yes\"}\n\n\n"
      "def _trace_attention_contract(\n",
      "attention trace const")
print("done")
