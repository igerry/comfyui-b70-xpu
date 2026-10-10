"""Resolve the 6 hunks that the 0.2.31 fork delta could not place on stock 0.2.37."""
import sys
from pathlib import Path
out = Path(sys.argv[1])

def sub(path, old, new, count=1):
    p = out / path
    s = p.read_text(encoding="utf-8")
    if new in s:
        print("skip", path, repr(new[:40])); return
    assert s.count(old) >= 1, (path, old)
    p.write_text(s.replace(old, new, count), encoding="utf-8")
    print("ok  ", path, repr(new.strip()[:50]))

# registry: xpu first; keep the stock order behind it
sub("registry.py", 'self._priority = ["ascend", "cuda", "triton", "eager"]',
    'self._priority = ["xpu", "ascend", "cuda", "triton", "eager"]')

# eager: fork-only ops
sub("backends/eager/__init__.py", '    "scaled_mm_svdquant_w4a4",\n    "stochastic_rounding_fp8",',
    '    "scaled_mm_svdquant_w4a4",\n    "svdquant_w4a16_linear",\n    "stochastic_rounding_fp8",')
sub("backends/eager/__init__.py", '    "int8_linear",\n    "w4a8_int8_linear",\n]',
    '    "int8_linear",\n    "mm_int8",\n    "w4a8_int8_linear",\n]')
sub("backends/eager/__init__.py", "from .na import na3d\n", "from .gguf import dequantize_gguf\nfrom .na import na3d\n")

# package __init__: version, xpu backend, fork-only public API
sub("__init__.py", "from .backends import triton as _triton_backend  # noqa: F401\n",
    "from .backends import triton as _triton_backend  # noqa: F401\n"
    "from .backends import xpu as _xpu_backend  # noqa: F401\n"
    "from ._version import __version__  # noqa: F401\n")
sub("__init__.py", "from .registry import registry\n",
    "from .gguf import dequantize_gguf, get_gguf_route_diagnostics\n"
    "from .svdquant_w4a16 import (\n    PreparedSVDQuantW4A16,\n    get_svdquant_w4a16_route_diagnostics,\n"
    "    prepare_svdquant_w4a16_for_xpu,\n    restore_svdquant_w4a16_source_,\n    svdquant_w4a16_linear,\n)\n"
    "from .registry import registry\n")
for r in out.rglob("*.rej"): r.unlink()
