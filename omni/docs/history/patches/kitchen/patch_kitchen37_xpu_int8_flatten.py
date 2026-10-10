"""XPU int8_linear: hand the native kernel a 2-D activation (2026-10-09).

omni_xpu_kernel's fast INT8 routes (fused ConvRot rotate+quantize, the SiLU*mul producer for
SwiGLU down-projections) are admitted only for x.ndim == 2. H3 already runs on (M, K) rows, but
Qwen-Image 2.1, Krea-2 and Music 3 call with (B, N, K), which fell through to the generic path:
materialize the bf16 activation, rotate, quantize, GEMM. Measured on B70 at Qwen-Image 2.1's
1280x720 shapes (N=3650): to_q 0.66 -> 0.54 ms, gate_up 3.09 -> 2.61, SwiGLU down 3.03 -> 1.80;
outputs differ by 2.5e-5..4e-5 relative (rounding inside the fused quantizer).
Flatten contiguous >2-D inputs to (rows, K) around the core call. Needs
patch_kitchen37_xpu_int8_linear.py applied first. Idempotent.
usage: patch_kitchen37_xpu_int8_flatten.py <comfy_kitchen pkg dir>
"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "backends" / "xpu" / "__init__.py"
s = p.read_text(encoding="utf-8")
if "_int8_lead_shape" in s:
    print("skip: already applied"); sys.exit(0)
old = ("            out = _int8_linear_core(\n"
       "                x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act\n"
       "            )\n")
assert s.count(old) == 1, "apply patch_kitchen37_xpu_int8_linear.py first"
new = ("            _int8_lead_shape = None\n"
       "            if x.ndim > 2 and x.is_contiguous():\n"
       "                _int8_lead_shape = x.shape[:-1]\n"
       "                x = x.reshape(-1, x.shape[-1])\n"
       + old +
       "            if _int8_lead_shape is not None:\n"
       "                out = out.reshape(*_int8_lead_shape, out.shape[-1])\n")
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched", p)
