"""comfy-kitchen 0.2.37 widened int8_linear (2026-10): the dispatcher now always passes
input_act_weight / input_act_eps / residual / residual_scale. The grafted XPU backend still
had the 0.2.31 signature, so every call died with TypeError. Keep the native kernel as the
core and handle the new arguments around it: "rms_norm" is applied eagerly before the
kernel (no fused XPU producer for it yet), the residual form is an addcmul after it.

usage: patch_kitchen37_xpu_int8_linear.py <comfy_kitchen pkg dir>     (idempotent)
"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "backends" / "xpu" / "__init__.py"
s = p.read_text(encoding="utf-8")
if "_int8_linear_core" in s:
    print("skip: already applied"); sys.exit(0)

head = "        def int8_linear(\n            x: torch.Tensor,\n            weight: torch.Tensor,\n"
tail = ("            return _compiled_int8_linear(\n                x,\n                weight,\n"
        "                weight_scale,\n                bias,\n                output_dtype_code,\n"
        "                convrot,\n                convrot_groupsize,\n                input_act_code,\n            )\n")
assert s.count(head) == 1 and s.count(tail) == 1
s = s.replace(head, head.replace("def int8_linear(", "def _int8_linear_core("))
s = s.replace(tail, tail + '''
        def int8_linear(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None = None,
            out_dtype: torch.dtype | None = None,
            convrot: bool = False,
            convrot_groupsize: int = 256,
            input_act: str | None = None,
            input_act_weight: torch.Tensor | None = None,
            input_act_eps: float = 0.0,
            residual: torch.Tensor | None = None,
            residual_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if input_act not in _INT8_LINEAR_INPUT_ACTS:
                from comfy_kitchen.backends._activations import apply_input_act

                x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
                input_act = None
            out = _int8_linear_core(
                x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act
            )
            if residual is None:
                return out
            from comfy_kitchen.backends._activations import apply_residual

            return apply_residual(out, residual, residual_scale)
''')
p.write_text(s, encoding="utf-8")
print("patched", p)
