"""phase-9 (2026-10-09): quantize_int8_convrot_silu_mul_split(..., gate_second=0).

The fused SiLU(gate)*up + Hadamard + rowwise INT8 producer reads a [M, 2K] row as [gate | up]
(H3's fc1, Qwen-Image 2.1's gate_up). MiniMax Music 3's GLU is `value, gate = proj(x).chunk(2)`,
i.e. [up | gate]. The kernel already takes the two halves as separate pointers, so the new
argument only swaps them. With it Music 3's `value * silu(gate)` (6.2 ms on strided chunk views
at batch 58 x 690 x 8192) and the down-projection's rotate+quantize collapse into one pass.
Idempotent. usage: patch_phase9_silu_mul_gate_second.py /root/omnibuild/src
"""
import sys
from pathlib import Path
src = Path(sys.argv[1]) / "omni_xpu_kernel" / "csrc"

def edit(name, pairs):
    p = src / name; s = p.read_text(encoding="utf-8")
    if "gate_second" in s:
        print("skip", name); return
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8"); print("patched", name)

edit("int8_convrot_quant_esimd.cpp", [
    ("std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(\n"
     "    torch::Tensor fc1_out,\n    int64_t group_size,\n    int64_t row_split) {\n",
     "// gate_second != 0: the row is [up | gate] (MiniMax Music 3 GLU) instead of [gate | up].\n"
     "std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(\n"
     "    torch::Tensor fc1_out,\n    int64_t group_size,\n    int64_t row_split,\n    int64_t gate_second) {\n"),
    ("        const auto* base = reinterpret_cast<const bf16*>(fc1_out.data_ptr());\n"
     "        if (group_size == 64) dispatch_silu_rowsplit<bf16, 64>(base, base + width, stride,",
     "        const auto* first = reinterpret_cast<const bf16*>(fc1_out.data_ptr());\n"
     "        const auto* base = gate_second ? first + width : first;\n"
     "        const auto* upp = gate_second ? first : first + width;\n"
     "        if (group_size == 64) dispatch_silu_rowsplit<bf16, 64>(base, upp, stride,"),
    ("        else dispatch_silu_rowsplit<bf16, 256>(base, base + width, stride,",
     "        else dispatch_silu_rowsplit<bf16, 256>(base, upp, stride,"),
    ("        const auto* base = reinterpret_cast<const fp16*>(fc1_out.data_ptr());\n"
     "        if (group_size == 64) dispatch_silu_rowsplit<fp16, 64>(base, base + width, stride,",
     "        const auto* first = reinterpret_cast<const fp16*>(fc1_out.data_ptr());\n"
     "        const auto* base = gate_second ? first + width : first;\n"
     "        const auto* upp = gate_second ? first : first + width;\n"
     "        if (group_size == 64) dispatch_silu_rowsplit<fp16, 64>(base, upp, stride,"),
    ("        else dispatch_silu_rowsplit<fp16, 256>(base, base + width, stride,",
     "        else dispatch_silu_rowsplit<fp16, 256>(base, upp, stride,"),
])
edit("bindings.cpp", [
    ("        torch::Tensor fc1_out, int64_t group_size, int64_t row_split);\n",
     "        torch::Tensor fc1_out, int64_t group_size, int64_t row_split, int64_t gate_second);\n"),
    ('        py::arg("fc1_out"), py::arg("group_size") = 256, py::arg("row_split") = 16);\n',
     '        py::arg("fc1_out"), py::arg("group_size") = 256, py::arg("row_split") = 16, py::arg("gate_second") = 0);\n'),
])
v = Path(sys.argv[1]) / "omni_xpu_kernel" / "_version.py"
s = v.read_text(encoding="utf-8")
if '"2.14"' not in s:
    old = '__supported_torch_minors__ = ("2.10", "2.11", "2.12", "2.13")'
    assert s.count(old) == 1
    v.write_text(s.replace(old, '__supported_torch_minors__ = ("2.10", "2.11", "2.12", "2.13", "2.14")'), encoding="utf-8"); print("whitelisted torch 2.14")
