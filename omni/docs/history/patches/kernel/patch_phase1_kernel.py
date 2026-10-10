#!/usr/bin/env python3
"""Phase-1 source patches for omni_xpu_kernel (run on 22 against /root/omnibuild/src).
Idempotent: each edit asserts its anchor and skips if already applied.
  1. bindings.cpp: export quantize_int8_convrot_fused (radix-4 in-register Hadamard + rowwise INT8)
  2. int8/__init__.py: use it for the activation path when OMNIXPU_CONVROT_FUSED != 0
  3. cute_fmha_torch.cpp: cache sm_count, skip zero-size workspace alloc; CUTE_FMHA_VAE_QTILE macro
  4. setup.py: honor OMNI_CUTE_EXTRA_FLAGS for variant builds (-DCUTE_FMHA_KV64 etc.)
"""
import re, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src")


def patch(path, old, new, tag):
    p = SRC / path
    s = p.read_text()
    if new in s:
        print(f"  [skip] {tag} already applied in {path}")
        return
    assert s.count(old) == 1, f"{tag}: anchor not unique/found in {path} (count={s.count(old)})"
    p.write_text(s.replace(old, new))
    print(f"  [ok]   {tag} -> {path}")


# ---- 1. bindings.cpp -------------------------------------------------------
patch("omni_xpu_kernel/csrc/bindings.cpp",
      "    torch::Tensor dequantize_int8_convrot_weight(\n        torch::Tensor q, torch::Tensor scale, int64_t group_size);\n",
      "    torch::Tensor dequantize_int8_convrot_weight(\n        torch::Tensor q, torch::Tensor scale, int64_t group_size);\n"
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(\n        torch::Tensor input, int64_t group_size);\n",
      "bindings decl")
patch("omni_xpu_kernel/csrc/bindings.cpp",
      '    int8.def("dequantize_int8_convrot_weight", &omni_xpu::int8_ops::dequantize_int8_convrot_weight,\n'
      '        "Dequantize INT8 ConvRot weight and apply the inverse orthogonal rotation",\n'
      '        py::arg("q"), py::arg("scale"), py::arg("group_size") = 256);\n',
      '    int8.def("dequantize_int8_convrot_weight", &omni_xpu::int8_ops::dequantize_int8_convrot_weight,\n'
      '        "Dequantize INT8 ConvRot weight and apply the inverse orthogonal rotation",\n'
      '        py::arg("q"), py::arg("scale"), py::arg("group_size") = 256);\n'
      '    int8.def("quantize_int8_convrot_fused", &omni_xpu::int8_ops::quantize_int8_convrot_fused,\n'
      '        "Fused radix-4 Hadamard rotation (in registers) + rowwise INT8 quantization for a 2D\\n"\n'
      '        "bf16/fp16 activation. Replaces rotate_convrot (matmul) + quantize_int8_rowwise.",\n'
      '        py::arg("input"), py::arg("group_size") = 256);\n',
      "bindings def")

# ---- 2. int8/__init__.py ---------------------------------------------------
patch("omni_xpu_kernel/int8/__init__.py",
      "_H3_ATTN_OUTPUT_FEATURES = 5376\n",
      "_H3_ATTN_OUTPUT_FEATURES = 5376\n"
      "# Phase-1 (2026-09-05): fused in-register ConvRot+quantize for activations.\n"
      "# OMNIXPU_CONVROT_FUSED=0 restores the matmul-Hadamard + separate quantize path.\n"
      "_CONVROT_FUSED = os.environ.get(\"OMNIXPU_CONVROT_FUSED\", \"1\") != \"0\"\n"
      "\n\ndef _can_quantize_convrot_fused(x, native, convrot_groupsize):\n"
      "    return bool(\n"
      "        _CONVROT_FUSED\n"
      "        and native is not None\n"
      "        and hasattr(native, \"quantize_int8_convrot_fused\")\n"
      "        and convrot_groupsize in (64, 256)\n"
      "        and x.dim() == 2\n"
      "        and x.dtype in (torch.bfloat16, torch.float16)\n"
      "        and x.is_contiguous()\n"
      "        and x.shape[1] % convrot_groupsize == 0\n"
      "    )\n",
      "int8 helper")
patch("omni_xpu_kernel/int8/__init__.py",
      "        # Rotate through the native cached Hadamard-matrix implementation.\n"
      "        if convrot:\n"
      "            if x.shape[-1] % convrot_groupsize != 0:\n"
      "                raise ValueError(\n"
      "                    f\"ConvRot group size {convrot_groupsize} does not divide \"\n"
      "                    f\"input features {x.shape[-1]}\"\n"
      "                )\n"
      "            if hasattr(native, \"rotate_convrot\"):\n"
      "                x = native.rotate_convrot(x, convrot_groupsize)\n",
      "        # Rotate through the native cached Hadamard-matrix implementation.\n"
      "        if convrot:\n"
      "            if x.shape[-1] % convrot_groupsize != 0:\n"
      "                raise ValueError(\n"
      "                    f\"ConvRot group size {convrot_groupsize} does not divide \"\n"
      "                    f\"input features {x.shape[-1]}\"\n"
      "                )\n"
      "            if _can_quantize_convrot_fused(x, native, convrot_groupsize):\n"
      "                x_int8, x_scale = native.quantize_int8_convrot_fused(\n"
      "                    x, convrot_groupsize\n"
      "                )\n"
      "                return native.int8_linear_prequantized(\n"
      "                    x_int8, x_scale, weight, weight_scale, bias, dtype_code\n"
      "                )\n"
      "            if hasattr(native, \"rotate_convrot\"):\n"
      "                x = native.rotate_convrot(x, convrot_groupsize)\n",
      "int8 main path")
# the two low-peak streaming routes (long 720p runs) do rotate+quantize per chunk
patch("omni_xpu_kernel/int8/__init__.py",
      "        activated = native.fused_silu_mul_exact_bf16(gate, up)\n"
      "        rotated = native.rotate_convrot(activated, 256)\n"
      "        x_int8, x_scale = native.quantize_int8_rowwise_fused(rotated)\n",
      "        activated = native.fused_silu_mul_exact_bf16(gate, up)\n"
      "        if _can_quantize_convrot_fused(activated, native, 256):\n"
      "            rotated = None\n"
      "            x_int8, x_scale = native.quantize_int8_convrot_fused(activated, 256)\n"
      "        else:\n"
      "            rotated = native.rotate_convrot(activated, 256)\n"
      "            x_int8, x_scale = native.quantize_int8_rowwise_fused(rotated)\n",
      "int8 stream swiglu")
patch("omni_xpu_kernel/int8/__init__.py",
      "        chunk = x[start:stop]\n"
      "        rotated = native.rotate_convrot(chunk, 256)\n"
      "        x_int8, x_scale = native.quantize_int8_rowwise_fused(rotated)\n",
      "        chunk = x[start:stop]\n"
      "        if _can_quantize_convrot_fused(chunk, native, 256):\n"
      "            rotated = None\n"
      "            x_int8, x_scale = native.quantize_int8_convrot_fused(chunk, 256)\n"
      "        else:\n"
      "            rotated = native.rotate_convrot(chunk, 256)\n"
      "            x_int8, x_scale = native.quantize_int8_rowwise_fused(rotated)\n",
      "int8 stream convrot")

# ---- 3. cute_fmha_torch.cpp -------------------------------------------------
patch("omni_xpu_kernel/cute/cute_fmha_torch.cpp",
      "  cutlass::KernelHardwareInfo hw_info;\n"
      "  hw_info.sm_count =\n"
      "      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(\n"
      "          hw_info.device_id);\n",
      "  cutlass::KernelHardwareInfo hw_info;\n"
      "  // Queried once: two get_info calls per attention call otherwise, and the\n"
      "  // individual tile scheduler ignores sm_count anyway.\n"
      "  static const int cached_sm_count =\n"
      "      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(\n"
      "          hw_info.device_id);\n"
      "  hw_info.sm_count = cached_sm_count;\n",
      "cute sm_count")
patch("omni_xpu_kernel/cute/cute_fmha_torch.cpp",
      "  size_t workspace_size = K::get_workspace_size(arguments);\n"
      "  auto opts = at::TensorOptions().dtype(at::kByte).device(at::kXPU);\n"
      "  at::Tensor workspace = at::empty({(long)workspace_size}, opts);\n",
      "  size_t workspace_size = K::get_workspace_size(arguments);\n"
      "  at::Tensor workspace;\n"
      "  if (workspace_size > 0) {\n"
      "    auto opts = at::TensorOptions().dtype(at::kByte).device(at::kXPU);\n"
      "    workspace = at::empty({(long)workspace_size}, opts);\n"
      "  }\n",
      "cute workspace")
patch("omni_xpu_kernel/cute/cute_fmha_torch.cpp",
      "  run_d128_tile<cutlass::half_t, 0, 0, 0, 0, 0, 64>(\n"
      "      q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(), B, H, L, L,\n",
      "  // -DCUTE_FMHA_VAE_QTILE=128|64 shrinks the Q tile for the short VAE tiles\n"
      "  // (L~1797: 256-row tiles leave the last tile 2% utilised); subgroup layout\n"
      "  // scales with it to keep 16 rows per subgroup.\n"
      "#ifndef CUTE_FMHA_VAE_QTILE\n"
      "#define CUTE_FMHA_VAE_QTILE 0\n"
      "#endif\n"
      "  run_d128_tile<cutlass::half_t, 0, CUTE_FMHA_VAE_QTILE,\n"
      "                (CUTE_FMHA_VAE_QTILE > 0 ? CUTE_FMHA_VAE_QTILE / 16 : 0),\n"
      "                0, 0, 64>(\n"
      "      q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(), B, H, L, L,\n",
      "cute vae qtile")
# workspace.data_ptr() is used later; make sure a null tensor is handled
s = (SRC / "omni_xpu_kernel/cute/cute_fmha_torch.cpp").read_text()
m = re.findall(r"workspace\.data_ptr\(\)", s)
print(f"  workspace.data_ptr() uses: {len(m)} (must tolerate undefined tensor -> patched below)")
patch("omni_xpu_kernel/cute/cute_fmha_torch.cpp",
      "workspace.data_ptr()",
      "(workspace.defined() ? workspace.data_ptr() : nullptr)",
      "cute workspace ptr") if len(m) == 1 else print("  [warn] check workspace.data_ptr() uses manually")

# ---- 4. setup.py -------------------------------------------------------------
patch("setup.py",
      '                    "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET",\n',
      '                    "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET",\n'
      '                    *os.environ.get("OMNI_CUTE_EXTRA_FLAGS", "").split(),\n',
      "setup cute extra flags")
print("done")
