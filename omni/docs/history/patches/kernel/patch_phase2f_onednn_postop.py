#!/usr/bin/env python3
"""Phase-2f: oneDNN post-ops on the scaled INT8 matmul (swish eltwise / binary_mul with a bf16
tensor) so H3's SwiGLU can run inside the fc1 GEMM epilogue: G = silu(x@Wg) [postop 1],
A = (x@Wu) * G [postop 2]. Adds int8_linear_prequantized_postop(x_int8, x_scale, w, w_scale,
out_dtype_code, postop, binary)."""
import sys
from pathlib import Path
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src")

def patch(rel, old, new, tag, count=1):
    p = SRC / rel; s = p.read_text()
    if new in s: print(f"[skip] {tag}"); return
    assert s.count(old) == count, f"{tag}: count={s.count(old)}"
    p.write_text(s.replace(old, new)); print(f"[ok] {tag}")

F = "omni_xpu_kernel/csrc/onednn_int8.cpp"
patch(F, "    dnnl::matmul primitive;\n    bool has_bias;\n    bool w_scale_is_scalar;\n};\n",
      "    dnnl::matmul primitive;\n    bool has_bias;\n    bool w_scale_is_scalar;\n    int postop = 0;  // 0 none, 1 swish, 2 binary_mul(binary_md)\n    dnnl::memory::desc binary_md;\n};\n", "state")
patch(F, "    bool has_bias;\n    bool w_scale_is_scalar;\n\n    bool operator==(const Int8ScaledCacheKey& other) const {\n        return device_index == other.device_index\n            && out_dtype == other.out_dtype\n            && m == other.m && k == other.k && n == other.n\n            && has_bias == other.has_bias\n            && w_scale_is_scalar == other.w_scale_is_scalar;\n",
      "    bool has_bias;\n    bool w_scale_is_scalar;\n    int postop;\n\n    bool operator==(const Int8ScaledCacheKey& other) const {\n        return device_index == other.device_index\n            && out_dtype == other.out_dtype\n            && m == other.m && k == other.k && n == other.n\n            && has_bias == other.has_bias\n            && w_scale_is_scalar == other.w_scale_is_scalar\n            && postop == other.postop;\n", "key")
patch(F, "        combine(std::hash<bool>{}(key.w_scale_is_scalar));\n        return seed;\n    }\n};\n\nstruct Int8PairCacheKey {",
      "        combine(std::hash<bool>{}(key.w_scale_is_scalar));\n        combine(std::hash<int>{}(key.postop));\n        return seed;\n    }\n};\n\nstruct Int8PairCacheKey {", "hash")
patch(F, "    bool w_scale_is_scalar,\n    const torch::Device& device,\n    const sycl::queue& queue\n) {\n    Int8ScaledCacheKey key{device.index(), out_dtype_code, m, k, n, has_bias, w_scale_is_scalar};\n",
      "    bool w_scale_is_scalar,\n    const torch::Device& device,\n    const sycl::queue& queue,\n    int postop = 0\n) {\n    Int8ScaledCacheKey key{device.index(), out_dtype_code, m, k, n, has_bias, w_scale_is_scalar, postop};\n", "get_or_create sig")
patch(F, "    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);\n\n    dnnl::matmul::primitive_desc pd = has_bias\n",
      "    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);\n"
      "    state->postop = postop;\n"
      "    if (postop == 1) {\n"
      "        dnnl::post_ops po;\n"
      "        po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);  // SiLU\n"
      "        attr.set_post_ops(po);\n"
      "    } else if (postop == 2) {\n"
      "        state->binary_md = dnnl::memory::desc({m, n}, dst_dt, dnnl::memory::format_tag::ab);\n"
      "        dnnl::post_ops po;\n"
      "        po.append_binary(dnnl::algorithm::binary_mul, state->binary_md);\n"
      "        attr.set_post_ops(po);\n"
      "    }\n\n"
      "    dnnl::matmul::primitive_desc pd = has_bias\n", "post-ops attr")
patch(F, "    std::optional<torch::Tensor> bias,\n    int64_t out_dtype_code,\n    std::optional<torch::Tensor> output_opt\n) {\n    TORCH_CHECK(x_int8.dim() >= 1,\n",
      "    std::optional<torch::Tensor> bias,\n    int64_t out_dtype_code,\n    std::optional<torch::Tensor> output_opt,\n    int postop = 0,\n    std::optional<torch::Tensor> binary = std::nullopt\n) {\n    TORCH_CHECK(x_int8.dim() >= 1,\n", "impl sig")
patch(F, "        m, k, n, static_cast<int>(out_dtype_code), has_bias,\n        w_scale_is_scalar, x_int8.device(), queue);\n",
      "        m, k, n, static_cast<int>(out_dtype_code), has_bias,\n        w_scale_is_scalar, x_int8.device(), queue, postop);\n", "impl get_or_create call")
patch(F, "    if (has_bias) {\n        args.emplace(\n            DNNL_ARG_BIAS,\n            dnnl::memory(state->bias_md, state->engine, bias_f32.data_ptr()));\n    }\n    state->primitive.execute(stream, args);\n",
      "    if (has_bias) {\n        args.emplace(\n            DNNL_ARG_BIAS,\n            dnnl::memory(state->bias_md, state->engine, bias_f32.data_ptr()));\n    }\n"
      "    if (postop == 2) {\n"
      "        TORCH_CHECK(binary.has_value(), \"binary_mul post-op needs a tensor\");\n"
      "        TORCH_CHECK(binary->scalar_type() == out_dtype && binary->is_contiguous() &&\n"
      "                    binary->numel() == m * n && binary->device() == x_int8.device(),\n"
      "                    \"binary post-op tensor must be a contiguous [M, N] tensor of the output dtype on the same device\");\n"
      "        args.emplace(\n"
      "            DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,\n"
      "            dnnl::memory(state->binary_md, state->engine, binary->data_ptr()));\n"
      "    }\n"
      "    state->primitive.execute(stream, args);\n", "impl exec")
patch(F, "torch::Tensor int8_linear_prequantized_out(\n",
      "torch::Tensor int8_linear_prequantized_postop(\n"
      "    torch::Tensor x_int8,\n    torch::Tensor x_scale,\n    torch::Tensor weight,\n    torch::Tensor weight_scale,\n"
      "    int64_t out_dtype_code,\n    int64_t postop,\n    std::optional<torch::Tensor> binary\n) {\n"
      "    TORCH_CHECK(postop == 1 || postop == 2, \"postop must be 1 (swish) or 2 (binary_mul)\");\n"
      "    return int8_linear_prequantized_impl(\n"
      "        x_int8, x_scale, weight, weight_scale, std::nullopt, out_dtype_code,\n"
      "        std::nullopt, static_cast<int>(postop), binary);\n}\n\n"
      "torch::Tensor int8_linear_prequantized_out(\n", "public fn")
B = "omni_xpu_kernel/csrc/bindings.cpp"
patch(B, "    torch::Tensor int8_linear_prequantized_out(\n",
      "    torch::Tensor int8_linear_prequantized_postop(\n        torch::Tensor x_int8, torch::Tensor x_scale, torch::Tensor weight,\n        torch::Tensor weight_scale, int64_t out_dtype_code, int64_t postop,\n        std::optional<torch::Tensor> binary);\n    torch::Tensor int8_linear_prequantized_out(\n", "bind decl")
patch(B, '    int8.def(\n        "int8_linear_prequantized_out",\n',
      '    int8.def("int8_linear_prequantized_postop", &omni_xpu::int8_ops::int8_linear_prequantized_postop,\n'
      '        "Scaled INT8 matmul with a fused oneDNN post-op: postop=1 swish (SiLU) on the dequantized\\n"\n'
      '        "accumulator, postop=2 elementwise multiply by `binary` ([M,N], output dtype).",\n'
      '        py::arg("x_int8"), py::arg("x_scale"), py::arg("weight"), py::arg("weight_scale"),\n'
      '        py::arg("out_dtype_code") = 2, py::arg("postop") = 1, py::arg("binary") = py::none());\n'
      '    int8.def(\n        "int8_linear_prequantized_out",\n', "bind def")
print("done")
