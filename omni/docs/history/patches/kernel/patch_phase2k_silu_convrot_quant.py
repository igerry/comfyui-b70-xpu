#!/usr/bin/env python3
"""Phase-2k: producer fusion for fc2's input: SiLU(gate)*up + ConvRot + rowwise INT8 quantize in one
ESIMD pass over the fc1 output (M, 2K) -> int8 (M, K) + scales. Removes the bf16 (M, K) materialization
(write 468 MB + read 468 MB per block at 480p). Rounding boundaries follow the stock path
(silu -> bf16, product -> bf16, rotated -> bf16) except SiLU uses the ESIMD exp/divide."""
import sys
from pathlib import Path
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src")

def patch(rel, old, new, tag):
    p = SRC / rel; s = p.read_text()
    if new in s: print(f"[skip] {tag}"); return
    assert s.count(old) == 1, f"{tag}: count={s.count(old)}"
    p.write_text(s.replace(old, new)); print(f"[ok] {tag}")

KERNEL = r'''
// SiLU(gate)*up -> rotated -> rounded group (fc2 producer fusion, 2026-09-06).
template<typename InputT, int GroupSize>
inline simd<InputT, GroupSize> load_silu_mul_rotated_rounded_vec(
    const InputT* __restrict__ gate,
    const InputT* __restrict__ up) {
    simd<InputT, GroupSize> gate_v = block_load<InputT, GroupSize>(gate);
    simd<InputT, GroupSize> up_v = block_load<InputT, GroupSize>(up);
    simd<float, GroupSize> x = gate_v;
    simd<float, GroupSize> silu =
        x / (1.0f + sycl::ext::intel::esimd::exp(-x));
    simd<InputT, GroupSize> silu_rounded = silu;
    simd<float, GroupSize> silu_f = silu_rounded;
    simd<float, GroupSize> up_f = up_v;
    simd<InputT, GroupSize> product = silu_f * up_f;
    simd<float, GroupSize> values = product;
    radix4_stage_vec<1, GroupSize>(values);
    values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));
    simd<InputT, GroupSize> rounded = values;
    return rounded;
}

template<typename InputT, int GroupSize, int WG>
void quantize_convrot_silu_mul_rowsplit_kernel(
    const InputT* __restrict__ gate,
    const InputT* __restrict__ up,
    int64_t in_row_stride,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int kCache = 4;
    const int gpw = static_cast<int>((groups + WG - 1) / WG);
    const bool use_cache = gpw <= kCache;
    sycl::range<2> global_size(static_cast<size_t>(rows), WG);
    sycl::range<2> local_size(1, WG);
    auto cgf = [&](sycl::handler& handle) {
        sycl::ext::oneapi::experimental::properties props{
            sycl::ext::intel::experimental::grf_size<OMNI_CONVROT_QUANT_GRF>};
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            props,
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                constexpr int kSlmBytes = (WG * 4 < 16) ? 16 : WG * 4;
                slm_init<kSlmBytes>();
                const int64_t row = item.get_global_id(0);
                const int t = static_cast<int>(item.get_local_id(1));
                const int64_t in_start = row * in_row_stride;
                const int64_t out_start = row * groups * GroupSize;
                const int g0 = t * gpw;
                const int g1 = static_cast<int>(
                    (g0 + gpw < groups) ? (g0 + gpw) : groups);

                simd<InputT, GroupSize> cache[kCache];
                float local_max = 0.0f;
                if (use_cache) {
#pragma unroll
                    for (int i = 0; i < kCache; ++i) {
                        const int g = g0 + i;
                        if (g < g1) {
                            cache[i] = load_silu_mul_rotated_rounded_vec<InputT, GroupSize>(
                                gate + in_start + g * GroupSize,
                                up + in_start + g * GroupSize);
                            simd<float, GroupSize> values = cache[i];
                            const float group_max = hmax<float>(
                                sycl::ext::intel::esimd::abs<float, GroupSize>(values));
                            local_max = group_max > local_max ? group_max : local_max;
                        }
                    }
                } else {
                    for (int g = g0; g < g1; ++g) {
                        simd<float, GroupSize> values =
                            load_silu_mul_rotated_rounded_vec<InputT, GroupSize>(
                                gate + in_start + g * GroupSize,
                                up + in_start + g * GroupSize);
                        const float group_max = hmax<float>(
                            sycl::ext::intel::esimd::abs<float, GroupSize>(values));
                        local_max = group_max > local_max ? group_max : local_max;
                    }
                }
                slm_block_store<float, 1>(t * sizeof(float), local_max);
                barrier();
                simd<float, WG> partials = slm_block_load<float, WG>(0);
                const float row_max = hmax<float>(partials);

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                if (t == 0) scales[row] = scale;

                if (use_cache) {
#pragma unroll
                    for (int i = 0; i < kCache; ++i) {
                        const int g = g0 + i;
                        if (g < g1) {
                            quantize_store_rounded<InputT, GroupSize>(
                                cache[i], output, out_start + g * GroupSize, inv_scale);
                        }
                    }
                } else {
                    for (int g = g0; g < g1; ++g) {
                        quantize_store_rounded<InputT, GroupSize>(
                            load_silu_mul_rotated_rounded_vec<InputT, GroupSize>(
                                gate + in_start + g * GroupSize,
                                up + in_start + g * GroupSize),
                            output, out_start + g * GroupSize, inv_scale);
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_silu_mul_quantize_rowsplit");
}

template<typename InputT, int GroupSize>
void dispatch_silu_rowsplit(
    const InputT* gate, const InputT* up, int64_t in_row_stride, int8_t* output, float* scales,
    int64_t rows, int64_t groups, int row_split, const at::Device& device) {
    switch (row_split) {
        case 4:  quantize_convrot_silu_mul_rowsplit_kernel<InputT, GroupSize, 4>(gate, up, in_row_stride, output, scales, rows, groups, device); break;
        case 16: quantize_convrot_silu_mul_rowsplit_kernel<InputT, GroupSize, 16>(gate, up, in_row_stride, output, scales, rows, groups, device); break;
        case 32: quantize_convrot_silu_mul_rowsplit_kernel<InputT, GroupSize, 32>(gate, up, in_row_stride, output, scales, rows, groups, device); break;
        default: quantize_convrot_silu_mul_rowsplit_kernel<InputT, GroupSize, 8>(gate, up, in_row_stride, output, scales, rows, groups, device); break;
    }
}

}  // namespace

// fc1_out: [M, 2K] bf16/fp16 contiguous ([gate | up]); returns int8 [M, K] + scales [M, 1].
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(
    torch::Tensor fc1_out,
    int64_t group_size,
    int64_t row_split) {
    TORCH_CHECK(fc1_out.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(fc1_out.dim() == 2, "input must be 2D [M, 2K]");
    TORCH_CHECK(
        fc1_out.scalar_type() == torch::kBFloat16 ||
            fc1_out.scalar_type() == torch::kFloat16,
        "input must be bf16 or fp16");
    TORCH_CHECK(group_size == 64 || group_size == 256, "group size must be 64 or 256");
    TORCH_CHECK(fc1_out.size(1) % 2 == 0 && (fc1_out.size(1) / 2) % group_size == 0,
                "half width must be divisible by group size");
    TORCH_CHECK(row_split == 4 || row_split == 8 || row_split == 16 || row_split == 32,
                "row_split must be 4, 8, 16 or 32");
    fc1_out = fc1_out.contiguous();
    const int64_t rows = fc1_out.size(0);
    const int64_t width = fc1_out.size(1) / 2;
    const int64_t groups = width / group_size;
    auto output = torch::empty({rows, width}, fc1_out.options().dtype(torch::kInt8));
    auto scales = torch::empty({rows, 1}, fc1_out.options().dtype(torch::kFloat));
    if (fc1_out.numel() == 0) return {output, scales};
    auto* out_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    auto* scale_ptr = scales.data_ptr<float>();
    const int rs = static_cast<int>(row_split);
    const int64_t stride = fc1_out.size(1);
    if (fc1_out.scalar_type() == torch::kBFloat16) {
        const auto* base = reinterpret_cast<const bf16*>(fc1_out.data_ptr());
        if (group_size == 64) dispatch_silu_rowsplit<bf16, 64>(base, base + width, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
        else dispatch_silu_rowsplit<bf16, 256>(base, base + width, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
    } else {
        const auto* base = reinterpret_cast<const fp16*>(fc1_out.data_ptr());
        if (group_size == 64) dispatch_silu_rowsplit<fp16, 64>(base, base + width, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
        else dispatch_silu_rowsplit<fp16, 256>(base, base + width, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
    }
    return {output, scales};
}

namespace {
'''
patch("omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp",
      "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n",
      KERNEL.lstrip("\n") + "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n",
      "silu-convrot-quant kernel")
B = "omni_xpu_kernel/csrc/bindings.cpp"
patch(B, "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n        torch::Tensor input, int64_t group_size, int64_t row_split);\n",
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n        torch::Tensor input, int64_t group_size, int64_t row_split);\n"
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(\n        torch::Tensor fc1_out, int64_t group_size, int64_t row_split);\n", "bind decl")
patch(B, '        py::arg("input"), py::arg("group_size") = 256, py::arg("row_split") = 8);\n    int8.def("fused_scaleback"',
      '        py::arg("input"), py::arg("group_size") = 256, py::arg("row_split") = 8);\n'
      '    int8.def("quantize_int8_convrot_silu_mul_split", &omni_xpu::int8_ops::quantize_int8_convrot_silu_mul_split,\n'
      '        "SiLU(gate)*up on a [M, 2K] fc1 output fused with radix-4 Hadamard rotation and rowwise INT8 quantization.",\n'
      '        py::arg("fc1_out"), py::arg("group_size") = 256, py::arg("row_split") = 16);\n'
      '    int8.def("fused_scaleback"', "bind def")
P = "omni_xpu_kernel/int8/__init__.py"
patch(P, "        ):\n            x = _apply_h3_swiglu_exact(x, native)\n        else:\n            x = _apply_input_act(x, input_act)\n",
      "        ):\n"
      "            if (\n"
      "                _CONVROT_FUSED and _CONVROT_SPLIT > 0\n"
      "                and hasattr(native, \"quantize_int8_convrot_silu_mul_split\")\n"
      "                and bias is None\n"
      "            ):\n"
      "                # fc2 producer fusion: SiLU*mul + rotate + quantize in one pass\n"
      "                x_int8, x_scale = native.quantize_int8_convrot_silu_mul_split(\n"
      "                    x, convrot_groupsize, _CONVROT_SPLIT\n"
      "                )\n"
      "                return native.int8_linear_prequantized(\n"
      "                    x_int8, x_scale, weight, weight_scale, bias, dtype_code\n"
      "                )\n"
      "            x = _apply_h3_swiglu_exact(x, native)\n        else:\n            x = _apply_input_act(x, input_act)\n", "python main path")
patch(P, "        chunk = x[start:stop]\n        gate, up = chunk.chunk(2, dim=-1)\n        activated = native.fused_silu_mul_exact_bf16(gate, up)\n        if _can_quantize_convrot_fused(activated, native, 256):\n",
      "        chunk = x[start:stop]\n"
      "        if _CONVROT_FUSED and _CONVROT_SPLIT > 0 and hasattr(native, \"quantize_int8_convrot_silu_mul_split\"):\n"
      "            gate = up = activated = rotated = None\n"
      "            x_int8, x_scale = native.quantize_int8_convrot_silu_mul_split(chunk, 256, _CONVROT_SPLIT)\n"
      "            output_chunk = output[start:stop]\n"
      "            native.int8_linear_prequantized_out(x_int8, x_scale, weight, weight_scale, None, dtype_code, output_chunk)\n"
      "            del output_chunk, x_scale, x_int8, chunk\n"
      "            continue\n"
      "        gate, up = chunk.chunk(2, dim=-1)\n        activated = native.fused_silu_mul_exact_bf16(gate, up)\n        if _can_quantize_convrot_fused(activated, native, 256):\n", "python stream path")
print("done")
