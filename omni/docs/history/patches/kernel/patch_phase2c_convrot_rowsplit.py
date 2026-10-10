#!/usr/bin/env python3
"""Phase-2c: row-split fused ConvRot+INT8 quantize kernel.
Intel's fused kernel gives one row per work-item (21-56 serial 256-wide groups, two passes) and
reaches ~200 GB/s on B70. This variant spreads one row over WG work-items (contiguous group ranges),
reduces the row max through SLM, then each work-item quantizes its own groups. Same math (radix-4
Hadamard in fp32, rounding boundary preserved) so int8 output is identical.
Exported as int8.quantize_int8_convrot_fused_split(input, group_size, row_split in {4,8,16,32}).
Python: OMNIXPU_CONVROT_SPLIT (default 8) selects it inside _can_quantize_convrot_fused's path."""
import sys
from pathlib import Path
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src")

def patch(rel, old, new, tag):
    p = SRC / rel; s = p.read_text()
    if new in s: print(f"[skip] {tag}"); return
    assert s.count(old) == 1, f"{tag}: count={s.count(old)}"
    p.write_text(s.replace(old, new)); print(f"[ok] {tag}")

KERNEL = r'''
// ---------------------------------------------------------------------------
// Row-split variant (2026-09-06): WG work-items share one row. Each handles a
// contiguous range of GroupSize-wide groups; the row max is reduced via SLM.
// ---------------------------------------------------------------------------
template<typename InputT, int GroupSize, int WG>
void quantize_convrot_rowsplit_kernel(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    const int gpw = static_cast<int>((groups + WG - 1) / WG);
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
                const int64_t row_start = row * groups * GroupSize;
                const int g0 = t * gpw;
                const int g1 = static_cast<int>(
                    (g0 + gpw < groups) ? (g0 + gpw) : groups);

                float local_max = 0.0f;
                for (int g = g0; g < g1; ++g) {
                    simd<float, GroupSize> values =
                        load_rotated<InputT, GroupSize>(
                            input + row_start + g * GroupSize);
                    simd<float, GroupSize> magnitudes =
                        sycl::ext::intel::esimd::abs<float, GroupSize>(values);
                    const float group_max = hmax<float>(magnitudes);
                    local_max = group_max > local_max ? group_max : local_max;
                }
                slm_block_store<float, 1>(t * sizeof(float), local_max);
                barrier();
                simd<float, WG> partials = slm_block_load<float, WG>(0);
                const float row_max = hmax<float>(partials);

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                if (t == 0) scales[row] = scale;

                for (int g = g0; g < g1; ++g) {
                    quantize_store_group<InputT, GroupSize>(
                        input, output, row_start + g * GroupSize, inv_scale);
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_quantize_rowsplit");
}

template<typename InputT, int GroupSize>
void dispatch_rowsplit(
    const InputT* input, int8_t* output, float* scales,
    int64_t rows, int64_t groups, int row_split, const at::Device& device) {
    switch (row_split) {
        case 4:  quantize_convrot_rowsplit_kernel<InputT, GroupSize, 4>(input, output, scales, rows, groups, device); break;
        case 16: quantize_convrot_rowsplit_kernel<InputT, GroupSize, 16>(input, output, scales, rows, groups, device); break;
        case 32: quantize_convrot_rowsplit_kernel<InputT, GroupSize, 32>(input, output, scales, rows, groups, device); break;
        default: quantize_convrot_rowsplit_kernel<InputT, GroupSize, 8>(input, output, scales, rows, groups, device); break;
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(
    torch::Tensor input,
    int64_t group_size,
    int64_t row_split) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 ||
            input.scalar_type() == torch::kFloat16,
        "input must be bf16 or fp16");
    TORCH_CHECK(
        group_size == 64 || group_size == 256,
        "fused ConvRot quantization supports group sizes 64 and 256");
    TORCH_CHECK(input.size(1) % group_size == 0, "invalid group size");
    TORCH_CHECK(
        row_split == 4 || row_split == 8 || row_split == 16 || row_split == 32,
        "row_split must be 4, 8, 16 or 32");
    input = input.contiguous();
    auto output = torch::empty_like(
        input, input.options().dtype(torch::kInt8));
    auto scales = torch::empty(
        {input.size(0), 1}, input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return {output, scales};
    const int64_t rows = input.size(0);
    const int64_t groups = input.size(1) / group_size;
    auto* out_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    auto* scale_ptr = scales.data_ptr<float>();
    const int rs = static_cast<int>(row_split);
    if (input.scalar_type() == torch::kBFloat16) {
        const auto* in_ptr = reinterpret_cast<const bf16*>(input.data_ptr());
        if (group_size == 64) dispatch_rowsplit<bf16, 64>(in_ptr, out_ptr, scale_ptr, rows, groups, rs, input.device());
        else dispatch_rowsplit<bf16, 256>(in_ptr, out_ptr, scale_ptr, rows, groups, rs, input.device());
    } else {
        const auto* in_ptr = reinterpret_cast<const fp16*>(input.data_ptr());
        if (group_size == 64) dispatch_rowsplit<fp16, 64>(in_ptr, out_ptr, scale_ptr, rows, groups, rs, input.device());
        else dispatch_rowsplit<fp16, 256>(in_ptr, out_ptr, scale_ptr, rows, groups, rs, input.device());
    }
    return {output, scales};
}

namespace {
'''
# insert the new kernel right before the existing "}  // namespace" that closes the anonymous
# namespace ahead of quantize_int8_convrot_fused (the file has exactly one such closer before it)
patch("omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp",
      "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(\n",
      KERNEL.lstrip("\n") + "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(\n",
      "rowsplit kernel")
patch("omni_xpu_kernel/csrc/bindings.cpp",
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(\n        torch::Tensor input, int64_t group_size);\n",
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(\n        torch::Tensor input, int64_t group_size);\n"
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n        torch::Tensor input, int64_t group_size, int64_t row_split);\n",
      "bindings decl")
patch("omni_xpu_kernel/csrc/bindings.cpp",
      '        py::arg("input"), py::arg("group_size") = 256);\n    int8.def("fused_scaleback"',
      '        py::arg("input"), py::arg("group_size") = 256);\n'
      '    int8.def("quantize_int8_convrot_fused_split", &omni_xpu::int8_ops::quantize_int8_convrot_fused_split,\n'
      '        "Row-split fused radix-4 Hadamard rotation + rowwise INT8 quantization (row_split work-items per row).",\n'
      '        py::arg("input"), py::arg("group_size") = 256, py::arg("row_split") = 8);\n'
      '    int8.def("fused_scaleback"',
      "bindings def")
patch("omni_xpu_kernel/int8/__init__.py",
      '_CONVROT_FUSED_MIN_K = int(os.environ.get("OMNIXPU_CONVROT_FUSED_MIN_K", "8192"))\n',
      '_CONVROT_FUSED_MIN_K = int(os.environ.get("OMNIXPU_CONVROT_FUSED_MIN_K", "8192"))\n'
      '# Row-split variant (0 = Intel one-row-per-work-item kernel).\n'
      '_CONVROT_SPLIT = int(os.environ.get("OMNIXPU_CONVROT_SPLIT", "8"))\n\n\n'
      'def _quantize_convrot_fused(x, native, convrot_groupsize):\n'
      '    if _CONVROT_SPLIT > 0 and hasattr(native, "quantize_int8_convrot_fused_split"):\n'
      '        return native.quantize_int8_convrot_fused_split(x, convrot_groupsize, _CONVROT_SPLIT)\n'
      '    return native.quantize_int8_convrot_fused(x, convrot_groupsize)\n',
      "python helper")
s = (SRC / "omni_xpu_kernel/int8/__init__.py").read_text()
n = s.count("native.quantize_int8_convrot_fused(")
s2 = s.replace("native.quantize_int8_convrot_fused(x, convrot_groupsize)", "_quantize_convrot_fused(x, native, convrot_groupsize)") \
      .replace("native.quantize_int8_convrot_fused(\n                    x, convrot_groupsize\n                )", "_quantize_convrot_fused(x, native, convrot_groupsize)") \
      .replace("native.quantize_int8_convrot_fused(activated, 256)", "_quantize_convrot_fused(activated, native, 256)") \
      .replace("native.quantize_int8_convrot_fused(chunk, 256)", "_quantize_convrot_fused(chunk, native, 256)")
(SRC / "omni_xpu_kernel/int8/__init__.py").write_text(s2)
print(f"[ok] python call sites rewired ({n} direct calls before, {s2.count('native.quantize_int8_convrot_fused(')} after)")
