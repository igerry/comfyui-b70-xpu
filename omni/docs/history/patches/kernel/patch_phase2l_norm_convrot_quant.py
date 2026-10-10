#!/usr/bin/env python3
"""Phase-2l: rms_norm_mod_convrot_quant -- optional gated residual (x += gate[mod]*other, in place),
RMSNorm, AdaLN scale/shift, radix-4 Hadamard and rowwise INT8 quantize in ONE ESIMD pass, producing
the int8 activation the following INT8 linear consumes. Removes the bf16 (M, hidden) materializations
between addcmul_/norm/rotate/quantize. Rounding boundaries follow the stock chain: residual -> bf16,
norm*mod -> bf16, rotated -> bf16."""
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
// [x += gate*other] -> RMSNorm -> *(1+scale)+shift -> Hadamard -> INT8 (2026-09-06)
// One row per work-group of WG work-items; thread t owns groups [t*gpw, ...).
// ---------------------------------------------------------------------------
template<typename InputT, int GroupSize, int WG, bool Residual>
void norm_mod_convrot_quant_kernel(
    InputT* __restrict__ x,
    const InputT* __restrict__ other,
    const InputT* __restrict__ gate,
    const InputT* __restrict__ weight,
    const InputT* __restrict__ scale_t,
    const InputT* __restrict__ shift_t,
    const int32_t* __restrict__ seg_end,
    const int32_t* __restrict__ seg_mod,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    float eps,
    const at::Device& device) {
    constexpr int kCache = 4;
    const int gpw = static_cast<int>((groups + WG - 1) / WG);
    TORCH_CHECK(gpw <= kCache, "norm_mod_convrot_quant: too many groups per work-item; raise row_split");
    const int64_t hidden = groups * GroupSize;
    sycl::range<2> global_size(static_cast<size_t>(rows), WG);
    sycl::range<2> local_size(1, WG);
    auto cgf = [&](sycl::handler& handle) {
        sycl::ext::oneapi::experimental::properties props{
            sycl::ext::intel::experimental::grf_size<OMNI_CONVROT_QUANT_GRF>};
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            props,
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                constexpr int kSlmBytes = (2 * WG * 4 < 32) ? 32 : 2 * WG * 4;
                slm_init<kSlmBytes>();
                const int64_t row = item.get_global_id(0);
                const int t = static_cast<int>(item.get_local_id(1));
                const int g0 = t * gpw;
                const int g1 = static_cast<int>(
                    (g0 + gpw < groups) ? (g0 + gpw) : groups);

                simd<int32_t, 8> ends = block_load<int32_t, 8>(seg_end);
                simd<int32_t, 8> mods = block_load<int32_t, 8>(seg_mod);
                int seg = 0;
#pragma unroll
                for (int s = 0; s < 7; ++s) {
                    if (row >= ends[s]) seg = s + 1;
                }
                const int64_t mod = mods[seg];
                const int64_t row_start = row * hidden;
                const int64_t mod_start = mod * hidden;

                simd<InputT, GroupSize> cache[kCache];
                float sumsq = 0.0f;
#pragma unroll
                for (int i = 0; i < kCache; ++i) {
                    const int g = g0 + i;
                    if (g < g1) {
                        const int64_t off = row_start + g * GroupSize;
                        simd<InputT, GroupSize> xv = block_load<InputT, GroupSize>(x + off);
                        if constexpr (Residual) {
                            simd<float, GroupSize> xf = xv;
                            simd<float, GroupSize> of = block_load<InputT, GroupSize>(other + off);
                            simd<float, GroupSize> gf = block_load<InputT, GroupSize>(gate + mod_start + g * GroupSize);
                            xv = xf + of * gf;  // bf16 rounding = stock addcmul_ output
                            block_store<InputT, GroupSize>(x + off, xv);
                        }
                        cache[i] = xv;
                        simd<float, GroupSize> xr = xv;
                        sumsq += sycl::ext::intel::esimd::detail::sum<float, float, GroupSize>(xr * xr);
                    }
                }
                slm_block_store<float, 1>(t * sizeof(float), sumsq);
                barrier();
                simd<float, WG> parts = slm_block_load<float, WG>(0);
                const float mean = sycl::ext::intel::esimd::detail::sum<float, float, WG>(parts) / static_cast<float>(hidden);
                const float inv_rms = rsqrt(mean + eps);

                float local_max = 0.0f;
#pragma unroll
                for (int i = 0; i < kCache; ++i) {
                    const int g = g0 + i;
                    if (g < g1) {
                        const int64_t col = g * GroupSize;
                        simd<float, GroupSize> xr = cache[i];
                        simd<float, GroupSize> wv = block_load<InputT, GroupSize>(weight + col);
                        simd<float, GroupSize> sv = block_load<InputT, GroupSize>(scale_t + mod_start + col);
                        simd<float, GroupSize> hv = block_load<InputT, GroupSize>(shift_t + mod_start + col);
                        simd<InputT, GroupSize> h = xr * inv_rms * wv * (sv + 1.0f) + hv;  // stock norm output boundary
                        simd<float, GroupSize> values = h;
                        radix4_stage_vec<1, GroupSize>(values);
                        values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));
                        simd<InputT, GroupSize> rounded = values;
                        cache[i] = rounded;
                        simd<float, GroupSize> rf = rounded;
                        const float gmax = hmax<float>(sycl::ext::intel::esimd::abs<float, GroupSize>(rf));
                        local_max = gmax > local_max ? gmax : local_max;
                    }
                }
                slm_block_store<float, 1>((WG + t) * sizeof(float), local_max);
                barrier();
                simd<float, WG> maxes = slm_block_load<float, WG>(WG * sizeof(float));
                const float row_max = hmax<float>(maxes);
                float qscale = row_max / 127.0f;
                if (qscale < 1e-30f) qscale = 1e-30f;
                const float inv_scale = 1.0f / qscale;
                if (t == 0) scales[row] = qscale;
#pragma unroll
                for (int i = 0; i < kCache; ++i) {
                    const int g = g0 + i;
                    if (g < g1) {
                        quantize_store_rounded<InputT, GroupSize>(
                            cache[i], output, row_start + g * GroupSize, inv_scale);
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, Residual ? "gate_norm_mod_convrot_quant" : "norm_mod_convrot_quant");
}

template<typename InputT, int GroupSize, bool Residual>
void dispatch_norm_quant(
    InputT* x, const InputT* other, const InputT* gate, const InputT* weight, const InputT* scale_t,
    const InputT* shift_t, const int32_t* seg_end, const int32_t* seg_mod, int8_t* output, float* scales,
    int64_t rows, int64_t groups, float eps, int row_split, const at::Device& device) {
    switch (row_split) {
        case 8:  norm_mod_convrot_quant_kernel<InputT, GroupSize, 8, Residual>(x, other, gate, weight, scale_t, shift_t, seg_end, seg_mod, output, scales, rows, groups, eps, device); break;
        case 32: norm_mod_convrot_quant_kernel<InputT, GroupSize, 32, Residual>(x, other, gate, weight, scale_t, shift_t, seg_end, seg_mod, output, scales, rows, groups, eps, device); break;
        default: norm_mod_convrot_quant_kernel<InputT, GroupSize, 16, Residual>(x, other, gate, weight, scale_t, shift_t, seg_end, seg_mod, output, scales, rows, groups, eps, device); break;
    }
}

}  // namespace

// x [M, H] (updated in place when `other` is given), other/gate optional, weight [H], scale/shift
// [mod_rows, H], seg_end/seg_mod int32[8]; returns (int8 [M, H], scales [M, 1]).
std::tuple<torch::Tensor, torch::Tensor> rms_norm_mod_convrot_quant(
    torch::Tensor x,
    std::optional<torch::Tensor> other,
    std::optional<torch::Tensor> gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor seg_end,
    torch::Tensor seg_mod,
    double eps,
    int64_t group_size,
    int64_t row_split) {
    TORCH_CHECK(x.device().is_xpu() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2D XPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16, "x must be bf16/fp16");
    TORCH_CHECK(group_size == 256 || group_size == 64, "group size must be 64 or 256");
    const int64_t rows = x.size(0), hidden = x.size(1);
    TORCH_CHECK(hidden % group_size == 0, "hidden must be divisible by group size");
    TORCH_CHECK(weight.is_contiguous() && weight.numel() == hidden && weight.scalar_type() == x.scalar_type(), "bad weight");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == hidden && scale.is_contiguous() && scale.scalar_type() == x.scalar_type(), "bad scale");
    TORCH_CHECK(shift.sizes() == scale.sizes() && shift.is_contiguous() && shift.scalar_type() == x.scalar_type(), "bad shift");
    TORCH_CHECK(seg_end.scalar_type() == torch::kInt32 && seg_end.numel() == 8 && seg_end.is_contiguous() &&
                seg_mod.scalar_type() == torch::kInt32 && seg_mod.numel() == 8 && seg_mod.is_contiguous(), "seg arrays must be int32[8]");
    const bool residual = other.has_value();
    if (residual) {
        TORCH_CHECK(gate.has_value(), "gate required with other");
        TORCH_CHECK(other->sizes() == x.sizes() && other->is_contiguous() && other->scalar_type() == x.scalar_type(), "bad other");
        TORCH_CHECK(gate->sizes() == scale.sizes() && gate->is_contiguous() && gate->scalar_type() == x.scalar_type(), "bad gate");
    }
    TORCH_CHECK(row_split == 8 || row_split == 16 || row_split == 32, "row_split must be 8, 16 or 32");
    auto output = torch::empty({rows, hidden}, x.options().dtype(torch::kInt8));
    auto scales = torch::empty({rows, 1}, x.options().dtype(torch::kFloat));
    if (x.numel() == 0) return {output, scales};
    const int64_t groups = hidden / group_size;
    auto* out_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    auto* sc_ptr = scales.data_ptr<float>();
    const auto* se = seg_end.data_ptr<int32_t>();
    const auto* sm = seg_mod.data_ptr<int32_t>();
    const int rs = static_cast<int>(row_split);
#define OMNI_NQ_DISPATCH(T, GS)                                                                        \
    {                                                                                                   \
        auto* xp = reinterpret_cast<T*>(x.data_ptr());                                                  \
        const T* op = residual ? reinterpret_cast<const T*>(other->data_ptr()) : nullptr;               \
        const T* gp = residual ? reinterpret_cast<const T*>(gate->data_ptr()) : nullptr;                \
        const auto* wp = reinterpret_cast<const T*>(weight.data_ptr());                                 \
        const auto* sp = reinterpret_cast<const T*>(scale.data_ptr());                                  \
        const auto* hp = reinterpret_cast<const T*>(shift.data_ptr());                                  \
        if (residual) dispatch_norm_quant<T, GS, true>(xp, op, gp, wp, sp, hp, se, sm, out_ptr, sc_ptr, rows, groups, static_cast<float>(eps), rs, x.device()); \
        else dispatch_norm_quant<T, GS, false>(xp, op, gp, wp, sp, hp, se, sm, out_ptr, sc_ptr, rows, groups, static_cast<float>(eps), rs, x.device()); \
    }
    if (x.scalar_type() == torch::kBFloat16) {
        if (group_size == 64) OMNI_NQ_DISPATCH(bf16, 64) else OMNI_NQ_DISPATCH(bf16, 256)
    } else {
        if (group_size == 64) OMNI_NQ_DISPATCH(fp16, 64) else OMNI_NQ_DISPATCH(fp16, 256)
    }
#undef OMNI_NQ_DISPATCH
    return {output, scales};
}

namespace {
'''
patch("omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp",
      "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n",
      KERNEL.lstrip("\n") + "}  // namespace\n\nstd::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_split(\n",
      "norm-convrot-quant kernel")
B = "omni_xpu_kernel/csrc/bindings.cpp"
patch(B, "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(\n        torch::Tensor fc1_out, int64_t group_size, int64_t row_split);\n",
      "    std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(\n        torch::Tensor fc1_out, int64_t group_size, int64_t row_split);\n"
      "    std::tuple<torch::Tensor, torch::Tensor> rms_norm_mod_convrot_quant(\n"
      "        torch::Tensor x, std::optional<torch::Tensor> other, std::optional<torch::Tensor> gate,\n"
      "        torch::Tensor weight, torch::Tensor scale, torch::Tensor shift, torch::Tensor seg_end,\n"
      "        torch::Tensor seg_mod, double eps, int64_t group_size, int64_t row_split);\n", "bind decl")
patch(B, '        py::arg("fc1_out"), py::arg("group_size") = 256, py::arg("row_split") = 16);\n    int8.def("fused_scaleback"',
      '        py::arg("fc1_out"), py::arg("group_size") = 256, py::arg("row_split") = 16);\n'
      '    int8.def("rms_norm_mod_convrot_quant", &omni_xpu::int8_ops::rms_norm_mod_convrot_quant,\n'
      '        "[x += gate[mod]*other] -> RMSNorm -> *(1+scale[mod])+shift[mod] -> Hadamard -> rowwise INT8, one pass.\\n"\n'
      '        "Returns (int8 [M,H], scales [M,1]); x is updated in place when other is given.",\n'
      '        py::arg("x"), py::arg("other") = py::none(), py::arg("gate") = py::none(), py::arg("weight"),\n'
      '        py::arg("scale"), py::arg("shift"), py::arg("seg_end"), py::arg("seg_mod"), py::arg("eps") = 1e-6,\n'
      '        py::arg("group_size") = 256, py::arg("row_split") = 8);\n'
      '    int8.def("fused_scaleback"', "bind def")
print("done")
