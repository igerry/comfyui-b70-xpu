#!/usr/bin/env python3
"""Phase-2h: rms_norm_mod = RMSNorm with the AdaLN scale/shift fused into the normalize pass.
H3's DiTBlock does norm -> bf16 -> mul_(1+scale) -> bf16 -> add_(shift) -> bf16 per segment
(two extra full passes over (M,5376), 3.9 ms/block for the two norms). Rows are mapped to a
modulation row through <= 8 contiguous segments (seg_end cumulative, seg_mod row index)."""
import sys
from pathlib import Path
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src")

def patch(rel, old, new, tag):
    p = SRC / rel; s = p.read_text()
    if new in s: print(f"[skip] {tag}"); return
    assert s.count(old) == 1, f"{tag}: count={s.count(old)}"
    p.write_text(s.replace(old, new)); print(f"[ok] {tag}")

KERNEL = r'''
// ============================================================================
// RMSNorm + AdaLN modulation (2026-09-06): out = norm(x)*w*(1+scale[mod]) + shift[mod]
// mod = seg_mod[first s with row < seg_end[s]]; <= 8 segments, arrays padded to 8.
// ============================================================================
template<typename IT, const int GS, const int BS>
void rms_norm_mod_kernel(
    const void* weight_ptr,
    const void* input_ptr,
    void* output_ptr,
    const void* scale_ptr,
    const void* shift_ptr,
    const int32_t* seg_end,
    const int32_t* seg_mod,
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    constexpr int slm_acc_align = ((GS * (int)sizeof(float) + 15) / 16) * 16;
    const int acc_offset = hidden_size * sizeof(IT);

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + slm_acc_align>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                simd<int32_t, 8> ends = block_load<int32_t, 8>(seg_end);
                simd<int32_t, 8> mods = block_load<int32_t, 8>(seg_mod);
                int seg = 0;
#pragma unroll
                for (int s = 0; s < 7; ++s) {
                    if (rid >= ends[s]) seg = s + 1;
                }
                // `seg` is the first segment whose end is beyond rid because ends are ascending.
                const int mod = mods[seg];

                const IT* weight = (const IT*)weight_ptr;
                const IT* input = (const IT*)input_ptr + hidden_size * (size_t)rid;
                const IT* sc = (const IT*)scale_ptr + hidden_size * (size_t)mod;
                const IT* sh = (const IT*)shift_ptr + hidden_size * (size_t)mod;
                IT* output = (IT*)output_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                simd<float, BS> accv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<IT, BS> xv = block_load<IT, BS>(input + i * BS);
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), xv);
                    simd<float, BS> xv_f32 = xv;
                    accv += xv_f32 * xv_f32;
                }
                float acc = sycl::ext::intel::esimd::detail::sum<float, float, BS>(accv) / hidden_size;

                float scale;
                if constexpr (GS == 1) {
                    scale = rsqrt(acc + eps);
                } else {
                    slm_block_store<float, 1>(acc_offset + tid * sizeof(float), acc);
                    barrier();
                    simd<float, GS> accs = slm_block_load<float, GS>(acc_offset);
                    float mean = sycl::ext::intel::esimd::detail::sum<float, float, GS>(accs);
                    scale = rsqrt(mean + eps);
                }
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                    simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                    simd<float, BS> sv = block_load<IT, BS>(sc + i * BS);
                    simd<float, BS> hv = block_load<IT, BS>(sh + i * BS);
                    simd<IT, BS> result = xv * scale * yv * (sv + 1.0f) + hv;
                    block_store<IT, BS>(output + i * BS, result);
                }
            }
        );
    };
    utils::submit_kernel(cgf, device, "rms_norm_mod_esimd");
}

template<typename IT, int BS>
using rms_mod_fn_t = void(*)(const void*, const void*, void*, const void*, const void*,
                             const int32_t*, const int32_t*, float, const int, const int, const at::Device&);

template<typename IT, int BS>
rms_mod_fn_t<IT, BS> select_rms_mod_kernel(int nb) {
    if (nb <= 1)  return rms_norm_mod_kernel<IT, 1,  BS>;
    if (nb <= 2)  return rms_norm_mod_kernel<IT, 2,  BS>;
    if (nb <= 4)  return rms_norm_mod_kernel<IT, 4,  BS>;
    if (nb <= 8)  return rms_norm_mod_kernel<IT, 8,  BS>;
    if (nb <= 16) return rms_norm_mod_kernel<IT, 16, BS>;
    return rms_norm_mod_kernel<IT, 32, BS>;
}

torch::Tensor rms_norm_mod(
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor seg_end,
    torch::Tensor seg_mod,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, hidden]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1), "weight must be [hidden]");
    TORCH_CHECK(scale.dim() == 2 && shift.dim() == 2 && scale.size(1) == input.size(1) &&
                shift.sizes() == scale.sizes(), "scale/shift must be [mod_rows, hidden]");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && scale.scalar_type() == input.scalar_type() &&
                shift.scalar_type() == input.scalar_type(), "dtypes must match");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scale.is_contiguous() && shift.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(seg_end.scalar_type() == torch::kInt32 && seg_mod.scalar_type() == torch::kInt32 &&
                seg_end.numel() == 8 && seg_mod.numel() == 8 && seg_end.is_contiguous() && seg_mod.is_contiguous() &&
                seg_end.device() == input.device() && seg_mod.device() == input.device(),
                "seg_end/seg_mod must be int32[8] on the input device");
    const int64_t input_size = input.size(0);
    const int64_t hidden_size = input.size(1);
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 8192 && hidden_size % 32 == 0,
                "hidden_size must be <= 8192 and divisible by 32");
    auto output = torch::empty({input_size, hidden_size},
        torch::device(input.device()).dtype(input.dtype()));
    if (input_size == 0) return output;
    const auto* se = seg_end.data_ptr<int32_t>();
    const auto* sm = seg_mod.data_ptr<int32_t>();
#define OMNI_RMS_MOD_DISPATCH(T, BSV)                                                     \
    {                                                                                      \
        auto fn = select_rms_mod_kernel<T, BSV>(hidden_size / BSV);                        \
        fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(), scale.data_ptr(),      \
           shift.data_ptr(), se, sm, static_cast<float>(eps), input_size, hidden_size,    \
           input.device());                                                                \
    }
    if (hidden_size % 64 == 0 && hidden_size >= 2048) {
        switch (input.scalar_type()) {
            case ST::Float: OMNI_RMS_MOD_DISPATCH(float, 64); break;
            case ST::Half: OMNI_RMS_MOD_DISPATCH(fp16, 64); break;
            case ST::BFloat16: OMNI_RMS_MOD_DISPATCH(bf16, 64); break;
            default: TORCH_CHECK(false, "Unsupported dtype");
        }
    } else {
        switch (input.scalar_type()) {
            case ST::Float: OMNI_RMS_MOD_DISPATCH(float, 32); break;
            case ST::Half: OMNI_RMS_MOD_DISPATCH(fp16, 32); break;
            case ST::BFloat16: OMNI_RMS_MOD_DISPATCH(bf16, 32); break;
            default: TORCH_CHECK(false, "Unsupported dtype");
        }
    }
#undef OMNI_RMS_MOD_DISPATCH
    return output;
}

}  // namespace norm
'''
patch("omni_xpu_kernel/csrc/norm.cpp", "}  // namespace norm\n}  // namespace omni_xpu\n",
      KERNEL.lstrip("\n") + "}  // namespace omni_xpu\n", "norm.cpp kernel+entry")
B = "omni_xpu_kernel/csrc/bindings.cpp"
s = (SRC / B).read_text()
decl_anchor = [l for l in s.splitlines() if "torch::Tensor rms_norm(" in l and "def" not in l]
assert len(decl_anchor) == 1, decl_anchor
line = decl_anchor[0]
patch(B, line + "\n", line + "\n    torch::Tensor rms_norm_mod(torch::Tensor weight, torch::Tensor input, torch::Tensor scale,\n        torch::Tensor shift, torch::Tensor seg_end, torch::Tensor seg_mod, double eps);\n", "bind decl")
patch(B, '    norm.def("rms_norm", &omni_xpu::norm::rms_norm,\n        "RMSNorm using ESIMD optimization",\n        py::arg("weight"), py::arg("input"), py::arg("eps") = 1e-6);\n',
      '    norm.def("rms_norm", &omni_xpu::norm::rms_norm,\n        "RMSNorm using ESIMD optimization",\n        py::arg("weight"), py::arg("input"), py::arg("eps") = 1e-6);\n'
      '    norm.def("rms_norm_mod", &omni_xpu::norm::rms_norm_mod,\n'
      '        "RMSNorm with fused AdaLN modulation: norm(x)*w*(1+scale[mod])+shift[mod], mod per row via <=8 segments\\n"\n'
      '        "(seg_end int32[8] cumulative row ends, seg_mod int32[8] modulation row index).",\n'
      '        py::arg("weight"), py::arg("input"), py::arg("scale"), py::arg("shift"), py::arg("seg_end"),\n'
      '        py::arg("seg_mod"), py::arg("eps") = 1e-6);\n', "bind def")
patch("omni_xpu_kernel/norm/__init__.py", "def rms_norm_gate_residual(\n",
      "def rms_norm_mod(weight, input, scale, shift, seg_end, seg_mod, eps=1e-6):\n"
      "    \"\"\"RMSNorm with fused AdaLN scale/shift (see native docstring).\"\"\"\n"
      "    return _get_native().rms_norm_mod(weight, input, scale, shift, seg_end, seg_mod, eps)\n\n\n"
      "def rms_norm_gate_residual(\n", "python wrapper")
print("done")
