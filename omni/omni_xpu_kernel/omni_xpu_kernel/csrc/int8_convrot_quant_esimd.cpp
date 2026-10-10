#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <tuple>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_CONVROT_QUANT_GRF
#if defined(OMNI_XPU_ARCH_BMG)
#define OMNI_CONVROT_QUANT_GRF 256
#else
#define OMNI_CONVROT_QUANT_GRF 128
#endif
#endif
#ifndef OMNI_CONVROT_QUANT_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_CONVROT_QUANT_WG_SIZE 1
#elif defined(OMNI_XPU_ARCH_BMG)
// The fused path remains disabled on BMG until it is measured there.
#define OMNI_CONVROT_QUANT_WG_SIZE 8
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<int Stride, int GroupSize>
inline void radix4_hadamard_stage(simd<float, GroupSize>& values) {
#pragma unroll
    for (int base = 0; base < GroupSize; base += 4 * Stride) {
        simd<float, Stride> a = values.template select<Stride, 1>(base);
        simd<float, Stride> b =
            values.template select<Stride, 1>(base + Stride);
        simd<float, Stride> c =
            values.template select<Stride, 1>(base + 2 * Stride);
        simd<float, Stride> d =
            values.template select<Stride, 1>(base + 3 * Stride);
        values.template select<Stride, 1>(base) = a + b + c - d;
        values.template select<Stride, 1>(base + Stride) = a + b - c + d;
        values.template select<Stride, 1>(base + 2 * Stride) = a - b + c + d;
        values.template select<Stride, 1>(base + 3 * Stride) = -a + b + c + d;
    }
    if constexpr (Stride * 4 < GroupSize) {
        radix4_hadamard_stage<Stride * 4, GroupSize>(values);
    }
}

template<typename InputT, int GroupSize>
inline simd<float, GroupSize> load_rotated(
    const InputT* __restrict__ input) {
    simd<InputT, GroupSize> source = block_load<InputT, GroupSize>(input);
    simd<float, GroupSize> values = source;
    radix4_hadamard_stage<1, GroupSize>(values);
    values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));

    // The composed path materializes the matmul result in the input dtype
    // before rowwise quantization. Preserve that rounding boundary exactly.
    simd<InputT, GroupSize> rounded = values;
    simd<float, GroupSize> result = rounded;
    return result;
}

template<typename InputT, int GroupSize>
SYCL_ESIMD_FUNCTION inline void quantize_store_group(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t offset,
    float inv_scale) {
    simd<float, GroupSize> values =
        load_rotated<InputT, GroupSize>(input + offset);
    simd<float, GroupSize> quantized =
        sycl::ext::intel::esimd::rnde<float, GroupSize>(values * inv_scale);
    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(
        sycl::ext::intel::esimd::min<float, GroupSize>(
            quantized, simd<float, GroupSize>(127.0f)),
        simd<float, GroupSize>(-128.0f));
    simd<int8_t, GroupSize> packed = quantized;
    block_store<int8_t, GroupSize>(output + offset, packed);
}

template<typename InputT, int GroupSize>
void quantize_convrot_kernel(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int WorkGroupSize = OMNI_CONVROT_QUANT_WG_SIZE;
    const int64_t padded =
        (rows + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;
    auto cgf = [&](sycl::handler& handle) {
        sycl::ext::oneapi::experimental::properties props{
            sycl::ext::intel::experimental::grf_size<OMNI_CONVROT_QUANT_GRF>};
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            props,
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                if (row >= rows) return;
                const int64_t row_start = row * groups * GroupSize;

                // Two independent load->Hadamard chains per iteration so the
                // second 512-byte load is issued before the first chain's math.
                float row_max = 0.0f;
                int64_t group = 0;
                for (; group + 1 < groups; group += 2) {
                    simd<float, GroupSize> values0 =
                        load_rotated<InputT, GroupSize>(
                            input + row_start + group * GroupSize);
                    simd<float, GroupSize> values1 =
                        load_rotated<InputT, GroupSize>(
                            input + row_start + (group + 1) * GroupSize);
                    simd<float, GroupSize> magnitudes =
                        sycl::ext::intel::esimd::max<float, GroupSize>(
                            sycl::ext::intel::esimd::abs<float, GroupSize>(values0),
                            sycl::ext::intel::esimd::abs<float, GroupSize>(values1));
                    const float group_max = hmax<float>(magnitudes);
                    row_max = group_max > row_max ? group_max : row_max;
                }
                for (; group < groups; ++group) {
                    simd<float, GroupSize> values =
                        load_rotated<InputT, GroupSize>(
                            input + row_start + group * GroupSize);
                    simd<float, GroupSize> magnitudes =
                        sycl::ext::intel::esimd::abs<float, GroupSize>(values);
                    const float group_max = hmax<float>(magnitudes);
                    row_max = group_max > row_max ? group_max : row_max;
                }

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                scales[row] = scale;

                group = 0;
                for (; group + 1 < groups; group += 2) {
                    quantize_store_group<InputT, GroupSize>(
                        input, output, row_start + group * GroupSize, inv_scale);
                    quantize_store_group<InputT, GroupSize>(
                        input, output, row_start + (group + 1) * GroupSize,
                        inv_scale);
                }
                for (; group < groups; ++group) {
                    quantize_store_group<InputT, GroupSize>(
                        input, output, row_start + group * GroupSize, inv_scale);
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_quantize_fused");
}

template<typename InputT>
void dispatch_group(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    int64_t group_size) {
    const int64_t rows = input.size(0);
    const int64_t groups = input.size(1) / group_size;
    const auto* input_ptr = reinterpret_cast<const InputT*>(input.data_ptr());
    auto* output_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    if (group_size == 64) {
        quantize_convrot_kernel<InputT, 64>(
            input_ptr, output_ptr, scales.data_ptr<float>(), rows, groups,
            input.device());
    } else {
        quantize_convrot_kernel<InputT, 256>(
            input_ptr, output_ptr, scales.data_ptr<float>(), rows, groups,
            input.device());
    }
}

// ---------------------------------------------------------------------------
// Row-split variant (2026-09-06): WG work-items share one row. Each handles a
// contiguous range of GroupSize-wide groups; the row max is reduced via SLM.
// Butterflies are 64-wide vector ops (2-D views); rotated values are cached in
// GRF between the two passes when a work-item owns <= kCache groups.
template<int Stride, int GroupSize>
inline void radix4_stage_vec(simd<float, GroupSize>& values) {
    constexpr int Rows = GroupSize / (4 * Stride);
    constexpr int Q = GroupSize / 4;
    auto m = values.template bit_cast_view<float, Rows, 4 * Stride>();
    simd<float, Q> a = m.template select<Rows, 1, Stride, 1>(0, 0);
    simd<float, Q> b = m.template select<Rows, 1, Stride, 1>(0, Stride);
    simd<float, Q> c = m.template select<Rows, 1, Stride, 1>(0, 2 * Stride);
    simd<float, Q> d = m.template select<Rows, 1, Stride, 1>(0, 3 * Stride);
    m.template select<Rows, 1, Stride, 1>(0, 0) = a + b + c - d;
    m.template select<Rows, 1, Stride, 1>(0, Stride) = a + b - c + d;
    m.template select<Rows, 1, Stride, 1>(0, 2 * Stride) = a - b + c + d;
    m.template select<Rows, 1, Stride, 1>(0, 3 * Stride) = -a + b + c + d;
    if constexpr (Stride * 4 < GroupSize) {
        radix4_stage_vec<Stride * 4, GroupSize>(values);
    }
}

// Rotated group rounded to the input dtype (the composed path's materialization boundary).
template<typename InputT, int GroupSize>
inline simd<InputT, GroupSize> load_rotated_rounded_vec(
    const InputT* __restrict__ input) {
    simd<InputT, GroupSize> source = block_load<InputT, GroupSize>(input);
    simd<float, GroupSize> values = source;
    radix4_stage_vec<1, GroupSize>(values);
    values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));
    simd<InputT, GroupSize> rounded = values;
    return rounded;
}

template<typename InputT, int GroupSize>
SYCL_ESIMD_FUNCTION inline void quantize_store_rounded(
    simd<InputT, GroupSize> rounded,
    int8_t* __restrict__ output,
    int64_t offset,
    float inv_scale) {
    simd<float, GroupSize> values = rounded;
    simd<float, GroupSize> quantized =
        sycl::ext::intel::esimd::rnde<float, GroupSize>(values * inv_scale);
    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(
        sycl::ext::intel::esimd::min<float, GroupSize>(
            quantized, simd<float, GroupSize>(127.0f)),
        simd<float, GroupSize>(-128.0f));
    simd<int8_t, GroupSize> packed = quantized;
    block_store<int8_t, GroupSize>(output + offset, packed);
}

template<typename InputT, int GroupSize, int WG>
void quantize_convrot_rowsplit_kernel(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int kCache = 4;  // groups kept in GRF per work-item (4 x 512 B bf16)
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
                const int64_t row_start = row * groups * GroupSize;
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
                            cache[i] = load_rotated_rounded_vec<InputT, GroupSize>(
                                input + row_start + g * GroupSize);
                            simd<float, GroupSize> values = cache[i];
                            const float group_max = hmax<float>(
                                sycl::ext::intel::esimd::abs<float, GroupSize>(values));
                            local_max = group_max > local_max ? group_max : local_max;
                        }
                    }
                } else {
                    for (int g = g0; g < g1; ++g) {
                        simd<float, GroupSize> values =
                            load_rotated_rounded_vec<InputT, GroupSize>(
                                input + row_start + g * GroupSize);
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
                                cache[i], output, row_start + g * GroupSize, inv_scale);
                        }
                    }
                } else {
                    for (int g = g0; g < g1; ++g) {
                        quantize_store_rounded<InputT, GroupSize>(
                            load_rotated_rounded_vec<InputT, GroupSize>(
                                input + row_start + g * GroupSize),
                            output, row_start + g * GroupSize, inv_scale);
                    }
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

// SiLU(gate)*up -> rotated -> rounded group (fc2 producer fusion, 2026-09-06).
template<typename InputT, int GroupSize>
inline simd<InputT, GroupSize> load_silu_mul_rotated_rounded_vec(
    const InputT* __restrict__ gate,
    const InputT* __restrict__ up) {
    simd<InputT, GroupSize> gate_v = block_load<InputT, GroupSize>(gate);
    simd<InputT, GroupSize> up_v = block_load<InputT, GroupSize>(up);
    simd<float, GroupSize> x = gate_v;
    simd<float, GroupSize> denominator = 1.0f + sycl::ext::intel::esimd::exp(-x);
    simd<float, GroupSize> silu = x / denominator;
    // Same refinement as h3_silu_mul.cpp: one Newton step on the divide for finite positive x.
    simd<float, GroupSize> refined = silu + (x - silu * denominator) / denominator;
    silu.merge(refined, x > 0.0f);
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
// gate_second != 0: the row is [up | gate] (MiniMax Music 3 GLU) instead of [gate | up].
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_silu_mul_split(
    torch::Tensor fc1_out,
    int64_t group_size,
    int64_t row_split,
    int64_t gate_second) {
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
        const auto* first = reinterpret_cast<const bf16*>(fc1_out.data_ptr());
        const auto* base = gate_second ? first + width : first;
        const auto* upp = gate_second ? first : first + width;
        if (group_size == 64) dispatch_silu_rowsplit<bf16, 64>(base, upp, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
        else dispatch_silu_rowsplit<bf16, 256>(base, upp, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
    } else {
        const auto* first = reinterpret_cast<const fp16*>(fc1_out.data_ptr());
        const auto* base = gate_second ? first + width : first;
        const auto* upp = gate_second ? first : first + width;
        if (group_size == 64) dispatch_silu_rowsplit<fp16, 64>(base, upp, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
        else dispatch_silu_rowsplit<fp16, 256>(base, upp, stride, out_ptr, scale_ptr, rows, groups, rs, fc1_out.device());
    }
    return {output, scales};
}

namespace {
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
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(
    torch::Tensor input,
    int64_t group_size) {
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
    input = input.contiguous();
    auto output = torch::empty_like(
        input, input.options().dtype(torch::kInt8));
    auto scales = torch::empty(
        {input.size(0), 1}, input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return {output, scales};
    if (input.scalar_type() == torch::kBFloat16) {
        dispatch_group<bf16>(input, output, scales, group_size);
    } else {
        dispatch_group<fp16>(input, output, scales, group_size);
    }
    return {output, scales};
}

}  // namespace int8_ops
}  // namespace omni_xpu
