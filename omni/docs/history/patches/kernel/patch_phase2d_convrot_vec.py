#!/usr/bin/env python3
"""Phase-2d: vectorized radix-4 Hadamard + register-cached second pass for the row-split kernel.
Intel's radix4_hadamard_stage iterates `select<Stride,1>` blocks: at Stride=1 that is 64 scalar
butterflies per 256-wide group (SIMD utilisation 1/16), at Stride=4 1/4 -- the kernel is
instruction-bound (~250 GB/s regardless of parallelism). Each stage is now four 64-wide vector ops
through a 2-D bit_cast_view. With WG work-items per row each owning <= CACHE groups, the rotated
(bf16-rounded) values stay in GRF between the max pass and the quantize pass."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp")
s = p.read_text()
if "radix4_stage_vec" in s:
    print("[skip]"); sys.exit(0)
start = s.index("// Row-split variant (2026-09-06)")
end = s.index("template<typename InputT, int GroupSize>\nvoid dispatch_rowsplit(")
new = r'''// Row-split variant (2026-09-06): WG work-items share one row. Each handles a
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
                    CUTLASS_PRAGMA_UNROLL_PLACEHOLDER
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
                    CUTLASS_PRAGMA_UNROLL_PLACEHOLDER
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

'''.replace("CUTLASS_PRAGMA_UNROLL_PLACEHOLDER", "#pragma unroll")
s = s[:start] + new + s[end:]
p.write_text(s); print("[ok] vectorized row-split kernel")
