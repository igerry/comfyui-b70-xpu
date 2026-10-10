#!/usr/bin/env python3
"""Phase-1d: ESIMD kernels cannot call a plain lambda that uses ESIMD intrinsics
(icpx: "SYCL kernel without 'sycl_explicit_simd' attribute can't call a function with this
attribute"). Replace the quantize_store lambda from phase-1c with a SYCL_ESIMD_FUNCTION
template helper."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp")
s = p.read_text()
if "quantize_store_group<" in s:
    print("[skip]"); sys.exit(0)
old_lambda = """                simd<float, GroupSize> clamp_lo(-128.0f);
                simd<float, GroupSize> clamp_hi(127.0f);
                auto quantize_store = [&](int64_t g) {
                    const int64_t offset = row_start + g * GroupSize;
                    simd<float, GroupSize> values =
                        load_rotated<InputT, GroupSize>(input + offset);
                    simd<float, GroupSize> quantized =
                        sycl::ext::intel::esimd::rnde<float, GroupSize>(
                            values * inv_scale);
                    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(
                        sycl::ext::intel::esimd::min<float, GroupSize>(
                            quantized, clamp_hi),
                        clamp_lo);
                    simd<int8_t, GroupSize> packed = quantized;
                    block_store<int8_t, GroupSize>(output + offset, packed);
                };
                group = 0;
                for (; group + 1 < groups; group += 2) {
                    quantize_store(group);
                    quantize_store(group + 1);
                }
                for (; group < groups; ++group) {
                    quantize_store(group);
                }
"""
new_body = """                group = 0;
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
"""
assert s.count(old_lambda) == 1
s = s.replace(old_lambda, new_body)
anchor = "template<typename InputT, int GroupSize>\nvoid quantize_convrot_kernel(\n"
helper = """template<typename InputT, int GroupSize>
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

"""
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)
p.write_text(s); print("[ok] lambda -> SYCL_ESIMD_FUNCTION helper")
