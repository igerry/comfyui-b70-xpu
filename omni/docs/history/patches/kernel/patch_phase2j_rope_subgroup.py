#!/usr/bin/env python3
"""Phase-2j: sub-group-per-row H3 RMSNorm+RoPE kernel.
Intel's BMG fast path uses a 64-work-item work-group per 128-element row (2 elements per work-item,
SLM tree reduction, 7 barriers; 1.83 M work-groups per call, 267 GB/s). This variant gives each
row to one 16-lane sub-group; lane l owns columns l+16j (j=0..7) so every split-half rotary pair
(c, c+48) sits in the same lane and the RMS is a sub-group reduction. Same arithmetic; only the
sum-of-squares association order differs. OMNI_H3_RMS_ROPE_SG=0 selects the old kernel."""
import sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/csrc/kitchen_rms_rope_sycl.cpp")
s = p.read_text()
if "launch_minimax_h3_rms_rope_sg" in s:
    print("[skip]"); sys.exit(0)

KERNEL = r'''
// Sub-group-per-row variant (2026-09-06). Lane l of a 16-lane sub-group owns
// columns l + 16*j (j = 0..7): pair (c, c+48) lives in one lane (j and j+3),
// columns 96..127 (j = 6, 7) are only normalized. No SLM, no barriers.
void launch_minimax_h3_rms_rope_sg(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    float epsilon) {
    const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
    const auto* f_ptr = reinterpret_cast<const bf16*>(freqs.data_ptr());
    const auto* qs_ptr = reinterpret_cast<const bf16*>(q_scale.data_ptr());
    const auto* ks_ptr = reinterpret_cast<const bf16*>(k_scale.data_ptr());
    auto* qo_ptr = reinterpret_cast<bf16*>(q.data_ptr());
    auto* ko_ptr = reinterpret_cast<bf16*>(k.data_ptr());
    const int64_t rows_per_operand = q.size(1) * H3_HEADS;
    const int64_t total_rows = 2 * rows_per_operand;
    constexpr int SG = 16;
    constexpr int ROWS_PER_WG = 16;
    constexpr int WG = SG * ROWS_PER_WG;
    const int64_t groups = (total_rows + ROWS_PER_WG - 1) / ROWS_PER_WG;

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(groups * WG)),
                sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_id()[0]);
                const int64_t global_row =
                    static_cast<int64_t>(item.get_group(0)) * ROWS_PER_WG +
                    static_cast<int64_t>(sg.get_group_id()[0]);
                if (global_row >= total_rows) return;  // uniform per sub-group

                const bool is_key = global_row >= rows_per_operand;
                const int64_t row = is_key
                    ? global_row - rows_per_operand
                    : global_row;
                const int64_t token = row / H3_HEADS;
                const int64_t head = row - token * H3_HEADS;
                const int64_t base =
                    token * H3_PACKED_SEQUENCE_STRIDE + head * H3_HEAD_DIM;
                const bf16* source = is_key ? k_ptr : q_ptr;
                bf16* destination = is_key ? ko_ptr : qo_ptr;
                const bf16* scale = is_key ? ks_ptr : qs_ptr;

                float v[8];
                float square_sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] = static_cast<float>(source[base + lane + 16 * j]);
                    square_sum += v[j] * v[j];
                }
                square_sum = sycl::reduce_over_group(sg, square_sum, sycl::plus<float>());
                const float inverse_rms =
                    sycl::rsqrt(square_sum / H3_HEAD_DIM + epsilon);

#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int col0 = lane + 16 * j;
                    const int col1 = col0 + static_cast<int>(H3_ROTARY_PAIRS);
                    const bf16 normalized0 = static_cast<bf16>(
                        v[j] * inverse_rms * static_cast<float>(scale[col0]));
                    const bf16 normalized1 = static_cast<bf16>(
                        v[j + 3] * inverse_rms * static_cast<float>(scale[col1]));
                    const float rotated0 = static_cast<float>(normalized0);
                    const float rotated1 = static_cast<float>(normalized1);
                    const int64_t freq_base =
                        token * H3_ROTARY_PAIRS * 4 + static_cast<int64_t>(col0) * 4;
                    const float f00 = static_cast<float>(f_ptr[freq_base]);
                    const float f01 = static_cast<float>(f_ptr[freq_base + 1]);
                    const float f10 = static_cast<float>(f_ptr[freq_base + 2]);
                    const float f11 = static_cast<float>(f_ptr[freq_base + 3]);
                    destination[base + col0] = static_cast<bf16>(
                        f00 * rotated0 + f01 * rotated1);
                    destination[base + col1] = static_cast<bf16>(
                        f10 * rotated0 + f11 * rotated1);
                }
#pragma unroll
                for (int j = 6; j < 8; ++j) {
                    const int col = lane + 16 * j;
                    destination[base + col] = static_cast<bf16>(
                        v[j] * inverse_rms * static_cast<float>(scale[col]));
                }
            });
    };
    utils::submit_kernel(cgf, q.device(), "kitchen_rms_rope_h3_subgroup_bmg");
}

inline bool use_h3_rms_rope_subgroup() {
    static const bool enabled = [] {
        const char* e = std::getenv("OMNI_H3_RMS_ROPE_SG");
        return !(e && e[0] == '0');
    }();
    return enabled;
}
'''
anchor = '        cgf, q.device(), "kitchen_rms_rope_h3_cached_input_bmg");\n}\n#endif\n'
assert s.count(anchor) == 1
s = s.replace(anchor, '        cgf, q.device(), "kitchen_rms_rope_h3_cached_input_bmg");\n}\n' + KERNEL + '#endif\n')
old_call = """        launch_minimax_h3_rms_rope(
            q, k, freqs_cis, q_scale, k_scale,
            static_cast<float>(epsilon));
        return {q_out, k_out};
"""
new_call = """        if (use_h3_rms_rope_subgroup()) {
            launch_minimax_h3_rms_rope_sg(
                q, k, freqs_cis, q_scale, k_scale,
                static_cast<float>(epsilon));
        } else {
            launch_minimax_h3_rms_rope(
                q, k, freqs_cis, q_scale, k_scale,
                static_cast<float>(epsilon));
        }
        return {q_out, k_out};
"""
assert s.count(old_call) == 1
s = s.replace(old_call, new_call)
if "#include <cstdlib>" not in s:
    s = s.replace("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n", 1)
p.write_text(s); print("[ok] sub-group RMS+RoPE kernel")
