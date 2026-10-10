#!/usr/bin/env python3
"""Phase-1c: make the fused ConvRot+quantize ESIMD kernel less latency-bound on BMG.
Measured 2026-09-05 (B70): 205 GB/s vs ~500 GB/s for the two-kernel path. The kernel does one
row per work-item and walks the 256-wide groups serially with one 512-byte load in flight.
Changes (int8_convrot_quant_esimd.cpp):
  1. per-kernel grf_size<256> property (OMNI_CONVROT_QUANT_GRF, default 256 on BMG) so the
     two in-flight simd<float,256> chains do not spill;
  2. 2-way unroll of both group loops (independent load->Hadamard chains overlap).
"""
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/omni_xpu_kernel/csrc/int8_convrot_quant_esimd.cpp")
s = p.read_text()
if "OMNI_CONVROT_QUANT_GRF" in s:
    print("[skip] already applied")
    sys.exit(0)
bak = p.with_name(p.name + ".pre-phase1c")
if not bak.exists():
    bak.write_text(s)


def rep(old, new, tag):
    global s
    assert s.count(old) == 1, f"{tag}: count={s.count(old)}"
    s = s.replace(old, new)
    print(f"[ok] {tag}")


rep("#ifndef OMNI_CONVROT_QUANT_WG_SIZE\n",
    "#ifndef OMNI_CONVROT_QUANT_GRF\n"
    "#if defined(OMNI_XPU_ARCH_BMG)\n"
    "#define OMNI_CONVROT_QUANT_GRF 256\n"
    "#else\n"
    "#define OMNI_CONVROT_QUANT_GRF 128\n"
    "#endif\n"
    "#endif\n"
    "#ifndef OMNI_CONVROT_QUANT_WG_SIZE\n",
    "grf macro")

rep("    auto cgf = [&](sycl::handler& handle) {\n"
    "        handle.parallel_for(\n"
    "            sycl::nd_range<1>(\n"
    "                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),\n"
    "            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {\n"
    "                const int64_t row = item.get_global_id(0);\n"
    "                if (row >= rows) return;\n"
    "                const int64_t row_start = row * groups * GroupSize;\n"
    "\n"
    "                float row_max = 0.0f;\n"
    "                for (int64_t group = 0; group < groups; ++group) {\n"
    "                    simd<float, GroupSize> values =\n"
    "                        load_rotated<InputT, GroupSize>(\n"
    "                            input + row_start + group * GroupSize);\n"
    "                    simd<float, GroupSize> magnitudes =\n"
    "                        sycl::ext::intel::esimd::abs<float, GroupSize>(values);\n"
    "                    const float group_max = hmax<float>(magnitudes);\n"
    "                    row_max = group_max > row_max ? group_max : row_max;\n"
    "                }\n",
    "    auto cgf = [&](sycl::handler& handle) {\n"
    "        sycl::ext::oneapi::experimental::properties props{\n"
    "            sycl::ext::intel::experimental::grf_size<OMNI_CONVROT_QUANT_GRF>};\n"
    "        handle.parallel_for(\n"
    "            sycl::nd_range<1>(\n"
    "                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),\n"
    "            props,\n"
    "            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {\n"
    "                const int64_t row = item.get_global_id(0);\n"
    "                if (row >= rows) return;\n"
    "                const int64_t row_start = row * groups * GroupSize;\n"
    "\n"
    "                // Two independent load->Hadamard chains per iteration so the\n"
    "                // second 512-byte load is issued before the first chain's math.\n"
    "                float row_max = 0.0f;\n"
    "                int64_t group = 0;\n"
    "                for (; group + 1 < groups; group += 2) {\n"
    "                    simd<float, GroupSize> values0 =\n"
    "                        load_rotated<InputT, GroupSize>(\n"
    "                            input + row_start + group * GroupSize);\n"
    "                    simd<float, GroupSize> values1 =\n"
    "                        load_rotated<InputT, GroupSize>(\n"
    "                            input + row_start + (group + 1) * GroupSize);\n"
    "                    simd<float, GroupSize> magnitudes =\n"
    "                        sycl::ext::intel::esimd::max<float, GroupSize>(\n"
    "                            sycl::ext::intel::esimd::abs<float, GroupSize>(values0),\n"
    "                            sycl::ext::intel::esimd::abs<float, GroupSize>(values1));\n"
    "                    const float group_max = hmax<float>(magnitudes);\n"
    "                    row_max = group_max > row_max ? group_max : row_max;\n"
    "                }\n"
    "                for (; group < groups; ++group) {\n"
    "                    simd<float, GroupSize> values =\n"
    "                        load_rotated<InputT, GroupSize>(\n"
    "                            input + row_start + group * GroupSize);\n"
    "                    simd<float, GroupSize> magnitudes =\n"
    "                        sycl::ext::intel::esimd::abs<float, GroupSize>(values);\n"
    "                    const float group_max = hmax<float>(magnitudes);\n"
    "                    row_max = group_max > row_max ? group_max : row_max;\n"
    "                }\n",
    "pass-1 unroll")

rep("                simd<float, GroupSize> clamp_lo(-128.0f);\n"
    "                simd<float, GroupSize> clamp_hi(127.0f);\n"
    "                for (int64_t group = 0; group < groups; ++group) {\n"
    "                    const int64_t offset = row_start + group * GroupSize;\n"
    "                    simd<float, GroupSize> values =\n"
    "                        load_rotated<InputT, GroupSize>(input + offset);\n"
    "                    simd<float, GroupSize> quantized =\n"
    "                        sycl::ext::intel::esimd::rnde<float, GroupSize>(\n"
    "                            values * inv_scale);\n"
    "                    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(\n"
    "                        sycl::ext::intel::esimd::min<float, GroupSize>(\n"
    "                            quantized, clamp_hi),\n"
    "                        clamp_lo);\n"
    "                    simd<int8_t, GroupSize> packed = quantized;\n"
    "                    block_store<int8_t, GroupSize>(output + offset, packed);\n"
    "                }\n",
    "                simd<float, GroupSize> clamp_lo(-128.0f);\n"
    "                simd<float, GroupSize> clamp_hi(127.0f);\n"
    "                auto quantize_store = [&](int64_t g) {\n"
    "                    const int64_t offset = row_start + g * GroupSize;\n"
    "                    simd<float, GroupSize> values =\n"
    "                        load_rotated<InputT, GroupSize>(input + offset);\n"
    "                    simd<float, GroupSize> quantized =\n"
    "                        sycl::ext::intel::esimd::rnde<float, GroupSize>(\n"
    "                            values * inv_scale);\n"
    "                    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(\n"
    "                        sycl::ext::intel::esimd::min<float, GroupSize>(\n"
    "                            quantized, clamp_hi),\n"
    "                        clamp_lo);\n"
    "                    simd<int8_t, GroupSize> packed = quantized;\n"
    "                    block_store<int8_t, GroupSize>(output + offset, packed);\n"
    "                };\n"
    "                group = 0;\n"
    "                for (; group + 1 < groups; group += 2) {\n"
    "                    quantize_store(group);\n"
    "                    quantize_store(group + 1);\n"
    "                }\n"
    "                for (; group < groups; ++group) {\n"
    "                    quantize_store(group);\n"
    "                }\n",
    "pass-2 unroll")

# make sure the property header is reachable
if "grf_size_properties.hpp" not in s:
    rep("#include <sycl/sycl.hpp>\n" if "#include <sycl/sycl.hpp>\n" in s else s.split("\n")[0] + "\n",
        ("#include <sycl/sycl.hpp>\n" if "#include <sycl/sycl.hpp>\n" in s else s.split("\n")[0] + "\n")
        + "#include <sycl/ext/intel/experimental/grf_size_properties.hpp>\n",
        "include")
p.write_text(s)
print("done")
