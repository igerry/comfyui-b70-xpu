#!/usr/bin/env python3
"""Phase-4a: let the AOT ocloc device be chosen independently of the arch macro / wheel tag.

`OMNI_XPU_DEVICE=bmg` drives three things at once in setup.py: the OMNI_XPU_ARCH_BMG macro,
the wheel's local-version tag (regex `+torchNNN.(bmg|ptlh)`), and the ocloc `-device` flag.
The plain `bmg` alias resolves to **bmg-g21** (Arc B580, IP 20.1.0). An Arc Pro B70 is
**bmg-g31** (IP 20.2.0) — device id 0xE223, and the runtime reports version 20.2.0.
So every AOT kernel in the wheel is compiled against the wrong die's machine model.

This adds OMNI_XPU_OCLOC_DEVICE, which overrides only the `-device` string passed to ocloc
(default: unchanged, i.e. BUILD_XPU_TARGET). Macro, overlay selection and wheel tag stay put,
so nothing downstream has to learn a new target name:

    OMNI_XPU_OCLOC_DEVICE=bmg-g31 CUTLASS_SYCL_ROOT=... OMNI_XPU_DEVICE=bmg pip wheel .

Idempotent; backup setup.py.pre-phase4a."""
import sys
from pathlib import Path

SP = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/omnibuild/src/setup.py")
s = SP.read_text()
if "OMNI_XPU_OCLOC_DEVICE" in s:
    print("[skip] already patched"); sys.exit(0)
bak = SP.with_name(SP.name + ".pre-phase4a")
if not bak.exists():
    bak.write_text(s)

# 1. Define the override next to the arch macro lookup.
anchor = "XPU_ARCH_MACRO = XPU_ARCH_MACROS[BUILD_XPU_TARGET]\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
# ocloc target for AOT codegen. BUILD_XPU_TARGET names the *family* (and picks the arch macro,
# the include overlay and the wheel tag); the alias "bmg" resolves to bmg-g21 (B580, IP 20.1.0).
# Set OMNI_XPU_OCLOC_DEVICE to compile for a specific die, e.g. bmg-g31 for Arc Pro B70 (IP 20.2.0).
# `ocloc ids <name>` prints the IP version; `ocloc compile -device 0x<pciid>` prints the match.
AOT_OCLOC_DEVICE = os.environ.get("OMNI_XPU_OCLOC_DEVICE", "").strip() or BUILD_XPU_TARGET
''')

# 2. Every place that spells the ocloc -device flag.
subs = [
    ('        f"-device {xpu_target}",\n',
     '        f"-device {AOT_OCLOC_DEVICE}",\n'),
    ('                    "-Xs", f"-device {BUILD_XPU_TARGET} -options -doubleGRF",\n',
     '                    "-Xs", f"-device {AOT_OCLOC_DEVICE} -options -doubleGRF",\n'),
    ('                    f"-device {BUILD_XPU_TARGET}",\n',
     '                    f"-device {AOT_OCLOC_DEVICE}",\n'),
    ('                    "-Xsycl-target-backend", f"-device {BUILD_XPU_TARGET}",\n',
     '                    "-Xsycl-target-backend", f"-device {AOT_OCLOC_DEVICE}",\n'),
]
total = 0
for old, new in subs:
    n = s.count(old)
    assert n >= 1, f"anchor not found: {old!r}"
    s = s.replace(old, new); total += n
assert "-device {BUILD_XPU_TARGET}" not in s and "-device {xpu_target}" not in s, "missed a -device site"
SP.write_text(s)
print(f"[ok] {SP} ({total} -device sites)")
