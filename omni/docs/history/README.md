# history/

The step-by-step patch scripts that produced `../../patches/vs-intel-0.2.0-b1/` and the two patches in `../../patches/`. They are here for the reasoning: each
file starts with what it changes, why, and what was measured. **You do not need them to build** —
the source directories and patches in `omni/` are the final state.

- `patches/port/` — moving Intel's sources from torch 2.11 / Python 3.12 to torch 2.13 / 3.13,
  and grafting the comfy-kitchen XPU backend onto newer comfy-kitchen releases.
- `patches/kernel/` — native kernel changes, in order (`phase1` … `phase9`). Phases 3 and 5–8 are
  attention-kernel experiments that ended flat or negative; they are compiled out by default.
- `patches/kitchen/` — the fork delta and our changes to the XPU backend.
- `omnixpu/patches/` — adapter changes and the hooks that load `ComfyUI-OmniXPU/fixes/*.py`.

Comments in these scripts are partly in Chinese.
