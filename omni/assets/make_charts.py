#!/usr/bin/env python3
"""Regenerate the README charts (plain SVG, no dependencies). Numbers are the ones in
omni/docs/BENCHMARKS.md; a range is drawn at its midpoint and labelled with the range.

    python3 omni/assets/make_charts.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT, MUTED, BASE, OURS, GAIN = "#768390", "#768390", "#8b949e", "#2f81f7", "#3fb950"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def chart(name, title, legend, rows):
    """rows: (label, sublabel, base_mid, base_text, ours_mid, ours_text)"""
    left, bar_w, row_h, top = 250, 420, 62, 86
    width, height = left + bar_w + 190, top + row_h * len(rows) + 14
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="{FONT}" role="img" aria-label="{title}">',
           f'<text x="0" y="24" font-size="18" font-weight="600" fill="{TEXT}">{title}</text>',
           f'<text x="0" y="46" font-size="13" fill="{MUTED}">seconds per job, lower is better · one Arc Pro B70 · each pair is scaled to its own baseline</text>']
    x = 0
    for colour, text in ((BASE, legend[0]), (OURS, legend[1])):
        out.append(f'<rect x="{x}" y="60" width="12" height="12" rx="2" fill="{colour}"/>')
        out.append(f'<text x="{x + 18}" y="71" font-size="13" fill="{TEXT}">{text}</text>')
        x += 30 + 7.2 * len(text)
    for i, (label, sub, b, bt, o, ot) in enumerate(rows):
        y = top + i * row_h
        out.append(f'<text x="0" y="{y + 20}" font-size="14" font-weight="600" fill="{TEXT}">{label}</text>')
        out.append(f'<text x="0" y="{y + 38}" font-size="12" fill="{MUTED}">{sub}</text>')
        ow = bar_w * o / b
        out.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w}" height="17" rx="3" fill="{BASE}"/>')
        out.append(f'<text x="{left + bar_w + 8}" y="{y + 19}" font-size="13" fill="{TEXT}">{bt} s</text>')
        out.append(f'<rect x="{left}" y="{y + 27}" width="{ow:.1f}" height="17" rx="3" fill="{OURS}"/>')
        out.append(f'<text x="{left + ow + 8:.1f}" y="{y + 40}" font-size="13" font-weight="600" fill="{OURS}">{ot} s</text>')
        pct = round((1 - o / b) * 100)
        out.append(f'<text x="{width}" y="{y + 31}" font-size="16" font-weight="700" fill="{GAIN}" text-anchor="end">−{pct}%</text>')
    out.append("</svg>")
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


chart("results-h3.svg", "MiniMax H3 video: Intel's image vs. this repo",
      ("intel/llm-scaler-omni:0.2.0-b1", "comfyui-b70-xpu"), [
    ("480×864, new prompt", "a real job: text encoder + model swap", 87.65, "87.5–87.8", 71.15, "69.9–72.4"),
    ("480×864, new seed", "kernels only", 59.3, "59.2–59.4", 55.85, "55.7–56.0"),
    ("704×1280, new prompt", "a real job", 212.1, "210.9–213.3", 185.15, "185.1–185.2"),
    ("704×1280, new seed", "kernels only", 181.6, "181.5–181.7", 169.65, "169.5–169.8"),
])
chart("results-int8.svg", "Other models on this stack: float weights vs. official INT8 ConvRot weights",
      ("float weights (bf16 / fp16)", "INT8 ConvRot weights"), [
    ("Krea-2 Turbo, new prompt", "8 steps, 1024²", 15.0, "14–16", 7.6, "7.6"),
    ("Krea-2 Turbo, new seed", "8 steps, 1024²", 9.9, "9.9", 7.1, "6.9–7.3"),
    ("Qwen-Image 2.1, 1280×720", "25 steps", 15.6, "15.4–15.8", 10.45, "10.3–10.6"),
    ("Qwen-Image 2.1, 2048²", "25 steps", 89.75, "89.7–89.8", 67.5, "67–68"),
    ("Music 3, same lyrics", "2 minutes, 30 steps", 79.0, "79.0", 53.2, "53.2"),
    ("Music 3, new song", "2 minutes, 30 steps", 111.5, "110–113", 88.5, "87–90"),
])
print("wrote results-h3.svg, results-int8.svg")
