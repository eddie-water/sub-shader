"""Tile the latest published figure PNGs into one labeled composite for review.

Run from repo root:  python research/dsplot/montage_all_figures.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE = Path("assets/images/dsp/figures/by_figure")
OUT = BASE / "_review" / "all_figures_composite_v1.png"

ITEMS = [
    ("fig 1  — main",     "fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_lower_longer2_v45.png"),
    ("fig 1  — hero",     "fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v1.png"),
    ("fig 1  — antihero", "fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_antihero_v1.png"),
    ("fig 2.4.1 — xy recombine",  "fig_2_4_1_xy_recombine/fig_2_4_1_xy_recombine_composite_v20.png"),
    ("fig 2.4.2 — a onto b",      "fig_2_4_2_a_onto_b/fig_2_4_2_a_onto_b_v8.png"),
    ("fig 2.4.3 — dot product 3d","fig_2_4_3_dot_product_3d/fig_2_4_3_dot_product_3d_inline_titles_v9.png"),
    ("fig 2.5 — sign accumulation","fig_2_5_sign_accumulation/fig_2_5_sign_accumulation_composite_v49_border_flush.png"),
    ("fig 2.6 — sine basis",       "fig_2_6_sine_basis/fig_2_6_sine_basis_2hz_10hz_v21.png"),
]

W = 1500          # common content width
LABEL_H = 46
PAD = 24
BG = (20, 20, 20)
FG = (238, 238, 238)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except Exception:
    font = ImageFont.load_default()

scaled = []
for label, rel in ITEMS:
    im = Image.open(BASE / rel).convert("RGB")
    h = int(im.height * W / im.width)
    scaled.append((label, im.resize((W, h), Image.LANCZOS)))

total_h = sum(LABEL_H + im.height + PAD for _, im in scaled) + PAD
canvas = Image.new("RGB", (W + 2 * PAD, total_h), BG)
draw = ImageDraw.Draw(canvas)

y = PAD
for label, im in scaled:
    draw.text((PAD, y + 8), label, fill=FG, font=font)
    y += LABEL_H
    canvas.paste(im, (PAD, y))
    y += im.height + PAD

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)
print("wrote", OUT, canvas.size)
