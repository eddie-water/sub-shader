"""Render every figure FRESH from current code, then tile into one dark-bg
montage for review. Restores the montage workflow (all fig-1 variants included)
but sourced from live renders, not stale published PNGs.

Run from repo root:  python research/dsplot/render_and_montage.py [round_tag]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

# Both worldviews on path: repo root so `from research.X import ...` resolves
# (fig-1 hero/antihero), and research/ so `import dsplot` resolves.
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("research"))

from PIL import Image

import glob
import re

_REVIEW_ROOT = "assets/images/dsp/figures/by_figure/_review"


def _next_round() -> str:
    """Next unused `round<letter(s)>` so a no-arg run never clobbers a prior
    round. Scans existing `_review/_round*` folders, takes the highest, and
    advances one (…, roundZ → roundAA → roundAB …). An explicit CLI arg always
    wins (use it for descriptive tags like `arrowhead`).
    """
    existing = []
    for p in glob.glob(os.path.join(_REVIEW_ROOT, "_round*")):
        m = re.fullmatch(r"round([A-Z]+)", os.path.basename(p)[1:])
        if m:
            existing.append(m.group(1))

    def _to_num(s: str) -> int:  # base-26 bijective: A=1 … Z=26, AA=27 …
        n = 0
        for ch in s:
            n = n * 26 + (ord(ch) - ord("A") + 1)
        return n

    def _to_alpha(n: int) -> str:
        out = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            out = chr(ord("A") + r) + out
        return out

    nxt = max((_to_num(s) for s in existing), default=0) + 1
    return f"round{_to_alpha(nxt)}"


# Explicit tag wins; otherwise auto-advance to the next unused round letter.
ROUND = sys.argv[1] if len(sys.argv) > 1 else _next_round()
OUT_DIR = f"{_REVIEW_ROOT}/_{ROUND}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"ROUND = {ROUND}  ->  {OUT_DIR}")


# Published Figure 1 (the user's chosen reference render) — used as-is in the
# montage rather than re-rendered, so the montage shows the canonical fig-1.
FIG1_PUBLISHED = (
    "assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/"
    "fig_1_fourier_vs_wavelet_lower_longer2_v47_sharedtile.png"
)


def _render_all():
    from dsplot.figures import (
        gen_figure_241_xy_recombine_independent as f241,
        gen_figure_242_a_onto_b as f242,
        gen_figure_243_dot_product_3d as f243,
        gen_figure_2_5_sign_accumulation as f25,
        gen_figure_2_6_sine_basis as f26,
    )
    items = []
    items.append(("fig 1 — fourier vs wavelet", os.path.abspath(FIG1_PUBLISHED)))
    items.append(("fig 2.4.1 — xy recombine",   f241.render(OUT_DIR, f"f241_{ROUND}.png")))
    items.append(("fig 2.4.2 — a onto b",       f242.render(OUT_DIR, f"f242_{ROUND}.png")))
    items.append(("fig 2.4.3 — dot product 3d", f243.render(OUT_DIR, f"f243_{ROUND}.png")))
    items.append(("fig 2.5 — sign accumulation", f25.render(OUT_DIR, f"f25_{ROUND}.png")))
    items.append(("fig 2.6 — sine basis",        f26.render(OUT_DIR, f"f26_{ROUND}.png")))
    return items


def _montage(items):
    # Tile at ONE uniform scale so grid squares stay equal across figures (the
    # whole point of the shared-square model). The WIDEST figure is scaled to
    # MAX_W; every other figure gets the SAME scale factor — so a narrow figure
    # (fewer columns) renders narrower, NOT stretched to match. Figures are
    # centered on a canvas sized to the widest. No per-figure width normalization.
    MAX_W = 2200
    PAD = 24
    # White PAGE background (the canvas the figures sit on) — the figures keep
    # their own dark panel backgrounds. No inter-figure caption strips — each
    # figure already carries its own title band, so the montage just stacks them.
    BG = (255, 255, 255)

    raw = [(label, Image.open(path).convert("RGB")) for label, path in items]
    widest = max(im.width for _, im in raw)
    scale = MAX_W / widest
    scaled = [
        (label, im.resize(
            (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
            Image.LANCZOS))
        for label, im in raw
    ]

    canvas_w = MAX_W + 2 * PAD
    total_h = sum(im.height + PAD for _, im in scaled) + PAD
    canvas = Image.new("RGB", (canvas_w, total_h), BG)
    y = PAD
    for label, im in scaled:
        x = (canvas_w - im.width) // 2  # center narrower figures
        canvas.paste(im, (x, y))
        y += im.height + PAD

    out = os.path.join(OUT_DIR, f"_montage_{ROUND}.png")
    canvas.save(out)
    return out, canvas.size


if __name__ == "__main__":
    items = _render_all()
    out, size = _montage(items)
    for label, path in items:
        print(f"  {label}: {path}")
    print(f"\nMONTAGE -> {out}  {size}")
