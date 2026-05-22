"""Temporary comparison: render dot-product symmetry for 3 candidate (a, b) pairs.

Each row in the output is one (a, b) candidate showing:
  - Left panel:  a onto b   (a's shadow on b's direction)
  - Right panel: b onto a   (b's shadow on a's direction)

Both labels on each row carry the same scalar a·b — that's the symmetry. The
visual the user is judging: how clearly do the two shadows differ in length?
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Reuse the project's plotting + style helpers
sys.path.insert(0, os.path.dirname(__file__))
from utilities import style
from utilities.plotting import setup_vector_axes, plot_vector


CANDIDATES = [
    {"label": "A",
     "a": (1.0, 0.5),
     "b": (1.0, 1.0),
     "note": "a·b = 1.50  |  shadow on b: 75% within  |  shadow on a: 120% past tip"},
    {"label": "B",
     "a": (1.0, 0.5),
     "b": (0.4, 1.2),
     "note": "a·b = 1.00  |  shadow on b: 63% within  |  shadow on a:  80% within"},
    {"label": "C",
     "a": (1.0, 0.5),
     "b": (0.3, 0.6),
     "note": "a·b = 0.60  |  shadow on b: 133% past tip  |  shadow on a: 48% within"},
]

_NEUTRAL = style.VECTOR_NEUTRAL_COLOR
_SPINE   = style.VECTOR_AXIS_COLOR
_A_COLOR = style.PALETTE_PRIMARY
_B_COLOR = style.PALETTE_SECONDARY


def _draw_proj(ax, *, source, target, source_color, target_color,
               source_label, target_label):
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    scale = float(np.dot(tgt, src)) / float(np.dot(tgt, tgt))
    foot = scale * tgt

    plot_vector(ax, target, color=target_color, label=target_label,
                alpha=1.0, zorder=2, linewidth=style.VECTOR_BOLD_LINEWIDTH)
    plot_vector(ax, source, color=source_color, label=source_label,
                alpha=1.0, zorder=3, linewidth=style.VECTOR_BOLD_LINEWIDTH)
    plot_vector(ax, tuple(foot), origin=(0.0, -0.04),
                color=_NEUTRAL,
                linewidth=style.VECTOR_LINEWIDTH + 0.4,
                alpha=0.9, zorder=4)
    ax.plot([foot[0], src[0]], [foot[1], src[1]],
            color=style.VECTOR_DROPLINE_COLOR,
            alpha=style.VECTOR_DROPLINE_ALPHA,
            linewidth=1.2, linestyle="--", zorder=1)


def main():
    n_rows = len(CANDIDATES)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(style.VECTOR_FIGSIZE_PER_PANEL * 1.25 * 2,
                 style.VECTOR_FIGSIZE_HEIGHT * n_rows),
        dpi=style.DEFAULT_DPI,
    )
    fig.patch.set_facecolor(style.BG_COLOR)

    axis_kwargs = dict(
        x_color=_SPINE, y_color=_SPINE,
        axis_alpha=0.55, axis_linewidth=1.2,
    )

    for row, cand in enumerate(CANDIDATES):
        a, b = cand["a"], cand["b"]
        # Bump lim per row so the longer-b candidates aren't cropped.
        max_extent = max(abs(c) for v in (a, b) for c in v)
        lim = max(1.25, max_extent * 1.15)

        ax_l, ax_r = axes[row]
        for ax in (ax_l, ax_r):
            ax.set_facecolor(style.BG_COLOR)

        setup_vector_axes(ax_l, lim=lim,
                          panel_title=f"Option {cand['label']}: a onto b",
                          **axis_kwargs)
        _draw_proj(ax_l, source=a, target=b,
                   source_color=_A_COLOR, target_color=_B_COLOR,
                   source_label="a", target_label="b")

        setup_vector_axes(ax_r, lim=lim,
                          panel_title=f"Option {cand['label']}: b onto a",
                          **axis_kwargs)
        _draw_proj(ax_r, source=b, target=a,
                   source_color=_B_COLOR, target_color=_A_COLOR,
                   source_label="b", target_label="a")

        # Row note pinned in the left panel's lower margin.
        ax_l.text(
            -0.02, -0.18,
            f"a={a}, b={b}   {cand['note']}",
            transform=ax_l.transAxes,
            color=_SPINE, fontsize=9,
            ha="left", va="top",
        )

    fig.subplots_adjust(hspace=0.55, wspace=0.20,
                        left=0.05, right=0.97,
                        top=0.96, bottom=0.06)
    out_path = "assets/images/diagnostics/dot_product_symmetry_candidates_temp.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, facecolor=style.BG_COLOR)
    print(f"Saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
