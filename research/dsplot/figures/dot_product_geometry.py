"""dot_product_geometry.png renderer — 4 canonical angle cases.

Each panel shows two vectors a, b plus a sign-of-result annotation beneath
the panel, so the reader sees angle → sign at a glance. Originals in
``research/dsp_figures.py::_plot_dot_product_geometry`` used the YELLOW /
CYAN legacy palette; this dsplot port uses the canonical PRIMARY/SECONDARY
identity colors (orange/purple) for visual consistency with the other §2.4
foundation figures.
"""
from __future__ import annotations

import os

from .. import Annotation, Figure, StaticPanel, Vector, style


_PANELS = [
    {
        "title": "Parallel, same direction",
        "result": "a · b  >  0",
        "a": (0.90, 0.45),
        "b": (0.50, 0.25),
        "b_origin": (0.0, -0.05),
    },
    {
        "title": "Parallel, opposite direction",
        "result": "a · b  <  0",
        "a": (0.90, 0.45),
        "b": (-0.55, -0.275),
        "b_origin": (0.0, 0.0),
    },
    {
        "title": "Perpendicular",
        "result": "a · b  =  0",
        "a": (0.90, 0.45),
        "b": (-0.45, 0.90),
        "b_origin": (0.0, 0.0),
    },
    {
        "title": "Oblique",
        "result": "a · b  >  0  (partial)",
        "a": (0.95, 0.20),
        "b": (0.55, 0.75),
        "b_origin": (0.0, 0.0),
    },
]


def render(output_dir: str, output_filename: str = "dot_product_geometry.png") -> str:
    """Render the 4-panel dot-product angle-cases figure.

    Each panel: ``a`` in PRIMARY, ``b`` in SECONDARY, plus an axes-relative
    annotation below the panel with the sign-of-result expression.
    """
    n_cols = len(_PANELS)
    # Panel widths sized to fit the longest title ("Parallel, opposite
    # direction") at DEFAULT_TITLE_FONT_SIZE without truncation.
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 1.4
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.1
    fig = Figure(
        n_rows=1, n_cols=n_cols,
        figsize=(panel_w * n_cols, panel_h),
    )

    for col, cfg in enumerate(_PANELS):
        panel = StaticPanel(
            title=cfg["title"],
            lim=1.15,
            axis_style="line",
            show_border=True,
        )
        panel.add(
            Vector(cfg["a"], color=style.PRIMARY_COLOR, label="a")
        )
        panel.add(
            Vector(cfg["b"], color=style.SECONDARY_COLOR, label="b",
                   origin=cfg["b_origin"])
        )
        panel.add(
            Annotation(
                cfg["result"],
                xy=(0.5, -0.08),
                transform="axes",
                ha="center", va="top",
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
            )
        )
        fig.add_panel(panel, row=0, col=col)

    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
