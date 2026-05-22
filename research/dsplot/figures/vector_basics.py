"""vector_basics.png renderer — 5 arrows radiating from origin.

Single-panel figure for §2.4.1 beat 1 ("a vector is an arrow with magnitude
and direction"). No math, no labels — just visual proof.

Per D-01 the figures subdir IS allowed to import non-library helpers, but
this module needs none — the figure is pure dsplot.
"""
from __future__ import annotations

import os

from .. import Figure, StaticPanel, Vector, style


# Hand-picked radiating hues for visual variety — not part of the canonical
# 3-role palette. dsplot.style intentionally does not carry these (the legacy
# names were ``VECTOR_PROJ_COLOR`` / ``GRID_WAVEFORM_COLOR`` in
# ``research/utilities/style.py``; figure-local here keeps subshader-specific
# names out of dsplot.style).
_RADIATING_ORANGE = "#ff8c66"
_RADIATING_YELLOW = "#fcfeaa"


def render(output_dir: str, output_filename: str = "vector_basics.png") -> str:
    """Render the 5-arrow basics figure into ``output_dir``.

    Returns the absolute output path.
    """
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 1.3
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.05
    fig = Figure(n_rows=1, n_cols=1, figsize=(panel_w, panel_h))

    panel = StaticPanel(
        lim=1.15,
        axis_style="line",
        show_border=True,
    )

    samples = [
        ((0.95, 0.55),  style.PRIMARY_COLOR),
        ((-0.70, 0.80), style.SECONDARY_COLOR),
        ((-0.85, -0.30), _RADIATING_ORANGE),
        ((0.30, -0.90), _RADIATING_YELLOW),
        ((0.50, 0.18),  style.NEUTRAL_COLOR),
    ]
    for vec, color in samples:
        panel.add(Vector(vec, color=color, alpha=0.95))

    fig.add_panel(panel, row=0, col=0)
    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
