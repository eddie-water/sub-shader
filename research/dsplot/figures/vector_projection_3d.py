"""vector_projection_3d_v2_combo5_palette.png renderer.

Per LOCKED D-02 + D-04: this figure uses the polymorphic Vector Plottable
with 3-tuples (no separate Vector3D class) on an Axes3D cell obtained via
``Figure.add_panel(panel, ..., projection="3d")``. The 3D-axis chrome
(spines through origin + axis labels) lives in StaticPanel3D, not in the
figure module — figures are composers of Plottables and Panels.

Three neutral spines + a neutral vector ``a`` + two dashed reconstruction
paths (x→y→z and z→y→x) showcase that the component arrows can be added in
any order and still reach the original vector tip.
"""
from __future__ import annotations

import os

from dsplot import Annotation, Figure, Vector, style
from dsplot.panels import StaticPanel3D

from .foundation_constants import A, A_Z, FOUND_LIM


def render(
    output_dir: str,
    output_filename: str = "vector_projection_3d_v2_combo5_palette.png",
) -> str:
    """Render the 3D foundation figure (single-panel)."""
    fig_size = style.DEFAULT_PANEL_SIZE_INCHES * 1.6
    fig = Figure(
        n_rows=1, n_cols=1,
        figsize=(fig_size, fig_size * 0.9),
    )

    a = (float(A[0]), float(A[1]), float(A_Z))

    panel = StaticPanel3D(
        lim_3d=FOUND_LIM,
        view_init=(38.0, -55.0),
    )

    seg_kwargs = dict(
        linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
        alpha=0.95,
        linestyle="--",
        show_tip=False,
        zorder=3,
    )

    # Path 1: x → y → z. Each segment colored by its dimension.
    panel.add(Vector((a[0], 0.0, 0.0),
                     origin=(0.0, 0.0, 0.0),
                     color=style.PRIMARY_COLOR, **seg_kwargs))
    panel.add(Vector((0.0, a[1], 0.0),
                     origin=(a[0], 0.0, 0.0),
                     color=style.SECONDARY_COLOR, **seg_kwargs))
    panel.add(Vector((0.0, 0.0, a[2]),
                     origin=(a[0], a[1], 0.0),
                     color=style.TERTIARY_COLOR, **seg_kwargs))

    # Path 2: z → y → x — same dimension-color rule applies.
    panel.add(Vector((0.0, 0.0, a[2]),
                     origin=(0.0, 0.0, 0.0),
                     color=style.TERTIARY_COLOR, **seg_kwargs))
    panel.add(Vector((0.0, a[1], 0.0),
                     origin=(0.0, 0.0, a[2]),
                     color=style.SECONDARY_COLOR, **seg_kwargs))
    panel.add(Vector((a[0], 0.0, 0.0),
                     origin=(0.0, a[1], a[2]),
                     color=style.PRIMARY_COLOR, **seg_kwargs))

    # Vector a — bold neutral, with scatter tip + 3D label.
    panel.add(Vector(
        a,
        color=style.NEUTRAL_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH + 0.6,
        show_tip=True,
        label="a",
        zorder=5,
    ))

    # Legend — placed at axes-relative coords via the Annotation Plottable
    # so the 3D scene controls placement, not the data extent.
    legend_kwargs = dict(
        color=style.NEUTRAL_COLOR,
        fontsize=style.DEFAULT_LABEL_FONT_SIZE - 4,
        transform="axes",
        ha="left", va="top",
    )
    panel.add(Annotation("Path 1:  x → y → z",
                          xy=(0.02, 0.97), **legend_kwargs))
    panel.add(Annotation("Path 2:  z → y → x",
                          xy=(0.02, 0.92), **legend_kwargs))

    fig.add_panel(panel, row=0, col=0, projection="3d")
    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
