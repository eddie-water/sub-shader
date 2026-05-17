"""projection_reconstruction_either_order_v9.png renderer.

Two-panel figure showing the symmetry of the dot product via projection:
panel 1 projects ``a`` onto ``b``, panel 2 projects ``b`` onto ``a``.
Both projections produce the same dot-product magnitude — the projection
"shadows" land on different axes but encode the same scalar.

Per LOCKED D-02, all Vectors here use 2-tuple inputs and are drawn on
regular 2D Axes via the polymorphic Vector Plottable.
"""
from __future__ import annotations

import os

import numpy as np

from dsplot import (
    Annotation,
    Dropline,
    Figure,
    StaticPanel,
    Vector,
    style,
)

from .foundation_constants import A, B, FOUND_LIM


def _projection(source: tuple[float, float],
                target: tuple[float, float]) -> tuple[float, float]:
    """Return the foot of the perpendicular from ``source`` onto the line
    through origin along ``target``. The foot is ``(target_hat · source) *
    target_hat`` — i.e. the projection of source onto target.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    scale = float(np.dot(tgt, src)) / float(np.dot(tgt, tgt))
    foot = scale * tgt
    return float(foot[0]), float(foot[1])


def _panel(
    *,
    source: tuple[float, float],
    target: tuple[float, float],
    source_color: str,
    target_color: str,
    source_label: str,
    target_label: str,
    title: str,
    subtitle: str,
) -> StaticPanel:
    """Build a 2D projection panel.

    Draws: target arrow, projection arrow (target's direction, neutral),
    perpendicular dropline from the projection foot to the source tip,
    and the source arrow on top.
    """
    panel = StaticPanel(
        title=title,
        subtitle=subtitle,
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )

    foot = _projection(source=source, target=target)

    # Target arrow (the "reference direction" — drawn bold in its own role color).
    panel.add(
        Vector(target,
               color=target_color,
               label=target_label,
               alpha=1.0,
               zorder=2,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )

    # Projection arrow (neutral white — the "shadow" of source along target).
    panel.add(
        Vector(foot,
               color=style.NEUTRAL_COLOR,
               linewidth=style.DEFAULT_VECTOR_LINEWIDTH + 0.4,
               alpha=0.9,
               zorder=3)
    )

    # Dropline from projection foot to source tip (closes the parallelogram).
    panel.add(Dropline(start=foot, end=source))

    # Source arrow on top.
    panel.add(
        Vector(source,
               color=source_color,
               label=source_label,
               alpha=1.0,
               zorder=4,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )

    return panel


def render(
    output_dir: str,
    output_filename: str = "projection_reconstruction_either_order_v9.png",
) -> str:
    """Render the 2-panel projection-symmetry figure."""
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 1.4
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.7
    fig = Figure(
        n_rows=1, n_cols=2,
        figsize=(panel_w * 2, panel_h),
        suptitle="Vector Projection Symmetry",
    )

    fig.add_panel(
        _panel(
            source=A, target=B,
            source_color=style.PRIMARY_COLOR,
            target_color=style.SECONDARY_COLOR,
            source_label="a", target_label="b",
            title="Project a onto b",
            subtitle=r"shadow of $\vec{a}$ along $\hat{b}$",
        ),
        row=0, col=0,
    )
    fig.add_panel(
        _panel(
            source=B, target=A,
            source_color=style.SECONDARY_COLOR,
            target_color=style.PRIMARY_COLOR,
            source_label="b", target_label="a",
            title="Project b onto a",
            subtitle=r"shadow of $\vec{b}$ along $\hat{a}$",
        ),
        row=0, col=1,
    )

    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, pad_inches=0.15)
    fig.close()
    return os.path.abspath(output_path)
