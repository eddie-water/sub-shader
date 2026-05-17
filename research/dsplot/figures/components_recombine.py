"""components_recombine_either_order_v18.png renderer — 3-panel projection figure.

Plus a regenerator for vector_xy_reconstruction.png (per LOCKED D-03 — orphan
retired). The historical PNG had no current generator; this module produces a
real dsplot-based render under the same filename.

Layout (3 panels, left → right):
  1. "Vector Projection onto Axes" — single vector ``a`` with solid white
     component arrows + droplines.
  2. "Tip-To-Tail Reconstruction" — vector ``a`` with parallelogram showing
     both x-then-y and y-then-x orderings (dashed components, no droplines).
  3. "Perpendicular Components" — vectors ``a`` and ``a'`` side-by-side
     showing that flipping the sign of aₓ gives the orthogonal sibling a'.

Foundation constants A, A_PRIME, FOUND_LIM live in foundation_constants.py.
"""
from __future__ import annotations

import os

from dsplot import (
    Annotation,
    Dropline,
    Figure,
    StaticPanel,
    Vector,
    VectorComponents,
    style,
)

from .foundation_constants import A, A_PRIME, FOUND_LIM


def _panel_projection_onto_axes() -> StaticPanel:
    """Panel 1: solid white component arrows + droplines."""
    panel = StaticPanel(
        title="Vector Projection onto Axes",
        subtitle=r"Projecting $\vec{a}$ onto $\hat{x}$ and $\hat{y}$",
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )

    ax_val, ay_val = A
    component_color = style.NEUTRAL_COLOR

    # Droplines from tip to each axis (dashed, drawn behind components).
    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0)))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val)))

    # Solid component arrows along x and y (white / NEUTRAL, bold).
    panel.add(
        Vector((ax_val, 0.0),
               color=component_color,
               alpha=0.95,
               zorder=2)
    )
    panel.add(
        Vector((0.0, ay_val),
               color=component_color,
               alpha=0.95,
               zorder=2)
    )

    # Component labels (orange, near each axis-aligned arrow tip).
    panel.add(
        Annotation("aₓ",
                   xy=(ax_val / 2.0, -0.35),
                   color=style.PRIMARY_COLOR,
                   fontweight="bold",
                   fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                   ha="center", va="top")
    )
    panel.add(
        Annotation("aᵧ",
                   xy=(ax_val + 0.30, ay_val / 2.0),
                   color=style.PRIMARY_COLOR,
                   fontweight="bold",
                   fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                   ha="left", va="center")
    )

    # Vector a on top.
    panel.add(
        Vector(A,
               color=style.PRIMARY_COLOR,
               label="a",
               alpha=1.0,
               zorder=3,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )
    return panel


def _panel_tip_to_tail() -> StaticPanel:
    """Panel 2: parallelogram with dashed component arrows in both orders."""
    panel = StaticPanel(
        title="Tip-To-Tail Reconstruction",
        subtitle=r"$\hat{x}$ then $\hat{y}$  |  $\hat{y}$ then $\hat{x}$",
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )

    ax_val, ay_val = A
    component_color = style.NEUTRAL_COLOR

    panel.add(
        VectorComponents(
            A,
            first_axis="x",
            show_droplines=False,
            component_color=component_color,
        )
    )
    panel.add(
        VectorComponents(
            A,
            first_axis="y",
            show_droplines=False,
            component_color=component_color,
        )
    )

    # Component labels at the four edge midpoints of the parallelogram.
    # Bottom (x leg, y=0), right (y leg, x=ax_val), top (x leg, y=ay_val),
    # left (y leg, x=0).
    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.35),
                          ha="center", va="top", **label_kwargs))
    panel.add(Annotation("aᵧ", xy=(ax_val + 0.30, ay_val / 2.0),
                          ha="left", va="center", **label_kwargs))
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.35),
                          ha="center", va="bottom", **label_kwargs))
    panel.add(Annotation("aᵧ", xy=(-0.30, ay_val / 2.0),
                          ha="right", va="center", **label_kwargs))

    panel.add(
        Vector(A,
               color=style.PRIMARY_COLOR,
               label="a",
               alpha=1.0,
               zorder=4,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )
    return panel


def _panel_perpendicular() -> StaticPanel:
    """Panel 3: vector a + its y-axis-mirrored sibling a'."""
    panel = StaticPanel(
        title="Perpendicular Components",
        subtitle="PLACEHOLDER",
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )

    ax_val, ay_val = A
    apx_val, apy_val = A_PRIME
    component_color = style.NEUTRAL_COLOR

    panel.add(
        VectorComponents(
            A,
            first_axis="x",
            show_droplines=False,
            component_color=component_color,
        )
    )
    panel.add(
        VectorComponents(
            A_PRIME,
            first_axis="x",
            show_droplines=False,
            component_color=component_color,
        )
    )

    # Labels for the four component arrows.
    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.35),
                          ha="center", va="top", **label_kwargs))
    panel.add(Annotation("aᵧ", xy=(ax_val + 0.30, ay_val / 2.0),
                          ha="left", va="center", **label_kwargs))
    panel.add(Annotation("a′ₓ", xy=(apx_val / 2.0, -0.35),
                          ha="center", va="top", **label_kwargs))
    panel.add(Annotation("a′ᵧ", xy=(apx_val - 0.30, apy_val / 2.0),
                          ha="right", va="center", **label_kwargs))

    # Vectors a and a' on top.
    panel.add(
        Vector(A,
               color=style.PRIMARY_COLOR,
               label="a",
               alpha=1.0,
               zorder=4,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )
    panel.add(
        Vector(A_PRIME,
               color=style.PRIMARY_COLOR,
               label="a′",
               alpha=1.0,
               zorder=4,
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    )
    return panel


def render(
    output_dir: str,
    output_filename: str = "components_recombine_either_order_v18.png",
) -> str:
    """Render the 3-panel components-recombine figure."""
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 1.3
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.55
    fig = Figure(
        n_rows=1, n_cols=3,
        figsize=(panel_w * 3, panel_h),
        suptitle="Basic Vector Projection",
    )
    fig.add_panel(_panel_projection_onto_axes(), row=0, col=0)
    fig.add_panel(_panel_tip_to_tail(),         row=0, col=1)
    fig.add_panel(_panel_perpendicular(),       row=0, col=2)
    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)


def render_vector_xy_reconstruction(
    output_dir: str,
    output_filename: str = "vector_xy_reconstruction.png",
) -> str:
    """Regenerate vector_xy_reconstruction.png (LOCKED D-03 — orphan retired).

    Two-panel figure showing A decomposed in both orders (x then y on the
    left, y then x on the right). Dashed component arrows + the original
    vector on top per panel demonstrate that the two orderings reconstruct
    the same tip.
    """
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 1.3
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.05
    fig = Figure(
        n_rows=1, n_cols=2,
        figsize=(panel_w * 2, panel_h),
        suptitle="Same Components, Either Order — Same Vector",
    )

    ax_val, ay_val = A
    component_color = style.PRIMARY_COLOR
    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )

    for col, first_axis in enumerate(("x", "y")):
        panel = StaticPanel(
            lim=FOUND_LIM,
            axis_style="arrow",
            axis_labels=True,
            show_border=False,
        )
        panel.add(
            VectorComponents(
                A,
                first_axis=first_axis,
                show_droplines=False,
                component_color=component_color,
            )
        )
        # Place labels per ordering — labels track the visible leg positions.
        if first_axis == "x":
            panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.35),
                                  ha="center", va="top", **label_kwargs))
            panel.add(Annotation("aᵧ", xy=(ax_val + 0.30, ay_val / 2.0),
                                  ha="left", va="center", **label_kwargs))
        else:
            panel.add(Annotation("aᵧ", xy=(-0.30, ay_val / 2.0),
                                  ha="right", va="center", **label_kwargs))
            panel.add(Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.35),
                                  ha="center", va="bottom", **label_kwargs))
        panel.add(
            Vector(A,
                   color=style.PRIMARY_COLOR,
                   label="a",
                   alpha=1.0,
                   zorder=4,
                   linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
        )
        fig.add_panel(panel, row=0, col=col)

    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
