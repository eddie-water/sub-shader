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
    DynamicPanel,
    Figure,
    StaticPanel,
    Vector,
    VectorComponents,
    style,
)

from .foundation_constants import A, A_PRIME, FOUND_LIM


# 6 frames: coincident → sweep through zero → mirror, then hold at mirror for
# one extra tick so the panel cycle length matches the 6-frame reconstruction
# panel and the two animations stay in lockstep.
_A_PRIME_X_SWEEP = [2.0, 1.0, 0.0, -1.0, -2.0, -2.0]


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


# === Notebook (dsp.ipynb cell 01) — §2.4 Figure 1 as a 1×3 mixed Figure =====
#
# Per CONTEXT.md D-08 amendment. Three panels:
#   col 0 — StaticPanel       — xy projection of vector a = (2, 3).
#   col 1 — DynamicPanel (6f) — reconstruction order: x-first then y-first;
#                                both orderings arrive at a.
#   col 2 — DynamicPanel (5f) — orthogonality: a stays anchored at (2, 3),
#                                a' sweeps x through [+2, +1, 0, -1, -2] with
#                                a'.y fixed at 3.0; y is invariant under x.
#
# Each builder returns a configured Panel; build_notebook_figure() composes
# them into the 1×3 Figure. The notebook cell stays a thin caller.


def _notebook_static_xy_projection() -> StaticPanel:
    panel = StaticPanel(
        title="Projection onto XY",
        subtitle=r"$\vec{a}$ on $\hat{x}$, $\hat{y}$",
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )
    ax_val, ay_val = A
    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0)))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val)))
    panel.add(Vector((ax_val, 0.0), color=style.NEUTRAL_COLOR, alpha=0.95,
                     linestyle="--", zorder=2))
    panel.add(Vector((0.0, ay_val), color=style.NEUTRAL_COLOR, alpha=0.95,
                     linestyle="--", zorder=2))
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.35),
                         color=style.PRIMARY_COLOR, fontweight="bold",
                         fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                         ha="center", va="top"))
    panel.add(Annotation("aᵧ", xy=(-0.30, ay_val / 2.0),
                         color=style.PRIMARY_COLOR, fontweight="bold",
                         fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                         ha="right", va="center"))
    panel.add(Vector(A,
                     color=style.PRIMARY_COLOR, label="a",
                     alpha=1.0, zorder=3,
                     linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH))
    return panel


def _reconstruction_frames() -> list[list]:
    ax_val, ay_val = A
    primary = style.PRIMARY_COLOR
    neutral = style.NEUTRAL_COLOR

    def x_leg():
        return Vector((ax_val, 0.0), color=neutral, alpha=0.95,
                      linestyle="--", zorder=2)

    def y_leg_from_x_tip():
        return Vector((0.0, ay_val), origin=(ax_val, 0.0),
                      color=neutral, alpha=0.95, linestyle="--", zorder=2)

    def y_leg():
        return Vector((0.0, ay_val), color=neutral, alpha=0.95,
                      linestyle="--", zorder=2)

    def x_leg_from_y_tip():
        return Vector((ax_val, 0.0), origin=(0.0, ay_val),
                      color=neutral, alpha=0.95, linestyle="--", zorder=2)

    def a_bold():
        return Vector(A,
                      color=primary, label="a",
                      alpha=1.0, zorder=4,
                      linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)

    # Component labels — each placed on the outer side of its leg so labels
    # never sit inside the parallelogram. Each label appears in the same
    # frame as the leg it labels and persists for the rest of the cycle.
    def label_bottom_x():
        return Annotation("aₓ", xy=(ax_val / 2.0, -0.30),
                          color=primary, fontweight="bold",
                          fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                          ha="center", va="top")

    def label_right_y():
        return Annotation("aᵧ", xy=(ax_val + 0.30, ay_val / 2.0),
                          color=primary, fontweight="bold",
                          fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                          ha="left", va="center")

    def label_left_y():
        return Annotation("aᵧ", xy=(-0.30, ay_val / 2.0),
                          color=primary, fontweight="bold",
                          fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                          ha="right", va="center")

    def label_top_x():
        return Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.24),
                          color=primary, fontweight="bold",
                          fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                          ha="center", va="bottom")

    # Cumulative buildup — never erase, just keep adding legs (and their
    # aₓ / aᵧ labels) until both orderings are visible side by side. The
    # figure clock holds the final full-parallelogram frame for the
    # trailing ticks while Panel 3 finishes.
    return [
        # Frame 0: destination teaser — a alone.
        [a_bold()],
        # Frame 1: start x-then-y — x-leg from origin.
        [a_bold(), x_leg(), label_bottom_x()],
        # Frame 2: complete x-then-y — y-leg from x-tip lands at a.
        [a_bold(), x_leg(), y_leg_from_x_tip(),
         label_bottom_x(), label_right_y()],
        # Frame 3: start y-then-x — y-leg from origin (x-then-y persists).
        [a_bold(), x_leg(), y_leg_from_x_tip(), y_leg(),
         label_bottom_x(), label_right_y(), label_left_y()],
        # Frame 4: complete y-then-x — x-leg from y-tip lands at a.
        # Both orderings now on screen; cycle holds here until reset.
        [a_bold(), x_leg(), y_leg_from_x_tip(), y_leg(), x_leg_from_y_tip(),
         label_bottom_x(), label_right_y(), label_left_y(), label_top_x()],
    ]


def _notebook_reconstruction_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frames=_reconstruction_frames(),
        interval_ms=1250,
        repeat=True,
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
        title="Reconstruction",
        subtitle="either order, same arrival",
    )


def _orthogonality_frame(apx_val: float) -> list:
    ax_val, ay_val = A
    primary = style.PRIMARY_COLOR
    neutral = style.NEUTRAL_COLOR

    a_components = VectorComponents(A, first_axis="x", show_droplines=False,
                                     component_color=neutral)
    ap_vec = (apx_val, ay_val)
    ap_components = VectorComponents(ap_vec, first_axis="x", show_droplines=False,
                                      component_color=neutral)
    bold_a = Vector(A,
                    color=primary, label="a",
                    alpha=1.0, zorder=5,
                    linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    # When a' coincides with a, suppress the a' label so the two centered
    # texts don't stack into a blurry overprint (matplotlib centers each
    # bbox; "a" and "a′" have different glyph widths, so the visible
    # characters land a few pixels apart). The bottom sweep label still
    # reports a′ₓ = +2.0 so the viewer knows the two are stacked.
    ap_label = None if apx_val == ax_val else "a′"
    bold_ap = Vector(ap_vec,
                     color=primary, label=ap_label,
                     alpha=1.0, zorder=5,
                     linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)
    # In-plot info labels: a' (sweep) in Q3 (lower-left), a (anchored) in
    # Q4 (lower-right). The upper half of the panel is occupied by the
    # vector pair; the lower half is empty, so it's a natural home for
    # the per-frame component readout.
    q3_label = Annotation(
        f"a′ₓ = {apx_val:+.1f}\na′ᵧ = {ay_val:+.1f}",
        xy=(-3.0, -3.0),
        color=style.PRIMARY_COLOR,
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
        fontweight="bold",
        ha="center", va="center",
    )
    q4_label = Annotation(
        f"aₓ = {ax_val:+.1f}\naᵧ = {ay_val:+.1f}",
        xy=(3.0, -3.0),
        color=style.PRIMARY_COLOR,
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
        fontweight="bold",
        ha="center", va="center",
    )
    # a's shadow component labels — always visible since a is anchored.
    label_kw = dict(
        color=style.PRIMARY_COLOR,
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    a_x_label = Annotation(
        "aₓ", xy=(ax_val / 2.0, -0.30),
        ha="center", va="top", **label_kw,
    )
    a_y_label = Annotation(
        "aᵧ", xy=(ax_val + 0.30, ay_val / 2.0),
        ha="left", va="center", **label_kw,
    )

    frame = [a_components, ap_components, bold_a, bold_ap,
             a_x_label, a_y_label, q3_label, q4_label]

    # On the held mirror frame (a' fully separated to the left), label a''s
    # shadow components too. The x-leg goes from origin to (-2, 0) — we
    # write "a'ₓ"; the y-leg is the same magnitude as a's so it's just "a'ᵧ".
    if apx_val == -ax_val:
        frame.append(Annotation(
            "a′ₓ", xy=(apx_val / 2.0, -0.30),
            ha="center", va="top", **label_kw,
        ))
        frame.append(Annotation(
            "a′ᵧ", xy=(apx_val - 0.30, ay_val / 2.0),
            ha="right", va="center", **label_kw,
        ))

    return frame


def _notebook_orthogonality_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frames=[_orthogonality_frame(apx) for apx in _A_PRIME_X_SWEEP],
        interval_ms=1250,
        repeat=True,
        lim=FOUND_LIM,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
        title="Perpendicular",
        subtitle=r"$a_y$ invariant under $a_x$",
    )


def build_notebook_figure() -> Figure:
    """Build §2.4 Figure 1 as a 1×3 mixed Figure for dsp.ipynb cell 01.

    Returns a configured but un-rendered Figure. The caller invokes
    `fig.render()` (and optionally `plt.show()` under `%matplotlib widget`).
    """
    # Notebook size — three knobs tuned together:
    #   panel_w / panel_h : figure aspect ratio. Slightly taller than wide
    #                       per panel so the suptitle band + axes both fit.
    #   dpi               : on-screen pixel size.
    #   suptitle_fontsize : scaled down for the shorter canvas so it
    #                       doesn't collide with panel titles.
    # At dpi=75 this renders to ~1012×412 px — fits a sidebar-narrowed
    # VS Code Jupyter cell without horizontal clip. Legacy static v18
    # export above keeps its taller 1.3×1.55 multipliers.
    panel_w = style.DEFAULT_PANEL_SIZE_INCHES * 0.9
    panel_h = style.DEFAULT_PANEL_SIZE_INCHES * 1.1
    fig = Figure(
        n_rows=1, n_cols=3,
        figsize=(panel_w * 3, panel_h),
        suptitle="Basic Vector Projection",
        suptitle_fontsize=22,
        dpi=75,
        # Default wspace (0.04) leaves panel titles touching adjacent
        # panels at this compact figsize. 0.15 gives breathing room.
        wspace=0.15,
    )
    fig.add_panel(_notebook_static_xy_projection(),  row=0, col=0)
    fig.add_panel(_notebook_reconstruction_dynamic(), row=0, col=1)
    fig.add_panel(_notebook_orthogonality_dynamic(),  row=0, col=2)
    return fig


def show() -> Figure:
    """Build, render, and display §2.4 Figure 1 in a Jupyter cell.

    Returns the rendered Figure so the caller can hold a reference (load-
    bearing for matplotlib's animation GC under standalone backends — the
    notebook's `%matplotlib widget` backend already pins it through the
    canvas widget).
    """
    import matplotlib.pyplot as plt
    fig = build_notebook_figure()
    fig.render()
    plt.show()
    return fig
