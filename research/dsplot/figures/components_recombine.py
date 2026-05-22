"""components_recombine_either_order_v19.png renderer — 3-panel §2.4.1 figure.

Layout (3 panels, left → right):
  1. "Projection Along Each Axis" (Figure 2.4.1.a) — vector ``a`` plus solid
     white component arrows + droplines onto the x/y axes.
  2. "Reconstruction in Any Order" (Figure 2.4.1.b) — vector ``a`` with the
     full parallelogram of both x-first and y-first orderings.
  3. "Completely Independent Components" (Figure 2.4.1.c) — vector ``a`` and
     its y-axis-mirrored sibling ``a'``. ``a`` and its x-component are alpha-
     muted so ``a'`` is the visual spotlight.

Chrome: v52 kitchen-sink template — boxed axes, integer ticks, light grid,
"x" / "y" axis labels at the spine edges. Title above each panel; subtitle
("Figure 2.4.1.a/b/c") just below the plot in italic; caption (one-line
explanatory line) below the subtitle in regular weight.

Foundation constants A, A_PRIME, FOUND_LIM live in foundation_constants.py
and are shared with the projection-reconstruction / 3D figures.
"""
from __future__ import annotations

import os

from .. import (
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


# Vector ``a`` label sits this far past the tip (axes units). Bumped above
# the library default of 0.30 so "a" doesn't crowd the arrowhead.
_LABEL_OFFSET = 0.55

# Panel-3 spotlight: vector ``a`` and its x-component fade to this alpha so
# ``a'`` reads as the new subject.
_MUTED_ALPHA = 0.35

# Figure.compose unit size (inches per 1-unit-wide panel). Slightly smaller
# than the legacy v18 layout but big enough that the new long titles and
# explanatory captions fit without overflowing panel cells.
_UNIT_INCHES = 4.2

# Notebook reconstruction animation — 9 frames per spec:
#   0 nothing
#   1 bottom-x leg
#   2 + right-y leg
#   3 + vector a
#   4 nothing (reset)
#   5 left-y leg
#   6 + top-x leg
#   7 + vector a
#   8 all components + vector a (held)
_RECONSTRUCTION_NUM_FRAMES = 9


def _common_panel_kwargs() -> dict:
    """Chrome shared by all three static panels — v52 kitchen-sink look."""
    return dict(
        lim=FOUND_LIM,
        axis_style="line",
        axis_labels=True,
        show_border=True,
        show_ticks=True,
        show_grid=True,
    )


def _panel_projection_onto_axes() -> StaticPanel:
    """Panel 1: solid white component arrows + droplines."""
    panel = StaticPanel(
        title="Projection Along Each Axis",
        subtitle="Figure 2.4.1.a",
        caption=(
            "Projection onto XY reveals the x and y\n"
            r"dimension components $a_x$, $a_y$ of $\vec{a}$"
        ),
        **_common_panel_kwargs(),
    )

    ax_val, ay_val = A
    component_color = style.NEUTRAL_COLOR

    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0)))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val)))

    panel.add(Vector((ax_val, 0.0), color=component_color, alpha=0.95, zorder=2))
    panel.add(Vector((0.0, ay_val), color=component_color, alpha=0.95, zorder=2))

    panel.add(
        Annotation(
            "aₓ",
            xy=(ax_val / 2.0, -0.45),
            color=style.PRIMARY_COLOR,
            fontweight="bold",
            fontsize=style.DEFAULT_LABEL_FONT_SIZE,
            ha="center", va="top",
        )
    )
    panel.add(
        Annotation(
            "aᵧ",
            xy=(ax_val + 0.35, ay_val / 2.0),
            color=style.PRIMARY_COLOR,
            fontweight="bold",
            fontsize=style.DEFAULT_LABEL_FONT_SIZE,
            ha="left", va="center",
        )
    )

    panel.add(
        Vector(
            A,
            color=style.PRIMARY_COLOR,
            label="a",
            label_offset=_LABEL_OFFSET,
            alpha=1.0,
            zorder=3,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    return panel


def _panel_tip_to_tail() -> StaticPanel:
    """Panel 2: parallelogram with dashed component arrows in both orders."""
    panel = StaticPanel(
        title="Reconstruction in Any Order",
        subtitle="Figure 2.4.1.b",
        caption=(
            "Rebuilding tip-to-tail, starting with x first then\n"
            "y, or y first then x, regardlessly end up with\n"
            "the same original"
        ),
        **_common_panel_kwargs(),
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

    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.45),
                          ha="center", va="top", **label_kwargs))
    panel.add(Annotation("aᵧ", xy=(ax_val + 0.35, ay_val / 2.0),
                          ha="left", va="center", **label_kwargs))
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.45),
                          ha="center", va="bottom", **label_kwargs))
    panel.add(Annotation("aᵧ", xy=(-0.35, ay_val / 2.0),
                          ha="right", va="center", **label_kwargs))

    panel.add(
        Vector(
            A,
            color=style.PRIMARY_COLOR,
            label="a",
            label_offset=_LABEL_OFFSET,
            alpha=1.0,
            zorder=4,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    return panel


def _panel_perpendicular() -> StaticPanel:
    """Panel 3: a' is the spotlight; a + its x-component fade to _MUTED_ALPHA."""
    panel = StaticPanel(
        title="Independent Components",
        subtitle="Figure 2.4.1.c",
        caption=(
            "Since x and y are at right angles to each other, we say\n"
            "they are orthogonal, or independent of each other —\n"
            "one cannot be described in terms of the other, meaning\n"
            "measuring one tells you nothing about the other"
        ),
        **_common_panel_kwargs(),
    )

    ax_val, ay_val = A
    apx_val, apy_val = A_PRIME
    component_color = style.NEUTRAL_COLOR

    # Spotlight = shared y-component. Both vectors and both x-legs fade to
    # muted; only the two y-legs (and their labels) stay at full opacity so
    # the eye locks onto "y is the same regardless of x".
    panel.add(
        Vector(
            (ax_val, 0.0), origin=(0.0, 0.0),
            color=component_color,
            linestyle="--",
            alpha=_MUTED_ALPHA,
            zorder=2,
        )
    )
    panel.add(
        Vector(
            (0.0, ay_val), origin=(ax_val, 0.0),
            color=component_color,
            linestyle="--",
            alpha=0.95,
            zorder=2,
        )
    )
    panel.add(
        Vector(
            (apx_val, 0.0), origin=(0.0, 0.0),
            color=component_color,
            linestyle="--",
            alpha=_MUTED_ALPHA,
            zorder=2,
        )
    )
    panel.add(
        Vector(
            (0.0, apy_val), origin=(apx_val, 0.0),
            color=component_color,
            linestyle="--",
            alpha=0.95,
            zorder=2,
        )
    )

    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )
    # x-component labels — muted (match their muted legs + muted vectors).
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.45),
                          ha="center", va="top",
                          alpha=_MUTED_ALPHA, **label_kwargs))
    panel.add(Annotation("a′ₓ", xy=(apx_val / 2.0, -0.45),
                          ha="center", va="top",
                          alpha=_MUTED_ALPHA, **label_kwargs))
    # y-component labels — full opacity (the spotlight).
    panel.add(Annotation("aᵧ", xy=(ax_val + 0.35, ay_val / 2.0),
                          ha="left", va="center", **label_kwargs))
    panel.add(Annotation("a′ᵧ", xy=(apx_val - 0.35, apy_val / 2.0),
                          ha="right", va="center", **label_kwargs))

    # Both vectors muted — they're the framing, not the subject.
    panel.add(
        Vector(
            A,
            color=style.PRIMARY_COLOR,
            label="a",
            label_offset=_LABEL_OFFSET,
            alpha=_MUTED_ALPHA,
            zorder=4,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    panel.add(
        Vector(
            A_PRIME,
            color=style.PRIMARY_COLOR,
            label="a′",
            label_offset=_LABEL_OFFSET,
            alpha=_MUTED_ALPHA,
            zorder=4,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    return panel


def render(
    output_dir: str,
    output_filename: str = "either_order_v19.png",
) -> str:
    """Render the 3-panel §2.4.1 figure with v52 kitchen-sink chrome."""
    fig = Figure.compose(
        rows=[[
            _panel_projection_onto_axes(),
            _panel_tip_to_tail(),
            _panel_perpendicular(),
        ]],
        suptitle="Basic Vector Projection",
        unit_inches=_UNIT_INCHES,
        show_cell_borders=True,
    )
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
    ax_val, ay_val = A
    component_color = style.PRIMARY_COLOR
    label_kwargs = dict(
        color=style.PRIMARY_COLOR,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )

    panels = []
    for first_axis in ("x", "y"):
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
        if first_axis == "x":
            panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.45),
                                  ha="center", va="top", **label_kwargs))
            panel.add(Annotation("aᵧ", xy=(ax_val + 0.35, ay_val / 2.0),
                                  ha="left", va="center", **label_kwargs))
        else:
            panel.add(Annotation("aᵧ", xy=(-0.35, ay_val / 2.0),
                                  ha="right", va="center", **label_kwargs))
            panel.add(Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.45),
                                  ha="center", va="bottom", **label_kwargs))
        panel.add(
            Vector(
                A,
                color=style.PRIMARY_COLOR,
                label="a",
                label_offset=_LABEL_OFFSET,
                alpha=1.0,
                zorder=4,
                linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
            )
        )
        panels.append(panel)

    fig = Figure.compose(
        rows=[panels],
        suptitle="Same Components, Either Order — Same Vector",
        unit_inches=_UNIT_INCHES,
        show_cell_borders=False,
    )
    fig.render()
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)


# === Notebook (dsp.ipynb) — §2.4 Figure 1 as a 1×3 mixed Figure ============
#
# Three panels:
#   col 0 — StaticPanel       — xy projection of vector a.
#   col 1 — DynamicPanel (9f) — reconstruction sequence per the 9-frame spec:
#                                nothing → x-leg → +y-leg → +a → nothing →
#                                y-leg → +x-leg → +a → all-components+a.
#   col 2 — DynamicPanel (5f) — orthogonality: a stays anchored at A; a'
#                                sweeps x through [+ax, +ax/2, 0, -ax/2, -ax]
#                                with a'.y fixed at ay; y is invariant under x.

# a'.x sweep — 5 evenly spaced points from +ax_val → -ax_val. The mirror
# frame is held one extra tick so the reconstruction panel finishes its
# 9-frame cycle in lockstep.
def _a_prime_x_sweep() -> list[float]:
    ax_val, _ = A
    return [ax_val, ax_val / 2.0, 0.0, -ax_val / 2.0, -ax_val]


def _notebook_static_xy_projection() -> StaticPanel:
    panel = StaticPanel(
        title="Projection Along Each Axis",
        subtitle="Figure 2.4.1.a",
        caption=(
            "Projection onto XY reveals the x and y\n"
            r"dimension components $a_x$, $a_y$ of $\vec{a}$"
        ),
        **_common_panel_kwargs(),
    )
    ax_val, ay_val = A
    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0)))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val)))
    panel.add(Vector((ax_val, 0.0), color=style.NEUTRAL_COLOR, alpha=0.95,
                     linestyle="--", zorder=2))
    panel.add(Vector((0.0, ay_val), color=style.NEUTRAL_COLOR, alpha=0.95,
                     linestyle="--", zorder=2))
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, -0.45),
                         color=style.PRIMARY_COLOR, fontweight="bold",
                         fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                         ha="center", va="top"))
    panel.add(Annotation("aᵧ", xy=(-0.35, ay_val / 2.0),
                         color=style.PRIMARY_COLOR, fontweight="bold",
                         fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                         ha="right", va="center"))
    panel.add(Vector(A,
                     color=style.PRIMARY_COLOR, label="a",
                     label_offset=_LABEL_OFFSET,
                     alpha=1.0, zorder=3,
                     linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH))
    return panel


def _reconstruction_frames() -> list[list]:
    """9-frame cycle per the locked spec:

      0 nothing                          | reset / pause
      1 x-leg from origin                | start x-then-y order
      2 + y-leg from x-tip               | complete x-then-y
      3 + bold vector a                  | confirm tip lands at a
      4 nothing                          | reset / pause
      5 y-leg from origin                | start y-then-x order
      6 + x-leg from y-tip               | complete y-then-x
      7 + bold vector a                  | confirm tip lands at a
      8 all four legs + bold a           | both orderings simultaneously
    """
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
                      label_offset=_LABEL_OFFSET,
                      alpha=1.0, zorder=4,
                      linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)

    label_kw = dict(
        color=primary,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )

    def label_bottom_x():
        return Annotation("aₓ", xy=(ax_val / 2.0, -0.45),
                          ha="center", va="top", **label_kw)

    def label_right_y():
        return Annotation("aᵧ", xy=(ax_val + 0.35, ay_val / 2.0),
                          ha="left", va="center", **label_kw)

    def label_left_y():
        return Annotation("aᵧ", xy=(-0.35, ay_val / 2.0),
                          ha="right", va="center", **label_kw)

    def label_top_x():
        return Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.45),
                          ha="center", va="bottom", **label_kw)

    return [
        # 0 — nothing
        [],
        # 1 — x-leg only
        [x_leg(), label_bottom_x()],
        # 2 — x-leg + y-leg from x-tip
        [x_leg(), y_leg_from_x_tip(), label_bottom_x(), label_right_y()],
        # 3 — same + bold a
        [x_leg(), y_leg_from_x_tip(), a_bold(),
         label_bottom_x(), label_right_y()],
        # 4 — reset
        [],
        # 5 — y-leg only (from origin)
        [y_leg(), label_left_y()],
        # 6 — y-leg + x-leg from y-tip
        [y_leg(), x_leg_from_y_tip(), label_left_y(), label_top_x()],
        # 7 — same + bold a
        [y_leg(), x_leg_from_y_tip(), a_bold(),
         label_left_y(), label_top_x()],
        # 8 — all four legs + bold a, both orderings on screen
        [x_leg(), y_leg_from_x_tip(), y_leg(), x_leg_from_y_tip(), a_bold(),
         label_bottom_x(), label_right_y(),
         label_left_y(), label_top_x()],
    ]


def _notebook_reconstruction_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frames=_reconstruction_frames(),
        interval_ms=1100,
        repeat=True,
        title="Reconstruction in Any Order",
        subtitle="Figure 2.4.1.b",
        caption=(
            "Rebuilding tip-to-tail, starting with x first then\n"
            "y, or y first then x, regardlessly end up with\n"
            "the same original"
        ),
        **_common_panel_kwargs(),
    )


def _orthogonality_frame(apx_val: float) -> list:
    """One animation frame at a given a'.x sweep value.

    Spotlight (matches the static panel-3): both vectors and both x-legs are
    muted; both y-legs and their labels stay full opacity so the shared
    y-component reads as the through-line invariant under x.
    """
    ax_val, ay_val = A
    primary = style.PRIMARY_COLOR
    neutral = style.NEUTRAL_COLOR

    # a's parallelogram legs — raw Vectors so x vs y can carry different alphas.
    a_x_leg = Vector(
        (ax_val, 0.0), origin=(0.0, 0.0),
        color=neutral, linestyle="--", alpha=_MUTED_ALPHA, zorder=2,
    )
    a_y_leg = Vector(
        (0.0, ay_val), origin=(ax_val, 0.0),
        color=neutral, linestyle="--", alpha=0.95, zorder=2,
    )

    # a''s parallelogram legs (apx_val sweeps).
    ap_x_leg = Vector(
        (apx_val, 0.0), origin=(0.0, 0.0),
        color=neutral, linestyle="--", alpha=_MUTED_ALPHA, zorder=2,
    )
    ap_y_leg = Vector(
        (0.0, ay_val), origin=(apx_val, 0.0),
        color=neutral, linestyle="--", alpha=0.95, zorder=2,
    )

    bold_a = Vector(
        A,
        color=primary, label="a",
        label_offset=_LABEL_OFFSET,
        alpha=_MUTED_ALPHA, zorder=5,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
    )
    # Suppress the a' label glyph when it sits directly on top of a's label
    # (overprint reads as a smudge). The y-leg label still reports the value.
    ap_label = None if apx_val == ax_val else "a′"
    bold_ap = Vector(
        (apx_val, ay_val),
        color=primary, label=ap_label,
        label_offset=_LABEL_OFFSET,
        alpha=_MUTED_ALPHA, zorder=5,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
    )

    label_kw = dict(
        color=style.PRIMARY_COLOR,
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    # x-component labels — muted.
    a_x_label = Annotation(
        "aₓ", xy=(ax_val / 2.0, -0.45),
        ha="center", va="top", alpha=_MUTED_ALPHA, **label_kw,
    )
    # y-component labels — full opacity (spotlight).
    a_y_label = Annotation(
        "aᵧ", xy=(ax_val + 0.35, ay_val / 2.0),
        ha="left", va="center", **label_kw,
    )

    frame = [a_x_leg, a_y_leg, ap_x_leg, ap_y_leg,
             bold_a, bold_ap, a_x_label, a_y_label]

    # a''s labels — appear once a' has separated from a (avoid overprinting).
    if apx_val != ax_val:
        ap_x_xy = (apx_val / 2.0, -0.45) if apx_val != 0.0 else (-0.45, -0.45)
        ap_x_ha = "center" if apx_val != 0.0 else "right"
        frame.append(Annotation(
            "a′ₓ", xy=ap_x_xy,
            ha=ap_x_ha, va="top",
            alpha=_MUTED_ALPHA, **label_kw,
        ))
        # a''s y-label sits on the opposite side of its y-leg (avoid colliding
        # with a's y-label on the right). When apx_val == 0 both y-legs sit on
        # the y-axis; place a'ᵧ on the left so it doesn't overprint aᵧ.
        ap_y_xy = (apx_val - 0.35, ay_val / 2.0)
        ap_y_ha = "right"
        frame.append(Annotation(
            "a′ᵧ", xy=ap_y_xy,
            ha=ap_y_ha, va="center", **label_kw,
        ))

    return frame


def _notebook_orthogonality_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frames=[_orthogonality_frame(apx) for apx in _a_prime_x_sweep()],
        interval_ms=1100,
        repeat=True,
        title="Independent Components",
        subtitle="Figure 2.4.1.c",
        caption=(
            "Since x and y are at right angles to each other, we say\n"
            "they are orthogonal, or independent of each other —\n"
            "one cannot be described in terms of the other, meaning\n"
            "measuring one tells you nothing about the other"
        ),
        **_common_panel_kwargs(),
    )


def build_notebook_figure() -> Figure:
    """Build §2.4 Figure 1 as a 1×3 mixed Figure for dsp.ipynb cell 01.

    Returns a configured but un-rendered Figure. The caller invokes
    `fig.render()` (and optionally `plt.show()` under `%matplotlib widget`).

    Renders at dpi=75 (vs. the library default of 150) so the cell output
    fits inside a sidebar-narrowed VS Code Jupyter cell without forcing a
    horizontal scroll / zoom-out. The static PNG export from `render()`
    keeps the library default DPI for print/web quality.
    """
    return Figure.compose(
        rows=[[
            _notebook_static_xy_projection(),
            _notebook_reconstruction_dynamic(),
            _notebook_orthogonality_dynamic(),
        ]],
        suptitle="Basic Vector Projection",
        unit_inches=_UNIT_INCHES,
        dpi=75,
        show_cell_borders=True,
    )


def show() -> Figure:
    """Build, render, and display §2.4 Figure 1 in a Jupyter cell."""
    import matplotlib.pyplot as plt
    fig = build_notebook_figure()
    fig.render()
    plt.show()
    return fig
