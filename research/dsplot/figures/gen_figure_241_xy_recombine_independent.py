"""figure_2_4_1 — §2.4.1 Basic Vector Projection (3 panels: projection / reconstruction / independence).

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

import math
import os
from contextlib import contextmanager

from .. import (
    Annotation,
    Dropline,
    DynamicPanel,
    Figure,
    StaticPanel,
    SuptitlePanel,
    Vector,
    VectorComponents,
    nb_compact_style,
    style,
)

from .foundation_constants import A, A_PRIME, FOUND_LIM


# Vector ``a`` label sits this far past the tip (axes units). Bumped above
# the library default of 0.30 so "a" doesn't crowd the arrowhead.
_LABEL_OFFSET = 0.40

# Panel-3 spotlight: vector ``a`` and its x-component fade to this alpha so
# ``a'`` reads as the new subject.
_MUTED_ALPHA = 0.35

# Figure.compose unit size (inches per 1-unit-wide panel). Slightly smaller
# than the legacy v18 layout but big enough that the new long titles and
# explanatory captions fit without overflowing panel cells.
_UNIT_INCHES = 4.2

# Header band font: the figure NUMBER + NAME on one line (the 2.5 / 2.6
# convention). Sized to sit in the top band without dominating like the prior
# 40pt suptitle did.
_HEADER_FONT_SIZE = 28

# Panel b — cumulative 5-beat reconstruction at four fixed angles. Fixed
# magnitude (3.0) — only the angle varies between reconstructions, and the
# angles cycle through one vector per quadrant.
#
# Beats (each frame is cumulative — adds to the previous):
#   0 vector at the cycle's angle
#   1 + x-leg from origin       (bottom; x-then-y starts)
#   2 + y-leg from x-tip        (right; x-then-y complete)
#   3 + y-leg from origin       (left; y-then-x starts — the "other y")
#   4 + x-leg from y-tip        (top; y-then-x complete, full parallelogram)
#
# Per-beat frame budget — each entry is how long that beat is held on
# screen before the next beat draws. [4,4,4,4,8] sums to 24 frames per
# reconstruction; 4 reconstructions × 24 = 96 (LCM-matches panel a). The
# climax beat (full parallelogram) gets a double-length linger before the
# next vector replaces it.
_RECON_MAG = 3.0
_RECON_ANGLES = [
    math.pi / 3.0,        # 60°  — Q1
    3 * math.pi / 4.0,    # 135° — Q2
    9 * math.pi / 7.0,    # ≈231° — Q3
    11 * math.pi / 6.0,   # 330° — Q4
]
_RECON_BEAT_FRAMES = [4, 4, 4, 4, 8]
_RECON_FRAMES_PER_RECONSTRUCTION = sum(_RECON_BEAT_FRAMES)  # 24
_RECON_RECONSTRUCTIONS_PER_CYCLE = len(_RECON_ANGLES)       # 4 × 24 = 96


def _vector_at_angle(theta: float) -> tuple[float, float]:
    return (_RECON_MAG * math.cos(theta), _RECON_MAG * math.sin(theta))


# Fill limits — asymmetric so the origin sits low-left and vector ``a`` (and
# ``a'``) fills the cell instead of stranding the lower / opposite quadrants as
# dead space. Equal RANGE on both axes (7.5) keeps equal-aspect geometry honest
# AND keeps ``a`` the same on-screen size across all three panels.
#   Panels 1 & 2 (Q1 content): origin low-left.
#   Panel 3 (a + mirrored a'): x symmetric (both signs present), y shares the
#   low-origin range.
def _common_panel_kwargs() -> dict:
    """Chrome shared by all three static panels — v52 kitchen-sink look."""
    return dict(
        lim=FOUND_LIM,
        axis_style="line",
        # In-plot italic "x"/"y" near the axis ends (data coords) instead of
        # external gutter labels — on the plot, off the padding. A touch
        # smaller than the library default so they read as quiet annotations.
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        # Spines OFF — the library CELL BORDER is the single frame for every
        # cell (data panels AND the header/footer bands), the unified 2.5 / 2.6
        # model. show_cell_borders=True in render() draws those boxes.
        show_border=False,
        # No inset tick marks — they read as faint dark-gray nubs on the cell
        # border and add nothing; the grid carries the scale.
        show_ticks=False,
        show_grid=True,
    )


# Bone-white unified chrome for the static PNG — matches gen_figure_2_5 / 2_6 so
# §2.4.1 shares their look. One colour (#EEEEEE) for every chrome text element
# (tick numbers, the in-plot x/y labels, the suptitle/footer bands); SPINE off so
# the gray library CELL BORDER (NEUTRAL_COLOR, also #EEEEEE) is the single panel
# frame instead of a competing inner spine box. In-plot vector/component labels
# keep their semantic colours (orange a, etc.) — those reference style.PRIMARY_*
# directly, not the chrome constants.
_STATIC_CHROME = {
    "TICK_LABEL_COLOR": "#EEEEEE",
    # Header/footer band text bone-white (#EEEEEE) so every figure's title reads
    # at the same bright weight across the montage.
    "SUPTITLE_COLOR": "#EEEEEE",
    # Origin crosshair ("spine") bone-white so §2.4.1's axis lines read at the
    # same bright weight as §2.4.2 — the axhline/axvline in setup_vector_axes
    # take their color from SPINE_COLOR. Mirrors gen_figure_242's _STATIC_CHROME
    # so the two figures' spines are common/identical.
    "SPINE_COLOR": "#EEEEEE",
    "DEFAULT_SPINE_LINEWIDTH": 2.0,
    # FRAME MODEL: the library cell border (#EEEEEE, DEFAULT_FRAME_LINEWIDTH=2.0)
    # is the single frame around EVERY cell — data panels AND the header/footer
    # bands — matching gen_figure_2_5 / 2_6. Per-axes spines are OFF
    # (show_border=False in _common_panel_kwargs); show_cell_borders=True in
    # render() draws the boxes.
    # Tight PHYSICAL-INCH spacing so the plots fill their cells; the cell border
    # then sits close to the data. Mirrors gen_figure_242. ROW gutter is
    # near-zero so the header/footer bands hug the panel grid (no floating gap
    # — the "undo the gutters" pass); COLUMN gutter keeps a little air between
    # the three side-by-side panels so their tick labels don't crowd.
    "DEFAULT_PAD_INCHES": 0.15,
    "DEFAULT_MARGIN_INCHES": 0.25,
    "DEFAULT_GUTTER_INCHES": 0.10,
    # Zero column gutter so adjacent body panels SHARE one border line instead of
    # each drawing its own a half-gutter apart (which read as a doubled border
    # between cells). Internal PAD keeps the plot content off the shared border.
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.0,
    # x/y axis glyphs and in-plot math labels at the shared 28"-canvas scale
    # (matches §2.4.2). Vectors use the shared bold weight (7.5) — no local
    # override — so 241 and 242 vectors read identically (the prior 4.6 read thin).
    "DEFAULT_AXIS_LABEL_SIZE": 42,
    # In-plot vector / component labels (a, aₓ, aᵧ) match the x/y axis glyphs
    # (axis_label_size = AXIS_LABEL_SIZE − 2 = 40) — they were reading too small.
    "DEFAULT_LABEL_FONT_SIZE": 40,
}


@contextmanager
def _static_chrome():
    orig = {k: getattr(style, k) for k in _STATIC_CHROME}
    try:
        for k, v in _STATIC_CHROME.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


def _panel_projection_onto_axes() -> StaticPanel:
    """Panel 1: dashed white component arrows + droplines."""
    panel = StaticPanel(
        **_common_panel_kwargs(),  # symmetric lim=FOUND_LIM, origin centered
    )

    ax_val, ay_val = A
    component_color = style.NEUTRAL_COLOR

    # Projection "shadow" droplines — same weight as vector a so they read as
    # the cast shadow of the vector onto each axis, not a faint guide.
    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0),
                        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val),
                        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH))

    panel.add(Vector((ax_val, 0.0), color=component_color, alpha=0.95, zorder=2,
                     linestyle="--"))
    panel.add(Vector((0.0, ay_val), color=component_color, alpha=0.95, zorder=2,
                     linestyle="--"))

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
        **_common_panel_kwargs(),  # symmetric lim=FOUND_LIM, origin centered
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
    panel.add(Annotation("aₓ", xy=(ax_val / 2.0, ay_val + 0.50),
                          ha="center", va="center", **label_kwargs))
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
        **_common_panel_kwargs(),  # symmetric lim=FOUND_LIM, origin centered
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
    """Render the 3-panel §2.4.1 figure with the bone-white unified chrome.

    Header convention matches gen_figure_2_5 / 2_6: the figure NUMBER + NAME sit
    on ONE line in a single top band ("Figure 2.4.1 - Basic Vector Projection"),
    folding in what used to be a separate giant bottom "Figure 2.4.1" footer band
    and freeing that vertical space for the plots. The per-panel descriptive names
    ("Projection Along Each Axis", …) stay as FOOTERS beneath each plot. The whole
    figure renders under _static_chrome so the chrome text is one bone-white tone
    and the gray cell border is each panel's single (now visible) frame.
    """
    panel_footers = [
        "Projection Along Each Axis",
        "Reconstruction in Any Order",
        "Independent Components",
    ]
    # Single header band: figure NUMBER + NAME on one line (the 2.5 / 2.6
    # convention). The old standalone bottom footer band is folded in here.
    header_row = [
        SuptitlePanel("Figure 2.4.1 - Basic Vector Projection",
                       units=(3, 1))
    ]
    # Per-panel descriptive footers — same font as the header band (SuptitlePanel
    # default = SUPTITLE_FONT_SIZE) so the bottom labels read at the same weight
    # and size as the title. auto_shrink (width + height) keeps the longest name
    # ("Reconstruction in Any Order") inside its 1-unit cell.
    footer_names_row = [
        SuptitlePanel(name, units=(1, 1))
        for name in panel_footers
    ]
    with _static_chrome():
        fig = Figure.compose(
            rows=[
                header_row,
                [
                    _panel_projection_onto_axes(),
                    _panel_tip_to_tail(),
                    _panel_perpendicular(),
                ],
                footer_names_row,
            ],
            # Footer band height == header band (HEADER_BAND_INCHES = 2.0" with
            # unit_height 10" → 0.20 row units) so top and bottom bands match.
            row_heights=[0.28, 1.0, 0.20],
            unit_inches=style.SHARED_UNIT_INCHES,
            header_band_inches=style.HEADER_BAND_INCHES,
            show_cell_borders=True,
            frame_inset=True,
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

# Panel c — square inscribed in the unit circle. The tip starts at π/4 and
# walks the four corners π/4 → 3π/4 → 5π/4 → 7π/4 → π/4. Each leg holds one
# component constant (top/bottom edges hold y; left/right edges hold x) so
# the eye sees component independence directly.
#
# Frame budget: 24 frames per sweep × 4 sweeps = 96 — LCM-matches panel a.
# Endpoints are excluded from the interpolation so corner N is owned by
# sweep N's frame 0 (no double-rendered corner between consecutive sweeps).
_ORTHO_CORNER_ANGLES = [math.pi / 4.0, 3 * math.pi / 4.0,
                        5 * math.pi / 4.0, 7 * math.pi / 4.0]
_ORTHO_CORNERS = [(math.cos(a), math.sin(a)) for a in _ORTHO_CORNER_ANGLES]
_ORTHO_NUM_SWEEPS = 4
_ORTHO_FRAMES_PER_SWEEP = 24
_ORTHO_NUM_FRAMES = _ORTHO_NUM_SWEEPS * _ORTHO_FRAMES_PER_SWEEP  # 96
_ORTHO_LIM = (-1.5, 1.5)  # unit-circle envelope + label headroom


def _ortho_tip(frame_idx: int) -> tuple[float, float]:
    sweep = frame_idx // _ORTHO_FRAMES_PER_SWEEP
    inner = frame_idx % _ORTHO_FRAMES_PER_SWEEP
    t = inner / _ORTHO_FRAMES_PER_SWEEP  # [0, 1) — endpoint excluded
    sx, sy = _ORTHO_CORNERS[sweep]
    ex, ey = _ORTHO_CORNERS[(sweep + 1) % _ORTHO_NUM_SWEEPS]
    return ((1.0 - t) * sx + t * ex, (1.0 - t) * sy + t * ey)


def _ortho_frame(frame_idx: int) -> list:
    vx, vy = _ortho_tip(frame_idx)
    return _overlay_frame_for(vx, vy)


# Panel-a animation parameters.
#   STEP_RAD       : angular step per frame (π/12 → 24 angles per revolution)
#   FRAMES_PER_REV : 24, derived from STEP_RAD
#   NUM_REVS       : 2 — first grows mag 0 → MAX_MAG, second shrinks back to 0
#   MIN_MAG/MAX_MAG: triangle-wave floor and apex magnitudes
#   LIM            : panel-a axis limit, tighter than FOUND_LIM=5 so the
#                    sweeping tip and live droplines fill the cell.
#   INTERVAL_MS    : master-clock interval; panel a is first in the row so
#                    this drives panels b and c via the shared FuncAnimation.
_PROJECTION_STEP_RAD = math.pi / 24.0
_PROJECTION_FRAMES_PER_REV = 48
_PROJECTION_NUM_REVS = 2
_PROJECTION_NUM_FRAMES = _PROJECTION_FRAMES_PER_REV * _PROJECTION_NUM_REVS
_PROJECTION_MIN_MAG = 2.0
_PROJECTION_MAX_MAG = 3.0
_PROJECTION_LIM = (-4.0, 4.0)
_PROJECTION_INTERVAL_MS = 175


def _projection_mag(frame_idx: int) -> float:
    """Triangle wave: MIN_MAG → MAX_MAG over rev 1, MAX_MAG → MIN_MAG over rev 2."""
    last = _PROJECTION_FRAMES_PER_REV - 1  # apex at end of first rev
    span = _PROJECTION_MAX_MAG - _PROJECTION_MIN_MAG
    if frame_idx <= last:
        fraction = frame_idx / last
    else:
        fraction = (_PROJECTION_NUM_FRAMES - 1 - frame_idx) / last
    return _PROJECTION_MIN_MAG + span * fraction


def _overlay_frame_for(vx: float, vy: float) -> list:
    """Frame contents for ANY vector with live projection overlay.

    Renders the vector itself plus droplines + component arrows + a_x/a_y
    labels positioned outside the parallelogram. Shared by panels a (triangle-
    wave sweep) and c (unit-circle square sweep) so the overlay convention
    stays consistent across panels.
    """
    mag = math.hypot(vx, vy)
    primary = style.PRIMARY_COLOR
    neutral = style.NEUTRAL_COLOR
    label_kw = dict(
        color=primary,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )

    frame: list = []
    # Droplines + component arrows + component labels only render when the
    # vector has meaningful length — at mag≈0 they collapse to the origin.
    if mag > 1e-3:
        frame.append(Dropline(start=(vx, vy), end=(vx, 0.0)))
        frame.append(Dropline(start=(vx, vy), end=(0.0, vy)))
        frame.append(Vector((vx, 0.0), color=neutral, alpha=0.95,
                            linestyle="--", zorder=2))
        frame.append(Vector((0.0, vy), color=neutral, alpha=0.95,
                            linestyle="--", zorder=2))
        # a_x sits outside the x-axis on the side opposite the tip's y.
        ax_label_y, ax_label_va = (-0.45, "top") if vy >= 0 else (0.45, "bottom")
        frame.append(Annotation("aₓ", xy=(vx / 2.0, ax_label_y),
                                ha="center", va=ax_label_va, **label_kw))
        # a_y sits outside the y-axis on the side opposite the tip's x.
        ay_label_x, ay_label_ha = (-0.35, "right") if vx >= 0 else (0.35, "left")
        frame.append(Annotation("aᵧ", xy=(ay_label_x, vy / 2.0),
                                ha=ay_label_ha, va="center", **label_kw))

    frame.append(Vector(
        (vx, vy),
        color=primary, label="a",
        label_offset=_LABEL_OFFSET,
        alpha=1.0, zorder=3,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
    ))
    return frame


def _projection_frame(frame_idx: int) -> list:
    """Panel a frame: triangle-wave sweep mapped through the shared overlay."""
    mag = _projection_mag(frame_idx)
    theta = (frame_idx % _PROJECTION_FRAMES_PER_REV) * _PROJECTION_STEP_RAD
    return _overlay_frame_for(mag * math.cos(theta), mag * math.sin(theta))


def _notebook_projection_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_projection_frame,
        num_frames=_PROJECTION_NUM_FRAMES,
        interval_ms=_PROJECTION_INTERVAL_MS,
        repeat=True,
        title="Projection Along Each Axis",
        lim=_PROJECTION_LIM,
        axis_style="line",
        # In-plot italic "x"/"y" near the axis ends — on the plot, off the
        # padding. A touch smaller than the library default.
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        show_border=True,
        show_ticks=True,
        show_grid=True,
    )


def _reconstruction_frames_for(vec: tuple[float, float]) -> list[list]:
    """Cumulative 5-beat reconstruction for a given vector tip.

    Each beat ADDS to the previous (no reset) — the eye watches the
    parallelogram assemble itself one edge at a time. Per-beat frame holds
    come from `_RECON_BEAT_FRAMES`. Labels flip placement based on the
    tip's sign so they sit outside the parallelogram in every quadrant.
    """
    vx, vy = vec
    primary = style.PRIMARY_COLOR
    neutral = style.NEUTRAL_COLOR

    def x_leg_bottom():
        return Vector((vx, 0.0), color=neutral, alpha=0.95,
                      linestyle="--", zorder=2)

    def y_leg_right():
        return Vector((0.0, vy), origin=(vx, 0.0),
                      color=neutral, alpha=0.95, linestyle="--", zorder=2)

    def x_leg_top():
        return Vector((vx, 0.0), origin=(0.0, vy),
                      color=neutral, alpha=0.95, linestyle="--", zorder=2)

    def y_leg_left():
        return Vector((0.0, vy), color=neutral, alpha=0.95,
                      linestyle="--", zorder=2)

    def a_bold():
        return Vector((vx, vy),
                      color=primary, label="a",
                      label_offset=_LABEL_OFFSET,
                      alpha=1.0, zorder=4,
                      linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)

    label_kw = dict(
        color=primary,
        fontweight="bold",
        fontsize=style.DEFAULT_LABEL_FONT_SIZE,
    )
    # Sign-aware label placement keeps text outside the parallelogram.
    bx_y, bx_va = (-0.45, "top") if vy >= 0 else (0.45, "bottom")
    ry_x, ry_ha = (vx + 0.35, "left") if vx >= 0 else (vx - 0.35, "right")
    ly_x, ly_ha = (-0.35, "right") if vx >= 0 else (0.35, "left")
    tx_y, tx_va = (vy + 0.45, "bottom") if vy >= 0 else (vy - 0.45, "top")

    def label_bottom_x():
        return Annotation("aₓ", xy=(vx / 2.0, bx_y),
                          ha="center", va=bx_va, **label_kw)

    def label_right_y():
        return Annotation("aᵧ", xy=(ry_x, vy / 2.0),
                          ha=ry_ha, va="center", **label_kw)

    def label_left_y():
        return Annotation("aᵧ", xy=(ly_x, vy / 2.0),
                          ha=ly_ha, va="center", **label_kw)

    def label_top_x():
        return Annotation("aₓ", xy=(vx / 2.0, tx_y),
                          ha="center", va=tx_va, **label_kw)

    def beat_contents(beat_idx: int) -> list:
        """Fresh plottables for cumulative beat 0..4."""
        items = [a_bold()]
        if beat_idx >= 1:
            items += [x_leg_bottom(), label_bottom_x()]
        if beat_idx >= 2:
            items += [y_leg_right(), label_right_y()]
        if beat_idx >= 3:
            items += [y_leg_left(), label_left_y()]
        if beat_idx >= 4:
            items += [x_leg_top(), label_top_x()]
        return items

    frames: list[list] = []
    for beat_idx, n_hold in enumerate(_RECON_BEAT_FRAMES):
        for _ in range(n_hold):
            frames.append(beat_contents(beat_idx))
    return frames


def _notebook_reconstruction_dynamic() -> DynamicPanel:
    all_frames: list[list] = []
    for theta in _RECON_ANGLES:
        all_frames.extend(_reconstruction_frames_for(_vector_at_angle(theta)))

    return DynamicPanel(
        frames=all_frames,
        interval_ms=1100,  # ignored — master clock uses panel a's 175ms
        repeat=True,
        title="Reconstruction in Any Order",
        **_common_panel_kwargs(),
    )


def _notebook_orthogonality_dynamic() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_ortho_frame,
        num_frames=_ORTHO_NUM_FRAMES,
        interval_ms=1100,  # ignored — master clock uses panel a's 175ms
        repeat=True,
        title="Independent Components",
        lim=_ORTHO_LIM,
        axis_style="line",
        # In-plot italic "x"/"y" near the axis ends — on the plot, off the
        # padding. A touch smaller than the library default.
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        show_border=True,
        show_ticks=True,
        show_grid=True,
    )


def build_notebook_figure(debug: bool = False) -> Figure:
    """Build §2.4 Figure 1 as a 1×3 mixed Figure for dsp.ipynb cell 01.

    Layout mirrors gen_figure_1_stft_vs_cwt's notebook structure: SuptitlePanel header row
    + 3-panel body + SuptitlePanel footer row, all sized via the shared
    notebook compact unit (2.5"). The wrapping `show()` enters
    `nb_compact_style()` so font / gutter / margin overrides apply during
    construction and render — visual tone matches every other notebook
    figure in the library.
    """
    suptitle_row = [SuptitlePanel("Basic Vector Projection", units=(3, 1))]
    subtitle_row = [
        SuptitlePanel("Figure 2.4.1.a", units=(1, 1),
                       font_size=style.DEFAULT_SUBTITLE_FONT_SIZE),
        SuptitlePanel("Figure 2.4.1.b", units=(1, 1),
                       font_size=style.DEFAULT_SUBTITLE_FONT_SIZE),
        SuptitlePanel("Figure 2.4.1.c", units=(1, 1),
                       font_size=style.DEFAULT_SUBTITLE_FONT_SIZE),
    ]
    footer_row = [SuptitlePanel("Figure 2.4.1", units=(3, 1))]
    return Figure.compose(
        rows=[
            suptitle_row,
            [
                _notebook_projection_dynamic(),
                _notebook_reconstruction_dynamic(),
                _notebook_orthogonality_dynamic(),
            ],
            subtitle_row,
            footer_row,
        ],
        row_heights=[0.25, 1.0, 0.12, 0.25],
        dpi=80,
        unit_inches=2.5,
        unit_height_inches=2.5,
        show_cell_borders=True,
        hold_ticks=0,
        debug_guides=debug,
    )


def show(debug: bool = False) -> Figure:
    """Build, render, and display §2.4 Figure 1 in a Jupyter cell.

    Mirrors gen_figure_1_stft_vs_cwt.show_hero — enters `nb_compact_style()` so the figure
    is constructed AND rendered under the notebook compact profile,
    constrains widget width to 75% of the cell, and suppresses the ipympl
    widget chrome ("Figure N" header + toolbar) so only the in-figure
    SuptitlePanel footer carries the figure number.
    """
    import matplotlib.pyplot as plt
    with nb_compact_style():
        fig = build_notebook_figure(debug=debug)
        fig._display_width = "75%"
        fig.render()
    canvas = fig._mpl_fig.canvas
    for attr in ("header_visible", "toolbar_visible", "footer_visible"):
        try:
            setattr(canvas, attr, False)
        except Exception:
            pass
    try:
        canvas.manager.set_window_title("")
    except Exception:
        pass
    plt.show()
    return fig
