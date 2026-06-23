"""figure_2_4_3 — §2.4.3 The Dot Product in 3D (4 synced panels).

The 3D sequel to §2.4.2, rebuilt to the same 4-panel template. All panels ride
ONE shared sweep clock (the figure's master FuncAnimation ticks every dynamic
panel from the same global frame index), so the 3D scenes, the stem bars, and
the live footer arithmetic move in lockstep.

A sweeps around B exactly as in §2.4.2 — its xy-magnitude triangle-waves
3 -> 2 -> 3 while its angle turns a full revolution — but now A also lifts off
the floor: a z-component grows 0 -> +3 -> 0 over rev 1 and 0 -> -3 -> 0 over
rev 2. B stays FIXED in the xy-plane (z = 0). The sweep pauses ~1s at every 45°.

The pedagogical spine: because B has no z-component, A's height contributes
NOTHING to a · b (a_z · b_z = a_z · 0 = 0). The dot product only "sees" A's
shadow on the xy-floor — and in panel D the b_z stem is always empty, so the
third product term visibly vanishes.

Two conceptual halves, each a 2-panel column group:

  COSINE FORM (A, B) — the relative angle controls the size of the projection:
    - Panel A: a (with its height, floor-shadow and vertical drop) and b.
    - Panel B: drop A to the floor, then project its shadow onto B's line —
      the white parallel component is the score.

  COMPONENT FORM (C, D) — the angle is already baked into the components:
    - Panel C: the x/y/z component staircases of a and b (b has no z leg).
    - Panel D: each product term as a side-by-side pair of "stem" arrows at an
      x / y / z slot (aₓ next to bₓ, …); b_z is empty, so the z-term vanishes.

Each panel carries a formula footer (symbolic for A/C, live substituted
arithmetic for B/D), the halves carry spanning group headers, and the figure
carries a suptitle above and a figure number below.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import (
    Annotation,
    DynamicPanel,
    DynamicPanel3D,
    DynamicTextPanel,
    Figure,
    Line,
    RichText,
    StemArrows,
    SuptitlePanel,
    Vector,
    nb_compact_style,
    style,
)


_LABEL_OFFSET = 0.55
_UNIT_INCHES = 4.2
_LIM_3D = 4.0
_LIM_2D = (-4.0, 4.0)
_VIEW_INIT = (24.0, -58.0)


# Shared sweep parameters — ONE clock for all four panels. Mirrors §2.4.2.
_SWEEP_STEP_RAD = math.pi / 24.0
_SWEEP_FRAMES_PER_REV = 48
_SWEEP_NUM_REVS = 2
_SWEEP_NUM_FRAMES = _SWEEP_FRAMES_PER_REV * _SWEEP_NUM_REVS  # 96 logical
_SWEEP_MIN_MAG = 2.0
_SWEEP_MAX_MAG = 3.0
_SWEEP_INTERVAL_MS = 175

_SWEEP_MAG_PERIOD = 2 * _SWEEP_FRAMES_PER_REV  # 96 logical frames

# Height (z) half-sine per revolution: rev 0 lifts 0 -> +3 -> 0, rev 1 dips
# 0 -> -3 -> 0. Peaks at mid-rev (a 45° pause), so the tallest/lowest A holds.
_Z_AMP = 3.0

# B — FIXED at 45° in the xy-plane (z = 0); A starts collinear with B, z = 0.
_B_SWEEP_MAG = 3.0
_B_SWEEP_ANGLE = math.pi / 4.0
_B_SWEEP = (_B_SWEEP_MAG * math.cos(_B_SWEEP_ANGLE),
            _B_SWEEP_MAG * math.sin(_B_SWEEP_ANGLE),
            0.0)
_A_START_ANGLE = math.pi / 4.0


_SWEEP_PAUSE_OFFSETS = (0, 6, 12, 18, 24, 30, 36, 42)
_SWEEP_PAUSE_HOLD_FRAMES = 6  # ~1s hold at every 45°


# White parallel component degeneracies are IN-PLANE (between A's floor-shadow
# a_xy and B). Drop near-collinear / near-anti-parallel, or when the shadow is
# too short to project meaningfully.
_PROJECTION_MIN_DEG = 8.0
_SHADOW_MIN_LEN = 0.2

# Footer formula font runs smaller than in-plot labels; the 3-term component
# arithmetic is the longest line, so this is sized for it.
_FOOTER_FONT_SCALE = 0.36

_NUMBER_FONT: str | None = "Ubuntu"
_NUMBER_FONT_WEIGHT = "normal"

_MUTED_ALPHA = style.DEFAULT_MUTED_ALPHA

# Faded ghost outline for the decomposed vectors in panel C — fainter than the
# standard muted alpha so the components are unambiguously the hero.
_GHOST_ALPHA = 0.16

# Dash pattern for every projection / component element (tighter than "--").
_PROJ_DASH = (0, (1.5, 1.5))

# Stem panel D: THREE slots — an "x", "y" and "z" slot, one per product term.
# At each slot the two multiplicands of that term sit side by side (aₓ next to
# bₓ; aᵧ next to bᵧ; a_z next to b_z), a-family left, b-family right, so the
# pairs the dot product multiplies read at a glance. b_z is always 0, so the
# z-slot's purple bar is absent — the third product term visibly vanishes.
_PAIR_SLOTS = (-2.5, 0.0, 2.5)         # x-term, y-term, z-term
_PAIR_OFFSET = 0.42
_SLOT_TICK_HALF = 0.16
_SLOT_LABEL_Y = -3.6
_SLOT_BASELINE_HALF = 3.4

# Group-header copy for the two conceptual halves (matches §2.4.2).
_COSINE_HEADER = ("Dot Product Cosine Form\n"
                  "The Relative Angle Controls the Size of the Projection")
_COMPONENT_HEADER = ("Dot Product Component Form\n"
                     "The Angle is Baked into The Components Already")


def _build_sweep_frame_map() -> list[int]:
    """Real frame index -> logical sweep frame, with 45° alignment holds."""
    pause_set = {
        rev * _SWEEP_FRAMES_PER_REV + offset
        for rev in range(_SWEEP_NUM_REVS)
        for offset in _SWEEP_PAUSE_OFFSETS
    }
    mapping: list[int] = []
    for logical in range(_SWEEP_NUM_FRAMES):
        mapping.append(logical)
        if logical in pause_set:
            mapping.extend([logical] * _SWEEP_PAUSE_HOLD_FRAMES)
    return mapping


_SWEEP_FRAME_MAP = _build_sweep_frame_map()
_SWEEP_REAL_NUM_FRAMES = len(_SWEEP_FRAME_MAP)


def _sweep_mag(logical_frame: int) -> float:
    """Triangle wave 3 -> 2 -> 3 on a 2-rev period (A's in-plane magnitude)."""
    cycle = logical_frame % _SWEEP_MAG_PERIOD
    last = _SWEEP_FRAMES_PER_REV - 1
    span = _SWEEP_MAX_MAG - _SWEEP_MIN_MAG
    if cycle <= last:
        fraction = cycle / last
    else:
        fraction = (_SWEEP_MAG_PERIOD - 1 - cycle) / last
    return _SWEEP_MAX_MAG - span * fraction


def _sweep_z(logical_frame: int) -> float:
    """Half-sine height per rev: rev 0 lifts 0->+3->0, rev 1 dips 0->-3->0."""
    rev = (logical_frame // _SWEEP_FRAMES_PER_REV) % _SWEEP_NUM_REVS
    frac = (logical_frame % _SWEEP_FRAMES_PER_REV) / (_SWEEP_FRAMES_PER_REV - 1)
    amp = _Z_AMP * math.sin(math.pi * frac)
    return amp if rev == 0 else -amp


def _ab_for_frame(frame_idx: int) -> tuple[tuple[float, float, float],
                                           tuple[float, float, float], int]:
    """Resolve (A, B, logical_frame) for a real frame index on the shared clock."""
    logical = _SWEEP_FRAME_MAP[frame_idx]
    mag = _sweep_mag(logical)
    theta = _A_START_ANGLE + (logical % _SWEEP_FRAMES_PER_REV) * _SWEEP_STEP_RAD
    a = (mag * math.cos(theta), mag * math.sin(theta), _sweep_z(logical))
    return a, _B_SWEEP, logical


# Geometry helpers (3D) =====================================================


def _has_len(v: tuple[float, float, float]) -> bool:
    return math.hypot(*v) > 1e-3


def _dot3(u, v) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _angle_between3(a, b) -> float:
    """Unsigned 3D angle (radians) between a and b; 0 when either is degenerate."""
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    cos_t = max(-1.0, min(1.0, _dot3(a, b) / (mag_a * mag_b)))
    return math.acos(cos_t)


def _projection_point3(u, v) -> tuple[float, float, float]:
    """Foot of the perpendicular from u's tip onto v's line through the origin."""
    v_norm_sq = _dot3(v, v)
    if v_norm_sq < 1e-12:
        return (0.0, 0.0, 0.0)
    scalar = _dot3(u, v) / v_norm_sq
    return (scalar * v[0], scalar * v[1], scalar * v[2])


def _floor_shadow(a) -> tuple[float, float, float]:
    return (a[0], a[1], 0.0)


def _in_plane_deg(a, b) -> float:
    """Angle (degrees) between A's floor-shadow a_xy and B (both in the plane)."""
    return math.degrees(_angle_between3(_floor_shadow(a), b))


def _vec_a(vec, *, alpha: float = 1.0, zorder: int = 4,
           label: str | None = "a") -> Vector:
    return Vector(
        vec, color=style.PRIMARY_COLOR, label=label,
        label_offset=_LABEL_OFFSET, alpha=alpha, zorder=zorder,
        show_arrowhead=True, show_tip=False,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH + 0.4,
    )


def _vec_b(vec, *, alpha: float = 1.0, zorder: int = 3,
           label: str | None = "b") -> Vector:
    return Vector(
        vec, color=style.SECONDARY_COLOR, label=label,
        label_offset=_LABEL_OFFSET, alpha=alpha, zorder=zorder,
        show_arrowhead=True, show_tip=False,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH + 0.4,
    )


def _drop_to_floor(a, *, alpha: float = 1.0, zorder: int = 2) -> Vector:
    """Vertical dashed dropline from A's tip straight down to the floor."""
    return Vector(
        (0.0, 0.0, -a[2]), origin=a, color=style.NEUTRAL_COLOR,
        linestyle=_PROJ_DASH, alpha=alpha, show_tip=False, zorder=zorder,
    )


def _shadow_vec(a, *, alpha: float, zorder: int = 2, dashed: bool = True) -> Vector:
    """A's floor-shadow vector a_xy = (ax, ay, 0) on the xy-plane."""
    return Vector(
        _floor_shadow(a), color=style.PRIMARY_COLOR,
        linestyle=_PROJ_DASH if dashed else "-", alpha=alpha,
        show_arrowhead=True, show_tip=False, zorder=zorder,
    )


# Axis-label anchor radii (data coords along each spine). The z anchor is pulled
# well in (3.8 -> 2.9) so the vertical z-axis label sits inside the cube instead
# of poking up through the inline panel title at the top of the cell.
_LABEL_ANCHORS = (3.8, 3.4, 2.9)  # (x, y, z)


def _common_3d_kwargs() -> dict:
    return dict(
        lim_3d=_LIM_3D,
        view_init=_VIEW_INIT,
        show_spines=True,
        show_spine_ticks=True,
        show_spine_tick_labels=False,
        label_anchors=_LABEL_ANCHORS,
        spine_alpha=0.9,
        spine_extension=1.0,  # spines reach the cube edge — which, with fill_cell
                              # + box_zoom below, is the cell edge. Longer spines
                              # would just bleed into the neighbor panel.
        # fill_cell expands the Axes3D to the whole gridspec cell (matching the 2D
        # sibling §2.4.2). WITHOUT it the axes is inset and mpl's box_zoom barely
        # grows the cube; WITH it box_zoom actually fills. This is the real fill
        # lever — box_zoom/lim/elevation only matter once fill_cell is on.
        fill_cell=True,
        box_zoom=2.40,   # fills the (now cell-sized) axes. Higher overflows the
                         # cell; lower leaves the legacy centered-cube margin.
        # No per-panel inner border: the figure's bright cell-grid border frames
        # each 3D panel; a second inner border just reads as clutter.
        show_border=False,
    )


# Panel A — cosine form: the vectors (and the angle between them) ============


def _panel_a_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [_vec_b(b)]
    if _has_len(a):
        frame.append(_shadow_vec(a, alpha=_MUTED_ALPHA))
        if abs(a[2]) > 1e-3:
            frame.append(_drop_to_floor(a))
    frame.append(_vec_a(a))
    return frame


def _panel_a() -> DynamicPanel3D:
    return DynamicPanel3D(
        frame_fn=_panel_a_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        title="The Angle Between a and b",
        **_common_3d_kwargs(),
    )


# Panel B — cosine form: drop to the floor, project onto b ===================


def _floor_overlay(a) -> list:
    """Drop A onto the xy-floor: its shadow a_xy + the vertical dropline."""
    overlay: list = []
    if math.hypot(a[0], a[1]) > 1e-3:
        overlay.append(_shadow_vec(a, alpha=1.0))
    if abs(a[2]) > 1e-3:
        overlay.append(_drop_to_floor(a))
    return overlay


def _onto_b_overlay(a, b) -> list:
    """Project the floor-shadow onto B's line: perpendicular + white component."""
    a_xy = _floor_shadow(a)
    foot = _projection_point3(a_xy, b)
    return [
        Vector(
            (foot[0] - a_xy[0], foot[1] - a_xy[1], 0.0), origin=a_xy,
            color=style.NEUTRAL_COLOR, linestyle=_PROJ_DASH,
            show_tip=False, zorder=3),
        Vector(
            foot, color="white", linestyle=_PROJ_DASH, alpha=1.0,
            show_arrowhead=True, show_tip=False, zorder=6),
    ]


def _panel_b_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [
        _vec_b(b, alpha=_MUTED_ALPHA),
        _vec_a(a, alpha=_MUTED_ALPHA),
    ]
    if not _has_len(a):
        return frame
    frame.extend(_floor_overlay(a))
    in_plane = _in_plane_deg(a, b)
    if (math.hypot(a[0], a[1]) >= _SHADOW_MIN_LEN
            and _PROJECTION_MIN_DEG <= in_plane <= 180.0 - _PROJECTION_MIN_DEG):
        frame.extend(_onto_b_overlay(a, b))
    return frame


def _panel_b() -> DynamicPanel3D:
    return DynamicPanel3D(
        frame_fn=_panel_b_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        title="The Shadow Is the Score",
        **_common_3d_kwargs(),
    )


# Panel C — component form: the x/y/z component staircases ===================


def _axis_legs(v, color, *, zorder: int) -> list:
    """Tip-to-tail x, y, z legs that rebuild v, each dashed in `color`.

    Skips a near-zero leg (no degenerate stub). b has no z-component, so its
    staircase is two legs; a's is three whenever it is off the floor.
    """
    vx, vy, vz = v
    legs: list = []
    ox = oy = oz = 0.0
    for delta in ((vx, 0.0, 0.0), (0.0, vy, 0.0), (0.0, 0.0, vz)):
        if math.hypot(*delta) > 1e-9:
            legs.append(Vector(
                delta, origin=(ox, oy, oz), color=color,
                linestyle=_PROJ_DASH, show_tip=False, zorder=zorder))
        ox, oy, oz = ox + delta[0], oy + delta[1], oz + delta[2]
    return legs


def _panel_c_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [
        _vec_b(b, alpha=_GHOST_ALPHA),
        _vec_a(a, alpha=_GHOST_ALPHA),
    ]
    if _has_len(b):
        frame.extend(_axis_legs(b, style.SECONDARY_COLOR, zorder=4))
    if _has_len(a):
        frame.extend(_axis_legs(a, style.PRIMARY_COLOR, zorder=5))
    return frame


def _panel_c() -> DynamicPanel3D:
    return DynamicPanel3D(
        frame_fn=_panel_c_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        title="The x / y / z Components",
        **_common_3d_kwargs(),
    )


# Panel D — component form: each multiplied pair as side-by-side arrows =======


def _slot_chrome() -> list:
    """The horizontal baseline plus an 'x', 'y' and 'z' tick — the three slots
    that hold each multiplied pair. Drawn fresh each frame (DynamicPanel clears)."""
    chrome: list = [
        Line(np.array([-_SLOT_BASELINE_HALF, _SLOT_BASELINE_HALF]),
             np.array([0.0, 0.0]), color=style.NEUTRAL_COLOR,
             linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=1),
    ]
    for slot, txt in zip(_PAIR_SLOTS, ("x", "y", "z")):
        chrome.append(Line(
            np.array([slot, slot]),
            np.array([-_SLOT_TICK_HALF, _SLOT_TICK_HALF]),
            color=style.NEUTRAL_COLOR,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=1))
        chrome.append(Annotation(
            txt, xy=(slot, _SLOT_LABEL_Y), color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE - 2, fontweight="bold",
            fontfamily="Ubuntu", zorder=5))
    return chrome


def _stem_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    ax_v, ay_v, az_v = a
    bx_v, by_v, bz_v = b
    sx, sy, sz = _PAIR_SLOTS
    x_positions = [
        sx - _PAIR_OFFSET, sx + _PAIR_OFFSET,
        sy - _PAIR_OFFSET, sy + _PAIR_OFFSET,
        sz - _PAIR_OFFSET, sz + _PAIR_OFFSET,
    ]
    return _slot_chrome() + [StemArrows(
        [ax_v, bx_v, ay_v, by_v, az_v, bz_v],
        colors=[style.PRIMARY_COLOR, style.SECONDARY_COLOR,
                style.PRIMARY_COLOR, style.SECONDARY_COLOR,
                style.PRIMARY_COLOR, style.SECONDARY_COLOR],
        labels=["$a_x$", "$b_x$", "$a_y$", "$b_y$", "$a_z$", "$b_z$"],
        x_positions=x_positions,
        baseline=0.0,
    )]


def _panel_d() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_stem_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        title="The Pairs You Multiply",
        lim=_LIM_2D,
        axis_style="none",
        axis_labels=False,
        show_border=True,
        show_ticks=False,
        show_grid=False,
    )


# Footer formula boxes ======================================================


_A = style.PRIMARY_COLOR
_B = style.SECONDARY_COLOR
_W = "white"


def _footer_richtext(parts: list, *, numeric: bool) -> RichText:
    fs = style.DEFAULT_LABEL_FONT_SIZE * _FOOTER_FONT_SCALE
    kw = dict(transform="axes", fontsize=fs)
    if numeric:
        kw = dict(kw, fontfamily=_NUMBER_FONT, fontweight=_NUMBER_FONT_WEIGHT)
    return RichText(parts, xy=(0.5, 0.5), **kw)


def _footer_a_frame(frame_idx: int) -> list:
    """Symbolic cosine form (static across frames)."""
    return [_footer_richtext(
        [("|a|", _A), ("|b|", _B), (" cos θ", _W), (" = ", _W),
         ("a", _A), (" · ", _W), ("b", _B)],
        numeric=False)]


def _footer_b_frame(frame_idx: int) -> list:
    """Live substituted cosine arithmetic (true 3D angle)."""
    a, b, _ = _ab_for_frame(frame_idx)
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    theta_deg = math.degrees(_angle_between3(a, b))
    dot_val = _dot3(a, b)
    if abs(dot_val) < 0.05:
        dot_val = 0.0
    return [_footer_richtext(
        [(f"({mag_a:.1f})", _A), (f"({mag_b:.1f})", _B),
         (f" cos {theta_deg:.0f}°", _W), (" = ", _W),
         (f"{dot_val:+.1f}", _W)],
        numeric=True)]


def _footer_c_frame(frame_idx: int) -> list:
    """Symbolic component form — three terms (static across frames)."""
    return [_footer_richtext(
        [("a", _A), (" · ", _W), ("b", _B), (" = ", _W),
         ("$a_x$", _A), ("$b_x$", _B), (" + ", _W),
         ("$a_y$", _A), ("$b_y$", _B), (" + ", _W),
         ("$a_z$", _A), ("$b_z$", _B)],
        numeric=False)]


def _footer_d_frame(frame_idx: int) -> list:
    """Live substituted component arithmetic — three terms (a_z·b_z -> 0)."""
    a, b, _ = _ab_for_frame(frame_idx)
    ax_v, ay_v, az_v = a
    bx_v, by_v, bz_v = b
    dot_val = _dot3(a, b)
    if abs(dot_val) < 0.05:
        dot_val = 0.0
    return [_footer_richtext(
        [(f"({ax_v:+.1f})", _A), (f"({bx_v:+.1f})", _B), (" + ", _W),
         (f"({ay_v:+.1f})", _A), (f"({by_v:+.1f})", _B), (" + ", _W),
         (f"({az_v:+.1f})", _A), (f"({bz_v:+.1f})", _B), (" = ", _W),
         (f"{dot_val:+.1f}", _W)],
        numeric=True)]


def _footer_panel(frame_fn) -> DynamicTextPanel:
    return DynamicTextPanel(
        frame_fn=frame_fn,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        show_border=True,
    )


# Figure assembly ===========================================================


def _rows() -> list:
    suptitle_row = [SuptitlePanel("The Dot Product in 3D", units=(4, 1))]
    header_row = [
        SuptitlePanel(_COSINE_HEADER, units=(2, 1),
                      font_size=style.DEFAULT_SUBTITLE_FONT_SIZE),
        SuptitlePanel(_COMPONENT_HEADER, units=(2, 1),
                      font_size=style.DEFAULT_SUBTITLE_FONT_SIZE),
    ]
    body_row = [_panel_a(), _panel_b(), _panel_c(), _panel_d()]
    footer_row = [
        _footer_panel(_footer_a_frame),
        _footer_panel(_footer_b_frame),
        _footer_panel(_footer_c_frame),
        _footer_panel(_footer_d_frame),
    ]
    letter_row = [
        SuptitlePanel("A", units=(1, 1)),
        SuptitlePanel("B", units=(1, 1)),
        SuptitlePanel("C", units=(1, 1)),
        SuptitlePanel("D", units=(1, 1)),
    ]
    caption_row = [SuptitlePanel("Figure 2.4.3", units=(4, 1))]
    return [suptitle_row, header_row, body_row,
            footer_row, letter_row, caption_row]


_ROW_HEIGHTS = [0.22, 0.30, 1.0, 0.22, 0.18, 0.20]


def build_notebook_figure(debug: bool = False) -> Figure:
    """Build §2.4.3 as a 4-column mixed Figure (three 3D panels + a stem panel)."""
    return Figure.compose(
        rows=_rows(),
        row_heights=_ROW_HEIGHTS,
        dpi=80,
        unit_inches=2.5,
        unit_height_inches=2.5,
        show_cell_borders=True,
        hold_ticks=0,
        debug_guides=debug,
    )


def show(debug: bool = False) -> Figure:
    """Build, render, and display §2.4.3 in a Jupyter cell."""
    import matplotlib.pyplot as plt
    with nb_compact_style():
        fig = build_notebook_figure(debug=debug)
        fig._display_width = "92%"
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


def render(output_dir: str,
           output_filename: str = "dot_product_3d/baseline.png") -> str:
    """Render a static PNG snapshot (a mid-rev frame: A lifted off the floor)."""
    fig = Figure.compose(
        rows=_rows(),
        row_heights=_ROW_HEIGHTS,
        unit_inches=_UNIT_INCHES,
        show_cell_borders=True,
    )
    fig.render()
    for panel in (p for p, *_ in fig.panels if isinstance(p, DynamicPanel)):
        panel.tick(12)
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
