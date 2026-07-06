"""figure_2_4_2 — §2.4.2 Projection onto Another Vector (4 synced panels).

Four panels driven by ONE shared sweep clock so they animate in lockstep (the
figure's master FuncAnimation ticks every DynamicPanel from the same global
frame index). The sweep runs TWO revolutions of A around B; B is fixed at 45°.
A starts collinear with B; rev 1 shrinks its magnitude 3 -> 2 over a full turn
and rev 2 grows it 2 -> 3 back. The sweep pauses ~1s at every 45° so the
projection lands cleanly at each. Then it repeats.

The figure is split into two conceptual halves, each a 2-panel column group:

  COSINE FORM (A, B) — the relative angle controls the size of the projection:
    - Panel A: a, b and the angle θ between them.
    - Panel B: a, b plus the live white projection shadow of a onto b's line.

  COMPONENT FORM (C, D) — the angle is already baked into the components:
    - Panel C: a, b with their x/y components drawn (the staircase view).
    - Panel D: the SAME components abstracted into a "stem" panel — each
      component (aₓ, aᵧ, bₓ, bᵧ) drawn as a colored arrow whose height is its
      value, so the bars read with the same arrow vocabulary as A/B/C.

Each panel carries a formula FOOTER box below it (symbolic for A/C, live
substituted arithmetic for B/D), the two halves carry a spanning group header,
and the figure carries a suptitle above and a figure number below.
"""
from __future__ import annotations

import math
import os

import numpy as np

from contextlib import contextmanager

from .. import (
    AccumulatorStrip,
    Annotation,
    CompositePanel,
    Dropline,
    DynamicPanel,
    DynamicTextPanel,
    Figure,
    Line,
    RichText,
    StaticPanel,
    StemArrows,
    SuptitlePanel,
    TextPanel,
    TimeSeriesPanel,
    Vector,
    VectorComponents,
    nb_compact_style,
    style,
)


_LABEL_OFFSET = 0.40
_UNIT_INCHES = 4.2
_LIM = (-4.0, 4.0)


# Shared sweep parameters — ONE clock for all four panels. STEP_RAD=pi/24 ->
# 48 angles/rev; TWO revolutions so the whole loop is 2 turns of A around B.
_SWEEP_STEP_RAD = math.pi / 24.0
_SWEEP_FRAMES_PER_REV = 48
_SWEEP_NUM_REVS = 2
_SWEEP_NUM_FRAMES = _SWEEP_FRAMES_PER_REV * _SWEEP_NUM_REVS  # 96 logical
_SWEEP_MIN_MAG = 2.0
_SWEEP_MAX_MAG = 3.0
_SWEEP_INTERVAL_MS = 175

# Magnitude triangle wave over the full 2-rev loop: rev 1 shrinks 3 -> 2, rev 2
# grows 2 -> 3. The period equals the whole sweep.
_SWEEP_MAG_PERIOD = 2 * _SWEEP_FRAMES_PER_REV  # 96 logical frames

# B — fixed reference at 45° (pi/4) for the whole sweep. Off-axis so b has BOTH
# an x- and a y-component; A starts collinear with B at the same angle.
_B_SWEEP_MAG = 3.0
_B_SWEEP_ANGLE = math.pi / 4.0
_B_SWEEP = (_B_SWEEP_MAG * math.cos(_B_SWEEP_ANGLE),
            _B_SWEEP_MAG * math.sin(_B_SWEEP_ANGLE))

_A_START_ANGLE = math.pi / 4.0


# Pause at every 45° so the projection lands cleanly at 0, 45, 90, …, 315.
_SWEEP_PAUSE_OFFSETS = (0, 6, 12, 18, 24, 30, 36, 42)
_SWEEP_PAUSE_HOLD_FRAMES = 6  # 6 extra frames * 175ms = ~1s hold


# Geometry knobs for the angle / projection markers.
_ANGLE_ARC_RADIUS = 0.55
_ANGLE_ARC_SAMPLES = 32
_RIGHT_ANGLE_SIZE = 0.42
_PERP_TOL_DEG = 1.0
_ANGLE_LABEL_RADIUS = 1.15
_ANGLE_LABEL_MIN_DEG = 5.0
# Θ now lives only on panel B (the projection panel) and reads as the driver of
# the projection size, so it runs large rather than as a small in-plot tick.
# Offset added to the (runtime, nb-overridable) axis-label size at draw time.
_THETA_FONT_OFFSET = 0
_ANGLE_LABEL_ACUTE_DEG = 32.0
_ANGLE_LABEL_OUTSIDE_MARGIN_DEG = 13.0
_ANGLE_LABEL_OUTSIDE_RADIUS = 1.5
# Near-collinear a & b: the white projection shadow stacks on the bold vectors
# and reads as a "chewed" arrow (it's degenerate there — equals a). Drop it.
_PROJECTION_MIN_DEG = 8.0

# Stem panel D: TWO slots — an "x" slot and a "y" slot. At each slot the two
# multiplicands of that term sit side by side (aₓ next to bₓ; aᵧ next to bᵧ) so
# the pairs the dot product multiplies read at a glance. a-family on the left of
# each slot, b-family on the right.
_PAIR_X_SLOT = -1.6
_PAIR_Y_SLOT = 1.6
_PAIR_LABEL_DX = 0.5   # a-label sits left of the slot, b-label right
_PAIR_LABEL_DY = 0.32  # label clears the arrow tip
_SLOT_TICK_HALF = 0.16
_SLOT_LABEL_Y = -3.6

# Faded ghost outline for the decomposed vectors in panel C — muted so the
# components stay the hero, but visible enough to read what's being decomposed.
_GHOST_ALPHA = 0.30

# Footer formula font runs smaller than in-plot labels so the substituted
# arithmetic line fits comfortably inside the small footer cell.
_FOOTER_FONT_SCALE = 0.42

# Numeric readouts use the shared monospace family at regular weight so digits
# read as plain data and column-align. Symbolic identity lines keep default bold.
_NUMBER_FONT: str | None = style.DEFAULT_MONO_FONT_FAMILY
_NUMBER_FONT_WEIGHT = "normal"

# Group-header copy for the two conceptual halves.
_COSINE_HEADER = ("Dot Product Cosine Form\n"
                  "The Relative Angle Controls the Size of the Projection")
_COMPONENT_HEADER = ("Dot Product Component Form\n"
                     "The Angle is Baked into The Components Already")


def _build_sweep_frame_map() -> list[int]:
    """Real frame index -> logical sweep frame, with alignment holds."""
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
    """Triangle wave 3 -> 2 -> 3 on a 2-rev period, repeating across all revs."""
    cycle = logical_frame % _SWEEP_MAG_PERIOD
    last = _SWEEP_FRAMES_PER_REV - 1
    span = _SWEEP_MAX_MAG - _SWEEP_MIN_MAG
    if cycle <= last:
        fraction = cycle / last
    else:
        fraction = (_SWEEP_MAG_PERIOD - 1 - cycle) / last
    return _SWEEP_MAX_MAG - span * fraction


def _ab_for_frame(frame_idx: int) -> tuple[tuple[float, float],
                                            tuple[float, float], int]:
    """Resolve (A, B, logical_frame) for a real frame index on the shared clock."""
    logical = _SWEEP_FRAME_MAP[frame_idx]
    mag = _sweep_mag(logical)
    theta = _A_START_ANGLE + (logical % _SWEEP_FRAMES_PER_REV) * _SWEEP_STEP_RAD
    a = (mag * math.cos(theta), mag * math.sin(theta))
    return a, _B_SWEEP, logical


def _dot(u: tuple[float, float], v: tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Unsigned angle (radians) between a and b; 0 when either is degenerate."""
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    cos_t = max(-1.0, min(1.0, _dot(a, b) / (mag_a * mag_b)))
    return math.acos(cos_t)


def _projection_point(u: tuple[float, float],
                      v: tuple[float, float]) -> tuple[float, float]:
    """Foot of the perpendicular from u's tip onto v's line through origin."""
    v_norm_sq = _dot(v, v)
    if v_norm_sq < 1e-12:
        return (0.0, 0.0)
    scalar = _dot(u, v) / v_norm_sq
    return (scalar * v[0], scalar * v[1])


def _projection_overlay(u: tuple[float, float],
                        v: tuple[float, float],
                        *,
                        component_zorder: int = 2) -> list:
    """Dropline (u's tip -> v's line) + white parallel component on v's line."""
    proj_pt = _projection_point(u, v)
    # Dropline and the white parallel component share the bold vector weight and
    # a "--" dash so they read as the SAME dashed gesture as the projection
    # lands on b's line (matplotlib scales the dash unit with linewidth, so
    # equal linewidth + "--" => identical dash cadence).
    return [
        Dropline(start=u, end=proj_pt,
                 linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH),
        Vector(proj_pt, color="white", alpha=1.0,
               zorder=component_zorder, linestyle="--",
               linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH),
    ]


def _vec_a(vec, *, alpha: float = 1.0, zorder: int = 3) -> Vector:
    return Vector(
        vec, color=style.PRIMARY_COLOR, label="a",
        label_offset=_LABEL_OFFSET, alpha=alpha, zorder=zorder,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
    )


def _vec_b(vec, *, alpha: float = 1.0, zorder: int = 3) -> Vector:
    return Vector(
        vec, color=style.SECONDARY_COLOR, label="b",
        label_offset=_LABEL_OFFSET, alpha=alpha, zorder=zorder,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
    )


def _common_panel_kwargs() -> dict:
    return dict(
        lim=_LIM,
        axis_style="line",
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        show_border=True,
        show_ticks=True,
        show_grid=True,
        fill_cell=True,
    )


def _cosine_panel_kwargs() -> dict:
    """Chrome for the cosine half (A, B) — x/y line axes, grid, and ticks, same
    as the component half so the two halves share a coordinate frame."""
    return dict(
        lim=_LIM,
        axis_style="line",
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        show_border=True,
        show_ticks=True,
        show_grid=True,
        fill_cell=True,
    )


# Angle markers (shared by panels A and B) ==================================


def _signed_angle_delta(theta_from: float, theta_to: float) -> float:
    """Short-way signed delta in (-pi, pi]. Picks +pi (not -pi) for anti-parallel."""
    delta = (theta_to - theta_from + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if delta == -math.pi else delta


def _angle_arc_points(a: tuple[float, float],
                      b: tuple[float, float],
                      radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample an arc of `radius` from A's direction to B's direction, short way."""
    theta_a = math.atan2(a[1], a[0])
    theta_b = math.atan2(b[1], b[0])
    delta = _signed_angle_delta(theta_a, theta_b)
    t = np.linspace(0.0, 1.0, _ANGLE_ARC_SAMPLES)
    angles = theta_a + delta * t
    return radius * np.cos(angles), radius * np.sin(angles)


def _right_angle_marker(a: tuple[float, float],
                        b: tuple[float, float],
                        size: float) -> list:
    """Square right-angle corner at the origin, between A's and B's directions."""
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return []
    leg_a = (a[0] / mag_a * size, a[1] / mag_a * size)
    leg_b = (b[0] / mag_b * size, b[1] / mag_b * size)
    corner = (leg_a[0] + leg_b[0], leg_a[1] + leg_b[1])
    xs = np.array([leg_a[0], corner[0], leg_b[0]])
    ys = np.array([leg_a[1], corner[1], leg_b[1]])
    return [Line(xs, ys, color=style.NEUTRAL_COLOR,
                 linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=4)]


def _angle_marker(a: tuple[float, float], b: tuple[float, float]) -> list:
    """Angle symbol (arc, or right-angle square at a perpendicular crossing)."""
    if math.hypot(*a) < 1e-9 or math.hypot(*b) < 1e-9:
        return []
    angle_deg = math.degrees(_angle_between(a, b))
    if abs(angle_deg - 90.0) < _PERP_TOL_DEG:
        return _right_angle_marker(a, b, _RIGHT_ANGLE_SIZE)
    arc_x, arc_y = _angle_arc_points(a, b, _ANGLE_ARC_RADIUS)
    return [Line(arc_x, arc_y, color=style.NEUTRAL_COLOR,
                 linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=4)]


def _angle_label(a: tuple[float, float], b: tuple[float, float]) -> list:
    """"θ" placed on the A→B bisector, just outside the arc marker."""
    if math.hypot(*a) < 1e-9 or math.hypot(*b) < 1e-9:
        return []
    delta_deg = math.degrees(_angle_between(a, b))
    if delta_deg < _ANGLE_LABEL_MIN_DEG:
        return []
    theta_a = math.atan2(a[1], a[0])
    theta_b = math.atan2(b[1], b[0])
    signed = _signed_angle_delta(theta_a, theta_b)
    mid = theta_a + signed / 2.0

    if delta_deg < _ANGLE_LABEL_ACUTE_DEG:
        half = abs(signed) / 2.0
        label_angle = mid + math.copysign(
            half + math.radians(_ANGLE_LABEL_OUTSIDE_MARGIN_DEG), signed
        )
        radius = _ANGLE_LABEL_OUTSIDE_RADIUS
    else:
        label_angle = mid
        radius = _ANGLE_LABEL_RADIUS

    xy = (radius * math.cos(label_angle), radius * math.sin(label_angle))
    # Bold mathtext lowercase theta — the capital "Θ" glyph read as an O with a
    # thin crossbar; the bold cursive theta has a heavier, unmistakable bar.
    return [Annotation(r"$\boldsymbol{\theta}$", xy=xy, color="white",
                       fontsize=style.DEFAULT_AXIS_LABEL_SIZE + _THETA_FONT_OFFSET,
                       zorder=5)]


# Panel A — cosine form: a, b and the angle between them =====================


def _angle_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [_vec_b(b)]
    if math.hypot(*a) > 1e-3:
        frame.extend(_angle_marker(a, b))
    frame.append(_vec_a(a))
    return frame


def _panel_a() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_angle_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        **_cosine_panel_kwargs(),
    )


# Panel B — cosine form: the projection shadow ==============================


def _projection_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [_vec_b(b)]
    if math.hypot(*a) > 1e-3:
        frame.extend(_angle_marker(a, b))
        frame.extend(_angle_label(a, b))
        if math.degrees(_angle_between(a, b)) >= _PROJECTION_MIN_DEG:
            frame.extend(_projection_overlay(a, b, component_zorder=4))
    frame.append(_vec_a(a))
    return frame


def _panel_b() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_projection_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        **_cosine_panel_kwargs(),
    )


# Panel C — component form: the x/y component staircase ======================


def _components_overlay(a: tuple[float, float],
                        b: tuple[float, float]) -> list:
    """x/y components of A and B stacked tip-to-tail (staircase), un-muted."""
    return [
        VectorComponents(
            b, from_origin=False, show_droplines=True,
            component_color=style.SECONDARY_COLOR,
            linewidth=style.DEFAULT_VECTOR_LINEWIDTH, alpha=1.0, zorder=4),
        VectorComponents(
            a, from_origin=False, show_droplines=True,
            component_color=style.PRIMARY_COLOR,
            linewidth=style.DEFAULT_VECTOR_LINEWIDTH, alpha=1.0, zorder=5),
    ]


def _component_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    frame: list = [
        _vec_b(b, alpha=_GHOST_ALPHA),
        _vec_a(a, alpha=_GHOST_ALPHA),
    ]
    if math.hypot(*a) > 1e-3:
        frame.extend(_components_overlay(a, b))
    return frame


def _panel_c() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_component_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        **_common_panel_kwargs(),
    )


# Panel D — component form: the components as a stem of arrows ===============


def _slot_chrome() -> list:
    """The horizontal baseline plus an 'x' and a 'y' tick — the two slots that
    hold each multiplied pair. Drawn fresh each frame (DynamicPanel clears)."""
    chrome: list = [
        Line(np.array([-3.6, 3.6]), np.array([0.0, 0.0]),
             color=style.NEUTRAL_COLOR,
             linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=1),
    ]
    for slot, txt in ((_PAIR_X_SLOT, "x"), (_PAIR_Y_SLOT, "y")):
        chrome.append(Line(
            np.array([slot, slot]),
            np.array([-_SLOT_TICK_HALF, _SLOT_TICK_HALF]),
            color=style.NEUTRAL_COLOR,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH, zorder=1))
        chrome.append(Annotation(
            txt, xy=(slot, _SLOT_LABEL_Y), color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE - 2, fontweight="bold",
            fontfamily=style.DEFAULT_FONT_FAMILY, zorder=5))
    return chrome


def _mid_label(text: str, x: float, value: float, color: str) -> Annotation:
    """Component label set beside the MIDDLE of its arrow (value / 2), so it
    rides alongside the component it names rather than chasing the tip. a-labels
    sit to the arrow's left, b-labels to its right."""
    # Match the a / b vector labels exactly: DEFAULT_LABEL_FONT_SIZE, bold, and
    # the default font family (no Ubuntu override) — Vector.draw uses the same.
    return Annotation(text, xy=(x, value / 2.0), color=color,
                      fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                      fontweight="bold", zorder=6)


def _overlap_pair(a_val: float, b_val: float, x_center: float,
                  a_label: str, b_label: str) -> list:
    """a- and b-components of one axis drawn OVERLAPPING at a single x-slot —
    the two multiplicands of one dot-product term. The shorter arrow draws in
    front so it stays visible against the taller one. Labels flank the slot
    (a on the left, b on the right) so they never collide."""
    a_short = abs(a_val) <= abs(b_val)
    a_z, b_z = (5, 4) if a_short else (4, 5)
    items: list = []
    if abs(a_val) > 1e-9:
        items.append(Vector((0.0, a_val), origin=(x_center, 0.0),
                            color=style.PRIMARY_COLOR,
                            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
                            zorder=a_z))
    if abs(b_val) > 1e-9:
        items.append(Vector((0.0, b_val), origin=(x_center, 0.0),
                            color=style.SECONDARY_COLOR,
                            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
                            zorder=b_z))
    items.append(_mid_label(a_label, x_center - _PAIR_LABEL_DX, a_val,
                            style.PRIMARY_COLOR))
    items.append(_mid_label(b_label, x_center + _PAIR_LABEL_DX, b_val,
                            style.SECONDARY_COLOR))
    return items


def _stem_frame(frame_idx: int) -> list:
    a, b, _ = _ab_for_frame(frame_idx)
    ax_v, ay_v = a
    bx_v, by_v = b
    return (_slot_chrome()
            + _overlap_pair(ax_v, bx_v, _PAIR_X_SLOT, "$a_x$", "$b_x$")
            + _overlap_pair(ay_v, by_v, _PAIR_Y_SLOT, "$a_y$", "$b_y$"))


def _panel_d() -> DynamicPanel:
    return DynamicPanel(
        frame_fn=_stem_frame,
        num_frames=_SWEEP_REAL_NUM_FRAMES,
        interval_ms=_SWEEP_INTERVAL_MS,
        repeat=True,
        lim=_LIM,
        axis_style="none",
        axis_labels=False,
        show_border=True,
        show_ticks=False,
        show_grid=False,
        fill_cell=True,
    )


# Footer formula boxes ======================================================


def _footer_richtext(parts: list, *, numeric: bool) -> RichText:
    fs = style.DEFAULT_LABEL_FONT_SIZE * _FOOTER_FONT_SCALE
    kw = dict(transform="axes", fontsize=fs)
    if numeric:
        kw = dict(kw, fontfamily=_NUMBER_FONT, fontweight=_NUMBER_FONT_WEIGHT)
    return RichText(parts, xy=(0.5, 0.5), **kw)


_A = style.PRIMARY_COLOR
_B = style.SECONDARY_COLOR
_W = "white"


def _footer_a_frame(frame_idx: int) -> list:
    """Symbolic cosine form (static across frames)."""
    return [_footer_richtext(
        [("|a|", _A), ("|b|", _B), (" cos θ", _W), (" = ", _W),
         ("a", _A), (" · ", _W), ("b", _B)],
        numeric=False)]


def _footer_b_frame(frame_idx: int) -> list:
    """Live substituted cosine arithmetic."""
    a, b, _ = _ab_for_frame(frame_idx)
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    theta_deg = math.degrees(_angle_between(a, b))
    dot_val = _dot(a, b)
    if abs(dot_val) < 0.05:
        dot_val = 0.0
    return [_footer_richtext(
        [(f"({mag_a:.1f})", _A), (f"({mag_b:.1f})", _B),
         (f" cos {theta_deg:.0f}°", _W), (" = ", _W),
         (f"{dot_val:+.1f}", _W)],
        numeric=True)]


def _footer_c_frame(frame_idx: int) -> list:
    """Symbolic component form (static across frames)."""
    return [_footer_richtext(
        [("a", _A), (" · ", _W), ("b", _B), (" = ", _W),
         ("aₓ", _A), ("bₓ", _B), (" + ", _W), ("aᵧ", _A), ("bᵧ", _B)],
        numeric=False)]


def _footer_d_frame(frame_idx: int) -> list:
    """Live substituted component arithmetic."""
    a, b, _ = _ab_for_frame(frame_idx)
    ax_v, ay_v = a
    bx_v, by_v = b
    dot_val = _dot(a, b)
    if abs(dot_val) < 0.05:
        dot_val = 0.0
    return [_footer_richtext(
        [(f"({ax_v:+.1f})", _A), (f"({bx_v:+.1f})", _B), (" + ", _W),
         (f"({ay_v:+.1f})", _A), (f"({by_v:+.1f})", _B), (" = ", _W),
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
    suptitle_row = [SuptitlePanel("Projection onto Another Vector - 2D",
                                  units=(4, 1))]
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
    caption_row = [SuptitlePanel("Figure 2.4.2", units=(4, 1))]
    return [suptitle_row, header_row, body_row,
            footer_row, letter_row, caption_row]


# Chrome rows sized as fractions of the body panel's square unit (body = 1.0):
#   suptitle 1/8 · group header 1/4 (two lines) · BODY 1 · equation footer 1/4
#   · A/B/C/D letter row 1/8 · figure-number footer 1/8.
_ROW_HEIGHTS = [0.25, 0.25, 1.0, 0.125, 0.25, 0.25]


def build_notebook_figure(debug: bool = False) -> Figure:
    """Build §2.4.2 as a 4-column mixed Figure for the dsp.ipynb cell."""
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
    """Build, render, and display §2.4.2 in a Jupyter cell."""
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


# ===========================================================================
# STATIC redesign — two stacked 2.5-style rows: [ visual | a·b strip | text ].
# Both rows show the SAME dot product so the accumulator bars line up vertically
# ("two forms, one answer"). The dynamic/notebook path above is unchanged.
# ===========================================================================

# Fixed static pose: clean integers. a·b = 2·3 + 3·1 = 9; component terms
# aₓbₓ = 6, aᵧbᵧ = 3; angle ≈ 38° (a clear acute angle, positive projection).
_A_STATIC = (2.0, 3.0)
_B_STATIC = (3.0, 1.0)

# Strip y-scale: spans the dot value with headroom for the readout below.
_ACCUM_YLIM = (-10.0, 10.0)
# Static gauge y-scale: SYMMETRIC about zero so the zero baseline sits dead-
# center of the square cell and the bar grows up from the middle — figure 2.5's
# accumulator look. ±1.4×value (matches 2.5's RUNNING_YLIM headroom factor), so
# a·b=+9 reaches near the top with the zero line centered.
_ACCUM_GAUGE_YLIM = (-12.6, 12.6)

# Bone-white unified chrome + tight spacing for the static PNG, mirroring
# gen_figure_241 / 2_5 so §2.4.2 shares their look. (To be lifted into the
# library as a shared chrome profile — see the visual-coherence plan.)
_STATIC_CHROME = {
    "TICK_LABEL_COLOR": "#EEEEEE",
    # Header text bone-white (#EEEEEE) so every figure's title reads at the same
    # bright weight across the montage.
    "SUPTITLE_COLOR": "#EEEEEE",
    # FRAME MODEL: the library cell border (#EEEEEE, DEFAULT_FRAME_LINEWIDTH=2.0)
    # frames EVERY cell — visuals, accum strips, text boxes, and the header band
    # — matching gen_figure_2_5 / 2_6. Per-panel spines/ghost-borders are OFF;
    # show_cell_borders=True in render() draws the boxes.
    "SPINE_COLOR": "#EEEEEE",
    "DEFAULT_SPINE_LINEWIDTH": 2.0,
    # Tight pad so the spine box sits close to the cell edge. ROW gutter is
    # near-zero so the header band hugs the panel grid (the "undo the gutters"
    # pass); COLUMN gutter keeps a little air between visual | strip | text.
    "DEFAULT_PAD_INCHES": 0.15,
    "DEFAULT_MARGIN_INCHES": 0.25,
    "DEFAULT_GUTTER_INCHES": 0.10,
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.30,
    # x/y axis glyphs and in-plot math labels sized for the 28" canvas (match
    # §2.4.1 / the shared scale); vectors at the shared bold weight (7.5) so
    # they read identically to §2.4.1 instead of thinner.
    "DEFAULT_AXIS_LABEL_SIZE": 42,
    # In-plot vector labels (a, b) match the x/y axis glyphs (axis_label_size =
    # AXIS_LABEL_SIZE − 2 = 40) — they were reading too small next to x/y.
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


# Header font for the single top band (figure NUMBER + NAME, the 2.5 convention).
# Larger than §2.4.1's 28pt because this figure is ~21" wide (5 units) vs 2.4.1's
# ~12.6" — the title must scale with the canvas to read at the same visual weight.
_HEADER_FONT_SIZE = 46
# RHS text-box copy (scaffold — final prose is the user's).
_COSINE_TEXT = (
    "Dot Product - Cosine Form\n\n"
    "a · b = |a| |b| cos θ. The angle between a and b sets how much of a lies "
    "along b — the projection. A wider angle shrinks it; at 90° it is zero."
)
_COMPONENT_TEXT = (
    "Dot Product - Component Form\n\n"
    "a · b = aₓbₓ + aᵧbᵧ. The angle is already baked into the components — "
    "multiply matching axes and add. Same answer, no angle measured."
)


# Static fill limits — SYMMETRIC so the origin (0,0) sits dead-center of the
# square visual cell with all four quadrants shown equally. a=(2,3), b=(3,1)
# (both in Q1, max coord 3) sit comfortably inside the +4 reach. Equal range on
# both axes keeps equal-aspect geometry honest.
_STATIC_FILL = (-4.0, 4.0)


def _static_visual_kwargs() -> dict:
    """Cosine/component StaticPanel chrome (no fill_cell — StaticPanel-safe).

    units=(1, 1) with the body rows at row_heights=1.0 makes each visual cell a
    true SQUARE on the shared-tile canvas (1·unit wide × 1·unit tall = fig-1's
    text-box tile), the same size as §2.4.1's plots and the accumulator / text
    squares beside it: [ vector | a·b | text ] = three equal tiles per row."""
    return dict(
        units=(1, 1),
        xlim=_STATIC_FILL,
        ylim=_STATIC_FILL,
        axis_style="line",
        axis_labels=True,
        axis_label_size=style.DEFAULT_AXIS_LABEL_SIZE - 2,
        # Spines OFF — the library cell border is the single frame for every
        # cell (the unified 2.5 / 2.6 model). show_cell_borders=True in render().
        show_border=False,
        # No inset tick nubs (matches §2.4.1); the grid carries the scale.
        show_ticks=False,
        show_grid=True,
    )


def _static_cosine_visual() -> StaticPanel:
    """Cosine row: a, b, the angle θ between them, and a's projection onto b."""
    a, b = _A_STATIC, _B_STATIC
    panel = StaticPanel(**_static_visual_kwargs())
    # zorder above b (3) so the white parallel component reads on TOP of b's
    # shaft instead of being hidden under it.
    for item in _projection_overlay(a, b, component_zorder=5):
        panel.add(item)
    for item in _angle_marker(a, b):
        panel.add(item)
    for item in _angle_label(a, b):
        panel.add(item)
    panel.add(_vec_b(b))
    panel.add(_vec_a(a))
    return panel


def _static_component_visual() -> StaticPanel:
    """Component row: a, b ghosted with their x/y components (staircase)."""
    a, b = _A_STATIC, _B_STATIC
    panel = StaticPanel(**_static_visual_kwargs())
    panel.add(_vec_b(b, alpha=_GHOST_ALPHA))
    panel.add(_vec_a(a, alpha=_GHOST_ALPHA))
    for item in _components_overlay(a, b):
        panel.add(item)
    return panel


def _accum_strip_panel() -> TimeSeriesPanel:
    """The a·b accumulator strip — identical value in both rows so bars align.

    Figure 2.5's accumulator look: the y-range is SYMMETRIC about zero so the
    baseline sits centered in the square cell and the bar grows up from the
    middle; the stem matches the shared hero linework weight so it reads at the
    same thickness as the data stems / vectors elsewhere."""
    dot = _dot(_A_STATIC, _B_STATIC)
    panel = TimeSeriesPanel(
        units=(1, 1), xticks=[], yticks=[],
        xlim=(-1.0, 1.0), ylim=_ACCUM_GAUGE_YLIM,
        show_xticklabels=False, show_yticklabels=False,
        # Spine OFF — the cell border frames this strip too (unified model).
        show_border=False,
        # No x/y crosshair — the AccumulatorStrip draws its OWN horizontal zero
        # baseline; the default "line" axis would add a stray vertical line at
        # x=0 that the centered bar sits on (reads as an artifact).
        axis_style="none",
    )
    panel.add(AccumulatorStrip(
        dot,
        # Bar centered in the strip cell; the readout drops into the empty well
        # below the baseline (AccumulatorStrip handles the clean placement).
        x=0.0,
        # Crisp bone-white bar at FULL opacity — the a·b RESULT reads as a neutral
        # gauge (not a colored accent), distinct from the orange vectors it scores.
        # Full alpha keeps it solid, not the faint washed-out pill alpha<1 gave.
        color="#EEEEEE", alpha=1.0,
        # Stem at the shared hero weight so the a·b bar reads at the same
        # thickness as the data stems / vectors (figure 2.5's accumulator).
        linewidth=style.DEFAULT_ACCUM_STEM_LINEWIDTH,
        markersize=style.DEFAULT_ACCUM_MARKERSIZE,
        readout_color="#EEEEEE", readout_font_size=34,
        # VERTICAL spine (no grid): a bone-white axis line down the strip center
        # that the bar rides — same color / alpha / weight as the vector panels'
        # axis lines. The +9 readout sits a fixed 1 unit from the origin in the
        # empty well below the bar.
        vertical_spine=True,
        readout_origin_offset=1.0,
        zero_line_color="#EEEEEE",
        zero_line_alpha=style.DEFAULT_AXIS_DECORATION_ALPHA,
        zero_line_width=style.DEFAULT_AXIS_DECORATION_LINEWIDTH,
    ))
    return panel


def _static_text_panel(text: str) -> TextPanel:
    """RHS caption box — Figure 1's _side_text_panel treatment exactly: one
    1-unit square tile (= fig-1's text box), uppercase head + ragged-right body,
    top-left anchored at the shared caption font size, normal weight, no inner
    box (the cell border is the frame). One type system across every figure."""
    return TextPanel(
        text, units=(1, 1),
        font_size=style.DEFAULT_CAPTION_FONT_SIZE,
        min_font_size=24.0,
        color=style.TICK_LABEL_COLOR, fontweight="bold",
        auto_shrink=False, justify=False, top_anchor=True,
        content_margin_frac=0.05,
        show_ghost_border=False, facecolor="none",
        # Promote the leading "DOT PRODUCT - …" line into a header band the height
        # of the figure header band, with the explanation below the divider.
        header_band=True,
    )


def _static_rows() -> list:
    header_row = [
        SuptitlePanel("Figure 2.4.2 - Projection onto Another Vector",
                       units=(3, 1))
    ]
    # Three bordered cells per row: [ visual | a·b strip | text ]. The strip is
    # its OWN cell with its own cell border (own left+right frame) — previously
    # it shared a borderless CompositePanel cell with the text, which left it
    # with no right border. Separating also gives the text its own framed box.
    cosine_row = [
        _static_cosine_visual(),
        _accum_strip_panel(),
        _static_text_panel(_COSINE_TEXT),
    ]
    component_row = [
        _static_component_visual(),
        _accum_strip_panel(),
        _static_text_panel(_COMPONENT_TEXT),
    ]
    return [header_row, cosine_row, component_row]


def render(output_dir: str,
           output_filename: str = "projection_onto_vector/baseline.png") -> str:
    """Render the static §2.4.2 — two stacked rows (cosine / component form),
    each [ visual | a·b strip | text ], both strips showing the same dot product."""
    with _static_chrome():
        fig = Figure.compose(
            rows=_static_rows(),
            row_heights=[0.32, 1.0, 1.0],
            unit_inches=style.SHARED_UNIT_INCHES,
            header_band_inches=style.HEADER_BAND_INCHES,
            show_cell_borders=True,
            frame_inset=True,
        )
        fig.render()
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    fig.close()
    return os.path.abspath(output_path)
