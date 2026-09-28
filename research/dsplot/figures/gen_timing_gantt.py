"""Timing figures - Start Up and Process Loop per-stage gantts (hero's zoom-in).

Sibling of ``gen_timing_hero``: the same borderless dark banner (no title band,
no cell borders - README prose does the captioning) hosting per-stage cascade
gantts. Two standalone README figures:

    render_startup - the Start Up cascade plus a final "Process Loop" leg: the
        first 186 ms deadline window as hero-language unit cells, first cell
        lit (one frame of work), the rest empty headroom.
    render_runtime - the Process Loop cascade alone, one stage per row: the
        startup figure's lit cell exploded.

Each stage row's y label is ONLY its boxed capital letter - a black square
outlined in the stage's module color, a miniature of its flowchart block.
One continuous capital sequence flows across the whole pipeline (Start Up
A–J, Process Loop K–X), the same capitals that sit under the
software-flowchart blocks; stage names live on the flowchart boxes, not here
(``assets/timing/subshader_startup.drawio`` /
``assets/timing/subshader_runtime.drawio``), so gantts and flowcharts
cross-reference directly. The flowcharts skip the transfer-edge letters
(F, M, P - arrows there, not blocks), so the gantt is where those are
spelled out; GPU Warmup is a gantt-only row and carries no letter.
`init:other`'s Setup row is dropped from the fine breakdown entirely - its
residual still rides inside Total, just unlettered and undrawn. gl_draw is
W, shared with the flowchart's separate GPU-side Shader block (both halves
of one shader-draw concept read W).
Letters are statically bound to stage keys - a missing stage leaves a gap;
rows never re-letter.

Both README figures use the same row grammar: cascade, then one solid white
Total bar beneath everything, then (startup only) the deadline-window leg -
row-height cells, first one lit - butt against the Total's end. The cell-echo
Total treatment survives only in the legacy combined ``render``
(timing_gantt.png), kept so previously published URLs keep resolving.

Data source: ``assets/timing/timing_results.csv`` (written by
``research/timing_live.py``). This module only renders. Stage wording mirrors
``utilities.timing_results._PIPE_LABEL`` / ``timing_report`` init names (the
reporting layer's source of truth - keep in sync; hero-precedent duplication
so the figure module stays importable in both dsplot worldviews).
"""
from __future__ import annotations

import csv
import os
from contextlib import contextmanager

from .. import (
    Figure,
    BarPanel,
    Barh,
    Line,
    Annotation,
    Vector,
    style,
)

PANEL_UNITS = (4, 1)

# --- geometry (y is in row units; one row = ROW_IN inches) --------------------
ROW_IN = 0.62                   # vertical pitch per gantt row
BAR_H = 0.75                    # bar thickness in row units - equal to the
                                # letter boxes, with the rest of the pitch as
                                # breathing room between rows
TITLE_HEAD = 1.85               # top headroom when a block name is drawn
HEAD_BARE = 0.75                # top headroom for the bare variant
AXIS_GAP = 0.6                  # last row → axis-line distance (snug)
AXIS_TAIL = 1.55                # rows kept below the axis line (tick labels)
LEGEND_DROP = 1.95              # below-axis legend row center (startup)
LEGEND_TAIL = 2.6               # axis tail when that legend row is present
CELL_PAD_FRAC = 0.07            # per-cell inset, fraction of one cell (hero's)
GUTTER_FRAC = 0.05              # left gutter, fraction of block span - holds
                                # only the boxed row letter (flowchart-boxes
                                # carry the stage names)
RIGHT_FRAC = 1.115              # x head-room past the Total for trailing labels
LEG_RIGHT_EXTRA = 0.1           # extra right head-room for the leg's note
MIN_BAR_FRAC = 0.005            # floor so sub-pixel stages stay visible
AXES_W_IN = 27.7                # composed panel width (28" canvas − banner pads)
                                # - used only to keep echo cells/swatches square

NAME_SIZE = style.DEFAULT_BAR_VALUE_FONT_SIZE      # 32 - block names (legacy tier)
MEASURE_SIZE = 28               # ONE type size: durations, ticks, letters
TICK_SIZE = MEASURE_SIZE
LETTER_SIZE = MEASURE_SIZE
LETTER_BOX_H = BAR_H            # letter box side = bar height - one row height
LETTER_BOX_EDGE = 3.0           # letter box outline weight (the flowchart's
                                # block stroke, in miniature)
LETTER_NUDGE = 0.07             # rows down (y grows downward) - optically
                                # centers capitals in their boxes: va="center"
                                # centers the full font bbox, whose descender
                                # space (unused by capitals) pushes glyphs high
AXIS_LINE_W = 3.5               # x/y axis rules + stub ticks - a step heavier
                                # than the 2.5 event rules

# Init stages in construction order → (label, color). Fine per-module sub-steps
# when the run recorded them, else the coarse module rows - same fallback as
# the hero. `init:cuda` (the first-cupy-call CUDA context creation cost) rides
# right after the audio module, its own gray "Init CUDA" row. Unlike the
# coarse fallback below, the fine breakdown drops `init:other`'s row entirely
# - its ~30 ms glue residual still counts inside Total, it just isn't drawn.
_FINE = [
    ("init:audio_reader", "Open Audio File", style.TERTIARY_COLOR),
    ("init:audio_player", "Audio Output Init", style.TERTIARY_COLOR),
    # Init CUDA rides with the DSP module (the flowchart strokes it DSP
    # purple - the user's v4 lifetime resolution, now applied everywhere).
    ("init:cuda", "Init CUDA", style.SECONDARY_COLOR),
    ("init:dsp_kernels", "Build Wavelet Kernels", style.SECONDARY_COLOR),
    ("init:dsp_fft", "Generate FFT Kernel Bank", style.SECONDARY_COLOR),
    ("init:dsp_upload", "Transfer Kernel Bank → GPU", style.SECONDARY_COLOR),
    # CircularFrameBuffer is np.zeros → CPU RAM, so "Frame Buffer" (the
    # flowchart's wording), not "GPU Buffer".
    ("init:render_buffer", "Allocate Frame Buffer", style.PRIMARY_COLOR),
    ("init:render_glcontext", "Init Graphics Context", style.PRIMARY_COLOR),
    ("init:render_shader", "Compile Shader + Texture", style.PRIMARY_COLOR),
    # prescan + prime fold into one row (see _gantt_data): the pre-scan warms
    # the CWT path while calibrating the color map, prime warms the renderer -
    # one "color map + GPU ready" leg on the diagram.
    ("init:prescan", "Color Map Setup", style.PRIMARY_COLOR),
]
_COARSE = [
    ("init:audio", "Audio Source", style.TERTIARY_COLOR),
    ("init:cuda", "Init CUDA", style.SECONDARY_COLOR),
    ("init:dsp", "DSP Stages", style.SECONDARY_COLOR),
    ("init:renderer", "Renderer", style.PRIMARY_COLOR),
    ("init:prescan", "Intensity Pre-scan", style.PRIMARY_COLOR),
    ("init:prime", "GPU Warmup", style.SPINE_COLOR),
    ("init:other", "Setup", style.SPINE_COLOR),
]
_COARSE_KEYS = {k for k, _, _ in _COARSE}

# Runtime loop stages in execution order → (label, module color). Wording
# mirrors utilities.timing_results._PIPE_LABEL (palette-3 flowchart boxes);
# gl_clear folds into Shader Draw at the reporting layer (sum, nothing lost).
_LOOP = [
    ("audio_read", "Fetch Audio Samples", style.TERTIARY_COLOR),
    ("fft_cpu", "FFT", style.SECONDARY_COLOR),
    ("upload", "Transfer → GPU", style.SECONDARY_COLOR),
    ("multiply", "Freq-Domain Multiply", style.SECONDARY_COLOR),
    ("ifft", "IFFT", style.SECONDARY_COLOR),
    ("download", "Transfer ← GPU", style.SECONDARY_COLOR),
    ("magnitude", "Compute Magnitude", style.SECONDARY_COLOR),
    ("edge_trim", "Discard Edges", style.SECONDARY_COLOR),
    ("hop_center", "Extract New Hop", style.SECONDARY_COLOR),
    ("downsample", "Down-sample", style.SECONDARY_COLOR),
    ("buf_push", "Store Into Frame Buffer", style.PRIMARY_COLOR),
    ("tex_upload", "Upload To Texture", style.PRIMARY_COLOR),
    ("gl_draw", "Shader Draw", style.PRIMARY_COLOR),
    ("gl_swap", "Update Display Buffer", style.PRIMARY_COLOR),
]

# Capital letters, statically bound to stage keys - identical to the letters
# under the flowchart blocks (subshader_startup.drawio /
# subshader_runtime.drawio). One continuous sequence: Start Up A–J, Process
# Loop K–X. GPU Warmup has no letter (gantt-only residual, absent from the
# flowchart); gl_draw is W, shared with the flowchart's separate GPU-side
# Shader block. Coarse-fallback init rows have no entry, so they render
# un-lettered rather than mis-lettered.
_FINE_NUM = {
    "init:audio_reader": "A", "init:audio_player": "B", "init:cuda": "C",
    "init:dsp_kernels": "D", "init:dsp_fft": "E", "init:dsp_upload": "F",
    "init:render_buffer": "G", "init:render_glcontext": "H",
    "init:render_shader": "I", "init:prescan": "J",
}
_LOOP_NUM = {
    "audio_read": "K", "fft_cpu": "L", "upload": "M", "multiply": "N",
    "ifft": "O", "download": "P", "magnitude": "Q", "edge_trim": "R",
    "hop_center": "S", "downsample": "T", "buf_push": "U",
    "tex_upload": "V", "gl_draw": "W", "gl_swap": "X",
}

_LEGEND = [
    ("Audio", style.TERTIARY_COLOR),
    ("DSP", style.SECONDARY_COLOR),
    ("Render", style.PRIMARY_COLOR),
    ("Setup", style.SPINE_COLOR),
]

# Headroom cells of the Process Loop leg - the hero's empty-square look.
EMPTY_FACE, EMPTY_EDGE = "#222222", "#3a3a3a"

# Banner-local layout scale - same save/restore mechanism as the hero: the
# README strip is full-bleed, so the family's 1.5" pads would waste the canvas.
# BG rides along: the README's timing figures sit next to the drawio flowchart
# exports, which render on pure black - the banner matches them (LIFETIME_BG),
# not the family's shared #1A1A1A.
_BANNER_OVERRIDES = {
    "DEFAULT_PAD_INCHES": 0.15,
    "DEFAULT_MARGIN_INCHES": 0.15,
    "DEFAULT_GUTTER_INCHES": 0.3,
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.3,
    "BG_COLOR": "#000000",
}


@contextmanager
def _banner_style():
    orig = {k: getattr(style, k) for k in _BANNER_OVERRIDES}
    try:
        for k, v in _BANNER_OVERRIDES.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _latest_run() -> list[dict]:
    """Rows of the latest detailed run (has render legs), else the last run."""
    path = os.path.join(_repo_root(), "assets", "timing", "timing_results.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ids: list[str] = []
    for r in rows:
        if r["run_id"] not in ids:
            ids.append(r["run_id"])
    for rid in reversed(ids):
        if any(r["stage"] == "gl_swap" for r in rows if r["run_id"] == rid):
            return [r for r in rows if r["run_id"] == rid]
    return [r for r in rows if r["run_id"] == ids[-1]] if ids else []


def _gantt_data() -> dict:
    """Both blocks' segments, derived from the CSV.

    Returns dict with:
        su_segments - [(numeral, label, dur_ms, color)] in construction order
        su_total    - one-time construction total (sum of coarse module rows)
        rt_segments - [(numeral, label, dur_ms, color)] in execution order
        rt_total    - TOTAL mean (one frame of work, the hero's cell)
        deadline    - the frame budget in ms (rt_margin × rt_total), or None
    """
    run = _latest_run()
    if not run:
        raise RuntimeError("No timing rows in assets/timing/timing_results.csv")
    ms = {r["stage"]: float(r["mean_ms"]) for r in run}

    # GPU warmup rides the color-map leg (Color Map Setup): the pre-scan warms
    # the CWT path, prime warms the renderer. Sum, nothing lost - same fold as
    # gl_clear → gl_draw at the reporting layer.
    if "init:prime" in ms:
        ms["init:prescan"] = ms.get("init:prescan", 0.0) + ms.pop("init:prime")

    fine = any(k in ms and ms[k] > 0 for k, _, _ in _FINE
               if k not in _COARSE_KEYS)
    order = _FINE if fine else _COARSE
    su_segments = [(_FINE_NUM.get(key) if fine else None, label, ms[key], color)
                   for key, label, color in order if ms.get(key, 0.0) > 0]
    su_total = sum(ms.get(k, 0.0) for k, _, _ in _COARSE)
    if su_total <= 0:
        su_total = sum(dur for _, _, dur, _ in su_segments)

    loop_ms = dict(ms)
    if "gl_clear" in loop_ms:
        loop_ms["gl_draw"] = loop_ms.get("gl_draw", 0.0) + loop_ms.pop("gl_clear")
    rt_segments = [(_LOOP_NUM.get(key), label, loop_ms[key], color)
                   for key, label, color in _LOOP
                   if loop_ms.get(key, 0.0) > 0]
    total_row = next((r for r in run if r["stage"] == "TOTAL"), None)
    rt_total = (float(total_row["mean_ms"]) if total_row
                else sum(dur for _, _, dur, _ in rt_segments))
    rt_str = total_row["rt_margin"] if total_row else ""
    try:
        deadline = float(rt_str) * rt_total if rt_str not in ("", None) else None
    except ValueError:
        deadline = None
    return dict(su_segments=su_segments, su_total=su_total,
                rt_segments=rt_segments, rt_total=rt_total, deadline=deadline)


def _dur_label(ms_val: float) -> str:
    """Compact duration: seconds past 1 s, whole milliseconds down to 1 ms,
    and a flat "< 1 ms" floor below that."""
    if ms_val >= 1000:
        return f"{ms_val / 1000:.1f} s"
    if ms_val < 1:
        return "< 1 ms"
    return f"{ms_val:.0f} ms"


def _total_label(ms_val: float) -> str:
    """Total rows keep one decimal ("10.2 ms") - the number the README quotes."""
    return f"{ms_val / 1000:.1f} s" if ms_val >= 1000 else f"{ms_val:.1f} ms"


def _label_backing_w(text: str, x_range: float,
                     fontsize: float = MEASURE_SIZE) -> float:
    """Width (in data ms) of the canvas-color strip drawn behind a time
    label so event rules pass behind the text, not through it. Estimated
    from character count (~0.66 em average advance, plus half a character
    of padding); slightly generous is fine - the strip is invisible against
    the background, it only widens the gap it cuts in a rule."""
    return (len(text) + 0.5) * 0.66 * (fontsize / 72.0) / AXES_W_IN * x_range


def _block_panel(title: str, segments, total, ticks_ms, tick_labels, *,
                 cell_ms: float | None = None, echo_colors: bool = False,
                 legend: list | None = None, leg: dict | None = None,
                 events: list | None = None, tail: dict | None = None,
                 span_ms: float | None = None, show_total_value: bool = True,
                 axis_tail: float = AXIS_TAIL, close_label: str | None = "t"):
    """One gantt block: block name, Total row, stage cascade, hero-style axis.

    ``segments`` rows are ``(numeral, name, dur_ms, color)``; a non-None
    numeral IS the row's y label - the same capital letter that sits under
    the matching flowchart block. ``name`` is kept for the data model but not
    drawn; un-lettered rows get no label.

    ``cell_ms`` set → the Total row renders as unit cells of that width (the
    cell-echo); None → one solid white Total bar. ``echo_colors`` colors the
    echo cells by stage module (the hero strip verbatim) instead of white.
    ``legend`` (a list of (name, color)) drops the module color key into the
    block's empty top-right corner.

    ``leg`` set → one extra row after the cascade: the first frame-budget
    window starting where the cascade ends, as hero-language unit cells -
    first cell lit white (one frame of work), the rest empty headroom.
    Expects ``{"label", "work", "n_cells", "note"}``.

    ``events`` - moments in time as the two-tier axis grammar's second tier:
    each ``{"x", "color", "dashed"?, "tag"?, "kind"?}``. kind "rule"
    (default) draws a thin full-height vertical through the field (solid =
    lifecycle boundary, dashed = deadline); kind "tick" draws a short anchor
    tick below the axis instead. A ``tag`` puts the event's value in the
    tick band, small and color-matched - scale ticks stay purely even.
    ``tail`` (``{"work", "period", "repeats"?}``) rides one row below the
    Total bar: one solid frame of work seated at the block's end, repeated
    at each of ``repeats`` following deadline windows - the lifetime
    rhythm. ``span_ms`` overrides the x span (events/tail land beyond the
    block total).
    ``show_total_value=False`` drops the Total bar's trailing text (its value
    lives in an event tag instead).
    """
    n = len(segments)
    span = span_ms if span_ms else total + (leg["n_cells"] * leg["work"] if leg else 0.0)
    gutter = GUTTER_FRAC * span
    if span_ms:
        x_hi = 1.02 * span      # events land inside the span; a hair of run-out
    else:
        x_hi = (RIGHT_FRAC + (LEG_RIGHT_EXTRA if leg else 0.0)) * span
    head = TITLE_HEAD if title else HEAD_BARE
    y_axis = n + (1 if (leg or tail) else 0) + AXIS_GAP
    panel = BarPanel(
        units=PANEL_UNITS,
        xlim=(-gutter, x_hi),
        ylim=(y_axis + axis_tail, -head),   # y grows downward
        xticks=[], xticklabels=[],
        show_xticklabels=False,
        show_border=False,
    )
    label_pad = 0.011 * (x_hi + gutter)
    x_range = x_hi + gutter

    # Phase boundary - a quiet vertical rule where Start Up ends and the
    # Process Loop begins (drawn first, so rows sit on top of it).
    if leg:
        panel.add(Line([total, total], [-0.55, y_axis],
                       color=style.SPINE_COLOR, linewidth=2.0))

    # Event rules - thin verticals through the field, drawn before the bars
    # so a bar paints over a rule inside its own lane (the bar's edge becomes
    # the marker there) while the rule reads clean through empty field. Rules
    # start level with the first row's top edge - never into the headroom.
    # Dashed rules use a long-dash pattern (points): nearly solid, still
    # visibly distinct from the solid lifecycle rules.
    for ev in (events or []):
        if ev.get("kind", "rule") == "rule":
            panel.add(Line([ev["x"], ev["x"]], [-BAR_H / 2.0, y_axis],
                           color=ev["color"], linewidth=2.5,
                           linestyle=(0, (16, 4)) if ev.get("dashed") else "-"))

    # Block name - legacy voice only; the README figures are bare (their
    # prose does the captioning).
    if title:
        panel.add(Annotation(title, (0.0, -1.15), ha="left", va="bottom",
                             color=style.NEUTRAL_COLOR, fontsize=NAME_SIZE,
                             fontweight="bold"))

    # Stage cascade - each stage on its own row, offset by its start time.
    # The y axis carries ONLY the stage's boxed capital letter - a miniature of
    # its flowchart block: black square, module-color outline, white letter.
    # Box and bar share the row's vertical center; the box runs slightly
    # taller so it frames its glyph the way a flowchart block frames its
    # label. Stage names live on the flowchart boxes. Un-lettered rows
    # (Setup) have no flowchart box and get no symbol.
    box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN   # physically square
    start = 0.0
    for i, (num, name, dur, color) in enumerate(segments):
        draw_w = max(dur, MIN_BAR_FRAC * total)
        panel.add(Barh([float(i)], [draw_w], lefts=[start],
                       colors=[color], height=BAR_H, edgewidth=0.0))
        if num:
            panel.add(Barh([float(i)], [box_w], lefts=[-label_pad - box_w],
                           color="#000000", height=LETTER_BOX_H,
                           edgecolor=color, edgewidth=LETTER_BOX_EDGE))
            panel.add(Annotation(num,
                                 (-label_pad - box_w / 2.0,
                                  float(i) + LETTER_NUDGE),
                                 ha="center", va="center", color="#ffffff",
                                 fontsize=LETTER_SIZE, fontweight="bold"))
        dur_text = _dur_label(dur)
        dur_pad = 0.5 * label_pad   # duration labels hug their bar's tail
        # Backing runs from the bar's end, so a rule can't peek through the
        # sliver between bar and text either.
        panel.add(Barh([float(i)],
                       [dur_pad + _label_backing_w(dur_text, x_range)],
                       lefts=[start + draw_w], color=style.BG_COLOR,
                       height=BAR_H, edgewidth=0.0))
        panel.add(Annotation(dur_text, (start + draw_w + dur_pad, float(i)),
                             ha="left", va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))
        start += dur

    # Total row - BELOW the cascade: the whole block as one bar, its trailing
    # duration the only text.
    y_tot = float(n)
    if cell_ms:
        # Physically square cells - the hero's unit, one cell = one frame of
        # work. Height derived from the cell's on-canvas width so the Total
        # row reads as the hero strip, not a striped bar.
        n_cells = max(1, int(round(total / cell_ms)))
        pad = CELL_PAD_FRAC * cell_ms
        cell_h = min(BAR_H, (cell_ms / x_range) * AXES_W_IN / ROW_IN)
        if echo_colors:
            # Hero strip verbatim: each cell takes the color of the stage
            # whose time-span its center falls in.
            bounds, acc = [], 0.0
            for _, _, dur, color in segments:
                acc += dur
                bounds.append((acc, color))
            colors = []
            for k in range(n_cells):
                center = (k + 0.5) * cell_ms
                colors.append(next((c for b, c in bounds if center <= b),
                                   bounds[-1][1]))
        else:
            colors = ["#ffffff"] * n_cells
        panel.add(Barh(
            [y_tot] * n_cells, [cell_ms - 2 * pad] * n_cells,
            lefts=[k * cell_ms + pad for k in range(n_cells)],
            colors=colors, height=cell_h, edgewidth=0.0,
        ))
    else:
        panel.add(Barh([y_tot], [total], color="#ffffff",
                       height=BAR_H, edgewidth=0.0))
    if show_total_value:
        tot_text = _total_label(total)
        panel.add(Barh([y_tot], [_label_backing_w(tot_text, x_range)],
                       lefts=[total + label_pad], color=style.BG_COLOR,
                       height=BAR_H, edgewidth=0.0))
        panel.add(Annotation(tot_text, (total + label_pad, y_tot),
                             ha="left", va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))

    # Lifetime tail - one row below the Total bar (its own lane, so the Total
    # bar still visibly ends at the ready rule): one solid frame of work
    # seated at the rule, repeated at each following deadline window.
    if tail:
        y_tail = y_tot + 1.0
        for k in range(tail.get("repeats", 1) + 1):
            _lifetime_interval_box(panel, total + k * tail["period"],
                                   tail["work"], y_tail, fade=False)

    # Process Loop leg - BELOW the Total, butt against its end: the first
    # frame-budget window. One lit cell of work, then empty headroom cells,
    # trailed by the window's duration; the runtime figure zooms the lit cell.
    if leg:
        y = float(n + 1)
        work, leg_cells = leg["work"], leg["n_cells"]
        pad = CELL_PAD_FRAC * work
        cell_h = BAR_H          # row-height cells - consistent with the bars
        lefts = [total + k * work + pad for k in range(leg_cells)]
        panel.add(Barh([y], [work - 2 * pad], lefts=[lefts[0]],
                       colors=["#ffffff"], height=cell_h, edgewidth=0.0))
        panel.add(Barh([y] * (leg_cells - 1), [work - 2 * pad] * (leg_cells - 1),
                       lefts=lefts[1:], color=EMPTY_FACE, height=cell_h,
                       edgecolor=EMPTY_EDGE, edgewidth=1.2))
        panel.add(Annotation(leg["note"],
                             (total + leg_cells * work + label_pad, y),
                             ha="left", va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))

    # Hero-style axis: one neutral rule with stub ticks, white numbers - plus
    # a matching y axis at t=0, from the first row's top edge down to the
    # axis line (on the runtime figure it doubles as the loop-start marker).
    panel.add(Line([0.0, 0.0], [-BAR_H / 2.0, y_axis],
                   color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
    panel.add(Line([0.0, span], [y_axis, y_axis],
                   color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
    for t, lbl in zip(ticks_ms, tick_labels):
        panel.add(Line([t, t], [y_axis, y_axis + 0.32],
                       color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
        panel.add(Annotation(lbl, (t, y_axis + 0.62), ha="center", va="top",
                             color="#ffffff", fontsize=TICK_SIZE))
    # Closing tick at the axis limit, labeled a bold "t" - the time
    # dimension itself, running on past the figure. ``close_label=None``
    # drops tick and label both (for spans that end snug against a tagged
    # event rule, where the "t" would collide).
    if span_ms and close_label:
        panel.add(Line([span, span], [y_axis, y_axis + 0.32],
                       color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
        panel.add(Annotation(close_label, (span, y_axis + 0.62), ha="center",
                             va="top", color="#ffffff", fontsize=TICK_SIZE,
                             fontweight="bold"))

    # Event tags - in the tick band at the tick type size, color-matched to
    # their rule; "tick"-kind events get a short anchor tick instead of a
    # rule (e.g. the loop end at the Total bar's edge).
    for ev in (events or []):
        if ev.get("kind") == "tick":
            panel.add(Line([ev["x"], ev["x"]],
                           [y_axis, y_axis + LIFETIME_LOOP_END_TICK_LEN],
                           color=ev["color"], linewidth=2.5))
        if ev.get("tag"):
            panel.add(Annotation(ev["tag"], (ev["x"], y_axis + 0.62),
                                 ha="center", va="top", color=ev["color"],
                                 fontsize=TICK_SIZE, fontweight="bold"))

    # Module color key - tucked into the cascade's empty top-right corner.
    # Swatches are physically square (same derivation as the echo cells).
    if legend:
        sw_h = 0.52
        sw_w = sw_h * ROW_IN * x_range / AXES_W_IN
        x0 = 0.965 * total - sw_w
        for j, (name, color) in enumerate(legend):
            y = 1.4 + 1.15 * j
            panel.add(Barh([y], [sw_w], lefts=[x0], colors=[color],
                           height=sw_h, edgewidth=0.0))
            panel.add(Annotation(name, (x0 + sw_w + label_pad, y), ha="left",
                                 va="center", color=style.NEUTRAL_COLOR,
                                 fontsize=MEASURE_SIZE))
    return panel


def _su_ticks(su_total):
    # Bare numbers - the "ms" unit lives on the bold event tags only.
    ticks = [t for t in (0, 250, 500, 750, 1000) if t <= su_total]
    return ticks, [f"{t:g}" for t in ticks]


def _rt_ticks(rt_total, deadline=None):
    # Bare numbers - the "ms" unit lives on the bold event tags only.
    hi = deadline if deadline else rt_total
    ticks = [t for t in (0, 50, 100, 150, 200) if t <= hi]
    return ticks, [f"{t:g}" for t in ticks]


def _compose_one(panel, n_rows, head=HEAD_BARE, tail=AXIS_TAIL) -> Figure:
    rows = head + n_rows + AXIS_GAP + tail
    return Figure.compose(
        rows=[[panel]],
        row_heights=[rows * ROW_IN],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=1.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


def _startup_leg(rt_total, deadline) -> dict | None:
    """The Process Loop leg spec: first deadline window, cell-quantized."""
    if not deadline or not rt_total:
        return None
    return dict(
        work=rt_total,
        n_cells=max(2, int(round(deadline / rt_total))),
        note=f"{deadline:.0f} ms",
    )


def _tail_legend(panel, span, x_range, y):
    """Horizontal line-key centered below the x axis - solid = Startup
    Complete, dashed = Processing Deadlines."""
    sample = 0.048 * span
    gap = 0.008 * span
    item_gap = 0.03 * span
    items = [("-", "Startup Complete"),
             ((0, (16, 4)), "Processing Deadlines")]
    widths = [sample + gap + _label_backing_w(text, x_range)
              for _, text in items]
    x = (span - sum(widths) - item_gap) / 2.0
    for (ls, text), w in zip(items, widths):
        panel.add(Line([x, x + sample], [y, y], color="#ffffff",
                       linewidth=2.5, linestyle=ls))
        panel.add(Annotation(text, (x + sample + gap, y), ha="left",
                             va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))
        x += w + item_gap


def build_startup_figure() -> Figure:
    """Start Up cascade + solid ready rule where construction ends, then the
    lifetime tail: one frame of work repeated at each processing-deadline
    window, closed by a trailing ellipsis (the loop runs on). A line-key
    legend names the solid/dashed rules. Falls back to the bare cascade when
    the run has no deadline."""
    d = _gantt_data()
    su_segments, su_total = d["su_segments"], d["su_total"]
    rt_total, deadline = d["rt_total"], d["deadline"]
    ticks, tick_labels = _su_ticks(su_total)
    if not (deadline and rt_total):
        panel = _block_panel("", su_segments, su_total, ticks, tick_labels)
        return _compose_one(panel, len(su_segments))
    reps = 3
    events = [dict(x=su_total, color="#ffffff", tag=f"{su_total:.0f} ms")]
    events += [dict(x=su_total + k * deadline, color="#ffffff", dashed=True,
                    tag=f"+{deadline:.0f} ms" if k == 1 else None)
               for k in range(1, reps + 1)]
    span = su_total + (reps + 0.77) * deadline    # ~2 s of pipeline lifetime
    panel = _block_panel(
        "", su_segments, su_total, ticks, tick_labels,
        events=events,
        tail=dict(work=rt_total, period=deadline, repeats=reps),
        span_ms=span,
        show_total_value=False,
        axis_tail=LEGEND_TAIL,
    )
    y_axis = len(su_segments) + 1 + AXIS_GAP
    _tail_legend(panel, span, (1.02 + GUTTER_FRAC) * span,
                 y_axis + LEGEND_DROP)
    # The repetition continues past the frame - say so with an ellipsis.
    last_end = su_total + reps * deadline + rt_total
    panel.add(Annotation("⋯", (last_end + 0.012 * span,
                               len(su_segments) + 1.0),
                         ha="left", va="center", color="#ffffff",
                         fontsize=MEASURE_SIZE, fontweight="bold"))
    return _compose_one(panel, len(su_segments) + 1, tail=LEGEND_TAIL)


def build_runtime_figure() -> Figure:
    """Process Loop cascade with the deadline inside its own axis: the y
    axis at t=0 doubles as the loop-start marker (this figure is one window
    of the Start Up tail, zoomed), the dashed deadline rule sits at the
    frame budget, and the Total bar's value is a small tag under its end -
    the empty span between the two is the headroom."""
    d = _gantt_data()
    rt_segments, rt_total = d["rt_segments"], d["rt_total"]
    deadline = d["deadline"]
    ticks, tick_labels = _rt_ticks(rt_total, deadline)
    events = [
        dict(x=rt_total, kind="tick", color="#ffffff",
             tag=_total_label(rt_total)),
    ]
    span = None
    if deadline:
        events.append(dict(x=deadline, color="#ffffff",
                           dashed=True, tag=f"{deadline:.0f} ms"))
        span = RUNTIME_LIFETIME_AXIS_PAD * deadline
    panel = _block_panel("", rt_segments, rt_total,
                         ticks, tick_labels, events=events, span_ms=span,
                         show_total_value=False)
    return _compose_one(panel, len(rt_segments))


def build_figure(echo_colors: bool = False) -> Figure:
    """Legacy combined figure - both blocks stacked (timing_gantt.png)."""
    d = _gantt_data()
    su_segments, su_total = d["su_segments"], d["su_total"]
    rt_segments, rt_total = d["rt_segments"], d["rt_total"]
    su_ticks, su_tick_labels = _su_ticks(su_total)
    rt_ticks, rt_tick_labels = _rt_ticks(rt_total)

    su_panel = _block_panel("Start Up", su_segments, su_total,
                            su_ticks, su_tick_labels,
                            cell_ms=rt_total, echo_colors=echo_colors,
                            legend=_LEGEND)
    rt_panel = _block_panel("Runtime Loop", rt_segments, rt_total,
                            rt_ticks, rt_tick_labels)

    su_rows = TITLE_HEAD + len(su_segments) + AXIS_GAP + AXIS_TAIL
    rt_rows = TITLE_HEAD + len(rt_segments) + AXIS_GAP + AXIS_TAIL
    return Figure.compose(
        rows=[[su_panel], [rt_panel]],
        row_heights=[su_rows * ROW_IN, rt_rows * ROW_IN],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=1.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


# --- "Lifetime" design iteration (v5) -----------------------------------
# User-directed variant of the Start Up figure: same cascade, same banner,
# but the bottom of the block trades the deadline-window leg's un-lettered
# headroom cells for a WHEN-shaped timeline built almost entirely out of
# EVENT MARKERS instead of words: the standard Total bar (geometry only, no
# trailing value text), one white interval box per loop period seated with
# zero inset at each rule (interval 1 full-brightness, interval 2 the same
# box dissolving smoothly left->right), and four events marked visually --
# end-of-Start-Up/loop-start (the one SOLID rule, the epoch boundary),
# real-time playback deadline (a DASHED rule in audio-amber, tying the
# deadline to the audio clock catching up), loop end (a short tick where
# interval 1's own box edge already sits), and the loop period width itself
# (a tiny tick label). All value text ("1.3 s" bar label, bold "186 ms")
# is gone from the data field, demoted to small sparse tick labels below the
# axis; only the cascade letter-boxes and the top-row "Start Up" /
# "Runtime Loop" captions remain as words. The lowest lane seats directly on
# the axis (no dead band -- v2's "seat on axis" fix, reapplied). Rules are
# added to the panel BEFORE the interval boxes, so equal-zorder insertion
# order lets a box paint over a rule within its own lane -- the box's crisp
# edge becomes the marker there, while the rule still reads normally above
# (through the cascade) and below (down to the axis). No title text, a
# smaller arrow mutation on the t axis, and a wider outer margin remain
# figure-local overrides scoped to this path only. Two renditions of the
# vertical layout share this code via `stagger`: v5 (stagger=True) keeps the
# Total bar and the interval boxes on separate rows; v5b_aligned
# (stagger=False) raises them onto the Total bar's own row, one continuous
# bottom lane. Deliberately self-contained (duplicates the cascade-row loop
# from `_block_panel` rather than adding branches to it) so `render_startup`
# / `timing_startup_gantt.png` stay byte-for-byte untouched by this
# experiment. Renders via `render_lifetime` -- defaults to
# timing_lifetime_gantt_v1.png for signature parity with the rest of the
# module, but this code is the v5 revision; callers pass an explicit
# output_path to land a new iteration file without touching an earlier one.
# Never the standard timing_startup_gantt.png.
LIFETIME_RIGHT_PAD_FRAC = 0.05      # headroom past the arrow tip for "t"
LIFETIME_LOOP_PERIODS = 1.75        # axis extends this many loop_periods past
                                    # su_total -- interval 2's own box ends at
                                    # ~1.44 loop_periods (loop_period +
                                    # rt_total), so 1.75 leaves a modest
                                    # ~0.31 loop_period (~58ms) run-out before
                                    # the arrow, within the requested 1.7-2.0
                                    # range (picked the low end -- a tighter
                                    # figure, still a clearly visible gap)
LIFETIME_FADE_STRIPS = 100          # sub-slices for interval 2's own smooth
                                    # left->right dissolve (spread across just
                                    # the box's own rt_total width, not a
                                    # separate tail) -- fine enough to read as
                                    # a continuous gradient, no banding
LIFETIME_ARROW_MUTATION = 12        # vs. style.DEFAULT_ARROW_MUTATION's 26 --
                                    # roughly half, a noticeably smaller
                                    # arrowhead (head_length/width scale
                                    # together with mutation_scale in
                                    # Vector's FancyArrowPatch)
LIFETIME_MARGIN_INCHES = 0.5        # vs. the banner override's 0.15 -- visible
                                    # breathing room on all four sides without
                                    # eating much of the 28" canvas
LIFETIME_MARKER_LABEL_SIZE = 18     # tiny event-marker tick labels below the
                                    # axis ("1.3 s" / "+81 ms" / "+186 ms") --
                                    # small and sparse vs. MEASURE_SIZE (28)
                                    # everywhere else, per the "almost
                                    # wordless" ask; not bold, unlike the
                                    # value text they replace
LIFETIME_LOOP_END_TICK_LEN = 0.45   # vs. a normal tick's 0.32 -- "slightly
                                    # longer" so the loop-end marker (which
                                    # anchors interval 1's own box edge down
                                    # to the axis) reads as a distinct event,
                                    # not just another regular tick

# Drawio-derived palette (figure-local ONLY -- never assigned into
# dsplot.style, which figure_1 and the rest of the family read from). Pulled
# straight from assets/timing/subshader_startup.drawio /
# subshader_runtime.drawio: each block's strokeColor is a `light-dark(light,
# dark)` pair -- the DARK side is used here, since both diagrams' exported
# PNGs render on pure black (confirmed by sampling the corner pixel of both
# .drawio.png files). Audio/DSP/Render turned out to be exact hex matches to
# style.TERTIARY_COLOR/SECONDARY_COLOR/PRIMARY_COLOR already; background and
# font color did not (see report). v3 mapped Init CUDA to a gray analog
# since no block in either diagram carried a true gray fill/stroke, but the
# user resolved that the OTHER way for v4: the drawio block (currently
# stroked #7B6FE1, DSP purple) is right, so init:cuda maps to
# LIFETIME_DSP here, not a separate gray.
LIFETIME_BG = "#000000"        # mxGraphModel background (both diagrams)
LIFETIME_FONT = "#FFFFFF"      # every block + letter-label fontColor
LIFETIME_AUDIO = "#FFD27D"     # Open Audio File / Audio Output Init / Fetch Audio
LIFETIME_DSP = "#7B6FE1"       # Build Wavelet Kernels / FFT / IFFT / Compute Mag /
                                # ... and now Init CUDA too (see note above)
LIFETIME_RENDER = "#FF5A1F"    # Allocate Frame Buffer / OpenGL Context / Shader / ...

# Cascade segment colors arrive from `_gantt_data()` pre-baked with
# style.TERTIARY_COLOR/SECONDARY_COLOR/PRIMARY_COLOR/SPINE_COLOR (`_FINE` /
# `_COARSE` are built once at import time) -- a style.* context-manager
# override at render time can't reach values already baked into those
# tuples, so the cascade loop below remaps each row's color through this
# dict explicitly instead. SPINE_COLOR (init:cuda's only current user) maps
# to LIFETIME_DSP per the user's v4 resolution above.
_LIFETIME_COLOR_MAP = {
    style.TERTIARY_COLOR: LIFETIME_AUDIO,
    style.SECONDARY_COLOR: LIFETIME_DSP,
    style.PRIMARY_COLOR: LIFETIME_RENDER,
    style.SPINE_COLOR: LIFETIME_DSP,
}

# BG_COLOR / DEFAULT_MARGIN_INCHES / DEFAULT_ARROW_MUTATION are all read live
# (BarPanel.render() / Figure.compose() / Vector.draw() each read style.* at
# call time, not at construction), so a temporary context-manager override
# DOES reach them -- same save/restore mechanism as `_banner_style`, scoped
# to `render_lifetime` only so render_startup/render_runtime/render keep the
# shared style untouched.
_LIFETIME_STYLE_OVERRIDES = {
    "BG_COLOR": LIFETIME_BG,
    "DEFAULT_MARGIN_INCHES": LIFETIME_MARGIN_INCHES,
    "DEFAULT_ARROW_MUTATION": LIFETIME_ARROW_MUTATION,
}


@contextmanager
def _lifetime_style():
    orig = {k: getattr(style, k) for k in _LIFETIME_STYLE_OVERRIDES}
    try:
        for k, v in _LIFETIME_STYLE_OVERRIDES.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


def _lifetime_interval_box(panel, x0, width, y, *, fade=False):
    """One real-time-deadline-interval box: zero-inset at x0 (its left edge
    IS the rule), width = the actual measured loop time (rt_total) -- no
    headroom squares, no dark outline; the idle remainder of the interval is
    just background (v4: dark boxes removed entirely per spec item 5).
    `fade=False` draws one solid full-brightness bar. `fade=True` draws the
    SAME box as `LIFETIME_FADE_STRIPS` fine per-strip-alpha slices, ramping
    ~1.0 at the box's own left edge down to ~0 at its own right edge -- a
    continuous-looking dissolve confined to the box, not a separate tail.
    One Barh call per strip: Barh always forwards its own `alpha=` to
    ax.barh (default 1.0), which OVERRIDES any per-bar alpha baked into an
    RGBA `colors` list passed to a single batched call -- confirmed by
    rendering during v2/v3.
    """
    if not fade:
        panel.add(Barh([y], [width], lefts=[x0], color=LIFETIME_FONT,
                       height=BAR_H, edgewidth=0.0))
        return
    strip_w = width / LIFETIME_FADE_STRIPS
    for k in range(LIFETIME_FADE_STRIPS):
        left = x0 + k * strip_w
        alpha = max(0.0, 1.0 - (k + 0.5) / LIFETIME_FADE_STRIPS)
        panel.add(Barh([y], [strip_w], lefts=[left], color=LIFETIME_FONT,
                       alpha=alpha, height=BAR_H, edgewidth=0.0))


def _lifetime_panel(segments, su_total, loop_period, rt_total, ticks_ms,
                    tick_labels, *, stagger=True):
    """Cascade unchanged; bottom is Total bar + two interval boxes + event
    markers, no floating value text (v5 -- see module note above). `stagger`
    picks the vertical rendition: True (v5) keeps the boxes on their own row
    below the Total bar; False (v5b_aligned) raises them onto the Total
    bar's own row, one lane. Returns (panel, y_axis) -- the caller needs
    y_axis to size the composed figure directly (see build_lifetime_figure),
    since seating the lowest lane's bottom edge ON the axis (no dead band)
    makes y_axis a fractional row value that no generic n_rows+AXIS_GAP
    formula can reproduce.
    """
    n = len(segments)
    x_end = su_total + LIFETIME_LOOP_PERIODS * loop_period
    gutter = GUTTER_FRAC * x_end
    x_hi = x_end * (1.0 + LIFETIME_RIGHT_PAD_FRAC)
    # Total bar keeps the cascade's standard row pitch (y=n). Staggered: the
    # interval boxes sit one full row below it (boxes_y = y_tot+1), same as
    # before. Aligned: boxes share the Total bar's own row (boxes_y = y_tot).
    # Either way, the axis now seats directly under the LOWEST lane's bottom
    # edge (y_axis = boxes_y + BAR_H/2) instead of floating a full
    # row+AXIS_GAP below it -- closes the dead band entirely (same fix as
    # the v2 lifetime row, reapplied to whichever lane is now lowest).
    y_tot = float(n)
    boxes_y = y_tot + 1.0 if stagger else y_tot
    y_axis = boxes_y + BAR_H / 2.0
    panel = BarPanel(
        units=PANEL_UNITS,
        xlim=(-gutter, x_hi),
        ylim=(y_axis + AXIS_TAIL, -TITLE_HEAD),   # y grows downward
        xticks=[], xticklabels=[],
        show_xticklabels=False,
        show_border=False,
    )
    label_pad = 0.011 * (x_hi + gutter)
    x_range = x_hi + gutter

    # Event-marker rules - added BEFORE the interval boxes (see module note:
    # equal-zorder insertion order lets a box paint over the rule within its
    # own lane, so the box's own edge reads as the marker there while the
    # rule still shows above/below it).
    rule2_x = su_total + loop_period       # real-time playback deadline
    loop_end_x = su_total + rt_total       # loop end (interval 1's own edge)
    # End of Start Up == Runtime Loop start (t=su_total): the ONE solid
    # rule, the epoch boundary.
    panel.add(Line([su_total, su_total], [-0.55, y_axis],
                   color=LIFETIME_FONT, linewidth=2.5))
    # Real-time playback deadline (t=su_total+loop_period): dashed, audio
    # amber -- the deadline IS the audio playback clock catching up.
    panel.add(Line([rule2_x, rule2_x], [-0.55, y_axis],
                   color=LIFETIME_AUDIO, linewidth=2.5, linestyle="--"))

    # y-axis - a vertical rule at x=0, same thickness/color/style as the
    # x-axis, spanning the plotted content height.
    panel.add(Line([0.0, 0.0], [-0.55, y_axis],
                   color=LIFETIME_FONT, linewidth=2.5))

    # Top captions - where the (now-removed) title used to sit: "Start Up"
    # over the startup region, "Runtime Loop" starting just past the first
    # rule, over the loop region. Free-floating text, not bars.
    panel.add(Annotation("Start Up", (label_pad, -1.15), ha="left",
                         va="bottom", color=LIFETIME_FONT,
                         fontsize=MEASURE_SIZE, fontweight="bold"))
    panel.add(Annotation("Runtime Loop", (su_total + label_pad, -1.15),
                         ha="left", va="bottom", color=LIFETIME_FONT,
                         fontsize=MEASURE_SIZE, fontweight="bold"))

    # Stage cascade - same geometry/row-grammar as `_block_panel`'s cascade
    # loop (boxed capital letter, module-color bar, trailing duration label).
    # Segment colors arrive pre-baked with the shared style.* module hexes
    # (see `_LIFETIME_COLOR_MAP` note above) - remapped to the drawio palette
    # here, once per row, so both the bar fill and its letter-box edge follow.
    box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN
    start = 0.0
    for i, (num, name, dur, color) in enumerate(segments):
        color = _LIFETIME_COLOR_MAP.get(color, color)
        draw_w = max(dur, MIN_BAR_FRAC * su_total)
        panel.add(Barh([float(i)], [draw_w], lefts=[start],
                       colors=[color], height=BAR_H, edgewidth=0.0))
        if num:
            panel.add(Barh([float(i)], [box_w], lefts=[-label_pad - box_w],
                           color=LIFETIME_BG, height=LETTER_BOX_H,
                           edgecolor=color, edgewidth=LETTER_BOX_EDGE))
            panel.add(Annotation(num,
                                 (-label_pad - box_w / 2.0,
                                  float(i) + LETTER_NUDGE),
                                 ha="center", va="center", color=LIFETIME_FONT,
                                 fontsize=LETTER_SIZE, fontweight="bold"))
        panel.add(Annotation(_dur_label(dur),
                             (start + draw_w + label_pad, float(i)),
                             ha="left", va="center", color=LIFETIME_FONT,
                             fontsize=MEASURE_SIZE, fontweight="bold"))
        start += dur

    # Total bar - standard row pitch, verbatim geometry from `_block_panel`'s
    # non-cell_ms branch. No trailing value text (v5 removes ALL floating
    # text from the data field - the event-marker grammar below replaces it
    # with a tiny tick label under the epoch-boundary rule instead).
    panel.add(Barh([y_tot], [su_total], color=LIFETIME_FONT,
                   height=BAR_H, edgewidth=0.0))

    # Interval 1 - one white box, zero inset: its left edge coincides
    # exactly with the first (solid) rule, width = the actual measured loop
    # time (rt_total, the same number the standard figure's Total row
    # uses), full brightness. No headroom squares, no dark outline - the
    # idle remainder of the interval is just background. Its own RIGHT edge
    # is the in-field loop-end marker (anchored to the axis by the tick
    # added below).
    _lifetime_interval_box(panel, su_total, rt_total, boxes_y, fade=False)

    # Interval 2 - the SAME box repeated with zero inset at the second
    # (dashed amber) rule, dissolving smoothly left->right across its own
    # width (no tiles, no separate tail strip past it) - nothing drawn
    # after it; the axis just runs out to the (smaller) arrowhead + t. No
    # tick for this one (spec: "skip a tick for the faded second box").
    _lifetime_interval_box(panel, rule2_x, rt_total, boxes_y, fade=True)

    # Loop end - a short tick just below the axis at interval 1's own right
    # edge, slightly longer than a normal tick: the field marker (the box
    # edge) is already there; this just anchors it down to the axis.
    panel.add(Line([loop_end_x, loop_end_x],
                   [y_axis, y_axis + LIFETIME_LOOP_END_TICK_LEN],
                   color=LIFETIME_FONT, linewidth=2.5))

    # Physics-style time axis: one arrow-terminated rule from 0 to x_end
    # (same color/thickness as the hero-style axis it replaces), closing
    # with a "t" label at the tip. Ticks kept exactly as before.
    panel.add(Vector((x_end, 0.0), origin=(0.0, y_axis),
                     color=LIFETIME_FONT, linewidth=2.5))
    panel.add(Annotation("t", (x_end + label_pad, y_axis), ha="left",
                         va="center", color=LIFETIME_FONT, fontsize=MEASURE_SIZE,
                         fontweight="bold"))
    for t, lbl in zip(ticks_ms, tick_labels):
        panel.add(Line([t, t], [y_axis, y_axis + 0.32],
                       color=LIFETIME_FONT, linewidth=2.5))
        panel.add(Annotation(lbl, (t, y_axis + 0.62), ha="center", va="top",
                             color=LIFETIME_FONT, fontsize=TICK_SIZE))

    # Tiny event-marker tick labels - small, sparse, nothing in the field:
    # "1.3 s" under the solid epoch-boundary rule, "+81 ms" under the
    # loop-end tick, "+186 ms" (in audio amber, matching its rule) under the
    # dashed deadline rule. "0" at the origin already exists (the first
    # entry in ticks_ms/tick_labels above), so it's not repeated here.
    for x, text, color in (
        (su_total, _total_label(su_total), LIFETIME_FONT),
        (loop_end_x, f"+{rt_total:.0f} ms", LIFETIME_FONT),
        (rule2_x, f"+{loop_period:.0f} ms", LIFETIME_AUDIO),
    ):
        panel.add(Annotation(text, (x, y_axis + 0.62), ha="center", va="top",
                             color=color, fontsize=LIFETIME_MARKER_LABEL_SIZE))

    return panel, y_axis


def build_lifetime_figure(*, stagger=True) -> Figure:
    """User design iteration v5 - 'Lifetime' framing of the Start Up cascade:
    standard Total bar (geometry only), an event-marker grammar (solid epoch
    rule, dashed audio-amber deadline rule, loop-end tick) replacing all
    floating value text, one full-brightness interval box seated with zero
    inset at each rule (the second dissolving smoothly across its own
    width), top-row "Start Up" / "Runtime Loop" captions (no title), under a
    physics-style arrow t axis with a smaller arrowhead. `stagger=True` (v5)
    keeps the interval boxes on their own row below the Total bar;
    `stagger=False` (v5b_aligned) raises them onto the Total bar's own row,
    with the lowest lane seated directly on the axis (no dead band) either
    way. Does NOT touch build_startup_figure / render_startup."""
    d = _gantt_data()
    su_segments, su_total, rt_total = d["su_segments"], d["su_total"], d["rt_total"]
    loop_period = d["deadline"]
    if not loop_period or not rt_total:
        raise RuntimeError(
            "No deadline/rt_margin/rt_total in the latest run - cannot place "
            "the interval boxes for the lifetime figure."
        )
    ticks, tick_labels = _su_ticks(su_total)
    panel, y_axis = _lifetime_panel(su_segments, su_total, loop_period,
                                    rt_total, ticks, tick_labels,
                                    stagger=stagger)
    # Bypass `_compose_one`'s generic head+n_rows+AXIS_GAP+AXIS_TAIL formula
    # - it assumes the old "lane floats above the axis" geometry. Composing
    # directly off the panel's own y_axis keeps ROW_IN the true inch-per-row
    # scale now that the lowest lane sits flush on the axis (fix 1).
    rows = TITLE_HEAD + y_axis + AXIS_TAIL
    return Figure.compose(
        rows=[[panel]],
        row_heights=[rows * ROW_IN],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=1.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


def render_lifetime(output_path: str | None = None, *, stagger=True) -> str:
    """'Lifetime' design iteration → timing_lifetime_gantt_v1.png by default
    (a NEW file - never overwrites timing_startup_gantt.png). Pass an
    explicit output_path to render a new iteration (e.g. _v5 /
    _v5b_aligned) without touching earlier iteration PNGs. `stagger` selects
    the vertical rendition (see `build_lifetime_figure`). Own render body
    (not `_render_to`) so it can layer `_lifetime_style()` (the
    drawio-derived BG_COLOR / margin / arrow-mutation overrides) on top of
    the shared `_banner_style()` layout override, without touching
    render_startup/render_runtime/render, which stay on the shared palette.
    """
    if output_path is None:
        output_path = os.path.join(_repo_root(), "assets", "timing",
                                   "timing_lifetime_gantt_v1.png")
    with _banner_style(), _lifetime_style():
        fig = build_lifetime_figure(stagger=stagger)
        fig.render()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return os.path.abspath(output_path)


# --- "Lifetime" treatment of the Process Loop figure ---------------------
# Runtime sibling of the lifetime iteration above, same figure-local
# grammar: drawio palette + black background, cascade rows unchanged, the
# Total bar as geometry only (its value demoted to a small tick label under
# the loop-end marker), a dashed audio-amber rule at the real-time playback
# deadline, and a physics-style arrow t axis with the smaller arrowhead.
# The loop's own start is this figure's origin, so the startup version's
# solid epoch rule is simply the y-axis rule here; there are no interval
# boxes -- the Total bar IS the loop's work, and the idle remainder of the
# deadline window is just background. Same self-containment rule as above:
# render_runtime / timing_runtime_gantt.png stay byte-for-byte untouched;
# iterations land in NEW timing_runtime_lifetime_gantt_v*.png files.
RUNTIME_LIFETIME_AXIS_PAD = 1.15    # axis extends this factor past the
                                    # deadline rule -- ~28 ms of run-out
                                    # before the arrowhead, proportionally
                                    # the startup figure's modest gap


def _rt_lifetime_ticks(deadline):
    ticks = [t for t in (0, 50, 100, 150) if t <= deadline]
    labels = [f"{t:g}" for t in ticks]
    labels[-1] += " ms"
    return ticks, labels


def _runtime_lifetime_panel(segments, rt_total, deadline, ticks_ms,
                            tick_labels):
    """Process Loop cascade in the lifetime grammar; bottom is the white
    Total bar seated directly on the axis (no dead band), a loop-end tick at
    its right edge, and the dashed deadline rule -- no floating value text.
    Returns (panel, y_axis), same contract as `_lifetime_panel`."""
    n = len(segments)
    x_end = deadline * RUNTIME_LIFETIME_AXIS_PAD
    gutter = GUTTER_FRAC * x_end
    x_hi = x_end * (1.0 + LIFETIME_RIGHT_PAD_FRAC)
    y_tot = float(n)
    y_axis = y_tot + BAR_H / 2.0
    panel = BarPanel(
        units=PANEL_UNITS,
        xlim=(-gutter, x_hi),
        ylim=(y_axis + AXIS_TAIL, -TITLE_HEAD),   # y grows downward
        xticks=[], xticklabels=[],
        show_xticklabels=False,
        show_border=False,
    )
    label_pad = 0.011 * (x_hi + gutter)
    x_range = x_hi + gutter

    # Real-time playback deadline: dashed, audio amber -- added before the
    # bars so a lane could paint over it, same insertion-order rule as the
    # startup figure (nothing overlaps it here; kept for parity).
    panel.add(Line([deadline, deadline], [-0.55, y_axis],
                   color=LIFETIME_AUDIO, linewidth=2.5, linestyle="--"))

    # y-axis rule at x=0 -- doubles as the loop-start marker (the startup
    # figure's solid epoch rule, which is this figure's origin).
    panel.add(Line([0.0, 0.0], [-0.55, y_axis],
                   color=LIFETIME_FONT, linewidth=2.5))

    # Top caption -- free-floating, where a title would sit.
    panel.add(Annotation("Runtime Loop", (label_pad, -1.15), ha="left",
                         va="bottom", color=LIFETIME_FONT,
                         fontsize=MEASURE_SIZE, fontweight="bold"))

    # Stage cascade -- verbatim row grammar from `_lifetime_panel`, colors
    # remapped through the drawio palette per row.
    box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN
    start = 0.0
    for i, (num, name, dur, color) in enumerate(segments):
        color = _LIFETIME_COLOR_MAP.get(color, color)
        draw_w = max(dur, MIN_BAR_FRAC * rt_total)
        panel.add(Barh([float(i)], [draw_w], lefts=[start],
                       colors=[color], height=BAR_H, edgewidth=0.0))
        if num:
            panel.add(Barh([float(i)], [box_w], lefts=[-label_pad - box_w],
                           color=LIFETIME_BG, height=LETTER_BOX_H,
                           edgecolor=color, edgewidth=LETTER_BOX_EDGE))
            panel.add(Annotation(num,
                                 (-label_pad - box_w / 2.0,
                                  float(i) + LETTER_NUDGE),
                                 ha="center", va="center", color=LIFETIME_FONT,
                                 fontsize=LETTER_SIZE, fontweight="bold"))
        panel.add(Annotation(_dur_label(dur),
                             (start + draw_w + label_pad, float(i)),
                             ha="left", va="center", color=LIFETIME_FONT,
                             fontsize=MEASURE_SIZE, fontweight="bold"))
        start += dur

    # Total bar -- geometry only, seated on the axis; its right edge is the
    # in-field loop-end marker, anchored down by the tick below.
    panel.add(Barh([y_tot], [rt_total], color=LIFETIME_FONT,
                   height=BAR_H, edgewidth=0.0))
    panel.add(Line([rt_total, rt_total],
                   [y_axis, y_axis + LIFETIME_LOOP_END_TICK_LEN],
                   color=LIFETIME_FONT, linewidth=2.5))

    # Physics-style time axis + ticks, as in `_lifetime_panel`.
    panel.add(Vector((x_end, 0.0), origin=(0.0, y_axis),
                     color=LIFETIME_FONT, linewidth=2.5))
    panel.add(Annotation("t", (x_end + label_pad, y_axis), ha="left",
                         va="center", color=LIFETIME_FONT, fontsize=MEASURE_SIZE,
                         fontweight="bold"))
    for t, lbl in zip(ticks_ms, tick_labels):
        panel.add(Line([t, t], [y_axis, y_axis + 0.32],
                       color=LIFETIME_FONT, linewidth=2.5))
        panel.add(Annotation(lbl, (t, y_axis + 0.62), ha="center", va="top",
                             color=LIFETIME_FONT, fontsize=TICK_SIZE))

    # Tiny event-marker labels: the loop's own total under its end tick, the
    # deadline (in its rule's amber) under the dashed rule. Absolute values
    # here, not "+" offsets -- the loop start IS this figure's t=0.
    for x, text, color in (
        (rt_total, _dur_label(rt_total), LIFETIME_FONT),
        (deadline, f"{deadline:.0f} ms", LIFETIME_AUDIO),
    ):
        panel.add(Annotation(text, (x, y_axis + 0.62), ha="center", va="top",
                             color=color, fontsize=LIFETIME_MARKER_LABEL_SIZE))

    return panel, y_axis


def build_runtime_lifetime_figure() -> Figure:
    """Lifetime treatment of the Process Loop cascade -- see the section
    note above. Does NOT touch build_runtime_figure / render_runtime."""
    d = _gantt_data()
    rt_segments, rt_total = d["rt_segments"], d["rt_total"]
    deadline = d["deadline"]
    if not deadline or not rt_total:
        raise RuntimeError(
            "No deadline/rt_margin/rt_total in the latest run - cannot place "
            "the deadline rule for the runtime lifetime figure."
        )
    ticks, tick_labels = _rt_lifetime_ticks(deadline)
    panel, y_axis = _runtime_lifetime_panel(rt_segments, rt_total, deadline,
                                            ticks, tick_labels)
    rows = TITLE_HEAD + y_axis + AXIS_TAIL
    return Figure.compose(
        rows=[[panel]],
        row_heights=[rows * ROW_IN],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=1.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


def render_runtime_lifetime(output_path: str | None = None) -> str:
    """Runtime lifetime figure → timing_runtime_lifetime_gantt_v1.png by
    default (a NEW file - never overwrites timing_runtime_gantt.png); pass
    an explicit output_path for later iterations. Same style layering as
    `render_lifetime`."""
    if output_path is None:
        output_path = os.path.join(_repo_root(), "assets", "timing",
                                   "timing_runtime_lifetime_gantt_v1.png")
    with _banner_style(), _lifetime_style():
        fig = build_runtime_lifetime_figure()
        fig.render()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return os.path.abspath(output_path)


def _render_to(build, default_name, output_path=None, **kwargs) -> str:
    if output_path is None:
        output_path = os.path.join(_repo_root(), "assets", "timing",
                                   default_name)
    with _banner_style():
        fig = build(**kwargs)
        fig.render()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return os.path.abspath(output_path)


def render_startup(output_path: str | None = None) -> str:
    """Start Up figure → timing_startup_gantt.png. Returns absolute path."""
    return _render_to(build_startup_figure, "timing_startup_gantt.png",
                      output_path)


def render_runtime(output_path: str | None = None) -> str:
    """Process Loop figure → timing_runtime_gantt.png. Returns absolute path."""
    return _render_to(build_runtime_figure, "timing_runtime_gantt.png",
                      output_path)


def render(output_path: str | None = None, *, echo_colors: bool = False) -> str:
    """Legacy combined figure → timing_gantt.png. Returns absolute path."""
    return _render_to(build_figure, "timing_gantt.png",
                      output_path, echo_colors=echo_colors)


if __name__ == "__main__":
    print(render_startup())
    print(render_runtime())
