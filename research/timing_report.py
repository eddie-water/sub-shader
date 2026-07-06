"""Visual timing report — turn the latest recorded run into clear pictures.

Figures, each its own PNG so the report can caption them one at a time:

    timing_startup.png            — startup, to-scale stacked bar + white total + key.
    timing_runtime.png            — one frame, high-level (Audio / DSP / Renderer).
    timing_perstage.png           — every DSP + render stage, in order of occurrence.
    timing_methods.png            — four transforms compared (log scale + resolution).
    timing_config.png             — compute per frame across the settings swept.

Bar geometry is pinned in inches (``_bar_fig`` sets the axes height to
``row_in × n`` exactly), so every bar renders at the *same* thickness across all
figures — except the dense per-stage breakdown, which runs at a quarter thickness
because it carries so many rows.

Styling follows the project's optichrome v52 palette (orange / purple / gold /
cyan on a dark ground) so the report matches the figures in DSP.md. The hexes are
copied here as constants rather than imported, to keep ``research/`` decoupled
from the ``dsplot`` library.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities import RESULTS_CSV, TIMING_DIR, render_markdown
from utilities.timing_results import (
    _read_rows, TOTAL_STAGE, _is_frame_stage, select_anatomy_run,
    _PIPE_ORDER, _PIPE_LABEL, _STAGE_MODULE,
    latest_method_rows, config_summary_rows, _fmt_ms,
)

STARTUP_PNG = os.path.join(TIMING_DIR, "timing_startup.png")
RUNTIME_PNG = os.path.join(TIMING_DIR, "timing_runtime.png")
PIPELINE_PNG = os.path.join(TIMING_DIR, "timing_pipeline.png")
LIFECYCLE_PNG = os.path.join(TIMING_DIR, "timing_lifecycle.png")
PERSTAGE_PNG = os.path.join(TIMING_DIR, "timing_perstage.png")
METHODS_PNG = os.path.join(TIMING_DIR, "timing_methods.png")
CONFIG_PNG = os.path.join(TIMING_DIR, "timing_config.png")

# --- optichrome v52 palette (copied from research/dsplot/style.py) -----------
_BG     = "#1A1A1A"   # BG_COLOR
_FG     = "#EEEEEE"   # NEUTRAL_COLOR — text / frame
_SPINE  = "#444444"   # SPINE_COLOR
_TICK   = "#888888"   # TICK_LABEL_COLOR
_AUDIO  = "#ffd27d"   # TERTIARY  — audio module
_DSP    = "#7b6fe1"   # SECONDARY — dsp module
_RENDER = "#ff5a1f"   # PRIMARY   — render module
_HILITE = "#22d3ee"   # HIGHLIGHT — calibration / accents

_MODULE_COLOR = {"audio": _AUDIO, "dsp": _DSP, "render": _RENDER}
_MODULE_NAME  = {"audio": "Grab Audio", "dsp": "DSP Stages", "render": "Renderer"}

# --- bar geometry, pinned in inches so every bar is the same thickness --------
_FIG_W      = 13.0    # figure width
_ROW_IN     = 0.62    # inches per data row (normal charts)
_ROW_IN_PS  = 0.34    # compact rows for the dense per-stage breakdown
_BAR_THICK  = 0.34    # bar thickness in inches (normal)
_PS_THICK   = _BAR_THICK / 4.0   # per-stage bars = quarter thickness
_TOP_IN     = 0.35    # top margin (no titles — MD headings name the charts)
_BOT_IN     = 0.62    # bottom margin (xlabel + ticks)
_BAR_H      = _BAR_THICK / _ROW_IN          # bar height in data units (normal)
_PS_BAR_H   = _PS_THICK / _ROW_IN_PS        # bar height in data units (per-stage)
_VAL_SIZE   = 10
_LBL_SIZE   = 10

_INIT_ORDER = ["init:audio", "init:dsp", "init:renderer", "init:prescan",
               "init:other", "init:build"]
_INIT_NAME = {
    "init:audio": "Audio Source", "init:dsp": "DSP Stages",
    "init:renderer": "Renderer", "init:prescan": "Intensity Pre-scan",
    "init:other": "Setup", "init:build": "Build Pipeline",
}
_INIT_COLOR = {
    "init:audio": _AUDIO, "init:dsp": _DSP, "init:renderer": _RENDER,
    "init:prescan": _HILITE, "init:other": _SPINE, "init:build": _DSP,
}

# Fine-grained init sub-stages (recorded by timed_block in the constructors;
# present only in runs taken after that instrumentation landed). When a run has
# them, the start-up gantt expands each module into its sub-steps; older runs
# fall back to the coarse module rows above. Colored by parent module.
_INIT_FINE_ORDER = [
    "init:audio_reader", "init:audio_player",
    "init:dsp_kernels", "init:dsp_fft", "init:dsp_upload",
    "init:render_buffer", "init:render_glcontext", "init:render_shader",
    "init:prescan", "init:other",
]
_INIT_FINE_NAME = {
    "init:audio_reader": "Open Audio File",
    "init:audio_player": "Audio Output Init",
    "init:dsp_kernels": "Build Wavelet Kernels",
    "init:dsp_fft": "Generate FFT Kernel Bank",
    "init:dsp_upload": "Transfer Kernel Bank → GPU",
    "init:render_buffer": "Allocate GPU Buffer",
    "init:render_glcontext": "Create Window + GL Context",
    "init:render_shader": "Compile Shader + Texture",
    "init:prescan": "Color Map Init",
    "init:other": "Setup",
}
_INIT_FINE_COLOR = {
    "init:audio_reader": _AUDIO, "init:audio_player": _AUDIO,
    "init:dsp_kernels": _DSP, "init:dsp_fft": _DSP, "init:dsp_upload": _DSP,
    "init:render_buffer": _RENDER, "init:render_glcontext": _RENDER,
    "init:render_shader": _RENDER,
    "init:prescan": _RENDER, "init:other": _SPINE,
}
# Keys that mark a run as carrying the fine breakdown (module sub-steps only —
# prescan/other exist in coarse runs too, so they don't count as a signal).
_INIT_FINE_SIGNAL = {k for k in _INIT_FINE_ORDER
                     if k not in ("init:prescan", "init:other")}


def _startup_segments(init):
    """(segments, total) for the start-up gantt.

    Prefer the fine per-module sub-stages when the run recorded them; otherwise
    fall back to the coarse module rows. The total is always the real one-time
    construction time (sum of the coarse module rows) so the white Total bar is
    correct regardless of which breakdown is shown.
    """
    coarse_total = sum(init.get(k, 0.0) for k in _INIT_ORDER)
    fine = any(k in init for k in _INIT_FINE_SIGNAL)
    order = _INIT_FINE_ORDER if fine else _INIT_ORDER
    names = _INIT_FINE_NAME if fine else _INIT_NAME
    colors = _INIT_FINE_COLOR if fine else _INIT_COLOR
    # init:other is the unaccounted construction residual (config glue, logging).
    # Every meaningful cost already has its own bar, so it's dropped as a row —
    # but kept in the Total so the white Total bar stays the true one-time cost.
    items = [(k, init[k]) for k in order
             if init.get(k, 0.0) > 0 and k != "init:other"]
    segments = [(names.get(k, k), v, colors.get(k, _SPINE)) for k, v in items]
    total = coarse_total if coarse_total > 0 else sum(v for _, v in items)
    return segments, total


def _ideal_text(hex_color):
    """Black or white in-bar text, whichever reads on `hex_color` (luminance)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return _BG if lum > 0.6 else "#ffffff"


def _style_axes(ax):
    """Apply the dark optichrome look to one axes."""
    ax.set_facecolor(_BG)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(_SPINE)
    ax.tick_params(colors=_TICK, labelcolor=_FG, labelsize=_LBL_SIZE)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def _bar_fig(n_units, row_in, left_in, *, top_in=_TOP_IN, bot_in=_BOT_IN,
             right_in=0.5, width=_FIG_W):
    """Figure whose axes is exactly `row_in × n_units` inches tall.

    Pinning the axes height in inches (not as a figure fraction) makes one data
    unit map to `row_in` inches in every chart, so bars of equal data-height
    render at equal physical thickness regardless of how many rows a chart has.
    """
    axes_h = row_in * n_units
    fig_h = axes_h + top_in + bot_in
    fig = plt.figure(figsize=(width, fig_h))
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([left_in / width, bot_in / fig_h,
                       1.0 - (left_in + right_in) / width, axes_h / fig_h])
    _style_axes(ax)
    return fig, ax


def _save(fig, out):
    os.makedirs(TIMING_DIR, exist_ok=True)
    fig.savefig(out, dpi=84, facecolor=_BG)
    plt.close(fig)
    return out


def _fits(text, seg_frac, axes_w_in, fs=9):
    """Rough check that `text` fits in a segment `seg_frac` wide (monospace est.)."""
    return len(text) * fs * 0.0072 <= seg_frac * axes_w_in


def _latest_run(csv=RESULTS_CSV):
    """Return (meta, ordered frame stages [(stage, mean, pct)], total, rt, init)."""
    rows = _read_rows(csv)
    if not rows:
        return None
    anatomy_id = select_anatomy_run(rows)
    latest = [r for r in rows if r["run_id"] == anatomy_id]
    meta = latest[0]

    means = {r["stage"]: float(r["mean_ms"]) for r in latest if _is_frame_stage(r["stage"])}
    # Fold the (non-blocking) glClear into the Shader Draw box to match the
    # software flowchart: clear+draw are one stage on the diagram, still two GL
    # calls in the renderer. Sum preserves the total; no measured time is lost.
    if "gl_clear" in means:
        means["gl_draw"] = means.get("gl_draw", 0.0) + means.pop("gl_clear")
    total_row = next((r for r in latest if r["stage"] == TOTAL_STAGE), None)
    total = float(total_row["mean_ms"]) if total_row else sum(means.values())
    rt = total_row["rt_margin"] if total_row else ""
    rt = float(rt) if rt not in ("", None) else 0.0

    init = {r["stage"]: float(r["mean_ms"]) for r in latest if r["stage"].startswith("init:")}

    ordered = [(s, means[s], 100.0 * means[s] / total if total else 0.0)
               for s in _PIPE_ORDER if s in means]
    return meta, ordered, total, rt, init


def _dur_label(ms):
    """Compact duration: seconds past 1 s, integer ms down to 10 ms, else 1 dp."""
    ms = float(ms)
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    if ms < 0.1:                 # sub-tenth: don't render as a misleading "0.0 ms"
        return "< 0.1 ms"
    return f"{ms:.1f} ms"


def _legend_below(ax, items, ncol, y=-0.55):
    """Horizontal color key beneath the axes (swatch → name), optichrome text."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=_SPINE, linewidth=0.8)
               for _, c in items]
    leg = ax.legend(handles, [n for n, _ in items],
                    loc="upper center", bbox_to_anchor=(0.5, y),
                    ncol=ncol, frameon=False, fontsize=11, handlelength=1.5,
                    handleheight=1.1, columnspacing=1.8, borderpad=0.0)
    for txt in leg.get_texts():
        txt.set_color(_FG)


def _legend_corner(ax, items, ncol=1, bbox=None):
    """Vertical color key in the top-right corner (matches the per-stage chart)."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=_SPINE, linewidth=0.8)
               for _, c in items]
    kw = dict(loc="upper right", frameon=False, fontsize=11, ncol=ncol,
              labelspacing=0.7, handlelength=1.6, handleheight=1.2,
              columnspacing=1.4, borderpad=0.9)
    if bbox is not None:
        kw["bbox_to_anchor"] = bbox
    leg = ax.legend(handles, [n for n, _ in items], **kw)
    for txt in leg.get_texts():
        txt.set_color(_FG)
    return leg


def _highlevel_figure(segments, total, total_label, xlabel, out):
    """Two to-scale bars + color key.

    Top bar: the segments stacked, each wide-enough slice labeled with its own
    time (white-on-color). Bottom bar: the full total in solid white with the
    grand total in black. A color key sits in the top-right corner (slices +
    Total), matching the per-stage chart.
    """
    left_in, right_in, top_in, bot_in = 0.5, 0.5, 0.30, 1.25
    items = [(name, color) for name, _, color in segments] + [("Total", "#ffffff")]
    # Headroom above the two bars (y=0,1) so the corner legend clears them.
    ncol = 1 if len(items) <= 4 else 2
    rows = -(-len(items) // ncol)            # ceil(len/ncol)
    leg_in = 0.34 * rows + 0.25              # legend height estimate, inches
    top_y = 1.3 + leg_in / _ROW_IN + 0.4     # top of y-range
    span = top_y + 0.7                       # y data range (-0.7 … top_y)
    axes_h = _ROW_IN * span
    fig_h = top_in + axes_h + bot_in
    fig = plt.figure(figsize=(_FIG_W, fig_h))
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([left_in / _FIG_W, bot_in / fig_h,
                       1.0 - (left_in + right_in) / _FIG_W, axes_h / fig_h])
    _style_axes(ax)
    axes_w_in = _FIG_W - left_in - right_in

    # Top row (y=1): stacked, to scale, labeled with per-segment time.
    left = 0.0
    for name, val, color in segments:
        ax.barh(1, val, height=_BAR_H, left=left, color=color,
                edgecolor=_BG, linewidth=1.2, zorder=3)
        label = _dur_label(val)
        if total and _fits(label, val / total, axes_w_in):
            ax.text(left + val / 2, 1, label, ha="center", va="center", fontsize=10,
                    color=_ideal_text(color), fontweight="bold", zorder=4)
        left += val

    # Bottom row (y=0): the whole total, solid white, total time in black.
    ax.barh(0, total, height=_BAR_H, left=0.0, color="#ffffff",
            edgecolor=_BG, linewidth=1.2, zorder=3)
    ax.text(total / 2, 0, total_label, ha="center", va="center", fontsize=11,
            color="#000000", fontweight="bold", zorder=4)

    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.7, top_y)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel(xlabel)

    _legend_corner(ax, items, ncol=ncol)
    return _save(fig, out)


def render_startup_figure(csv=RESULTS_CSV, out=STARTUP_PNG):
    """Startup: to-scale stacked bar (time per stage) + white total bar + key."""
    info = _latest_run(csv)
    if info is None:
        return None
    _, _, _, _, init = info
    items = [(k, init[k]) for k in _INIT_ORDER if init.get(k, 0.0) > 0]
    items += [(k, v) for k, v in init.items() if k not in _INIT_ORDER and v > 0]
    if not items:
        return None
    total = sum(v for _, v in items)
    segments = [(_INIT_NAME.get(k, k), v, _INIT_COLOR.get(k, _SPINE)) for k, v in items]
    return _highlevel_figure(segments, total, _time_block(total),
                             "milliseconds  (one-time, to scale)", out)


def render_runtime_figure(csv=RESULTS_CSV, out=RUNTIME_PNG):
    """Runtime high-level: one frame to scale as module blocks + white total bar."""
    info = _latest_run(csv)
    if info is None:
        return None
    _, stages, total, _, _ = info
    subtotals = {}
    for stage, mean, _ in stages:
        module = _STAGE_MODULE.get(stage, "dsp")
        subtotals[module] = subtotals.get(module, 0.0) + mean
    segments = [(_MODULE_NAME[m], subtotals[m], _MODULE_COLOR[m])
                for m in ("audio", "dsp", "render") if subtotals.get(m, 0) > 0]
    return _highlevel_figure(segments, total, _time_block(total),
                             "milliseconds  (one frame, to scale)", out)


def _gantt_axes(fig, rect):
    """Borderless, tick-free axes spanning 0…1 (fraction of the block total)."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(_BG)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    return ax


def _draw_gantt(ax, segments, total, tick_step=None, tick_unit="ms"):
    """One Gantt block: a full-width white Total bar, then each stage on its own
    row offset by its start time (cascade). x is normalized so the block total
    fills the axes; every bar carries its own duration label.

    tick_step (in ms, same units as `total`) adds a real-time x-axis with ticks
    every tick_step; tick_unit ('s' or 'ms') sets the label units."""
    rows = len(segments) + 1
    bar_h = 0.58   # thick bars
    # Every duration label: white text set just after the end of its bar. The
    # x-range is padded on the right so trailing labels on the longest bars stay
    # on-canvas.
    pad = 0.012
    # Row 0 (top): the whole total, solid white, total time after the bar.
    ax.barh(0, 1.0, height=bar_h, left=0.0, color="#ffffff",
            edgecolor=_BG, linewidth=1.0, zorder=3)
    ax.text(1.0 + pad, 0, _time_block(total), ha="left", va="center_baseline", fontsize=14,
            color="#ffffff", fontweight="bold", zorder=4)
    # Stage rows, cascading by cumulative start time. Sub-pixel stages get a
    # thin minimum-width strip (same bar height) so they stay visible.
    strip_min = 0.006
    start = 0.0
    for i, (name, dur, color) in enumerate(segments, start=1):
        frac = dur / total if total else 0.0
        x0 = start / total if total else 0.0
        draw_frac = max(frac, strip_min) if dur > 0 else frac
        ax.barh(i, draw_frac, height=bar_h, left=x0, color=color,
                edgecolor=_BG, linewidth=1.0, zorder=3)
        ax.text(x0 + draw_frac + pad, i, _dur_label(dur), ha="left", va="center_baseline",
                fontsize=13, color="#ffffff", fontweight="bold", zorder=4)
        start += dur
    ax.set_xlim(0, 1.16)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_yticks(range(rows))
    labels = ax.set_yticklabels(["Total"] + [name for name, _, _ in segments])
    labels[0].set_fontweight("bold")   # emphasize the Total row label
    ax.tick_params(axis="y", length=0, labelcolor=_FG, labelsize=14)

    # Real-time x-axis: ticks every tick_step (ms) along the cascade timeline.
    if tick_step:
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_SPINE)
        n_ticks = int(total // tick_step)
        times = [tick_step * k for k in range(n_ticks + 1)]
        positions = [t / total for t in times]
        if tick_unit == "s":
            tick_labels = [f"{t / 1000:g}" for t in times]
            ax.set_xlabel("s", color=_FG, fontsize=15)
        else:
            tick_labels = [f"{t:g}" for t in times]
            ax.set_xlabel("ms", color=_FG, fontsize=15)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", length=4, color=_SPINE, labelcolor=_FG,
                       labelsize=13)


def render_pipeline_figure(csv=RESULTS_CSV, out=PIPELINE_PNG):
    """Start Up + Runtime as one figure — two Gantt blocks, each to scale.

    Each block leads with a full-width white Total bar, then cascades its stages
    along its own timeline (offset by start time). The two blocks are scaled
    independently so both read clearly despite the ~80× difference in total time.
    """
    info = _latest_run(csv)
    if info is None:
        return None
    _, stages, rt_total, _, init = info

    su_segments, su_total = _startup_segments(init)

    # Runtime block: every individual stage, in execution order, cascading and
    # colored by its module (audio/dsp/render).
    rt_segments = [(_PIPE_LABEL.get(s, s), mean,
                    _MODULE_COLOR.get(_STAGE_MODULE.get(s), _DSP))
                   for s, mean, _ in stages
                   if _STAGE_MODULE.get(s) in ("audio", "dsp", "render")]
    if not su_segments or not rt_segments:
        return None

    row_in = 0.36   # tighter row spacing so the bars sit close together
    su_rows, rt_rows = len(su_segments) + 1, len(rt_segments) + 1
    left_in, right_in = 3.1, 0.5   # room for long per-stage labels on the left
    title_in, gap_in, top_in, bot_in = 0.45, 0.0, 0.20, 0.20
    xaxis_in = 0.55    # room for each block's real-time x-axis ticks + label
    legend_in = 0.55   # horizontal color key below both blocks
    su_h, rt_h = row_in * su_rows, row_in * rt_rows
    fig_h = (top_in + title_in + su_h + xaxis_in + gap_in
             + title_in + rt_h + xaxis_in + legend_in + bot_in)
    fig = plt.figure(figsize=(_FIG_W, fig_h))
    fig.patch.set_facecolor(_BG)
    aw = 1.0 - (left_in + right_in) / _FIG_W
    ax_x = left_in / _FIG_W

    rt_bottom = bot_in + legend_in + xaxis_in
    su_bottom = rt_bottom + rt_h + title_in + gap_in + xaxis_in
    ax1 = _gantt_axes(fig, [ax_x, su_bottom / fig_h, aw, su_h / fig_h])
    _draw_gantt(ax1, su_segments, su_total, tick_step=250, tick_unit="s")
    fig.text(ax_x, (su_bottom + su_h + 0.06) / fig_h,
             "Start Up — Pipeline Construction", color=_FG, fontsize=14,
             fontweight="bold", ha="left", va="bottom")

    ax2 = _gantt_axes(fig, [ax_x, rt_bottom / fig_h, aw, rt_h / fig_h])
    _draw_gantt(ax2, rt_segments, rt_total, tick_step=3, tick_unit="ms")
    fig.text(ax_x, (rt_bottom + rt_h + 0.06) / fig_h,
             "Runtime Loop", color=_FG, fontsize=14,
             fontweight="bold", ha="left", va="bottom")

    # Module color key — horizontal, centered below both blocks.
    key = [("Audio", _AUDIO), ("DSP", _DSP), ("Render", _RENDER)]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=_SPINE,
                             linewidth=0.8) for _, c in key]
    leg = fig.legend(handles, [n for n, _ in key], loc="center",
                     bbox_to_anchor=(ax_x + aw / 2, (bot_in + legend_in / 2) / fig_h),
                     ncol=3, frameon=False, fontsize=12, columnspacing=2.2,
                     handlelength=1.5, handleheight=1.1, handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(_FG)
    return _save(fig, out)


def _draw_lifecycle(ax, su_segments, rt_segments, su_total, rt_total,
                    min_frac=0.014):
    """One continuous Gantt on a single scale: start-up cascades to scale, then
    the runtime loop is appended right after it. Runtime is ~80x shorter, so its
    bars would be sub-pixel — each is floored to `min_frac` of the width so it
    stays visible, and labelled to the left to keep the chip on-canvas."""
    combined = su_total + rt_total
    chip = dict(boxstyle="square,pad=0.25", facecolor="#000000", edgecolor="none")
    # Build rows: (label, duration, color, true_start, is_runtime).
    rows = []
    start = 0.0
    for name, dur, color in su_segments:
        rows.append((name, dur, color, start, False))
        start += dur
    boundary = start / combined if combined else 1.0   # where start-up ends
    for name, dur, color in rt_segments:
        rows.append((name, dur, color, start, True))
        start += dur

    n = len(rows) + 1
    bar_h = 0.62
    # Row 0: combined total, solid white, total time on a black chip.
    ax.barh(0, 1.0, height=bar_h, left=0.0, color="#ffffff",
            edgecolor=_BG, linewidth=1.0, zorder=3)
    ax.text(0.5, 0, _time_block(combined), ha="center", va="center", fontsize=11,
            color="#ffffff", fontweight="bold", zorder=4, bbox=chip)

    for i, (name, dur, color, t0, is_rt) in enumerate(rows, start=1):
        true_frac = dur / combined if combined else 0.0
        frac = max(true_frac, min_frac)
        x0 = t0 / combined if combined else 0.0
        x0 = min(x0, 1.0 - frac)                       # keep the bar on-canvas
        ax.barh(i, frac, height=bar_h, left=x0, color=color,
                edgecolor=_BG, linewidth=1.0, zorder=3)
        label = _dur_label(dur)
        if true_frac < min_frac:                       # tiny: label to the left
            ax.text(x0 - 0.006, i, label, ha="right", va="center", fontsize=10,
                    color="#ffffff", fontweight="bold", zorder=4, bbox=chip)
        else:
            ax.text(x0 + frac / 2, i, label, ha="center", va="center", fontsize=10,
                    color="#ffffff", fontweight="bold", zorder=4, bbox=chip)

    # Dashed boundary between the one-time start-up and the per-frame loop.
    ax.axvline(boundary, ymin=0.0, ymax=0.93, color=_HILITE, linestyle="--",
               linewidth=1.2, zorder=2)

    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(["Total"] + [name for name, *_ in rows])
    ax.tick_params(axis="y", length=0, labelcolor=_FG, labelsize=_LBL_SIZE)


def render_lifecycle_figure(csv=RESULTS_CSV, out=LIFECYCLE_PNG):
    """Start-up + runtime as ONE to-scale timeline (start-up dominates; runtime
    bars floored to a visible minimum)."""
    info = _latest_run(csv)
    if info is None:
        return None
    _, stages, rt_total, _, init = info

    su_segments, su_total = _startup_segments(init)

    subtotals = {}
    for stage, mean, _ in stages:
        module = _STAGE_MODULE.get(stage, "dsp")
        subtotals[module] = subtotals.get(module, 0.0) + mean
    rt_segments = [(_MODULE_NAME[m], subtotals[m], _MODULE_COLOR[m])
                   for m in ("audio", "dsp", "render") if subtotals.get(m, 0) > 0]
    if not su_segments or not rt_segments:
        return None

    row_in = 0.55
    rows = len(su_segments) + len(rt_segments) + 1
    left_in, right_in = 1.6, 0.5
    title_in, top_in, bot_in = 0.45, 0.30, 0.20
    block_h = row_in * rows
    fig_h = top_in + title_in + block_h + bot_in
    fig = plt.figure(figsize=(_FIG_W, fig_h))
    fig.patch.set_facecolor(_BG)
    aw = 1.0 - (left_in + right_in) / _FIG_W
    ax_x = left_in / _FIG_W

    ax = _gantt_axes(fig, [ax_x, bot_in / fig_h, aw, block_h / fig_h])
    _draw_lifecycle(ax, su_segments, rt_segments, su_total, rt_total)
    fig.text(ax_x, (bot_in + block_h + 0.06) / fig_h,
             "Start Up → Runtime Loop  (one timeline, to scale)", color=_FG,
             fontsize=14, fontweight="bold", ha="left", va="bottom")
    return _save(fig, out)


def render_perstage_figure(csv=RESULTS_CSV, out=PERSTAGE_PNG):
    """Every DSP + render stage named, in order of occurrence (quarter thickness)."""
    info = _latest_run(csv)
    if info is None:
        return None
    _, stages, _, _, _ = info
    detail = [(s, m) for s, m, _ in stages
              if _STAGE_MODULE.get(s) in ("audio", "dsp", "render")]
    if not detail:
        return None
    n = len(detail)
    fig, ax = _bar_fig(n, _ROW_IN_PS, 3.0)
    y = list(range(n))
    ax.barh(y, [m for _, m in detail], height=_PS_BAR_H,
            color=[_MODULE_COLOR.get(_STAGE_MODULE.get(s), _DSP) for s, _ in detail],
            edgecolor=_BG, linewidth=0.5, zorder=3)
    for i, (s, m) in enumerate(detail):
        ax.text(m, i, f"  {m:.2f} ms", va="center", ha="left", fontsize=9, color=_FG)
    ax.set_yticks(y)
    ax.set_yticklabels([_PIPE_LABEL.get(s, s) for s, _ in detail])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("milliseconds  (per frame)")
    ax.margins(x=0.18)
    ax.grid(axis="x", alpha=0.18, color=_SPINE)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    present = [m for m in ("audio", "dsp", "render")
               if any(_STAGE_MODULE.get(s) == m for s, _ in detail)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=_MODULE_COLOR[m]) for m in present]
    leg = ax.legend(handles, [_MODULE_NAME[m] for m in present], loc="upper right",
                    frameon=False, fontsize=11, labelspacing=0.8, handlelength=1.6,
                    handleheight=1.2, borderpad=0.9)
    for txt in leg.get_texts():
        txt.set_color(_FG)
    return _save(fig, out)


# Log-axis ticks: one decade per step, ms below a second, seconds above.
_TIME_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
_TIME_TICK_LABELS = ["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]


def _method_label(r):
    """Display name for a method row (CuPy = GPU, NumPy = CPU)."""
    m, b = r["method"], r["backend"]
    if m == "CWT":
        return "SubShader CuPy CWT" if b == "GPU" else "SubShader NumPy CWT"
    return m  # "SciPy STFT", "PyWavelet CWT"


def _res_side(r):
    """Short resolution note for the right-hand column."""
    if r.get("res_kind") == "log":
        return "1 semitone / octave"
    parts = (r.get("native_res") or "").split()
    hz = parts[0] if parts else "?"
    return f"{hz} Hz · fixed"


def _time_block(ms):
    """In-bar time string — seconds past 1 s, else milliseconds."""
    ms = float(ms)
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.1f} ms"


def render_methods_figure(out=METHODS_PNG):
    """Fourier vs Wavelet — time per frame (log scale) with resolution in-bar."""
    rows = latest_method_rows()
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["mean_ms"]))  # fastest at top
    n = len(rows)
    vals = [float(r["mean_ms"]) for r in rows]

    def _color(r):
        # Match the flowchart lanes: GPU = orange, CPU = purple.
        return _RENDER if r["backend"] == "GPU" else _DSP

    fig, ax = _bar_fig(n, _ROW_IN, 2.7)
    y = list(range(n))
    ax.barh(y, vals, height=_BAR_H, color=[_color(r) for r in rows],
            edgecolor=_BG, linewidth=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, 5_000_000)        # headroom on the right for the resolution column
    ax.set_xticks(_TIME_TICKS)
    ax.set_xticklabels(_TIME_TICK_LABELS)
    # Time trailing each bar in bold white; resolution as a quiet right column.
    for i, r in enumerate(rows):
        ms = float(r["mean_ms"])
        ax.text(ms * 1.18, i, _time_block(ms), va="center_baseline", ha="left",
                fontsize=13, fontweight="bold", color="#ffffff", zorder=4)
        ax.text(0.995, i, _res_side(r), transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=11, color=_TICK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([_method_label(r) for r in rows])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("time per frame — log scale (ms / s)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13, length=0)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", which="major", alpha=0.28, color=_SPINE)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    key = [("GPU", _RENDER), ("CPU", _DSP)]
    # Top-right corner, inset left of the resolution column so they don't collide.
    _legend_corner(ax, key, bbox=(0.80, 0.97))
    return _save(fig, out)


def render_config_figure(out=CONFIG_PNG):
    """Per-frame compute across the configs measured (GPU orange, CPU purple)."""
    rows = config_summary_rows()
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["mean_ms"])
    n = len(rows)
    fig, ax = _bar_fig(n, _ROW_IN, 3.0)
    y = list(range(n))
    colors = [_RENDER if "Gpu" in r["backend"] else _DSP for r in rows]
    ax.barh(y, [r["mean_ms"] for r in rows], height=_BAR_H, color=colors,
            edgecolor=_BG, linewidth=1.0, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r["mean_ms"], i, f"  {r['mean_ms']:.1f} ms · {r['rt']:.0f}× real-time",
                va="center_baseline", ha="left", fontsize=13, fontweight="bold",
                color="#ffffff", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['backend']} · {r['chunk']} · {r['freqs']}f" for r in rows])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("compute per frame (ms)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13, length=0)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(0, max(r["mean_ms"] for r in rows) * 1.55)  # room for trailing labels
    ax.grid(axis="x", alpha=0.18, color=_SPINE)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def update_report(csv=RESULTS_CSV):
    """Regenerate every figure and the Markdown report from the latest data."""
    render_pipeline_figure(csv=csv)
    render_methods_figure()
    render_config_figure()
    return render_markdown(csv_path=csv)


if __name__ == "__main__":
    update_report()
