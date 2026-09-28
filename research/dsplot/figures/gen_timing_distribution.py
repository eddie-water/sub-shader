"""Timing distribution figures - per-sample dot clouds over the same cascade.

Companion to ``gen_timing_gantt``: evolves each gantt row from one mean bar
into a cloud of its raw per-sample measurements, so the report can show the
spread the mean bar hides (the campaign's Process Loop capture in particular
turned out bimodal - see the module docstring further down). Rows stay
anchored at the SAME cascade offset the standard gantt draws (own row's mean
duration accumulates the next row's start), reuse the same boxed-letter axis
and dark banner aesthetic, and mark each row's mean with a bright tick so the
cloud and the standard gantt's bar read as the same number.

Candidate variants (2-3, per the plan - beeswarm-class only, nothing
elaborate):
    render_startup_beeswarm  - Start Up init samples (N=12), deterministic
        non-overlapping swarm stacking (sparse - few points per row).
    render_startup_jitter    - Start Up init samples (N=12), random jitter
        + mean tick (the same data, a different simple read).
    render_runtime_beeswarm  - Process Loop per-frame samples (N=300+ frames),
        same swarm algorithm at higher density - small low-alpha dots so
        overlapping clusters read as weight rather than a solid blob.

Data source: ``assets/timing/timing_init_samples.csv`` (per-sample init
timings, written by ``research/timing_campaign.py``) and
``assets/timing/timing_iterations.csv`` (per-frame loop timings, written by
``research/utilities/timing_results.py::TimingRecorder``). Both are read for
the SAME run_id ``gen_timing_gantt._latest_run`` resolves for the standard
gantts, so this figure and the mean-bar gantt always describe one run.

This module is figure-script-level by default (per the plan's own stated
default) and reuses ``gen_timing_gantt``'s stage tables, cascade math, and
banner/composition helpers by import rather than duplicating them. The one
new primitive it needs - a per-sample scatter "dot cloud" - is defined LOCAL
to this module rather than added to the shared ``dsplot`` library (the
project's standing convention: don't bend the shared library for one
figure); it subclasses the same ``Plottable`` contract every other dsplot
primitive does, so it composes onto a ``BarPanel`` exactly like ``Barh`` or
``Line`` do.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .. import Figure, BarPanel, Barh, Line, Annotation, style
from ..plottables.base import Plottable
from .gen_timing_gantt import (
    _FINE, _FINE_NUM, _LOOP, _LOOP_NUM,
    _latest_run, _su_ticks, _repo_root,
    _dur_label, _total_label, _compose_one, _render_to,
    PANEL_UNITS, ROW_IN, BAR_H, TITLE_HEAD, AXIS_GAP, AXIS_TAIL,
    GUTTER_FRAC, RIGHT_FRAC, AXES_W_IN, LETTER_BOX_H, LETTER_BOX_EDGE,
    LETTER_SIZE, MEASURE_SIZE, TICK_SIZE, NAME_SIZE,
)

INIT_SAMPLES_CSV = os.path.join(_repo_root(), "assets", "timing", "timing_init_samples.csv")
ITERATIONS_CSV = os.path.join(_repo_root(), "assets", "timing", "timing_iterations.csv")

# Physical dot footprint (inches) - converted to data-space width per row the
# same way gen_timing_gantt derives its physically-square letter box
# (`box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN`), so beeswarm binning
# reflects how wide a dot actually renders on the canvas.
DOT_DIAM_IN = {"startup": 0.10, "runtime": 0.055}
DOT_SIZE_PT = {"startup": 170.0, "runtime": 26.0}   # matplotlib scatter `s`
DOT_ALPHA = {"startup": 0.85, "runtime": 0.45}
MEAN_TICK_COLOR = "#ffffff"
MEAN_TICK_WIDTH = 3.2
ROW_SPAN_FRAC = 0.82          # fraction of BAR_H the dot cloud may occupy


class _DotCloud(Plottable):
    """One row's raw per-sample points. Kept local to this figure module -
    not added to the shared dsplot plottable vocabulary (project convention:
    don't bend the shared library for one figure) - but implements the same
    ``Plottable`` contract so it composes onto a ``BarPanel`` like any other
    primitive (``panel.add(...)``, lazy style resolution at draw()).
    """

    def __init__(self, x, y, *, color=None, size=60.0, alpha=0.8, zorder=4):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.size = size

    def draw(self, ax):
        color = self.color if self.color is not None else style.PRIMARY_COLOR
        ax.scatter(self.x, self.y, s=self.size, c=color,
                   edgecolors=style.BG_COLOR, linewidths=0.9,
                   alpha=self.alpha, zorder=self.zorder)


def _beeswarm_offsets(values, row_span, dot_diam):
    """Deterministic non-overlapping-ish swarm: sort, bin by dot width, stack
    each bin's points in a zigzag (0, +1, -1, +2, -2, ...) around the row
    center. Offsets are clipped to the row's span, so a bin far denser than
    the row is tall (e.g. the loop capture's throttled-cluster spike) settles
    into overlap at the row edges rather than colliding with neighbor rows.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    offsets = np.zeros(n)
    if n == 0:
        return offsets
    bin_w = max(dot_diam, 1e-9)
    order = np.argsort(values)
    bin_of = np.floor((values - values.min()) / bin_w).astype(int)
    seen = {}
    for idx in order:
        b = int(bin_of[idx])
        k = seen.get(b, 0)
        seen[b] = k + 1
        level = (k + 1) // 2
        sign = 1 if k % 2 == 1 else -1
        offsets[idx] = sign * level * dot_diam
    half = row_span / 2.0
    return np.clip(offsets, -half, half)


def _jitter_offsets(n, row_span, seed):
    """Deterministic (seeded) random jitter within the row's span."""
    if n == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    half = row_span / 2.0
    return rng.uniform(-half, half, size=n)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _samples_by_stage(rows, run_id):
    """{stage: np.array([ms, ...])} for one run_id from a per-sample sidecar."""
    out = {}
    for r in rows:
        if r["run_id"] != run_id:
            continue
        out.setdefault(r["stage"], []).append(float(r["ms"]))
    return {k: np.array(v) for k, v in out.items()}


def _fold(sample_map, absorb_key, into_key):
    """Elementwise-sum `absorb_key`'s per-sample array into `into_key`'s (same
    fold gen_timing_gantt._gantt_data applies to the mean: prime -> prescan,
    gl_clear -> gl_draw), then drop `absorb_key`. Per-sample arrays pair by
    index (sample i / frame i), so the sum stays meaningful per-sample.
    """
    if absorb_key not in sample_map:
        return
    absorbed = sample_map.pop(absorb_key)
    base = sample_map.get(into_key)
    if base is None:
        sample_map[into_key] = absorbed
        return
    n = min(len(base), len(absorbed))
    sample_map[into_key] = base[:n] + absorbed[:n]


def _cascade_rows(order, num_map, sample_map):
    """Walk `order` [(key, label, color), ...], keep keys present in
    `sample_map`, and accumulate cascade offsets by each row's OWN mean -
    identical anchoring convention to gen_timing_gantt._block_panel's bar
    cascade (`start += dur`), so a row's dot cloud sits exactly where the
    standard gantt's mean bar for that stage starts.
    """
    rows = []
    start = 0.0
    for key, _label, color in order:
        samples = sample_map.get(key)
        if samples is None or samples.size == 0:
            continue
        rows.append((num_map.get(key), key, color, start, samples))
        start += float(np.mean(samples))
    return rows, start


def _load_startup_rows(run_id):
    sample_map = _samples_by_stage(_read_csv(INIT_SAMPLES_CSV), run_id)
    _fold(sample_map, "init:prime", "init:prescan")
    return _cascade_rows(_FINE, _FINE_NUM, sample_map)


def _load_runtime_rows(run_id):
    sample_map = _samples_by_stage(_read_csv(ITERATIONS_CSV), run_id)
    _fold(sample_map, "gl_clear", "gl_draw")
    return _cascade_rows(_LOOP, _LOOP_NUM, sample_map)


def _rt_dist_ticks(total_ms):
    """Adaptive ms ticks for the loop block. gen_timing_gantt._rt_ticks caps
    at 9 ms - tuned for the ~10 ms steady-state loop the short captures used
    to show. This campaign's 300+ frame capture measured a much larger total
    (idle-gap GPU clock ramp-up under sustained load - see this figure's
    module docstring), so this distribution figure picks its own round tick
    step instead of inheriting that fixed cap.
    """
    if total_ms <= 0:
        return [0], ["0 ms"]
    for step in (5, 10, 15, 20, 25, 50, 100):
        if total_ms / step <= 6:
            break
    ticks = list(range(0, int(total_ms) + step, step))
    ticks = [t for t in ticks if t <= total_ms + step * 0.2] or [0]
    labels = [f"{t:g}" for t in ticks]
    labels[-1] += " ms"
    return ticks, labels


def _dist_panel(title, rows, total_end, ticks_ms, tick_labels, *,
                 kind, mode, seed=0):
    """One distribution block: boxed-letter axis, per-row dot cloud + mean
    tick, hero-style axis. Mirrors gen_timing_gantt._block_panel's geometry
    (gutter/xlim/ylim/label placement) so the two figure families sit flush;
    the cascade bar itself is swapped for a scatter cloud.
    """
    n = len(rows)
    span = total_end
    gutter = GUTTER_FRAC * span
    # Right headroom: the standard gantt's RIGHT_FRAC assumes labels trail a
    # bar's END (<= span); a per-sample cloud can scatter PAST span (e.g. an
    # outlier sample), so size x_hi off the actual rightmost point (plus its
    # trailing duration label) rather than a fixed fraction of the total.
    data_max = max((start + float(samples.max()) for _, _, _, start, samples in rows),
                   default=span)
    x_hi = max(RIGHT_FRAC * span, data_max * 1.14)
    head = TITLE_HEAD
    y_axis = n + AXIS_GAP
    panel = BarPanel(
        units=PANEL_UNITS,
        xlim=(-gutter, x_hi),
        ylim=(y_axis + AXIS_TAIL, -head),
        xticks=[], xticklabels=[],
        show_xticklabels=False,
        show_border=False,
    )
    label_pad = 0.011 * (x_hi + gutter)
    x_range = x_hi + gutter

    panel.add(Annotation(title, (0.0, -1.15), ha="left", va="bottom",
                         color=style.NEUTRAL_COLOR, fontsize=NAME_SIZE,
                         fontweight="bold"))

    box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN
    dot_w_data = DOT_DIAM_IN[kind] * x_range / AXES_W_IN
    row_span = BAR_H * ROW_SPAN_FRAC
    dot_size = DOT_SIZE_PT[kind]
    alpha = DOT_ALPHA[kind]

    for i, (num, _key, color, start, samples) in enumerate(rows):
        mean_x = start + float(np.mean(samples))
        x_vals = start + samples
        if mode == "beeswarm":
            y_off = _beeswarm_offsets(samples, row_span, dot_w_data * 1.05)
        else:
            y_off = _jitter_offsets(len(samples), row_span, seed=seed + i)
        y_vals = np.full(samples.shape, float(i)) + y_off
        panel.add(_DotCloud(x_vals, y_vals, color=color, size=dot_size, alpha=alpha))

        panel.add(Line([mean_x, mean_x], [i - 0.42, i + 0.42],
                       color=MEAN_TICK_COLOR, linewidth=MEAN_TICK_WIDTH, zorder=6))

        if num:
            panel.add(Barh([float(i)], [box_w], lefts=[-label_pad - box_w],
                           color="#000000", height=LETTER_BOX_H,
                           edgecolor=color, edgewidth=LETTER_BOX_EDGE))
            panel.add(Annotation(num, (-label_pad - box_w / 2.0, float(i)),
                                 ha="center", va="center", color="#ffffff",
                                 fontsize=LETTER_SIZE, fontweight="bold"))

        label_x = max(float(x_vals.max()), mean_x) + label_pad
        panel.add(Annotation(_dur_label(mean_x - start), (label_x, float(i)),
                             ha="left", va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))

    # Total reference - a quiet dashed rule where the cascade ends (no solid
    # Total bar/leg here; this figure's job is the per-row spread, not a
    # restatement of the standard gantt's Total block).
    panel.add(Line([span, span], [-0.5, y_axis], color=style.SPINE_COLOR,
                   linewidth=2.0, linestyle="--"))
    panel.add(Annotation(f"{_total_label(span)} total", (span - label_pad, -0.65),
                         ha="right", va="bottom", color=style.NEUTRAL_COLOR,
                         fontsize=MEASURE_SIZE * 0.75))

    panel.add(Line([0.0, span], [y_axis, y_axis],
                   color=style.NEUTRAL_COLOR, linewidth=2.5))
    for t, lbl in zip(ticks_ms, tick_labels):
        panel.add(Line([t, t], [y_axis, y_axis + 0.32],
                       color=style.NEUTRAL_COLOR, linewidth=2.5))
        panel.add(Annotation(lbl, (t, y_axis + 0.62), ha="center", va="top",
                             color="#ffffff", fontsize=TICK_SIZE))
    return panel


def build_startup_figure(mode: str) -> Figure:
    run = _latest_run()
    run_id = run[0]["run_id"]
    rows, total = _load_startup_rows(run_id)
    ticks, labels = _su_ticks(total)
    n = rows[0][4].size if rows else 0
    title = f"Start Up - Per-Sample Init Timing (N={n})"
    panel = _dist_panel(title, rows, total, ticks, labels,
                        kind="startup", mode=mode, seed=1)
    return _compose_one(panel, len(rows), head=TITLE_HEAD)


def build_runtime_figure(mode: str) -> Figure:
    run = _latest_run()
    run_id = run[0]["run_id"]
    rows, total = _load_runtime_rows(run_id)
    ticks, labels = _rt_dist_ticks(total)
    n = rows[0][4].size if rows else 0
    title = f"Process Loop - Per-Frame Timing (N={n})"
    panel = _dist_panel(title, rows, total, ticks, labels,
                        kind="runtime", mode=mode, seed=7)
    return _compose_one(panel, len(rows), head=TITLE_HEAD)


def render_startup_beeswarm(output_path=None) -> str:
    """_render_to already wraps the build/save in _banner_style()."""
    return _render_to(lambda: build_startup_figure("beeswarm"),
                      "timing_startup_dist_beeswarm_v1.png", output_path)


def render_startup_jitter(output_path=None) -> str:
    return _render_to(lambda: build_startup_figure("jitter"),
                      "timing_startup_dist_jitter_v1.png", output_path)


def render_runtime_beeswarm(output_path=None) -> str:
    return _render_to(lambda: build_runtime_figure("beeswarm"),
                      "timing_runtime_dist_beeswarm_v1.png", output_path)


if __name__ == "__main__":
    print(render_startup_beeswarm())
    print(render_startup_jitter())
    print(render_runtime_beeswarm())
