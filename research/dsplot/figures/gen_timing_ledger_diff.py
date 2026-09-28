"""Ledger diff figure - one fix, before vs after, and the subtraction.

Three rows on ONE shared work-scale axis (the before run's total):

    before  - the Process Loop's stages end to end, module colors
    after   - the same, from the fix's after run
    saved   - before minus after, stage by stage: only the stages that
              changed get a segment, each carrying its letter chip
              (labelled "cost" and flipped to after minus before when the
              whole frame got slower - a fix that spends headroom)

Same data grammar as ``gen_timing_gantt`` (stage keys, letters, colors), but
reads TWO explicit run_ids instead of the latest run, so a ledger entry pins
the exact runs its tables came from. Iteration outputs land in NEW
``timing_ledger_diff_v*.png`` files - never overwrite an existing PNG.
"""
from __future__ import annotations

import csv
import os

from .. import Figure, BarPanel, Barh, Line, Annotation, style
from . import gen_timing_gantt as gtg
from .gen_timing_gantt import (
    _LOOP, _LOOP_NUM, _repo_root, BAR_H, ROW_IN, AXES_W_IN, HEAD_BARE,
    AXIS_GAP, AXIS_TAIL, MEASURE_SIZE, TICK_SIZE, LETTER_SIZE, LETTER_BOX_H,
    LETTER_BOX_EDGE, LETTER_NUDGE, AXIS_LINE_W, PANEL_UNITS,
)
from .gen_timing_deadline_split import _dur_label_fine

ROW_LABELS = ("before", "after", "saved")
GUTTER_FRAC = 0.11            # room for the word labels (wider than the
                              # letter-only gutter of the cascades)
RIGHT_FRAC = 1.13
CHANGE_FLOOR_MS = 0.05        # a stage must move this much to earn a "saved"
                              # segment - below it is run-to-run noise
SAVED_ROW_GAP = 0.35          # extra breathing room above the saved row


def _run_rows(run_id: str) -> list[dict]:
    path = os.environ.get("TIMING_RESULTS_CSV") or os.path.join(
        _repo_root(), "assets", "timing", "timing_results.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["run_id"] == run_id]
    if not rows:
        raise SystemExit(f"run_id {run_id!r} not in timing_results.csv")
    return rows


def _loop_segments(run_id: str):
    """[(letter, key, dur_ms, color)] in execution order, plus TOTAL mean."""
    rows = _run_rows(run_id)
    ms = {r["stage"]: float(r["mean_ms"]) for r in rows}
    if "gl_clear" in ms:
        ms["gl_draw"] = ms.get("gl_draw", 0.0) + ms.pop("gl_clear")
    segs = [(_LOOP_NUM.get(key), key, ms[key], color)
            for key, _, color in _LOOP if ms.get(key, 0.0) > 0]
    total_row = next((r for r in rows if r["stage"] == "TOTAL"), None)
    total = float(total_row["mean_ms"]) if total_row else sum(s[2] for s in segs)
    return segs, total


def _draw_stack(panel, y, segs, x_range, *, chips: bool):
    """One row: the segments laid end to end. ``chips`` puts each segment's
    letter box centered above it (the saved row's few segments)."""
    box_w = LETTER_BOX_H * ROW_IN * x_range / AXES_W_IN
    gap = 0.004 * x_range if chips else 0.0   # seam between same-color pieces
    start = 0.0
    for letter, _, dur, color in segs:
        panel.add(Barh([y], [max(dur - gap, 0.0)], lefts=[start],
                       colors=[color], height=BAR_H, edgewidth=0.0))
        if chips and letter:
            cx = start + dur / 2.0
            panel.add(Barh([y - 1.0], [box_w], lefts=[cx - box_w / 2.0],
                           color="#000000", height=LETTER_BOX_H,
                           edgecolor=color, edgewidth=LETTER_BOX_EDGE))
            panel.add(Annotation(letter, (cx, y - 1.0 + LETTER_NUDGE),
                                 ha="center", va="center", color="#ffffff",
                                 fontsize=LETTER_SIZE, fontweight="bold"))
        start += dur
    return start


def build_ledger_diff_figure(run_before: str, run_after: str) -> Figure:
    before, total_b = _loop_segments(run_before)
    after, total_a = _loop_segments(run_after)
    dur_a = {key: dur for _, key, dur, _ in after}
    # Third row follows the whole-frame direction: a net speedup collects
    # the stages that shrank ("saved"); a net slowdown collects the stages
    # that grew ("cost") - e.g. a visual fix that deliberately spends headroom.
    sign = 1.0 if total_b >= total_a else -1.0
    third = [(letter, key, sign * (dur - dur_a.get(key, 0.0)), color)
             for letter, key, dur, color in before
             if sign * (dur - dur_a.get(key, 0.0)) > CHANGE_FLOOR_MS]
    third_label = "saved" if sign > 0 else "cost"
    total_s = sign * (total_b - total_a)

    span = max(total_b, total_a)
    gutter = GUTTER_FRAC * span
    x_hi = RIGHT_FRAC * span
    x_range = x_hi + gutter
    label_pad = 0.011 * x_range

    ys = [0.0, 1.0, 2.0 + 1.0 + SAVED_ROW_GAP]   # saved row leaves a chip lane
    y_axis = ys[-1] + AXIS_GAP + 0.5
    panel = BarPanel(units=PANEL_UNITS, xlim=(-gutter, x_hi),
                     ylim=(y_axis + AXIS_TAIL, -HEAD_BARE),
                     xticks=[], xticklabels=[], show_xticklabels=False,
                     show_border=False)

    for y, label, segs, total in zip(ys, (*ROW_LABELS[:2], third_label),
                                     (before, after, third),
                                     (total_b, total_a, total_s)):
        panel.add(Annotation(label, (-label_pad, y), ha="right", va="center",
                             color="#ffffff", fontsize=MEASURE_SIZE,
                             fontweight="bold"))
        end = _draw_stack(panel, y, segs, x_range, chips=(label == third_label))
        text = _dur_label_fine(total)
        # Keep the measure clear of the after-row rule when it lies past the end
        # of this row (a slower after run crosses the before row's label).
        lx = end + label_pad
        if label != third_label and end < total_a:
            lx = total_a + label_pad
        panel.add(Annotation(text, (lx, y), ha="left",
                             va="center", color="#ffffff",
                             fontsize=MEASURE_SIZE, fontweight="bold"))

    # After row's end carried up/down as a thin white rule - the subtraction
    # made visible: what sticks out past it on the before row is what the
    # saved row collects.
    panel.add(Line([total_a, total_a], [-BAR_H / 2.0, ys[1] + BAR_H / 2.0],
                   color="#ffffff", linewidth=2.5, linestyle=(0, (16, 4))))

    step = 1 if span <= 5 else 2 if span <= 16 else 5
    ticks = list(range(0, int(span) + 1, step))
    panel.add(Line([0.0, 0.0], [-BAR_H / 2.0, y_axis],
                   color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
    panel.add(Line([0.0, span], [y_axis, y_axis],
                   color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
    for t in ticks:
        panel.add(Line([t, t], [y_axis, y_axis + 0.32],
                       color=style.NEUTRAL_COLOR, linewidth=AXIS_LINE_W))
        panel.add(Annotation(f"{t:g}", (t, y_axis + 0.62), ha="center",
                             va="top", color="#ffffff", fontsize=TICK_SIZE))

    rows = HEAD_BARE + y_axis + AXIS_TAIL
    return Figure.compose(rows=[[panel]], row_heights=[rows * ROW_IN],
                          total_width_inches=style.FIGURE_WIDTH_INCHES,
                          unit_height_inches=1.0, dpi=style.FIGURE_DPI,
                          show_cell_borders=False)


def render_ledger_diff(run_before: str, run_after: str,
                       output_path: str | None = None) -> str:
    with gtg._banner_style():
        fig = build_ledger_diff_figure(run_before, run_after)
        fig.render()
        path = output_path or os.path.join(_repo_root(), "assets", "timing",
                                           "timing_ledger_diff_v1.png")
        fig.savefig(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    import sys
    print(render_ledger_diff(sys.argv[1], sys.argv[2],
                             sys.argv[3] if len(sys.argv) > 3 else None))
