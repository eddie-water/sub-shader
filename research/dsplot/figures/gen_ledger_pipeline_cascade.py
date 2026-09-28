"""Ledger pipeline diagram - the Process Loop cascade, before vs after a fix.

Two K-X stage cascades (the runtime gantt's grammar: letter chips, module
colors, hero-style axis) stacked on ONE shared work-scale axis, each drawn
from an explicit run_id - the ledger's zoomed-in companion to the runtime
gantt, which always renders the latest run against the full 186 ms deadline.

Stage ORDER is per-block: the caller can hand each block its own execution
order, so a fix that moves a stage (Fix 4: the download now happens after
down-sample, GPU-resident post in between) shows the move in the diagram
itself - same letters, new position. Iteration outputs land in NEW
``ledger_fix*_pipeline_v*.png`` files - never overwrite an existing PNG.
"""
from __future__ import annotations

import os

from .. import Figure, style
from . import gen_timing_gantt as gtg
from .gen_timing_gantt import (
    _block_panel, _total_label, _repo_root,
    _LOOP, _LOOP_NUM, ROW_IN, TITLE_HEAD, AXIS_GAP, AXIS_TAIL,
)
from .gen_timing_ledger_diff import _run_rows

SPAN_PAD = 1.12               # x span past the slower block's total


def _reorder(order, key: str, after_key: str):
    """Move ``key``'s row to sit right after ``after_key``'s row."""
    row = next(r for r in order if r[0] == key)
    rest = [r for r in order if r[0] != key]
    at = next(i for i, r in enumerate(rest) if r[0] == after_key) + 1
    return rest[:at] + [row] + rest[at:]


# Fix 4's true execution order: post stages run on the GPU, only the finished
# frame downloads - P (Transfer ← GPU) moves from after O (IFFT) to after
# T (Down-sample). Letters stay bound to their stages.
LOOP_AFTER_FIX4 = _reorder(_LOOP, "download", "downsample")


def _segments(run_id: str, order):
    """[(letter, label, dur_ms, color)] in ``order``, plus the TOTAL mean."""
    rows = _run_rows(run_id)
    ms = {r["stage"]: float(r["mean_ms"]) for r in rows}
    if "gl_clear" in ms:
        ms["gl_draw"] = ms.get("gl_draw", 0.0) + ms.pop("gl_clear")
    segs = [(_LOOP_NUM.get(key), label, ms[key], color)
            for key, label, color in order if ms.get(key, 0.0) > 0]
    total_row = next((r for r in rows if r["stage"] == "TOTAL"), None)
    total = float(total_row["mean_ms"]) if total_row else sum(s[2] for s in segs)
    return segs, total


def build_pipeline_cascade_figure(run_before: str, run_after: str,
                                  order_after=None) -> Figure:
    order_after = order_after if order_after is not None else LOOP_AFTER_FIX4
    blocks = [("before", *_segments(run_before, _LOOP)),
              ("after", *_segments(run_after, order_after))]
    span = SPAN_PAD * max(total for _, _, total in blocks)
    step = 1 if span <= 5 else 2 if span <= 16 else 5

    panels, heights = [], []
    for title, segs, total in blocks:
        # Skip any scale tick sitting under this block's total tag - the tag
        # is the precise number there; a colliding round tick just smudges it.
        ticks = [t for t in range(0, int(span) + 1, step)
                 if abs(t - total) > 0.05 * span]
        tick_labels = [f"{t:g}" for t in ticks]
        events = [dict(x=total, kind="tick", color="#ffffff",
                       tag=_total_label(total))]
        panels.append(_block_panel(title, segs, total, ticks, tick_labels,
                                   events=events, span_ms=span,
                                   show_total_value=False))
        heights.append(TITLE_HEAD + len(segs) + AXIS_GAP + AXIS_TAIL)

    return Figure.compose(
        rows=[[p] for p in panels],
        row_heights=[h * ROW_IN for h in heights],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=1.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


def render_pipeline_cascade(run_before: str, run_after: str,
                            output_path: str | None = None) -> str:
    with gtg._banner_style():
        fig = build_pipeline_cascade_figure(run_before, run_after)
        fig.render()
        path = output_path or os.path.join(
            _repo_root(), "assets", "timing", "ledger_pipeline_cascade_v1.png")
        fig.savefig(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    import sys
    print(render_pipeline_cascade(sys.argv[1], sys.argv[2],
                                  sys.argv[3] if len(sys.argv) > 3 else None))
