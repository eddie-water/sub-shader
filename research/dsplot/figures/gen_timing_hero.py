"""Timing figure - Start Up vs Runtime Loop (the pipeline-timing hero).

Sibling of ``gen_timing_methods``: the same dark optichrome template - title
band, white-bordered panel grid, right-hand caption column - hosting the
discretized timing strip. One square = one loop of work; start-up runs as a
single proportional row of squares, then the "playback starts" divider, then
ONE frame budget (a work square plus the measured-idle squares) with an
ellipsis - it repeats identically forever.

Data source: ``assets/timing/timing_results.csv`` (written by
``research/timing_live.py``). This module only renders. All counts derive from
the CSV at render time: cell = TOTAL mean, budget = round(deadline / cell),
init = each init stage rounded to cells.
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
    style,
)

PANEL_UNITS = (4, 1)

CELL_PAD = 0.07                       # inset per square so the grid reads
EMPTY_FACE, EMPTY_EDGE = "#222222", "#3a3a3a"

# In-strip type tiers (28" canvas): names ride the bar-value tier, measures the
# in-bar tier - the same two voices the methods figure uses.
NAME_SIZE = style.DEFAULT_BAR_VALUE_FONT_SIZE      # 32
MEASURE_SIZE = style.DEFAULT_BAR_INBAR_FONT_SIZE   # 30

# Init stages in construction order → module color. Fine per-module sub-stages
# when the run recorded them, else the coarse module rows (same fallback as the
# timing report's gantt).
_FINE = [
    ("init:audio_reader", style.TERTIARY_COLOR),
    ("init:audio_player", style.TERTIARY_COLOR),
    ("init:dsp_kernels", style.SECONDARY_COLOR),
    ("init:dsp_fft", style.SECONDARY_COLOR),
    ("init:dsp_upload", style.SECONDARY_COLOR),
    ("init:render_buffer", style.PRIMARY_COLOR),
    ("init:render_glcontext", style.PRIMARY_COLOR),
    ("init:render_shader", style.PRIMARY_COLOR),
    ("init:prescan", style.PRIMARY_COLOR),
    # prime (GPU warmup) folds into the grey Setup catch-all - one cyan square
    # read as noise, and it's construction overhead like the rest of `other`.
    ("init:prime", style.SPINE_COLOR),
    ("init:other", style.SPINE_COLOR),
]
_COARSE = [
    ("init:audio", style.TERTIARY_COLOR),
    ("init:dsp", style.SECONDARY_COLOR),
    ("init:renderer", style.PRIMARY_COLOR),
    ("init:prescan", style.PRIMARY_COLOR),
    ("init:prime", style.SPINE_COLOR),
    ("init:other", style.SPINE_COLOR),
]
_COARSE_KEYS = {k for k, _ in _COARSE}
_LEGEND = [
    ("Audio", style.TERTIARY_COLOR),
    ("DSP", style.SECONDARY_COLOR),
    ("Render", style.PRIMARY_COLOR),
    ("Setup", style.SPINE_COLOR),
]


# Banner-local layout scale: the README hero is a borderless full-bleed strip,
# so the family's 1.5" pad would waste most of the canvas. Same save/restore
# mechanism as style.render_profile - applied around build+render only, so the
# globals other figures read are untouched.
_BANNER_OVERRIDES = {
    "DEFAULT_PAD_INCHES": 0.15,
    "DEFAULT_MARGIN_INCHES": 0.15,
    "DEFAULT_GUTTER_INCHES": 0.3,
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.3,
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


def _strip_data() -> dict:
    """Everything the strip needs, derived from the CSV."""
    run = _latest_run()
    if not run:
        raise RuntimeError("No timing rows in assets/timing/timing_results.csv")
    ms = {r["stage"]: float(r["mean_ms"]) for r in run}
    total_row = next(r for r in run if r["stage"] == "TOTAL")
    work = float(total_row["mean_ms"])
    rt = float(total_row["rt_margin"])
    deadline = rt * work
    wait = ms.get("wait:next_chunk", 0.0)
    fine = any(k in ms and ms[k] > 0 for k, _ in _FINE
               if k not in _COARSE_KEYS)
    order = _FINE if fine else _COARSE
    init_cells: list[str] = []
    for key, color in order:
        init_cells.extend([color] * int(round(ms.get(key, 0.0) / work)))
    init_total = sum(ms.get(k, 0.0) for k, _ in _COARSE)
    return dict(
        work=work, deadline=deadline, wait=wait,
        budget_cells=max(2, int(round(deadline / work))),
        init_cells=init_cells, init_total=init_total,
    )


def _fmt_total(ms_val: float) -> str:
    return f"{ms_val / 1000:.1f} s" if ms_val >= 1000 else f"{ms_val:.0f} ms"


class StripPanel(BarPanel):
    """BarPanel whose data units render physically SQUARE.

    After the stock render, the y-range is recomputed from the axes box's real
    aspect so one x-unit equals one y-unit on canvas - the squares stay square
    regardless of how compose sizes the cell. Figure-local subclass; nothing in
    the library references it.
    """

    def __init__(self, *, content_center: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.content_center = content_center

    def render(self) -> None:
        super().render()
        ax = self.ax
        fig = ax.figure
        pos = ax.get_position()
        w_in = pos.width * fig.get_figwidth()
        h_in = pos.height * fig.get_figheight()
        x0, x1 = ax.get_xlim()
        yspan = (x1 - x0) * (h_in / w_in)
        ax.set_ylim(self.content_center - yspan / 2.0,
                    self.content_center + yspan / 2.0)


def _squares(xs: list[float], y: float, colors: list[str]) -> Barh:
    """A run of unit squares (bottom edge at y) as one Barh call."""
    side = 1.0 - 2.0 * CELL_PAD
    return Barh(
        [y + 0.5] * len(xs), [side] * len(xs),
        lefts=[x + CELL_PAD for x in xs],
        colors=colors, height=side, edgewidth=0.0,
    )


def _empty_squares(xs: list[float], y: float) -> Barh:
    side = 1.0 - 2.0 * CELL_PAD
    return Barh(
        [y + 0.5] * len(xs), [side] * len(xs),
        lefts=[x + CELL_PAD for x in xs],
        color=EMPTY_FACE, height=side,
        edgecolor=EMPTY_EDGE, edgewidth=1.2,
    )


def _ruler(panel: BarPanel, xa: float, xb: float, y: float) -> None:
    panel.add(Line([xa, xb], [y, y], color=style.SPINE_COLOR, linewidth=2.0))
    for xx in (xa, xb):
        panel.add(Line([xx, xx], [y, y + 0.35],
                       color=style.SPINE_COLOR, linewidth=2.0))


def build_figure() -> Figure:
    """PULSE layout - one continuous timeline, butt-to-butt.

    Start-up runs as a solid strip of squares; the moment it ends, playback
    starts and the first work square lands - then work squares pulse along a
    bare track, one per frame budget. Headroom is the emptiness between
    pulses. A visible x-axis (ticks at the joint and each hop) and a bare
    y-axis frame the timeline; no legend, no idle outlines, white type.
    """
    d = _strip_data()
    deadline = d["deadline"]
    init_cells, budget_cells = d["init_cells"], d["budget_cells"]

    n = len(init_cells)
    pulses = 3                          # the joint frame + two more hops
    track_x1 = n + (pulses - 1) * budget_cells + 1.0
    x_hi = track_x1 + 0.4

    ty = 3.4          # names row (va bottom)
    ax_y = -1.5       # x-axis line
    ly = -3.2         # measures row (va top)

    strip = StripPanel(
        units=PANEL_UNITS,
        xlim=(-0.4, x_hi),
        ylim=(-6.0, 6.4),               # replaced by the square-aspect ylim
        content_center=0.2,
        xticks=[], xticklabels=[],
        show_xticklabels=False,
        show_border=False,
    )

    # the timeline: start-up squares, then work squares pulsing on a bare track
    strip.add(_squares(list(range(n)), 0.0, init_cells))
    strip.add(Line([n, track_x1], [0.5, 0.5], color=style.SPINE_COLOR,
                   linewidth=2.0, zorder=2))
    strip.add(_squares([n + k * budget_cells for k in range(pulses)], 0.0,
                       ["#ffffff"] * pulses))

    # axes: visible x with ticks at the joint + each hop; bare y at the origin
    strip.add(Line([0, track_x1], [ax_y, ax_y], color=style.NEUTRAL_COLOR,
                   linewidth=2.5))
    for xx in [0, n] + [n + k * budget_cells for k in range(1, pulses)]:
        strip.add(Line([xx, xx], [ax_y, ax_y - 0.5],
                       color=style.NEUTRAL_COLOR, linewidth=2.5))
    strip.add(Line([0, 0], [ax_y, 2.4], color=style.NEUTRAL_COLOR,
                   linewidth=2.5))

    # names above - each anchored where its section starts
    strip.add(Annotation("Start Up", (0.6, ty), ha="left", va="bottom",
                         color=style.NEUTRAL_COLOR, fontsize=NAME_SIZE,
                         fontweight="bold"))
    strip.add(Annotation("Runtime Loop", (n, ty), ha="left", va="bottom",
                         color=style.NEUTRAL_COLOR, fontsize=NAME_SIZE,
                         fontweight="bold"))

    # measures below, white
    strip.add(Annotation(f"{_fmt_total(d['init_total'])} · once", (n / 2.0, ly),
                         ha="center", va="top", color=style.NEUTRAL_COLOR,
                         fontsize=MEASURE_SIZE))
    strip.add(Annotation(f"{deadline:.0f} ms",
                         (n + budget_cells / 2.0, ly), ha="center", va="top",
                         color=style.NEUTRAL_COLOR, fontsize=MEASURE_SIZE))

    # README banner: no title band, no borders, no legend - the timeline fills
    # the canvas. The README's own prose does the captioning.
    return Figure.compose(
        rows=[[strip]],
        row_heights=[1.0],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=3.0,
        dpi=style.FIGURE_DPI,
        show_cell_borders=False,
    )


def render(output_path: str | None = None) -> str:
    """Build, render, save. Returns absolute output path."""
    if output_path is None:
        output_path = os.path.join(_repo_root(), "assets", "timing",
                                   "timing_hero.png")
    with _banner_style():
        fig = build_figure()
        fig.render()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render())
