"""Timing figures - the Process Loop split into its two stories.

The single runtime gantt tried to show headroom (needs the 186 ms axis) and
composition (needs an ~8 ms axis) on one scale; post clock-lock the stage bars
collapsed into slivers and most rows read "< 1 ms". Two figures now, one claim
each:

    render_deadline  - the high-level plot: one frame of work as a lit cell,
        the rest of the 186 ms window as empty headroom cells (the startup
        figure's leg grammar, standalone), dashed deadline rule. The only
        runtime figure that shows 186 ms.
    render_workscale - the stage cascade on the work's own axis (~8 ms), so
        the bars are legible; duration labels gain a sub-ms decimal tier.

Same data source, banner style, and row grammar as ``gen_timing_gantt`` -
everything layout is imported from there, nothing canonical is re-rendered.
Iteration outputs land in NEW ``timing_runtime_deadline_v*.png`` /
``timing_runtime_workscale_v*.png`` files - never overwrite an existing PNG.
"""
from __future__ import annotations

from contextlib import contextmanager

from matplotlib.patches import FancyBboxPatch, Rectangle

from ..plottables.base import Plottable
from . import gen_timing_gantt as gtg
from .gen_timing_gantt import (
    _gantt_data, _block_panel, _compose_one, _render_to, _total_label,
    _rt_ticks, BAR_H, CELL_PAD_FRAC,
)

HEADROOM_EDGE = "#ffffff"       # dashed outline of an empty headroom cell
HEADROOM_EDGE_W = 1.6
HEADROOM_DASHES = (4, 3)
DEADLINE_AXIS_END = 190.0       # t-axis end (ms) - snug past the 186 ms rule,
                                # so the lit cell keeps some width on canvas
Y_AXIS_TOP = -0.68              # y-axis reach above the cell row (row units,
                                # y grows downward; headroom tops out at -0.75)
GUTTER_FRAC_BARE = 0.02         # left gutter override - the family's 5% gutter
                                # exists to hold the row letter boxes, which
                                # this figure has none of; match the right
                                # run-out (x_hi = 1.02 x span) for an even
                                # border around the content


class _DashedCells(Plottable):
    """Empty headroom cells - outline only, dashed. Kept local to this figure
    (project convention: don't bend the shared library for one figure) but
    implements the ``Plottable`` contract so it composes onto a ``BarPanel``
    like any other primitive."""

    def __init__(self, lefts, width, y, height, *, zorder=3):
        super().__init__(color=HEADROOM_EDGE, zorder=zorder)
        self.lefts = lefts
        self.width = width
        self.y = y
        self.height = height

    def draw(self, ax) -> None:
        for left in self.lefts:
            ax.add_patch(Rectangle(
                (left, self.y - self.height / 2.0), self.width, self.height,
                facecolor="none", edgecolor=self.color,
                linewidth=HEADROOM_EDGE_W, linestyle=(0, HEADROOM_DASHES),
                zorder=self.zorder,
            ))


def _dur_label_fine(ms_val: float) -> str:
    """Work-scale duration tier: one decimal below 10 ms ("0.4 ms"), whole
    milliseconds above, and the flat floor pushed down to "< 0.1 ms"."""
    if ms_val >= 1000:
        return f"{ms_val / 1000:.1f} s"
    if ms_val >= 9.95:
        return f"{ms_val:.0f} ms"
    if ms_val < 0.05:
        return "< 0.1 ms"
    return f"{ms_val:.1f} ms"


@contextmanager
def _fine_labels():
    orig = gtg._dur_label
    gtg._dur_label = _dur_label_fine
    try:
        yield
    finally:
        gtg._dur_label = orig


def build_deadline_figure():
    """One frame of work vs the frame budget - the headroom made countable.

    The Total row renders as a single lit cell (cell_ms = the whole frame's
    work), then every following whole frame that would also fit before the
    deadline is drawn as an empty cell, and the dashed rule marks the budget
    itself. No cascade - composition lives in the work-scale figure.
    """
    d = _gantt_data()
    rt_total, deadline = d["rt_total"], d["deadline"]
    if not (deadline and rt_total):
        raise SystemExit("No deadline/rt_total in the latest run - cannot "
                         "place the deadline figure's headroom cells.")
    ticks, tick_labels = _rt_ticks(rt_total, deadline)
    events = [
        dict(x=rt_total, kind="tick", color="#ffffff",
             tag=_total_label(rt_total)),
        dict(x=deadline, color="#ffffff", dashed=True,
             tag=f"{deadline:.0f} ms"),
    ]
    span = DEADLINE_AXIS_END
    panel = _block_panel("", [], rt_total, ticks, tick_labels,
                         cell_ms=rt_total, events=events, span_ms=span,
                         show_total_value=False, close_label=None)
    n_cells = int(deadline // rt_total)
    pad = CELL_PAD_FRAC * rt_total
    lefts = [k * rt_total + pad for k in range(1, n_cells)]
    panel.add(_DashedCells(lefts, rt_total - 2 * pad, 0.0, BAR_H))
    panel.add(gtg.Line([0.0, 0.0], [Y_AXIS_TOP, gtg.AXIS_GAP],
                       color=gtg.style.NEUTRAL_COLOR,
                       linewidth=gtg.AXIS_LINE_W))
    panel.xlim = (-GUTTER_FRAC_BARE * span, 1.02 * span)
    return _compose_one(panel, 0)


def build_workscale_figure():
    """The stage cascade on the work's own axis - build_runtime_figure minus
    the deadline (that story now lives in the deadline figure), so ~8 ms
    fills the canvas and every bar is legible."""
    d = _gantt_data()
    rt_segments, rt_total = d["rt_segments"], d["rt_total"]
    step = 1 if rt_total <= 5 else 2 if rt_total <= 16 else 5
    ticks = list(range(0, int(rt_total) + 1, step))
    tick_labels = [f"{t:g}" for t in ticks]
    with _fine_labels():
        panel = _block_panel("", rt_segments, rt_total, ticks, tick_labels)
    return _compose_one(panel, len(rt_segments))


# Consolidated tier - the README's high-level stages: the key players (CWT,
# Sync Display) kept whole, everything small bucketed. Fresh single letters
# A-E; the detailed K-X alphabet lives in the detailed chart (TIMING.md).
# Each entry: (letter, name, detailed letters the stage buckets).
_CONSOLIDATED = [
    ("A", "Fetch Audio",  ("K",)),
    ("B", "CWT",          ("L", "M", "N", "O", "P")),
    ("C", "Post-process", ("Q", "R", "S", "T")),
    ("D", "Draw",         ("U", "V", "W")),
    ("E", "Sync Display", ("X",)),
]


def build_consolidated_figure():
    """The work-scale cascade at the README's zoom - one row per high-level
    stage, durations summed over the detailed letters each stage buckets.
    Single letters, so the chips are _block_panel's own."""
    d = _gantt_data()
    by_num = {num: (dur, color) for num, _, dur, color in d["rt_segments"]}
    segments = []
    for letter, name, members in _CONSOLIDATED:
        present = [by_num[m] for m in members if m in by_num]
        if not present:
            continue
        segments.append((letter, name, sum(p[0] for p in present),
                         present[0][1]))
    rt_total = d["rt_total"]
    step = 1 if rt_total <= 5 else 2 if rt_total <= 16 else 5
    ticks = list(range(0, int(rt_total) + 1, step))
    tick_labels = [f"{t:g}" for t in ticks]
    with _fine_labels():
        panel = _block_panel("", segments, rt_total, ticks, tick_labels)
    return _compose_one(panel, len(segments))


# --- Block tier: the flowchart's own symbols as row labels ------------------
# With five rows the y label can be the flowchart block itself (rounded
# square, module-color stroke, stage name inside, letter beneath - the
# mock_module_grammar_v5 symbol verbatim) and the rows can breathe. Geometry
# is patched onto gen_timing_gantt's module globals for the build only.
BLOCK_ROW_IN = 1.75             # row pitch (inches) - ~2.8x the fine gantt
BLOCK_BAR_H = 0.5               # bar thickness in row units (~0.9")
BLOCK_BOX_H = 0.74              # box side in row units (~1.3")
BLOCK_ROUND_FRAC = 0.14         # corner radius, fraction of side (mock: 1.8/13)
BLOCK_STROKE = 3.2              # box outline (mock's 2.8, one step heavier)
BLOCK_NAME_SIZE = 18
BLOCK_LETTER_SIZE = 26
BLOCK_GUTTER_FRAC = 0.075       # left gutter holds the box + its pad
BLOCK_LETTER_GAP = 0.04         # rows between box bottom and letter top

_BLOCK_NAMES = {
    "Fetch Audio": "Fetch\nAudio", "Post-process": "Post-\nprocess",
    "Sync Display": "Sync\nDisplay",
}


class _StageBlocks(Plottable):
    """The flowchart symbols, one per row, in the y-label gutter."""

    def __init__(self, rows, box_w, box_h, x_right, aspect, *, zorder=4):
        super().__init__(color="#ffffff", zorder=zorder)
        self.rows, self.box_w, self.box_h = rows, box_w, box_h
        self.x_right, self.aspect = x_right, aspect

    def draw(self, ax) -> None:
        x0 = self.x_right - self.box_w
        r = BLOCK_ROUND_FRAC * self.box_w
        for i, (letter, name, color) in enumerate(self.rows):
            cy = float(i)
            ax.add_patch(FancyBboxPatch(
                (x0, cy - self.box_h / 2.0), self.box_w, self.box_h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                mutation_aspect=self.aspect,
                facecolor="#000000", edgecolor=color,
                linewidth=BLOCK_STROKE, zorder=self.zorder))
            ax.text(x0 + self.box_w / 2.0, cy + 0.02,
                    _BLOCK_NAMES.get(name, name), color="#ffffff",
                    fontsize=BLOCK_NAME_SIZE, fontweight="bold",
                    ha="center", va="center", linespacing=1.0,
                    zorder=self.zorder + 1)
            ax.text(x0 + self.box_w / 2.0,
                    cy + self.box_h / 2.0 + BLOCK_LETTER_GAP, letter,
                    color="#ffffff", fontsize=BLOCK_LETTER_SIZE,
                    fontweight="bold", ha="center", va="top",
                    zorder=self.zorder + 1)


@contextmanager
def _block_geometry():
    patch = {"ROW_IN": BLOCK_ROW_IN, "BAR_H": BLOCK_BAR_H,
             "LETTER_BOX_H": BLOCK_BOX_H, "GUTTER_FRAC": BLOCK_GUTTER_FRAC}
    orig = {k: getattr(gtg, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(gtg, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(gtg, k, v)


def build_consolidated_blocks_figure():
    """build_consolidated_figure with the flowchart symbols as row labels
    and roomier rows - chart and flowchart sharing literal geometry."""
    d = _gantt_data()
    by_num = {num: (dur, color) for num, _, dur, color in d["rt_segments"]}
    rows, segments = [], []
    for letter, name, members in _CONSOLIDATED:
        present = [by_num[m] for m in members if m in by_num]
        if not present:
            continue
        dur, color = sum(p[0] for p in present), present[0][1]
        rows.append((letter, name, color))
        segments.append((None, name, dur, color))    # no built-in letter box
    rt_total = d["rt_total"]
    step = 1 if rt_total <= 5 else 2 if rt_total <= 16 else 5
    ticks = list(range(0, int(rt_total) + 1, step))
    tick_labels = [f"{t:g}" for t in ticks]
    with _fine_labels(), _block_geometry():
        panel = _block_panel("", segments, rt_total, ticks, tick_labels)
        x_lo, x_hi = panel.xlim
        x_range = x_hi - x_lo
        ms_per_in = x_range / gtg.AXES_W_IN
        box_w = BLOCK_BOX_H * BLOCK_ROW_IN * ms_per_in       # square
        aspect = (1.0 / BLOCK_ROW_IN) / ms_per_in            # rows/ms ratio
        label_pad = 0.011 * x_range
        panel.add(_StageBlocks(rows, box_w, BLOCK_BOX_H, -label_pad, aspect))
        return _compose_one(panel, len(segments))


def render_deadline(output_path=None) -> str:
    return _render_to(build_deadline_figure,
                      "timing_runtime_deadline_v5.png", output_path)


def render_workscale(output_path=None) -> str:
    return _render_to(build_workscale_figure,
                      "timing_runtime_workscale_v1.png", output_path)


def render_consolidated(output_path=None) -> str:
    return _render_to(build_consolidated_figure,
                      "timing_runtime_consolidated_v2.png", output_path)


def render_consolidated_blocks(output_path=None) -> str:
    return _render_to(build_consolidated_blocks_figure,
                      "timing_runtime_blocks_v2.png", output_path)


if __name__ == "__main__":
    print(render_deadline())
    print(render_workscale())
