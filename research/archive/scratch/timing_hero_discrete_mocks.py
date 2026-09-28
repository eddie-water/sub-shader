"""Mocks: discretized proportional hero - init vs runtime in frame budgets.

Base unit: 1 cell = one runtime loop (13.4 ms).
Frame budget (deadline) = 186 ms = 14 cells (13.88 rounded, +0.9%).
Start up = 1.7 s = 126 cells = exactly 9 frame budgets (-0.5%).

Variants:
    a - horizontal strip, every cell gridded (init + runtime)
    b - horizontal strip, init as solid module bands, runtime gridded
    c - waffle wrap, one row per frame budget

Numbers hardcoded from the latest anatomy run (timing_results.csv):
    init: audio 503.7 ms, dsp 393.1 ms, render 735 ms, setup ~68 ms
    loop: 13.4 ms total
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "scratch-images")

# --- optichrome v52 ----------------------------------------------------------
BG, FG, SPINE, TICK = "#1A1A1A", "#EEEEEE", "#444444", "#888888"
AUDIO, DSP, RENDER = "#ffd27d", "#7b6fe1", "#ff5a1f"
LOOP = "#ffffff"
SETUP = "#5a5a5a"
EMPTY_FACE, EMPTY_EDGE = "#222222", "#3a3a3a"

# --- discretization ----------------------------------------------------------
CELLS_PER_BUDGET = 14          # 186 ms deadline / 13.4 ms loop
INIT_BUDGETS = 9               # 126 cells = 1.69 s (actual 1.70 s)
INIT_CELL_COLORS = [AUDIO] * 38 + [DSP] * 29 + [RENDER] * 55 + [SETUP] * 4
assert len(INIT_CELL_COLORS) == INIT_BUDGETS * CELLS_PER_BUDGET

SLOT_GAP = 0.55                # gap between budgets, in cell units
PAD = 0.07                     # inset per cell so the grid reads

LEGEND = [("Audio", AUDIO), ("DSP", DSP), ("Render", RENDER), ("Setup", SETUP),
          ("One loop - 13.4 ms", LOOP), ("Headroom", EMPTY_FACE)]

CAPTION = ("one cell = one runtime loop (13.4 ms)   ·   "
           "one column of 14 cells = one frame budget (186 ms)")


def _cell(ax, x, y, color, w=1.0, h=1.0, empty=False):
    kw = (dict(facecolor=EMPTY_FACE, edgecolor=EMPTY_EDGE, linewidth=0.8)
          if empty else dict(facecolor=color, edgecolor=BG, linewidth=0.0))
    ax.add_patch(Rectangle((x + PAD, y + PAD), w - 2 * PAD, h - 2 * PAD,
                           zorder=3, **kw))


def _bracket(ax, xa, xb, y, text, drop=0.18, size=13, align="center"):
    ax.plot([xa, xa, xb, xb], [y - drop, y, y, y - drop],
            color=TICK, linewidth=1.2, zorder=2, solid_capstyle="butt")
    tx = xa if align == "left" else (xa + xb) / 2
    ax.text(tx, y + 0.14, text, ha=align, va="bottom",
            fontsize=size, fontweight="bold", color=FG)


def _legend(fig, y_frac, items=LEGEND):
    handles = [Rectangle((0, 0), 1, 1, facecolor=c,
                         edgecolor=EMPTY_EDGE if c in (EMPTY_FACE,) else SPINE,
                         linewidth=0.8) for _, c in items]
    leg = fig.legend(handles, [n for n, _ in items], loc="center",
                     bbox_to_anchor=(0.5, y_frac), ncol=len(items),
                     frameon=False, fontsize=11, columnspacing=1.6,
                     handlelength=1.4, handleheight=1.1, handletextpad=0.55)
    for t in leg.get_texts():
        t.set_color(FG)


def _budget_x(b):
    return b * (CELLS_PER_BUDGET + SLOT_GAP)


def _strip(out_name, gridded_init):
    """Variants a/b: one horizontal strip, 9 init budgets + 3 runtime + ellipsis."""
    rt_budgets = 3
    total_budgets = INIT_BUDGETS + rt_budgets
    x_end = _budget_x(total_budgets) - SLOT_GAP

    fig = plt.figure(figsize=(16.0, 3.1))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.015, 0.30, 0.955, 0.52])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])

    # init cells
    if gridded_init:
        for i, color in enumerate(INIT_CELL_COLORS):
            b, c = divmod(i, CELLS_PER_BUDGET)
            _cell(ax, _budget_x(b) + c, 0.0, color)
    else:
        # merged module bands, still broken at budget boundaries
        for b in range(INIT_BUDGETS):
            x0 = _budget_x(b)
            row = INIT_CELL_COLORS[b * CELLS_PER_BUDGET:(b + 1) * CELLS_PER_BUDGET]
            start = 0
            for c in range(1, CELLS_PER_BUDGET + 1):
                if c == CELLS_PER_BUDGET or row[c] != row[start]:
                    _cell(ax, x0 + start, 0.0, row[start], w=c - start)
                    start = c

    # runtime budgets: 1 loop cell + 13 headroom cells
    for b in range(INIT_BUDGETS, total_budgets):
        x0 = _budget_x(b)
        _cell(ax, x0, 0.0, LOOP)
        for c in range(1, CELLS_PER_BUDGET):
            _cell(ax, x0 + c, 0.0, None, empty=True)

    ax.text(x_end + 1.2, 0.5, "⋯", ha="left", va="center",
            fontsize=22, color=TICK, fontweight="bold")

    _bracket(ax, _budget_x(0), _budget_x(INIT_BUDGETS) - SLOT_GAP, 1.32,
             "Start Up - one-time · 1.7 s ≈ 9 frame budgets")
    _bracket(ax, _budget_x(INIT_BUDGETS), x_end, 1.32,
             "Runtime Loop - 13.4 ms per frame")

    ax.text((_budget_x(INIT_BUDGETS) + x_end) / 2, -0.42,
            "1 of 14 cells used every frame → 14× under the deadline",
            ha="center", va="top", fontsize=10.5, color=TICK, style="italic")

    ax.set_xlim(-0.4, x_end + 4.0)
    ax.set_ylim(-1.0, 2.35)

    fig.text(0.5, 0.115, CAPTION, ha="center", va="center",
             fontsize=11, color=TICK)
    _legend(fig, 0.035)

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return out


def _waffle(out_name):
    """Variant c: one row per frame budget, wrapping downward."""
    rt_budgets = 4
    rows = INIT_BUDGETS + rt_budgets

    fig = plt.figure(figsize=(11.0, 6.6))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.04, 0.185, 0.62, 0.72])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])

    def row_y(r):
        return rows - 1 - r

    for i, color in enumerate(INIT_CELL_COLORS):
        r, c = divmod(i, CELLS_PER_BUDGET)
        _cell(ax, c, row_y(r), color)
    for r in range(INIT_BUDGETS, rows):
        _cell(ax, 0, row_y(r), LOOP)
        for c in range(1, CELLS_PER_BUDGET):
            _cell(ax, c, row_y(r), None, empty=True)
        ax.text(CELLS_PER_BUDGET + 0.4, row_y(r) + 0.5, "13.4 ms",
                ha="left", va="center", fontsize=10.5, color=TICK)

    ax.text(CELLS_PER_BUDGET / 2, row_y(rows) + 0.55, "⋮  every frame after",
            ha="center", va="center", fontsize=13, color=TICK)

    # right-side group braces
    bx = CELLS_PER_BUDGET + 2.6
    for r0, r1, text in [
            (0, INIT_BUDGETS - 1, "Start Up\n1.7 s - paid once"),
            (INIT_BUDGETS, rows - 1, "Runtime Loop\n13.4 ms per frame\n14× under deadline")]:
        ya, yb = row_y(r1) + PAD, row_y(r0) + 1 - PAD
        ax.plot([bx, bx + 0.25, bx + 0.25, bx],
                [ya, ya, yb, yb], color=TICK, linewidth=1.2,
                solid_capstyle="butt")
        ax.text(bx + 0.75, (ya + yb) / 2, text, ha="left", va="center",
                fontsize=13, fontweight="bold", color=FG, linespacing=1.5)

    ax.text(CELLS_PER_BUDGET / 2, rows + 0.35,
            "→ one row = one frame budget (186 ms) = 14 cells",
            ha="center", va="bottom", fontsize=12, color=TICK)

    ax.set_xlim(-0.4, bx + 6.0)
    ax.set_ylim(row_y(rows) - 0.3, rows + 1.0)
    ax.set_aspect("equal")

    fig.text(0.5, 0.075, "one cell = one runtime loop (13.4 ms)",
             ha="center", va="center", fontsize=11, color=TICK)
    _legend(fig, 0.033)

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def _mesh(out_name):
    """Variant d: init as a compact block of squares (seconds, pre-playback),
    then a linear run of frame budgets. One square = one loop of work (13.4 ms).

    Init wraps at 18 cells (126 = 18 × 7 exactly) - NOT 14, so the block does
    not read as frame budgets: no deadline exists before playback starts.
    Headroom cells are the measured wait:next_chunk idle (~172 ms ≈ 13 cells).
    """
    INIT_W = 18                      # 126 cells = 18 wide × 7 tall, exact
    init_rows = len(INIT_CELL_COLORS) // INIT_W
    rt_budgets = 3
    div_gap = 3.2                    # gap holding the "playback starts" divider

    def budget_x(b):
        return INIT_W + div_gap + b * (CELLS_PER_BUDGET + SLOT_GAP)

    x_end = budget_x(rt_budgets) - SLOT_GAP

    fig = plt.figure(figsize=(16.0, 4.6))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.015, 0.24, 0.97, 0.72])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")

    # init block: reading order, top row first, baseline shared with runtime row
    for i, color in enumerate(INIT_CELL_COLORS):
        r, c = divmod(i, INIT_W)
        _cell(ax, c, (init_rows - 1) - r, color)

    # divider: playback starts
    dx = INIT_W + div_gap / 2
    ax.plot([dx, dx], [-0.55, init_rows + 0.25], color=TICK, linewidth=1.2,
            linestyle=(0, (4, 3)))
    ax.text(dx, -0.95, "▶ playback\nstarts", ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=FG, linespacing=1.4)

    # runtime budgets on the baseline: 1 work cell + 13 measured-idle cells
    for b in range(rt_budgets):
        x0 = budget_x(b)
        _cell(ax, x0, 0.0, LOOP)
        for c in range(1, CELLS_PER_BUDGET):
            _cell(ax, x0 + c, 0.0, None, empty=True)
    ax.text(x_end + 0.9, 0.5, "⋯", ha="left", va="center", fontsize=20,
            color=TICK, fontweight="bold")

    # labels
    _bracket(ax, 0, INIT_W, init_rows + 0.45,
             "Start Up - one-time construction · 1.7 s (no audio yet)",
             size=12.5, align="left")
    _bracket(ax, budget_x(0), x_end, 1.6,
             "Runtime Loop - 13.4 ms of work per 186 ms frame budget",
             size=12.5)
    # deadline ruler under the first budget
    y = -0.55
    ax.plot([budget_x(0), budget_x(0) + CELLS_PER_BUDGET, ],
            [y, y], color=TICK, linewidth=1.2, solid_capstyle="butt")
    for xx in (budget_x(0), budget_x(0) + CELLS_PER_BUDGET):
        ax.plot([xx, xx], [y, y + 0.18], color=TICK, linewidth=1.2)
    # footnotes: stacked, centered under the runtime run
    fx = (budget_x(0) + x_end) / 2
    ax.text(fx, y - 0.32,
            "one frame budget = 186 ms  (8,192-sample hop ÷ 44.1 kHz)",
            ha="center", va="top", fontsize=10.5, color=TICK)
    ax.text(fx, y - 0.95,
            "13 idle squares per budget = measured wait for the next hop (~172 ms)",
            ha="center", va="top", fontsize=10.5, color=TICK, style="italic")

    ax.set_xlim(-0.4, x_end + 3.2)
    ax.set_ylim(-2.75, init_rows + 1.9)

    fig.text(0.5, 0.115, "one square = one loop of work (13.4 ms)",
             ha="center", va="center", fontsize=11, color=TICK)
    _legend(fig, 0.04, items=[("Audio", AUDIO), ("DSP", DSP), ("Render", RENDER),
                              ("Setup", SETUP), ("Work - 13.4 ms", LOOP),
                              ("Idle (headroom)", EMPTY_FACE)])

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(_strip("timing_hero_mock_a_strip_grid.png", gridded_init=True))
    print(_strip("timing_hero_mock_b_strip_bands.png", gridded_init=False))
    print(_waffle("timing_hero_mock_c_waffle.png"))
    print(_mesh("timing_hero_mock_d_block_plus_linear.png"))
