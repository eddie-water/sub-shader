"""Four layout variants of the timing hero banner (run from research/).

All: dsplot family style, no ellipsis, tight edges, generous air BETWEEN
elements. Story: the one-time setup cost buys computational headroom at runtime.

    a_inline   - refined one-line strip (start up · divider · one budget)
    b_stacked  - cost row above, payoff row below, annotations in the empty right
    c_equal    - same-length rows: init vs the ~6 frame budgets it equals
    d_pulse    - init block, then work squares pulsing on an empty timeline

Outputs: research/archive/scratch-images/timing_hero_v2_{a,b,c,d}.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "research")))

from dsplot import Figure, Barh, Line, Annotation, style
from dsplot.figures.gen_timing_hero import (
    StripPanel, _strip_data, _squares, _empty_squares, _ruler,
    _fmt_total, _banner_style, _LEGEND, EMPTY_FACE, EMPTY_EDGE,
    NAME_SIZE, MEASURE_SIZE,
)

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "scratch-images"))

AXES_W = 26.0    # usable axes width (inches) under the banner style overrides


def _name(p, text, x, y, ha):
    p.add(Annotation(text, (x, y), ha=ha, va="bottom",
                     color=style.NEUTRAL_COLOR, fontsize=NAME_SIZE,
                     fontweight="bold"))


def _quiet(p, text, x, y, ha="center", va="top", size=MEASURE_SIZE):
    p.add(Annotation(text, (x, y), ha=ha, va=va, fontsize=size))


def _legend(p, y, x_span, items):
    swatch = 2.0
    widths = [swatch + 1.2 + 1.6 * len(n) + 3.5 for n, _ in items]
    lx = (x_span[0] + x_span[1] - sum(widths)) / 2.0
    for (nm, color), w in zip(items, widths):
        if color == EMPTY_FACE:
            p.add(Barh([y + swatch / 2.0], [swatch], lefts=[lx],
                       color=EMPTY_FACE, height=swatch,
                       edgecolor=EMPTY_EDGE, edgewidth=1.2))
        else:
            p.add(Barh([y + swatch / 2.0], [swatch], lefts=[lx],
                       color=color, height=swatch, edgewidth=0.0))
        p.add(Annotation(nm, (lx + swatch + 1.2, y + swatch / 2.0),
                         ha="left", va="center", fontsize=MEASURE_SIZE))
        lx += w


def _panel(x_lo, x_hi, y_lo, y_hi):
    return StripPanel(
        units=(4, 1), xlim=(x_lo, x_hi), ylim=(y_lo, y_hi),
        content_center=(y_lo + y_hi) / 2.0,
        xticks=[], xticklabels=[], show_xticklabels=False, show_border=False,
    ), (x_hi - x_lo), (y_hi - y_lo)


def _compose(panel, xspan, yspan):
    uh = yspan * AXES_W / xspan + 0.8 + 0.3
    return Figure.compose(rows=[[panel]], row_heights=[1.0],
                          total_width_inches=style.FIGURE_WIDTH_INCHES,
                          unit_height_inches=uh, dpi=style.FIGURE_DPI,
                          show_cell_borders=False)


def _items(init_cells, extra):
    base = [(n, c) for n, c in _LEGEND if c in set(init_cells)]
    return base + extra


# --- a: refined in-line ------------------------------------------------------
def build_a(d):
    ic, bc = d["init_cells"], d["budget_cells"]
    n = len(ic)
    bx0 = n + 3.5
    bx1 = bx0 + bc
    p, xs, ys = _panel(-0.5, bx1 + 0.5, -9.0, 5.9)

    p.add(_squares(list(range(n)), 0.0, ic))
    dx = n + 1.75
    p.add(Line([dx, dx], [-0.7, 1.7], color=style.SPINE_COLOR,
               linewidth=2.0, linestyle="--"))
    p.add(_squares([bx0], 0.0, ["#ffffff"]))
    p.add(_empty_squares([bx0 + c for c in range(1, bc)], 0.0))

    _name(p, "Start Up", 0, 3.4, "left")
    _name(p, "Runtime Loop", bx1, 3.4, "right")
    _ruler(p, 0, n, -1.5)
    _ruler(p, bx0, bx1, -1.5)
    _quiet(p, f"{_fmt_total(d['init_total'])} · once", n / 2.0, -2.4)
    _quiet(p, "playback starts ▶", dx - 1.0, -2.4, ha="right", size=22)
    _quiet(p, f"{d['deadline']:.0f} ms", (bx0 + bx1) / 2.0, -2.4)
    _legend(p, -8.8, (0, bx1), _items(ic, [(f"Work · {d['work']:.1f} ms",
                                            "#ffffff"), ("Idle", EMPTY_FACE)]))
    return p, xs, ys


# --- b: stacked cost / payoff ------------------------------------------------
def build_b(d):
    ic, bc = d["init_cells"], d["budget_cells"]
    n = len(ic)
    ry = -7.5                      # runtime row bottom edge
    p, xs, ys = _panel(-0.5, n + 0.5, -14.9, 4.6)

    _name(p, "Start Up", 0, 2.1, "left")
    _quiet(p, f"{_fmt_total(d['init_total'])} · paid once", n, 2.1,
           ha="right", va="bottom")
    p.add(_squares(list(range(n)), 0.0, ic))

    _name(p, "Runtime Loop", 0, ry + 1.7, "left")
    p.add(_squares([0], ry, ["#ffffff"]))
    p.add(_empty_squares(list(range(1, bc)), ry))
    _quiet(p, f"{d['work']:.1f} ms of work every {d['deadline']:.0f} ms"
              f" - {d['deadline'] / d['work']:.0f}× headroom",
           bc + 2.5, ry + 0.5, ha="left", va="center")
    _ruler(p, 0, bc, ry - 1.5)
    _quiet(p, f"{d['deadline']:.0f} ms", bc / 2.0, ry - 2.4)
    _legend(p, -14.6, (0, n), _items(ic, [(f"Work · {d['work']:.1f} ms",
                                           "#ffffff"), ("Idle", EMPTY_FACE)]))
    return p, xs, ys


# --- c: equal spans - init vs the frames it equals ---------------------------
def build_c(d):
    ic, bc = d["init_cells"], d["budget_cells"]
    n = len(ic)
    groups = max(1, round(n / bc))          # ~6 budgets ≈ the init span
    gap = 1.2
    row_w = groups * bc + (groups - 1) * gap
    x_hi = max(n, row_w) + 0.5
    ry = -7.5
    p, xs, ys = _panel(-0.5, x_hi, -14.9, 4.6)

    _name(p, "Start Up", 0, 2.1, "left")
    _quiet(p, f"{_fmt_total(d['init_total'])} · paid once", n, 2.1,
           ha="right", va="bottom")
    p.add(_squares(list(range(n)), 0.0, ic))

    _name(p, "Runtime Loop", 0, ry + 1.7, "left")
    _quiet(p, f"the same span ≈ {groups} frames · "
              f"{100.0 * (1 - d['work'] / d['deadline']):.0f}% headroom",
           row_w, ry + 1.7, ha="right", va="bottom")
    for g in range(groups):
        x0 = g * (bc + gap)
        p.add(_squares([x0], ry, ["#ffffff"]))
        p.add(_empty_squares([x0 + c for c in range(1, bc)], ry))
    _ruler(p, 0, bc, ry - 1.5)
    _quiet(p, f"{d['deadline']:.0f} ms", bc / 2.0, ry - 2.4)
    _legend(p, -14.6, (0, x_hi), _items(ic, [(f"Work · {d['work']:.1f} ms",
                                              "#ffffff"), ("Idle", EMPTY_FACE)]))
    return p, xs, ys


# --- d: pulse - work squares on an empty timeline ----------------------------
def build_d(d):
    ic, bc = d["init_cells"], d["budget_cells"]
    n = len(ic)
    pulses = 3                                  # 2 intervals after the divider
    tx0 = n + 3.0
    tx1 = tx0 + (pulses - 1) * bc + 1.0
    p, xs, ys = _panel(-0.5, tx1 + 0.5, -9.0, 5.9)

    p.add(_squares(list(range(n)), 0.0, ic))
    dx = n + 1.5
    p.add(Line([dx, dx], [-0.7, 1.7], color=style.SPINE_COLOR,
               linewidth=2.0, linestyle="--"))
    p.add(Line([tx0, tx1], [0.5, 0.5], color=style.SPINE_COLOR,
               linewidth=2.0))
    p.add(_squares([tx0 + k * bc for k in range(pulses)], 0.0,
                   ["#ffffff"] * pulses))

    _name(p, "Start Up", 0, 3.4, "left")
    _name(p, "Runtime Loop", tx1, 3.4, "right")
    _ruler(p, 0, n, -1.5)
    _ruler(p, tx0, tx0 + bc, -1.5)
    _quiet(p, f"{_fmt_total(d['init_total'])} · once", n / 2.0, -2.4)
    _quiet(p, "playback starts ▶", dx - 1.0, -2.4, ha="right", size=22)
    _quiet(p, f"{d['deadline']:.0f} ms", tx0 + bc / 2.0, -2.4)
    _legend(p, -8.8, (0, tx1), _items(ic, [(f"Work · {d['work']:.1f} ms",
                                            "#ffffff")]))
    return p, xs, ys


if __name__ == "__main__":
    d = _strip_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, builder in [("a_inline", build_a), ("b_stacked", build_b),
                         ("c_equal", build_c), ("d_pulse", build_d)]:
        with _banner_style():
            panel, xspan, yspan = builder(d)
            fig = _compose(panel, xspan, yspan)
            fig.render()
            out = os.path.join(OUT_DIR, f"timing_hero_v2_{tag}.png")
            fig.savefig(out)
        print(out)
