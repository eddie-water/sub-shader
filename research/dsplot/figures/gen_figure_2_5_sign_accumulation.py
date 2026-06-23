"""gen_figure_2_5_sign_accumulation — §2.5 of DSP.md.

The dot product as a *pattern detector*, watched one pair at a time. This
section comes BEFORE sine waves enter the story (§2.6), so the figure works on
plain sets of values — the "independent values" framing §2.4 left off on. Two
sequences are compared sample by sample: each pair is multiplied, the SIGN of
that product says whether the pair agrees (+), opposes (−), or abstains (0,
when one value is zero), and the products accumulate into a single running sum.
That running sum *is* the correlation — evidence of a shared pattern.

Layout — three COMPOSITE ROWS, one per relationship. None is a pure case —
correlation is a NET tendency, not an all-or-nothing rule:

    row 1  similar     mostly agrees    → sum climbs  (positive correlation)
    row 2  opposite    mostly opposes   → sum dives   (negative correlation)
    row 3  unrelated   agree ≈ disagree → sum ≈ 0     (no correlation)

Each composite row reads left → right:

    LEFT    a CompositePanel stacking TWO same-size plots on the shared sample
            axis —
              1. the two input sequences a, b (overlaid stems, no signs)
              2. their per-sample products a·b (stems, with the +/−/0 verdict —
                 agree / oppose / abstain — on the zero line)
    MIDDLE  a tall narrow "Sum" strip — the products collapsed into a single
            heavy stem rising (or falling) from zero to the running total, with
            the numeric dot product parked at its tip
    RIGHT   a "Text box" — the case name and the factual outcome

Dynamic (notebook) form — `show()`:
    Each frame *processes the next pair*. Every plot's full sequence is laid out
    up front as faint ghosts; as the master clock advances, each pair lights up
    in turn — the a/b stems brighten, their product stem and its +/−/0 verdict
    appear, and the Sum strip's single stem extends toward the running total.
    The strip's final height is the dot product — large +, large −, or ≈ 0. The
    stacked left plots live inside a CompositePanel, but the figure master clock
    reaches them so every panel ticks in lockstep. No slider: the same
    FuncAnimation master clock the other notebook figures use, on time-series
    axes via DynamicTimeSeriesPanel.

Static (doc) form — `render()`:
    The fully processed final frame as a PNG, for inline embedding where the
    animation can't run.

The palette stays minimal: orange/purple for the two input sequences, a single
neutral wash for the products and the Sum strip; sign is carried by geometry
(the +/−/0 glyphs, the stem direction, the Sum stem climbing or falling), never
a second colour.

Three-mode contract (see dsplot/figures/__init__.py):
    render(output_dir, output_filename) -> str   # production PNG (final frame)
    show() -> Figure                              # animated notebook display
    embed(target=None) -> Figure                  # caller-provided container
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Optional

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import matplotlib.transforms as mtransforms
import numpy as np

from .. import (
    CompositePanel,
    DynamicTextPanel,
    DynamicTimeSeriesPanel,
    Figure,
    SuptitlePanel,
    TextPanel,
    TimeSeriesPanel,
    nb_compact_style,
    style,
)
from ..plottables.base import Plottable


# ---------------------------------------------------------------------------
# Data — three independent (a, b) pairs (N=16). Values are now UNIT magnitude
# only (−1, 0, +1): it is the *agreement of signs*, not their strength, that
# drives the correlation. Each pair is one of three kinds:
#     agree    sign(a) == sign(b)   → product = +1   (+)
#     disagree sign(a) != sign(b)   → product = −1   (−)
#     abstain  a == 0  or  b == 0   → product =  0   (0)  ← both cases shown
#
# Each row's layout follows the same story: a run of positive- and negative-
# side agreements, two abstentions (one a=0, one b=0), two disagreements, then
# more agreements — with the agree/disagree balance set per scenario:
#     P1 Strong Agreement     12 agree,  2 disagree, 2 abstain → Σ = +10
#     P2 Strong Disagreement   2 agree, 12 disagree, 2 abstain → Σ = −10
#     P3 Weak Agreement        7 agree,  7 disagree, 2 abstain → Σ =   0
# ---------------------------------------------------------------------------
N = 16
N_IDX = np.arange(N, dtype=float)

# P1 — Strong Agreement → Strong Positive Correlation
A1 = [1, 1, 1, -1, -1, -1,  0, 1,  1, -1,  1, 1, -1, -1, 1, -1]
B1 = [1, 1, 1, -1, -1, -1,  1, 0, -1,  1,  1, 1, -1, -1, 1, -1]
# P2 — Strong Disagreement → Strong Negative Correlation
A2 = [1, 1, 1, -1, -1, -1,  0, -1,  1, -1,  1, 1, -1, -1, 1, -1]
B2 = [-1, -1, -1, 1, 1, 1, -1,  0,  1, -1, -1, -1, 1, 1, -1, 1]
# P3 — Weak Agreement → No Correlation (balanced mix, cancels to 0)
A3 = [1, -1, 1, -1, 1, -1,  0, 1, -1,  1, -1, 1, -1, 1, -1, 1]
B3 = [1, -1, -1, 1, 1, 1,  1, 0, -1, -1,  1, 1, -1, -1, 1, 1]


def _scenario(a_vals, b_vals) -> dict:
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    products = a * b
    running = np.cumsum(products)
    return {
        "a_vals": a,
        "b_vals": b,
        "products": products,
        "running": running,
        "kind": np.sign(products).astype(int),   # +1 agree, −1 disagree, 0 abstain
        "dot": float(running[-1]),
    }


SCENARIOS = [
    {"key": "p1", "title": "Strong Agreement - Strong Positive Correlation",
     **_scenario(A1, B1)},
    {"key": "p2", "title": "Strong Disagreement - Strong Negative Correlation",
     **_scenario(A2, B2)},
    {"key": "p3", "title": "Weak Agreement - No Correlation",
     **_scenario(A3, B3)},
]

# --- layout / scale ---------------------------------------------------------
# Three COMPOSITE ROWS, one per case. Each row is, left → right:
#     LEFT    a CompositePanel stacking TWO same-size plots, both on the shared
#             sample axis (so they read top-to-bottom as one computation):
#               1. the two input sequences a, b (overlaid stems, no signs)
#               2. their per-sample products a·b (stems + the +/−/0 verdict)
#     MIDDLE  a tall narrow "Sum" strip — one heavy stem collapsing the products
#             into the running total, with the numeric result at its tip
#     RIGHT   a "Text box" carrying the case name + the outcome
# Proportions are 4 : 1 : 3 (stem stack : Sum strip : text box).
LEFT_UNITS = 4             # stem stack — wide, landscape (two stacked plots)
SUM_UNITS = 1             # Sum strip — tall and narrow
CASE_ROW_HEIGHT = 2.0      # two stacked plots → each ≈ one unit tall
# Text box is a SQUARE cell. It spans TEXT_UNITS columns, so its rendered width
# is TEXT_UNITS·unit_w PLUS the (TEXT_UNITS−1) inter-column gutters it absorbs;
# its height is CASE_ROW_HEIGHT·unit_h. Square base units alone do NOT make it
# square (the absorbed gutter widens it) — instead `_square_unit_height()`
# derives the unit_h that makes width == height. See render()/save_gif()/show().
TEXT_UNITS = 2
WIDTH_UNITS = LEFT_UNITS + SUM_UNITS + TEXT_UNITS   # = 7

X_LIM = (-0.9, N - 0.1)
X_TICKS = [0, 5, 10, 15]     # evenly spaced, ending on the final sample (n=15)

SIGNAL_YLIM = (-1.7, 1.7)    # unit-magnitude stems with a little headroom
SIGNAL_YTICKS = [-1, 0, 1]

# Shared running-total scale across all three rows so the contrast reads at a
# glance: P1 climbs to +10, P2 dives to −10, P3 barely leaves zero.
_run_peak = max(float(np.abs(s["running"]).max()) for s in SCENARIOS)
# Generous headroom (1.4×) so the big numeric readout parked at the ±peak stem
# tips clears the top/bottom spines instead of clipping.
RUNNING_YLIM = (-1.4 * _run_peak, 1.4 * _run_peak)
RUNNING_YTICKS = [-round(_run_peak), 0, round(_run_peak)]

# The Sum strip's own x-axis is meaningless (one bar) — a unit-wide span with the
# single stem parked at its centre.
ACCUM_XLIM = (0.0, 1.0)
ACCUM_X_CENTER = 0.5

TITLE = "Figure 2.5 - Sign Agreement"

# Short case names for the text box heading (case name + correlation outcome).
CASE_TITLES = [
    "Strong Agreement\nPositive Correlation",
    "Strong Disagreement\nNegative Correlation",
    "Weak Agreement\nNo Correlation",
]

# Quick blurb beneath each heading — a plain-language read of what the plot is
# doing this cycle. SCAFFOLD copy; the user authors the final prose.
CASE_BLURBS = [
    "Most pairs share a sign, so their products are positive. "
    "The running sum climbs steadily to +10 — strong positive correlation.",
    "Most pairs have opposite signs, so their products are negative. "
    "The running sum falls steadily to −10 — strong negative correlation.",
    "Agreements and disagreements roughly cancel, so the products offset. "
    "The running sum drifts and ends near 0 — no real correlation.",
]

# Row y-axis captions — currently disabled (no y-labels for now); kept here so
# they can be wired back onto the leftmost column when wanted.
ROW_Y_LABELS = ("a , b", "a · b", "running Σ")

# One text colour for everything — chrome (titles, ticks, axis labels) AND the
# in-plot glyphs/readout — matching figure 1's "single consistent value"
# approach. The chrome side is driven through _unified_style() (a temporary
# style override, restored after build); in-plot text references TEXT_COLOR.
TEXT_COLOR = "#EEEEEE"
IN_PLOT_FONTSIZE = 22        # sign glyphs AND the running-sum readout — one size
TEXT_FONTSIZE = 16           # cycling text box (DynamicTextPanel _CaseText)
TEXT_MARGIN_FRAC = 0.06      # text inset from its cell border (breathing room
                             # matching the plots) — shared by the static
                             # TextPanel and the animated _CaseText
CASE_TITLE_COLOR = TEXT_COLOR

# --- weights / palette ------------------------------------------------------
GHOST_ALPHA = 0.16           # un-processed samples sit faint behind the action
BRIGHT_ALPHA = 0.9           # a / b once their pair has been processed
SAMPLE_STEM_LINEWIDTH = 5.0
SAMPLE_MARKERSIZE = 8.0
ACCUM_STEM_LINEWIDTH = SAMPLE_STEM_LINEWIDTH   # Sum strip stem matches the others
ACCUM_MARKERSIZE = SAMPLE_MARKERSIZE
PRODUCT_COLOR = style.NEUTRAL_COLOR   # products / running sum: sign by direction
ZERO_LINE_WIDTH = 1.8
SPINE_LINEWIDTH = 2.6

# Spacing follows figure 1's philosophy: gutters/margins are PHYSICAL INCHES set
# through a temporary style override (below), and `Figure.compose` derives the
# gridspec fractions from them proportionally (= inch_gutter × n / Σ cell_sizes).
# We do NOT pass raw wspace/hspace — that bypasses the derivation and makes the
# gutter diverge from the value chrome-title placement reads (titles overflow).
# Values match figure 1's CONTENDER_TIGHT_STYLE so the two figures rhyme.
# One gutter size everywhere — column gutters (stack ↔ strip ↔ text) and row
# gutters (between cases) use the SAME physical inch value so every gap reads
# identically. compose derives the per-axis gridspec fractions from this.
GUTTER_INCHES = 0.60
COL_GUTTER_INCHES = GUTTER_INCHES   # between stem stack, Sum strip, and text box
ROW_GUTTER_INCHES = GUTTER_INCHES   # between the three case rows
MARGIN_INCHES = 0.55       # panel-border → figure edge; wider than figure 1's
                           # 0.25 because our y-tick labels (−1 / −10) render
                           # OUTSIDE the plot into this margin (figure 1 put its
                           # labels in an in-plot strip, so it could go tighter)
PAD_INCHES = 0.40          # cell-fill inset (axis labels sit close to the border)

CURSOR_COLOR = style.NEUTRAL_COLOR
CURSOR_ALPHA = 0.16

INTERVAL_MS = 640            # per-frame dwell on the master clock (a touch slower)
HOLD_TICKS = 8              # completed-figure pause before the loop wraps
                             # (≈ HOLD_TICKS × INTERVAL_MS ms each cycle)

# Cycling animation (single composite row, three cases in sequence): each case
# processes its N+1 pairs, then HOLDS on the resolved result for CASE_HOLD_TICKS
# frames before the next case takes over. One full loop is all three segments.
CASE_HOLD_TICKS = 6
PER_CASE_FRAMES = N + 1                              # pairs 0..N
SEGMENT_FRAMES = PER_CASE_FRAMES + CASE_HOLD_TICKS   # process + resolved hold
TOTAL_CYCLE_FRAMES = 3 * SEGMENT_FRAMES              # case 1 → 2 → 3

# Temporary style override applied during build + render (then restored). Two
# jobs, both mirroring figure 1's CONTENDER_TIGHT_STYLE mechanism:
#   1. one consistent colour for every text element, and
#   2. tight PHYSICAL-INCH spacing (margin / gutters / cell-fill pad) so
#      `Figure.compose` derives proportional gridspec gutters the figure-1 way.
# SPINE_COLOR = TEXT_COLOR keeps the plot spines white where they're drawn (the
# animated paths bold them via _bolden_spines). The STATIC PNG's panel frame is
# the figure-1 cell border, not the spine — see render().
_UNIFIED_TEXT_STYLE = {
    "TICK_LABEL_COLOR": TEXT_COLOR,
    "SUPTITLE_COLOR": TEXT_COLOR,
    "SPINE_COLOR": TEXT_COLOR,
    "DEFAULT_MARGIN_INCHES": MARGIN_INCHES,
    "DEFAULT_PAD_INCHES": PAD_INCHES,
    "DEFAULT_GUTTER_INCHES": ROW_GUTTER_INCHES,
    "DEFAULT_COLUMN_GUTTER_INCHES": COL_GUTTER_INCHES,
    "DEFAULT_INNER_GUTTER_INCHES": GUTTER_INCHES,   # stem-stack inner gap == all others
}


@contextmanager
def _unified_style():
    orig = {k: getattr(style, k) for k in _UNIFIED_TEXT_STYLE}
    try:
        for k, v in _UNIFIED_TEXT_STYLE.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


# Static-PNG chrome overrides (figure 1 / 2.6). Two jobs:
#   1. Border model: the gray cell border (NEUTRAL_COLOR, drawn by
#      show_cell_borders) is the SINGLE frame for every panel. Spines off so no
#      white inner box competes — every panel reads with one identical border.
#      The animated paths keep SPINE_COLOR white (no cell borders in a single
#      row, so the spine IS their frame), so this stays scoped to the static PNG.
#   2. Chrome font scale: the global defaults (tick numbers 30, axis label 26)
#      dwarf this figure's content (text box 16, in-plot glyphs/readout 22), so
#      the axis chrome looks oversized. Pull tick numbers down to the body scale
#      and the axis label to the in-plot scale — a clean tier (in-plot 22 ≥ axis
#      label 22 > tick numbers / body 16). The GIF gets its own compact scale via
#      nb_compact_style, so this override is static-only by design.
_STATIC_CHROME_STYLE = {
    "SPINE_COLOR": "none",
    "DEFAULT_TICK_LABEL_SIZE": TEXT_FONTSIZE,       # tick numbers == text-box body (16)
    "DEFAULT_LABEL_FONT_SIZE": IN_PLOT_FONTSIZE,    # "n" axis label == in-plot glyphs (22)
}


@contextmanager
def _static_chrome_style():
    orig = {k: getattr(style, k) for k in _STATIC_CHROME_STYLE}
    try:
        for k, v in _STATIC_CHROME_STYLE.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


@contextmanager
def _cell_border_chrome():
    """Spines off so the gray cell border is the SINGLE panel frame — the static
    PNG's border model, applied to the GIF too so the two share one look. Unlike
    _static_chrome_style this touches ONLY the spine colour, leaving the font
    scale to the caller (the GIF keeps nb_compact_style's compact sizes)."""
    orig = style.SPINE_COLOR
    try:
        style.SPINE_COLOR = "none"
        yield
    finally:
        style.SPINE_COLOR = orig


# ---------------------------------------------------------------------------
# Plottables — small enough to compose into both the static panels and the
# per-frame lists the DynamicTimeSeriesPanel draws.
# ---------------------------------------------------------------------------
class _ZeroLine(Plottable):
    """Faint horizontal reference at y = 0 so above/below (the sign) reads."""

    def __init__(self, *, color, alpha=0.6, linewidth=ZERO_LINE_WIDTH, zorder=1):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.linewidth = linewidth

    def draw(self, ax):
        ax.axhline(0.0, color=self.color, alpha=self.alpha,
                   linewidth=self.linewidth, zorder=self.zorder)


# A vertical stem and its a/b twin share the same x=n line, so when a and b are
# EXACTLY equal they coincide and one colour hides the other. `split_side` slices
# only those coinciding stems (and their tip markers) down the centerline so both
# colours show: side="left" keeps the x ≤ n half, side="right" the x ≥ n half.
# Stems that do NOT coincide are drawn whole, untouched (they're already both
# visible). `split_mask` is the per-point boolean of where a == b; only masked
# points are sliced. The cut is a per-stem TransformedBbox(data half-plane,
# transData) — it tracks the transform, so it stays aligned at any DPI / figure
# size / animation frame (a frozen pixel Bbox would drift at savefig dpi). Same
# clipping technique as the split-vector arrow; here the shapes are already
# vertical, so an axis-aligned Bbox is exactly the right cut (no rotated path).
_CLIP_BIG = 1.0e6   # ±∞ in data space for the open sides of a half-plane box


class _Stems(Plottable):
    """Stem plot drawn with explicit line + marker artists (no autoscale —
    the panel pins its own limits). Used for both the faint ghost layer and
    the bright processed layer. With `split_side` ('left'/'right') plus a
    `split_mask`, the masked stems are clipped to one side of their own x so a
    coinciding a/b pair shows both colours; unmasked stems are drawn whole."""

    def __init__(self, x, y, *, color, alpha, linewidth=SAMPLE_STEM_LINEWIDTH,
                 markersize=SAMPLE_MARKERSIZE, zorder=3, split_side=None,
                 split_mask=None):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.linewidth = linewidth
        self.markersize = markersize
        self.split_side = split_side
        self.split_mask = None if split_mask is None else np.asarray(split_mask, dtype=bool)

    def _splits(self, i) -> bool:
        """Whether point i should be sliced — only where a/b coincide."""
        if self.split_side is None:
            return False
        return True if self.split_mask is None else bool(self.split_mask[i])

    def _clip_to_half(self, ax, artist, xi):
        """Clip one artist to the half-plane on `split_side` of x=xi."""
        if self.split_side == "left":
            box = mtransforms.Bbox.from_extents(-_CLIP_BIG, -_CLIP_BIG, xi, _CLIP_BIG)
        else:
            box = mtransforms.Bbox.from_extents(xi, -_CLIP_BIG, _CLIP_BIG, _CLIP_BIG)
        artist.set_clip_box(mtransforms.TransformedBbox(box, ax.transData))
        artist.set_clip_on(True)

    def draw(self, ax):
        # ONE artist per stem — the line AND its tip marker (markevery=[1] marks
        # only the top vertex). Drawing them as a single Line2D means the marker
        # shares the line's exact coordinate, so it can't round to a different
        # sub-pixel column than the stroke (two separate artists drift ~½px apart
        # — magnified ~5× on zoom, that reads as the circle sitting off-centre).
        # snap=False additionally stops matplotlib pixel-snapping the thin stroke
        # away from the (unsnapped) marker. Split stems add fillstyle so the tip
        # shows the correct half-circle, then clip the whole artist (line + tip)
        # to one side of x=xi; the unsplit half-plane and the fillstyle half agree,
        # so nothing fights. See _AccumStem.draw for the same single-artist tip.
        for i, (xi, yi) in enumerate(zip(self.x, self.y)):
            splits = self._splits(i)
            kwargs = dict(color=self.color, alpha=self.alpha,
                          linewidth=self.linewidth, solid_capstyle="round",
                          marker="o", markevery=[1], markersize=self.markersize,
                          zorder=self.zorder)
            if splits:
                kwargs.update(fillstyle=self.split_side,
                              markerfacecolor=self.color, markeredgewidth=0.0)
            stem, = ax.plot([xi, xi], [0.0, yi], **kwargs)
            stem.set_snap(False)
            if splits:
                self._clip_to_half(ax, stem, xi)


class _SignStrip(Plottable):
    """The +/−/0 verdict for each processed pair, centred on the x-axis (the
    zero line). A background box matching the panel bg punches each glyph out
    of any stem passing through zero, so a − can't read as a + when a vertical
    stem crosses it."""

    def __init__(self, x, kinds, *, y=0.0, fontsize=IN_PLOT_FONTSIZE,
                 color=TEXT_COLOR, zorder=6):
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.x = np.asarray(x, dtype=float)
        self.kinds = np.asarray(kinds, dtype=int)
        self.y = y
        self.fontsize = fontsize

    def draw(self, ax):
        glyph = {1: "+", -1: "−", 0: "0"}
        bbox = dict(facecolor=style.BG_COLOR, edgecolor="none",
                    boxstyle="square,pad=0.15")
        for xi, k in zip(self.x, self.kinds):
            ax.text(float(xi), self.y, glyph[int(k)],
                    color=self.color, fontsize=self.fontsize, fontweight="bold",
                    ha="center", va="center", zorder=self.zorder, bbox=bbox)


class _Cursor(Plottable):
    """A faint vertical highlight on the pair being processed this frame."""

    def __init__(self, x, *, color=CURSOR_COLOR, alpha=CURSOR_ALPHA, zorder=0):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.x = float(x)

    def draw(self, ax):
        ax.axvspan(self.x - 0.5, self.x + 0.5, color=self.color,
                   alpha=self.alpha, linewidth=0.0, zorder=self.zorder)


class _AccumStem(Plottable):
    """The Sum strip's single heavy stem — the products collapsed into one bar
    rising (positive sum) or falling (negative sum) from the zero line to the
    current running total, parked at the centre of the unit-wide strip."""

    def __init__(self, value, *, x=ACCUM_X_CENTER, color=PRODUCT_COLOR,
                 alpha=BRIGHT_ALPHA, linewidth=ACCUM_STEM_LINEWIDTH,
                 markersize=ACCUM_MARKERSIZE, zorder=3):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.value = float(value)
        self.x = float(x)
        self.linewidth = linewidth
        self.markersize = markersize

    def draw(self, ax):
        # ONE artist (line + tip marker via markevery=[1]) so the circle shares
        # the stem's exact coordinate and can't round off-centre; snap=False stops
        # the stroke pixel-snapping away from it. See _Stems.draw.
        stem, = ax.plot([self.x, self.x], [0.0, self.value], color=self.color,
                        alpha=self.alpha, linewidth=self.linewidth,
                        solid_capstyle="round", marker="o", markevery=[1],
                        markersize=self.markersize, zorder=self.zorder)
        stem.set_snap(False)


class _SumReadout(Plottable):
    """The running total as a big number printed on the side of the zero baseline
    OPPOSITE the stem's growth — UNDER the baseline for a positive (upward) stem,
    ABOVE it for a negative (downward) stem. The number lands in the empty half
    of the strip, so it never sits on the stem."""

    def __init__(self, value, *, x=ACCUM_X_CENTER, color=TEXT_COLOR, zorder=7):
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.value = float(value)
        self.x = float(x)

    def draw(self, ax):
        positive = self.value >= 0
        ax.annotate(
            f"{round(self.value):+d}",
            xy=(self.x, 0.0),
            xytext=(0, -10 if positive else 10), textcoords="offset points",
            ha="center", va="top" if positive else "bottom",
            color=self.color, fontsize=IN_PLOT_FONTSIZE, fontweight="bold",
            zorder=self.zorder,
        )


class _CaseText(Plottable):
    """The text box copy for the current case, drawn top-left in axes fraction
    coords so it animates inside a DynamicTextPanel as the cycle switches cases.
    Mirrors TextPanel's top-anchored, left-aligned look (no border). The first
    paragraph (the heading) is kept verbatim; later paragraphs (the blurb) are
    word-wrapped to ``wrap_width`` chars since ax.text does not wrap itself."""

    def __init__(self, text, *, color=TEXT_COLOR, fontsize=TEXT_FONTSIZE,
                 wrap_width=26, zorder=5):
        # Use the TEXT_COLOR constant (#EEEEEE), NOT style.TICK_LABEL_COLOR:
        # animation frames are drawn during _anim.save(), AFTER _unified_style()
        # has exited and restored TICK_LABEL_COLOR to its #888888 default — so a
        # style.* lookup (eager or lazy) would render gray. TEXT_COLOR equals
        # what TICK_LABEL_COLOR resolves to under the unified style, so the GIF
        # caption matches the static panel's colour.
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.text = text
        self.fontsize = fontsize
        self.wrap_width = wrap_width

    def draw(self, ax):
        import textwrap
        heading, _, blurb = self.text.partition("\n\n")
        rendered = heading
        if blurb:
            rendered += "\n\n" + textwrap.fill(blurb, self.wrap_width)
        # Normal weight + tick-label colour, matching figure 1's caption. Inset
        # by TEXT_MARGIN_FRAC so it doesn't hug the border (matches the static
        # TextPanel's content_margin_frac).
        ax.text(TEXT_MARGIN_FRAC, 1.0 - TEXT_MARGIN_FRAC, rendered,
                transform=ax.transAxes, ha="left", va="top", color=self.color,
                fontsize=self.fontsize, fontweight="normal", linespacing=1.5,
                zorder=self.zorder)


# ---------------------------------------------------------------------------
# Frame content — `count` pairs processed (indices 0..count-1).
# ---------------------------------------------------------------------------
def _coincide_mask(scn: dict) -> np.ndarray:
    """Per-sample boolean — where a and b are exactly equal (so their stems
    coincide and must be split to show both). Elsewhere a/b differ and draw
    whole, since both are already visible."""
    return np.isclose(scn["a_vals"], scn["b_vals"])


def _ab_frame(scn: dict, count: int) -> list:
    """Row 1 — the input sequences a, b light up pair by pair. No signs here."""
    items: list = []
    if count > 0:
        items.append(_Cursor(N_IDX[count - 1]))
        idx = slice(0, count)
        coincide = _coincide_mask(scn)[idx]
        items.append(_Stems(N_IDX[idx], scn["a_vals"][idx],
                            color=style.PRIMARY_COLOR, alpha=BRIGHT_ALPHA,
                            zorder=3, split_side="left", split_mask=coincide))
        items.append(_Stems(N_IDX[idx], scn["b_vals"][idx],
                            color=style.SECONDARY_COLOR, alpha=BRIGHT_ALPHA,
                            zorder=4, split_side="right", split_mask=coincide))
    return items


def _product_frame(scn: dict, count: int) -> list:
    """Row 2 — the per-sample products a·b appear as stems, with the +/−/0
    verdict on the zero line."""
    items: list = []
    if count > 0:
        items.append(_Cursor(N_IDX[count - 1]))
        idx = slice(0, count)
        items.append(_Stems(N_IDX[idx], scn["products"][idx],
                            color=PRODUCT_COLOR, alpha=BRIGHT_ALPHA, zorder=3))
        items.append(_SignStrip(N_IDX[idx], scn["kind"][idx]))
    return items


def _running_so_far(scn: dict, count: int) -> float:
    """The running total after `count` pairs processed (0 before any)."""
    return float(scn["running"][count - 1]) if count > 0 else 0.0


def _sum_frame(scn: dict, count: int) -> list:
    """The Sum strip — one heavy stem growing toward the running total as pairs
    are processed, with the numeric result tracking its tip."""
    val = _running_so_far(scn, count)
    items: list = [_AccumStem(val)]
    if count > 0:
        items.append(_SumReadout(val))
    return items


def _ghost_ab(scn: dict) -> list:
    """Faint full a/b sequences behind row 1 — fixes the layout and previews
    what is coming before it is processed."""
    return [
        _ZeroLine(color=style.DROPLINE_COLOR),
        _Stems(N_IDX, scn["a_vals"], color=style.PRIMARY_COLOR, alpha=GHOST_ALPHA,
               zorder=2, split_side="left", split_mask=_coincide_mask(scn)),
        _Stems(N_IDX, scn["b_vals"], color=style.SECONDARY_COLOR,
               alpha=GHOST_ALPHA, zorder=2, split_side="right",
               split_mask=_coincide_mask(scn)),
    ]


def _ghost_products(scn: dict) -> list:
    """Faint full product sequence behind row 2."""
    return [
        _ZeroLine(color=style.DROPLINE_COLOR),
        _Stems(N_IDX, scn["products"], color=PRODUCT_COLOR, alpha=GHOST_ALPHA,
               zorder=2),
    ]


# ---------------------------------------------------------------------------
# Panels — one composite row per case: [ stem-stack (a,b / products) | Sum strip
# | text box ]. A single `dynamic` flag picks animated (DynamicTimeSeriesPanel +
# frame_fn) vs static (TimeSeriesPanel + final-frame plottables); both share the
# same axis kwargs so the two modes stay in lockstep. The two left plots share
# the sample x-axis and are sized identically (only their y-scale differs); the
# Sum strip is a narrow companion on its own one-bar axis.
# ---------------------------------------------------------------------------
def _ab_panel(scn: dict, *, dynamic: bool):
    """Stack row 1 — input sequences a, b (no signs)."""
    axis = dict(units=(LEFT_UNITS, 1), xticks=X_TICKS, yticks=SIGNAL_YTICKS,
                xlim=X_LIM, ylim=SIGNAL_YLIM,
                show_xticklabels=False, show_yticklabels=True)
    if dynamic:
        return DynamicTimeSeriesPanel(
            frame_fn=lambda k, s=scn: _ab_frame(s, k),
            num_frames=N + 1, interval_ms=INTERVAL_MS,
            base_plottables=_ghost_ab(scn), **axis)
    panel = TimeSeriesPanel(**axis)
    panel.add(_ZeroLine(color=style.DROPLINE_COLOR))
    coincide = _coincide_mask(scn)
    panel.add(_Stems(N_IDX, scn["a_vals"], color=style.PRIMARY_COLOR,
                     alpha=BRIGHT_ALPHA, zorder=3, split_side="left",
                     split_mask=coincide))
    panel.add(_Stems(N_IDX, scn["b_vals"], color=style.SECONDARY_COLOR,
                     alpha=BRIGHT_ALPHA, zorder=4, split_side="right",
                     split_mask=coincide))
    return panel


def _product_panel(scn: dict, *, dynamic: bool):
    """Stack row 2 (bottom) — per-sample products a·b + the +/−/0 verdict. Being
    the bottom of the stack, this plot carries the shared x-axis chrome."""
    axis = dict(units=(LEFT_UNITS, 1), x_label="n", xticks=X_TICKS,
                yticks=SIGNAL_YTICKS, xlim=X_LIM, ylim=SIGNAL_YLIM,
                show_xticklabels=True, show_yticklabels=True)
    if dynamic:
        return DynamicTimeSeriesPanel(
            frame_fn=lambda k, s=scn: _product_frame(s, k),
            num_frames=N + 1, interval_ms=INTERVAL_MS,
            base_plottables=_ghost_products(scn), **axis)
    panel = TimeSeriesPanel(**axis)
    panel.add(_ZeroLine(color=style.DROPLINE_COLOR))
    panel.add(_Stems(N_IDX, scn["products"], color=PRODUCT_COLOR,
                     alpha=BRIGHT_ALPHA, zorder=3))
    panel.add(_SignStrip(N_IDX, scn["kind"]))
    return panel


def _sum_panel(scn: dict, *, dynamic: bool):
    """The Sum strip — a tall narrow companion to the stem stack carrying a
    single heavy stem (the running total) on its own one-bar x-axis. Shares the
    running-total y-scale across all three rows so the bars are comparable."""
    # No y-tick labels: the Sum strip is narrow, so a 10/0/−10 axis would strand
    # those numbers in the gutter across the neighbouring cell border. The big
    # readout at the stem tip carries the magnitude; the zero line carries sign;
    # the shared y-scale keeps the three bars comparable.
    axis = dict(units=(SUM_UNITS, 1), xticks=[], yticks=[],
                xlim=ACCUM_XLIM, ylim=RUNNING_YLIM,
                show_xticklabels=False, show_yticklabels=False)
    if dynamic:
        return DynamicTimeSeriesPanel(
            frame_fn=lambda k, s=scn: _sum_frame(s, k),
            num_frames=N + 1, interval_ms=INTERVAL_MS,
            base_plottables=[_ZeroLine(color=style.DROPLINE_COLOR)], **axis)
    panel = TimeSeriesPanel(**axis)
    panel.add(_ZeroLine(color=style.DROPLINE_COLOR))
    panel.add(_AccumStem(scn["dot"]))
    panel.add(_SumReadout(scn["dot"]))
    return panel


def _case_body(title: str) -> str:
    """The text-box copy for one case: the heading + a quick what's-happening
    blurb. (Tally/Σ are dropped — the plots already show them.)"""
    i = CASE_TITLES.index(title)
    # Uppercase header + blank line + body, matching figure 1's caption layout
    # (`_side_text_panel`: f"{title.upper()}\n\n{caption}").
    return f"{title.upper()}\n\n{CASE_BLURBS[i]}"


def _text_panel(scn: dict, title: str) -> TextPanel:
    """The text box — mirrors figure 1's right-hand caption (`_side_text_panel`)
    so the two figures share one text treatment: an UPPERCASE section label +
    blank line + body paragraph, all at NORMAL weight in the tick-label colour
    (one type system with the axis chrome), top-anchored and left-justified. The
    cell border (show_cell_borders) is the frame; facecolor="none" lets it sit on
    the dark bg without a competing fill. Scaffold; final prose is the user's."""
    return TextPanel(
        _case_body(title),
        units=(TEXT_UNITS, 1),
        color=style.TICK_LABEL_COLOR,
        fontweight="normal",
        auto_shrink=True,
        min_font_size=12.0,
        cell_padding_frac=0.0,
        # Uniform inset from the cell border so the text doesn't hug it — gives
        # the same internal breathing room the plots have inside their borders
        # (top_anchor starts the first line at this inset, so it sets the top gap).
        content_margin_frac=TEXT_MARGIN_FRAC,
        justify=True,   # flush both margins (figure 1's body treatment)
        # No inner box — the gray cell border (show_cell_borders) is the ONE and
        # only frame, identical for every panel (figure 1's model: SPINE_COLOR is
        # "none" in the static path, so plots have no inner spine box either).
        show_ghost_border=False,
        top_anchor=True,
        facecolor="none",
    )


def _case_row(scn: dict, title: str, *, dynamic: bool) -> list:
    """One composite row: the two-plot stem stack, the Sum strip, the text box."""
    stack = CompositePanel(
        units=(LEFT_UNITS, 1),
        rows=[[_ab_panel(scn, dynamic=dynamic)],
              [_product_panel(scn, dynamic=dynamic)]],
        share_x=False,          # gap between the two plots…
        hspace=None,            # …derived from DEFAULT_INNER_GUTTER_INCHES so it
                                #   matches every other gutter in the figure
    )
    return [stack, _sum_panel(scn, dynamic=dynamic), _text_panel(scn, title)]


# ---------------------------------------------------------------------------
# Cycling animation — ONE composite row that plays case 1 → 2 → 3 in sequence.
# A single global frame index maps to (case, pair); the ghosts and text now live
# in the per-frame functions because they change with the active case. Every
# panel runs TOTAL_CYCLE_FRAMES frames so the figure master clock keeps the
# stem-stack, Sum strip, and text box in lockstep.
# ---------------------------------------------------------------------------
def _cycle_case_pair(idx: int) -> tuple[int, int]:
    """Map a global frame index to (case 0..2, pairs-processed 0..N). During a
    case's trailing hold the pair count stays pinned at N (resolved)."""
    case = (idx // SEGMENT_FRAMES) % 3
    local = idx % SEGMENT_FRAMES
    return case, min(local, N)


def _ab_cycle(idx: int) -> list:
    case, pair = _cycle_case_pair(idx)
    scn = SCENARIOS[case]
    return _ghost_ab(scn) + _ab_frame(scn, pair)


def _product_cycle(idx: int) -> list:
    case, pair = _cycle_case_pair(idx)
    scn = SCENARIOS[case]
    return _ghost_products(scn) + _product_frame(scn, pair)


def _sum_cycle(idx: int) -> list:
    case, pair = _cycle_case_pair(idx)
    return _sum_frame(SCENARIOS[case], pair)


def _text_cycle(idx: int) -> list:
    case, _ = _cycle_case_pair(idx)
    return [_CaseText(_case_body(CASE_TITLES[case]))]


def _cycle_row(*, text_border: bool = True) -> list:
    """The single cycling composite row: stem stack | Sum strip | text box.

    `text_border` draws the text box's own content border; pass False when the
    figure-1 cell border frames the cell instead (the GIF's cell-border path),
    so the text box isn't double-framed."""
    common = dict(num_frames=TOTAL_CYCLE_FRAMES, interval_ms=INTERVAL_MS)
    ab = DynamicTimeSeriesPanel(
        frame_fn=_ab_cycle, base_plottables=[],
        units=(LEFT_UNITS, 1), xticks=X_TICKS, yticks=SIGNAL_YTICKS,
        xlim=X_LIM, ylim=SIGNAL_YLIM,
        show_xticklabels=False, show_yticklabels=True, **common)
    product = DynamicTimeSeriesPanel(
        frame_fn=_product_cycle, base_plottables=[],
        units=(LEFT_UNITS, 1), x_label="n", xticks=X_TICKS, yticks=SIGNAL_YTICKS,
        xlim=X_LIM, ylim=SIGNAL_YLIM,
        show_xticklabels=True, show_yticklabels=True, **common)
    stack = CompositePanel(
        units=(LEFT_UNITS, 1), rows=[[ab], [product]],
        share_x=False, hspace=None,
    )
    sum_strip = DynamicTimeSeriesPanel(
        frame_fn=_sum_cycle,
        base_plottables=[_ZeroLine(color=style.DROPLINE_COLOR)],
        units=(SUM_UNITS, 1), xticks=[], yticks=[],
        xlim=ACCUM_XLIM, ylim=RUNNING_YLIM,
        show_xticklabels=False, show_yticklabels=False, **common)
    text = DynamicTextPanel(
        frame_fn=_text_cycle, units=(TEXT_UNITS, 1), show_border=text_border,
        **common)
    return [stack, sum_strip, text]


def _compose_cycle(*, unit_inches=None, dpi=None, unit_height_inches=None,
                   hold_ticks=None, show_cell_borders=False,
                   text_border=True) -> Figure:
    """Single cycling case row under the full-width suptitle, above the footer."""
    rows = [
        [SuptitlePanel(TITLE, units=(WIDTH_UNITS, 1))],
        _cycle_row(text_border=text_border),
    ]
    row_heights = [0.50, CASE_ROW_HEIGHT]
    kwargs = dict(
        rows=rows, row_heights=row_heights,
        unit_inches=unit_inches, dpi=dpi,
        show_cell_borders=show_cell_borders,
    )
    if unit_height_inches is not None:
        kwargs["unit_height_inches"] = unit_height_inches
    if hold_ticks is not None:
        kwargs["hold_ticks"] = hold_ticks
    return Figure.compose(**kwargs)


# ---------------------------------------------------------------------------
# Composition + three-mode contract
# ---------------------------------------------------------------------------
def _square_unit_height(unit_w: float) -> float:
    """The unit_height_inches that makes the text cell a true square at a given
    unit width. The text cell spans TEXT_UNITS columns and absorbs the
    (TEXT_UNITS−1) inter-column gutters, so its width is
    ``TEXT_UNITS·unit_w + (TEXT_UNITS−1)·GUTTER_INCHES``; its height is
    ``CASE_ROW_HEIGHT·unit_h``. Setting them equal and solving for unit_h:"""
    text_w = TEXT_UNITS * unit_w + (TEXT_UNITS - 1) * GUTTER_INCHES
    return text_w / CASE_ROW_HEIGHT


def _compose(*, dynamic, unit_inches=None, dpi=None,
             unit_height_inches=None, hold_ticks=None,
             show_cell_borders=False) -> Figure:
    """Three composite case rows under a full-width suptitle (no footer)."""
    rows = [[SuptitlePanel(TITLE, units=(WIDTH_UNITS, 1))]]
    row_heights = [0.50]
    for scn, title in zip(SCENARIOS, CASE_TITLES):
        rows.append(_case_row(scn, title, dynamic=dynamic))
        row_heights.append(CASE_ROW_HEIGHT)

    # No wspace/hspace: let compose derive the gridspec gutters from the
    # physical-inch values in _UNIFIED_TEXT_STYLE (the figure-1 approach), so
    # the gutters stay proportional to the actual cell sizes and chrome titles
    # land correctly.
    kwargs = dict(
        rows=rows,
        row_heights=row_heights,
        unit_inches=unit_inches,
        dpi=dpi,
        show_cell_borders=show_cell_borders,
    )
    if unit_height_inches is not None:
        kwargs["unit_height_inches"] = unit_height_inches
    if hold_ticks is not None:
        kwargs["hold_ticks"] = hold_ticks
    return Figure.compose(**kwargs)


def _build_dynamic_figure(unit_inches=None, dpi=None,
                          unit_height_inches=None,
                          show_cell_borders=False,
                          text_border=True) -> Figure:
    # Animated form is the SINGLE cycling row (case 1 → 2 → 3), not the 3-row
    # static layout render() uses. Two border worlds: the live notebook view
    # (show()/embed()) bolds white spine boxes (_bolden_spines); the GIF mirrors
    # the static PNG via cell borders (show_cell_borders=True, text_border=False).
    return _compose_cycle(unit_inches=unit_inches, dpi=dpi,
                          unit_height_inches=unit_height_inches,
                          hold_ticks=HOLD_TICKS,
                          show_cell_borders=show_cell_borders,
                          text_border=text_border)


def _build_static_figure(unit_inches=None, dpi=None,
                         unit_height_inches=None,
                         show_cell_borders=False) -> Figure:
    return _compose(dynamic=False, unit_inches=unit_inches, dpi=dpi,
                    unit_height_inches=unit_height_inches,
                    show_cell_borders=show_cell_borders)


def _bolden_spines(fig: Figure) -> None:
    """Animated paths only — thicken the spines the panels actually drew (the
    text panel strips its own). The static PNG uses the figure-1 cell-border
    frame instead (see render()), so it leaves spines at their default weight;
    forcing heavy spines there would compete with the cell border and read as a
    second, mismatched panel frame. Mirrors gen_figure_2_6's _bolden_spines.

    Text panels don't frame themselves with an axes spine — they draw a
    fill=False Rectangle (base.py `_draw_content_border`) at DEFAULT_SPINE_LINEWIDTH.
    Bold that too, so the text box border matches the plot spine weight instead
    of rendering as a thinner, mismatched frame."""
    import matplotlib.patches as mpatches
    for ax in fig._mpl_fig.axes:
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(SPINE_LINEWIDTH)
        for patch in ax.patches:
            if isinstance(patch, mpatches.Rectangle) and not patch.get_fill():
                patch.set_linewidth(SPINE_LINEWIDTH)


def _draw_stack_dividers(fig: Figure, *, color, linewidth, alpha) -> None:
    """Draw the missing border line BETWEEN the two stacked stem plots (a,b
    above, products below). They live in one CompositePanel, so the cell-border
    system frames the pair with a single outer rectangle and leaves no divider
    where the two plots meet. This adds it: a horizontal line at the midpoint of
    their inter-plot gap, spanning the same width as the stack's outer cell
    border (figure edge on the left → gutter centre on the right) so it tiles
    flush with that border instead of floating inside it."""
    import matplotlib.lines as mlines
    mpl = fig._mpl_fig
    fig_w = mpl.get_size_inches()[0]
    half_gutter = (COL_GUTTER_INCHES / 2.0) / fig_w
    # All stem plots share the leftmost column, so identify them by xlim and pair
    # them by vertical adjacency: sorted bottom→top they alternate products, a/b,
    # products, a/b, …, so consecutive twos are the (products, a/b) of one case.
    stems = [ax for ax in mpl.axes
             if abs(ax.get_xlim()[0] - X_LIM[0]) < 1e-6
             and abs(ax.get_xlim()[1] - X_LIM[1]) < 1e-6]
    stems.sort(key=lambda a: a.get_position().y0)  # bottom-first
    for i in range(0, len(stems) - 1, 2):
        lower = stems[i].get_position()            # products (lower plot)
        upper = stems[i + 1].get_position()        # a/b (upper plot)
        y = (lower.y1 + upper.y0) / 2.0            # centre of the inter-plot gap
        cell_left = 0.0                            # stack is the leftmost column
        cell_right = upper.x1 + half_gutter        # gutter centre toward Sum strip
        mpl.add_artist(mlines.Line2D(
            [cell_left, cell_right], [y, y], transform=mpl.transFigure,
            color=color, linewidth=linewidth, alpha=alpha,
            zorder=50, clip_on=False))


def _uniform_borders(fig: Figure) -> None:
    """Make every gray cell border read at ONE thickness and ONE brightness.

    The library tiles a fill=False Rectangle per cell (figure.py _draw_cell_borders),
    which leaves two inconsistencies the user can see:

      1. Thickness — perimeter sides snap to the figure edge (x/y = 0 or 1), so
         half each stroke falls off-canvas and that side renders at HALF width
         (≈1px) while interior dividers show full width (≈2px). Fix: nudge any
         side sitting on the figure edge inward by half a stroke (converted to
         figure fraction) so the whole stroke lands on-canvas.
      2. Brightness — adjacent cells each draw their shared interior edge, so
         that seam is painted TWICE; at alpha<1 the two passes stack brighter
         (0.6 → ≈0.84) than a singly-drawn perimeter side, reading as a doubled
         border. Fix: set the borders opaque so a second pass is idempotent —
         overlaps and perimeter then match exactly.

    Post-process (mirrors _inset_ticks / _draw_stack_dividers) so the shared
    library stays untouched. Applied to BOTH render() and save_gif() so the
    static PNG and the GIF keep the same border treatment."""
    import matplotlib.patches as mpatches
    mpl = fig._mpl_fig
    fig_w_in, fig_h_in = mpl.get_size_inches()
    eps = 1e-6
    for art in list(mpl.artists):
        if not isinstance(art, mpatches.Rectangle) or art.get_fill():
            continue
        if art.get_zorder() != 50:          # the cell-border rects, not stray patches
            continue
        art.set_alpha(1.0)                  # opaque → double-drawn seams stop brightening
        lw_pt = art.get_linewidth()
        # Inset edge-snapped sides by HALF a stroke width: a line is drawn centered
        # on its edge coordinate, so a side sitting at fraction 0 (or 1) spills half
        # its stroke off-canvas and reads at half width. Nudging the edge in by half
        # a stroke seats the stroke's CENTER at half-stroke from the canvas, so its
        # outer edge lands flush with pixel 0 (no dark gap) and its full width is
        # on-canvas (matches the interior dividers). A full inset, by contrast,
        # leaves a ~1px dark strip outside the border on the left/top edges.
        ins_x = (lw_pt / 72.0) / fig_w_in / 2.0
        ins_y = (lw_pt / 72.0) / fig_h_in / 2.0
        x0, y0 = art.get_xy()
        x1, y1 = x0 + art.get_width(), y0 + art.get_height()
        if x0 <= eps:       x0 = ins_x
        if y0 <= eps:       y0 = ins_y
        if x1 >= 1.0 - eps: x1 = 1.0 - ins_x
        if y1 >= 1.0 - eps: y1 = 1.0 - ins_y
        art.set_bounds(x0, y0, x1 - x0, y1 - y0)


def _inset_ticks(fig: Figure) -> None:
    """Static path only — point every tick mark INWARD.

    StaticPanel draws ticks with direction="inout" (static_panel.py), so the
    outward half pokes past the (invisible) spine and crosses the gray cell
    border that frames each cell — the n-axis marks straddle the top/bottom
    border, the y marks straddle the left. The cell border is the single frame
    here (see render()), so the ticks must live inside it. Tick direction isn't
    a style knob and style.py is off-limits, so we retarget it after build,
    mirroring _bolden_spines."""
    for ax in fig._mpl_fig.axes:
        ax.tick_params(axis="both", which="both", direction="in")


# The figure's curated home under by_figure/. The latest composite (PNG) and
# cycle (GIF) live at the top level named ``{FIG_SLUG}_{kind}_v{N}.{ext}``; every
# superseded version is moved into archive/ by _publish_path(). One slug names
# both the folder and the file stem so the tree self-documents.
FIG_SLUG = "fig_2_5_sign_accumulation"


def _default_output_dir() -> str:
    research_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(os.path.join(
        research_dir, "..", "assets", "images", "dsp", "figures",
        "by_figure", FIG_SLUG))


def _publish_path(kind: str, ext: str, label: Optional[str] = None) -> str:
    """Path for a NEW render of `kind` ('composite' | 'cycle'), auto-versioned,
    with any existing top-level version of that kind moved into archive/ first.

    Naming: ``{FIG_SLUG}_{kind}_v{N}[_{label}].{ext}`` where N is one past the
    highest v-number seen across the live dir AND archive/ (so versions are
    monotonic — they never collide or reset). Net effect: the live folder always
    holds just the newest composite + cycle, and prior versions accumulate in
    archive/ — the by_figure convention, automated."""
    import re
    import shutil
    live = _default_output_dir()
    archive = os.path.join(live, "archive")
    os.makedirs(archive, exist_ok=True)
    pat = re.compile(rf"^{re.escape(FIG_SLUG)}_{re.escape(kind)}_v(\d+)")

    def _max_version(directory: str) -> int:
        best = 0
        if os.path.isdir(directory):
            for name in os.listdir(directory):
                match = pat.match(name)
                if match:
                    best = max(best, int(match.group(1)))
        return best

    version = max(_max_version(live), _max_version(archive)) + 1
    # Retire any current top-level version(s) of this kind into archive/.
    for name in os.listdir(live):
        src = os.path.join(live, name)
        if pat.match(name) and os.path.isfile(src):
            shutil.move(src, os.path.join(archive, name))
    suffix = f"_{label}" if label else ""
    return os.path.join(live, f"{FIG_SLUG}_{kind}_v{version}{suffix}.{ext}")


def render(
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    *,
    label: Optional[str] = None,
    show_cell_borders: bool = True,
) -> str:
    """Render the fully-processed final frame as a static PNG. Returns the path.

    By default publishes into the figure's by_figure/ home as the next
    ``{FIG_SLUG}_composite_v{N}.png`` and archives the prior version (see
    _publish_path); pass `label` to tag the version (e.g. label="border_fix").
    Pass an explicit output_dir/output_filename to write a one-off literal path
    instead (no versioning, no archiving).

    Figure-1 treatment: every panel's frame is the library CELL BORDER — the
    rectangle that wraps the whole gridspec cell (plot + its tick / axis labels),
    matching gen_figure_1 and gen_figure_2_6. Spines stay at their default thin
    weight so they don't compete with the cell border as a second frame; the
    leftmost column's cell border reaches the figure edge, so its y-tick labels
    sit inside the frame instead of stranded in the margin.
    """
    with _unified_style(), _static_chrome_style():
        # unit_height derived so the (gutter-absorbing) 2-column text cell is a
        # true square. Stem stack ≈ 8.2" wide; Sum strip ≈ 1.6" wide; text box
        # square. (4 : 1 : 2 — the square text cell supersedes the earlier 4:1:3.)
        uw = 1.6
        fig = _build_static_figure(unit_inches=uw,
                                   unit_height_inches=_square_unit_height(uw),
                                   show_cell_borders=show_cell_borders)
        fig.render()
        _inset_ticks(fig)
        _uniform_borders(fig)
        # Divider between the two stacked stem plots, in the same gray cell-border
        # style (NEUTRAL_COLOR, 2× spine width) so it reads as one system with
        # every other border — opaque to match _uniform_borders' flattened alpha.
        _draw_stack_dividers(fig, color=style.NEUTRAL_COLOR,
                             linewidth=style.DEFAULT_SPINE_LINEWIDTH * 2.0,
                             alpha=1.0)
    if output_filename is None and output_dir is None:
        output_path = _publish_path("composite", "png", label)
    else:
        output_dir = output_dir or _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename or f"{FIG_SLUG}_composite.png")
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def save_gif(
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    fps: Optional[int] = None,
    *,
    label: Optional[str] = None,
) -> str:
    """Render the animation to a looping GIF (every pair processed in turn).

    By default publishes into the figure's by_figure/ home as the next
    ``{FIG_SLUG}_cycle_v{N}.gif`` and archives the prior version (see
    _publish_path); pass `label` to tag the version. Pass an explicit
    output_dir/output_filename for a one-off literal path (no versioning).

    Plays anywhere — embeds in DSP.md and renders on mobile/web without the
    notebook widget backend the inline FuncAnimation needs. The trailing
    HOLD_TICKS frames are identical, so Pillow folds them into one long final
    frame — the between-loops pause. fps defaults to match INTERVAL_MS so the
    GIF runs at the same pace as the live animation. Returns the path.
    """
    from matplotlib.animation import PillowWriter
    if fps is None:
        fps = max(1, round(1000.0 / INTERVAL_MS))
    # Same border model as the static PNG (render()): the gray library CELL
    # BORDER is the single frame for every panel, spines off (_cell_border_chrome),
    # text box framed by the cell border not its own (text_border=False), n-axis
    # ticks turned inward (_inset_ticks) and the inter-plot divider drawn in the
    # matching gray (_draw_stack_dividers). These are figure-level artists / tick
    # params, so they persist across animation frames (blit=False, no ax.cla()).
    with nb_compact_style(), _unified_style(), _cell_border_chrome():
        fig = _build_dynamic_figure(unit_inches=1.4,
                                    unit_height_inches=_square_unit_height(1.4),
                                    dpi=90, show_cell_borders=True,
                                    text_border=False)
        fig.render()
        _inset_ticks(fig)
        _uniform_borders(fig)
        _draw_stack_dividers(fig, color=style.NEUTRAL_COLOR,
                             linewidth=style.DEFAULT_SPINE_LINEWIDTH * 2.0,
                             alpha=1.0)
    if output_filename is None and output_dir is None:
        output_path = _publish_path("cycle", "gif", label)
    else:
        output_dir = output_dir or _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename or f"{FIG_SLUG}_cycle.gif")
    fig._anim.save(output_path, writer=PillowWriter(fps=fps))
    fig.close()
    return os.path.abspath(output_path)


def show() -> Figure:
    """Build, render, and display the animated §2.5 figure in a Jupyter cell."""
    import matplotlib.pyplot as plt
    with nb_compact_style(), _unified_style():
        fig = _build_dynamic_figure(unit_inches=1.4,
                                    unit_height_inches=_square_unit_height(1.4),
                                    dpi=72)
        fig.render()
        _bolden_spines(fig)
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


def embed(target: object | None = None) -> Figure:
    """Drop the animated figure into a caller-provided container (target=None
    returns the rendered Figure with its animation reference held)."""
    with _unified_style():
        fig = _build_dynamic_figure()
        fig.render()
        _bolden_spines(fig)
    return fig


if __name__ == "__main__":
    print(render())
