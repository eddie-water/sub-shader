"""gen_figure_2_6_sine_basis — §2.6 of DSP.md.

The dot product as a *frequency detector*, swept across every frequency to build
a spectrum. §2.5 ran the same machinery on plain ±1 sequences; §2.6 swaps the
arbitrary partner `b` for a **pure sine** — a *basis function* — and then sweeps
that sine's frequency across the whole band, one measurement at a time.

ONE signal:

    signal = 1.0·sin(2 Hz) + 0.5·sin(8 Hz)

is compared against a pure-sine reference whose frequency steps 1 → 10 Hz. Each
dot product drops one bar into a spectrum:

    reference 2 Hz  → coeff 1.0     (present, strong)
    reference 8 Hz  → coeff 0.5     (present, weaker)
    everything else → coeff 0.0     (measured, came back empty)

A 1 s window pins integer-Hz sines orthogonal (bin spacing = 1/duration = 1 Hz),
so absent references score *exactly* zero and present ones score exactly their
amplitude. Sampling at 24 Hz keeps Nyquist (12 Hz) clear of the whole sweep —
an 8 Hz tone would vanish at the earlier 16 Hz rate. That collection of
coefficients across all frequencies IS the spectrum — the bridge into §3, where
this same sweep becomes the Fourier transform. §2.6 measures *every* frequency
by hand; §3 names the operation.

Coefficient = normalized dot product, signal · reference / (N/2): a unit sine at
an integer Hz scores 1.0 (mirrors gen_figure_3_1._coefficient).

Animation: one frame per reference frequency. The top panel re-sines to the
current frequency (smooth purple trace over the static orange signal); the
bottom spectrum grows the bar that frequency just earned, with the freshest bar
branded. Static / GIF / inline-widget modes share the build via the §2.5
three-mode contract.

    render(output_dir, output_filename) -> str   # production PNG (full spectrum)
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

import numpy as np

from .. import (
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
# Data — ONE signal swept against pure-sine references at every integer Hz.
#
# A 1 s window sampled at 16 Hz (N=16) makes every integer-Hz sine orthogonal,
# so the measurements are clean: present references score their exact amplitude,
# absent ones score exactly zero. Coefficients are normalized by N/2 so a unit
# sine scores 1.0 — identical convention to gen_figure_3_1.
# ---------------------------------------------------------------------------
N = 24
SAMPLE_RATE_HZ = 24.0                     # Nyquist 12 Hz — keeps the whole 1–10 sweep
DURATION_S = 1.0                          # (and the 10 Hz tone) below the aliasing limit
T = np.arange(N) / SAMPLE_RATE_HZ
N_IDX = np.arange(N, dtype=float)

TONE_A_HZ, TONE_A_AMP = 2.0, 1.0        # present, strong
TONE_B_HZ, TONE_B_AMP = 10.0, 0.5       # present, weaker

SWEEP_HZ = list(range(1, 11))            # reference frequencies 1 … 10 Hz
NORM = N / 2.0                            # unit sine at integer Hz → coeff 1.0

STATIC_HIGHLIGHT_HZ = 2.0                 # which reference the static PNG shows up top


def _sine(freq_hz: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq_hz * T)


def _signal() -> np.ndarray:
    return _sine(TONE_A_HZ, TONE_A_AMP) + _sine(TONE_B_HZ, TONE_B_AMP)


def _coefficient(freq_hz: float) -> float:
    """Normalized dot product of the signal against a unit sine at freq_hz."""
    return float(np.dot(_signal(), _sine(freq_hz)) / NORM)


SIGNAL = _signal()
COEFFS = {f: _coefficient(f) for f in SWEEP_HZ}
FINAL = COEFFS

# Per-sample running coefficient and product-sign for every reference frequency.
# RUNNING[f][n] is the normalized partial dot product after n+1 samples — the
# value the spectrum bar shows while frequency f is being accumulated. It climbs
# to FINAL[f]; for an absent reference it wobbles around zero and settles there.
RUNNING = {f: np.cumsum(SIGNAL * _sine(f)) / NORM for f in SWEEP_HZ}
KIND = {f: np.sign(SIGNAL * _sine(f)).astype(int) for f in SWEEP_HZ}


# --- layout / scale ---------------------------------------------------------
WIDTH_UNITS = 4

X_LIM = (-0.9, N - 0.1)
X_TICKS = [0, 6, 12, 18]
SIGNAL_YLIM = (-1.7, 1.7)
SIGNAL_YTICKS = [-1, 0, 1]

SPEC_XLIM = (SWEEP_HZ[0] - 0.6, SWEEP_HZ[-1] + 0.6)
SPEC_XTICKS = SWEEP_HZ
# Symmetric about zero (−1 … 1) for aesthetic balance: the zero line sits
# centred, and the absent references' negative running-sum dips read against the
# present references' positive climbs.
SPEC_YLIM = (-1.12, 1.12)
SPEC_YTICKS = [-1.0, 0.0, 1.0]

TITLE = "Figure 2.6 - Measuring Frequencies"
TOP_SUBTITLE = "The signal, compared against a sweeping pure-sine reference"
SPEC_SUBTITLE = "Dot product at each reference frequency  →  spectrum"

# Right-hand caption column width (units). Plot stays WIDTH_UNITS wide; the
# descriptive text box sits beside it instead of stacked in a band above.
TEXT_UNITS_W = 1
TEXT_UNITS = (TEXT_UNITS_W, 1)
TOTAL_WIDTH_UNITS = WIDTH_UNITS + TEXT_UNITS_W

# Captions are pre-broken into balanced lines and centred (h+v) in their square
# cell. Manual breaks (not auto-wrap) so the simple centered ax.text path can be
# used — the wrap pipeline only left-aligns / justifies, never centres.
TOP_SUBTITLE_LINES = "The signal, compared\nagainst a sweeping\npure-sine reference"
SPEC_SUBTITLE_LINES = "Dot product at each\nreference frequency\n→  spectrum"

# One text colour for everything (chrome + in-plot), matching §2.5.
TEXT_COLOR = "#EEEEEE"
IN_PLOT_FONTSIZE = 22
FREQ_LABEL_FONTSIZE = 20
ROW_TITLE_COLOR = TEXT_COLOR

# --- weights / palette ------------------------------------------------------
# The top time-domain panel reads as a spotlight: processed pairs and the
# reference sine sit at full strength, upcoming samples wait as a clearly-present
# (not washed-out) ghost, and the cursor lifts the pair under the multiply.
GHOST_ALPHA = 0.34
BRIGHT_ALPHA = 1.0
SINE_GHOST_ALPHA = 0.62
SINE_GHOST_LINEWIDTH = 3.2
SAMPLE_STEM_LINEWIDTH = 5.0
SAMPLE_MARKERSIZE = 9.0     # dome the 5pt stem at the same ~1.8 marker:stem ratio as the spectrum
ZERO_LINE_WIDTH = 1.8
SPINE_LINEWIDTH = 2.6
SPEC_STEM_LW = 6.0
SPEC_MARKERSIZE = 11.0     # clearly domes the 6pt stem (matches the signal panel's ~1.6 marker:stem ratio); a near-equal marker reads as off-centre under anti-aliasing
SPEC_ALPHA = 0.95          # every coefficient stem at one weight — uniform, keep adding
SPEC_LABEL_GAP = 0.12      # running-sum number sits this far across the zero line
COL_WSPACE = 0.18
ROW_HSPACE = 0.34

# Static-PNG-only layout (figure-1 treatment): every cell framed and flush so
# the title bands read as bordered text cells touching their plots. Thinner
# bands hug the plots; a wider gutter + compact axis typography + a tucked-in
# x-label inset keep each plot's tick numbers AND axis label inside the
# half-gutter above the mid-gutter cell border (otherwise the border line draws
# straight through "sample n"). Scoped to render() only — the animated
# notebook/GIF path keeps nb_compact_style's smaller-cell sizing untouched.
STATIC_ROW_HEIGHTS = [0.26, 0.78, 0.78]
STATIC_ROW_HSPACE = 0.5
# One font size for every chrome element in the static figure — title, axis
# labels, tick labels, and the right-hand captions all render at STATIC_FONT_SIZE.
# Applied via the scoped style block (only the static render path), so the
# compact GIF/notebook path keeps nb_compact_style's own smaller sizes.
STATIC_FONT_SIZE = 24
STATIC_LAYOUT_STYLE = {
    "DEFAULT_TICK_LABEL_SIZE": STATIC_FONT_SIZE,
    "DEFAULT_AXIS_LABEL_SIZE": STATIC_FONT_SIZE,
    "SUPTITLE_FONT_SIZE": STATIC_FONT_SIZE,
    "DEFAULT_SUBTITLE_FONT_SIZE": STATIC_FONT_SIZE,
    "DEFAULT_X_AXIS_LABEL_INSET_INCHES": 0.58,
    # This is a single full-width column, so each cell's left/right borders ARE
    # the figure perimeter. Pull that perimeter in close to the y-axis chrome so
    # the y-label + tick numbers sit just inside the left border (template look)
    # instead of stranded in a wide margin. Sized to clear the rotated y-label
    # ("signal , sine") + tick numbers without clipping at the edge.
    "DEFAULT_MARGIN_INCHES": 1.25,
}

CURSOR_COLOR = style.NEUTRAL_COLOR
CURSOR_ALPHA = 0.24
SIGN_FONTSIZE = 18

# Nested clock: each reference frequency gets its own per-sample accumulation,
# then a pause on the finished coefficient, then the next frequency begins.
#   local 0          → fresh reference shown, nothing multiplied yet
#   local 1..N       → speed-run the multiplies, the bar grows pair by pair
#   local N+1..N+P   → pause on the final coefficient
INTERVAL_MS = 250           # inner tick — deliberate, matching §2.5's cadence
PAUSE_TICKS = 9             # hold frames on each finished coefficient
SEG = N + 1 + PAUSE_TICKS
TOTAL_FRAMES = len(SWEEP_HZ) * SEG
END_HOLD_TICKS = 4          # brief breath before the loop wraps


def _decode(k: int) -> tuple[int, int]:
    """Global frame k → (frequency index, samples processed 0..N)."""
    fi = min(k // SEG, len(SWEEP_HZ) - 1)
    local = k - fi * SEG
    count = local if local <= N else N
    return fi, count

_UNIFIED_TEXT_STYLE = {
    "TICK_LABEL_COLOR": TEXT_COLOR,
    "SUPTITLE_COLOR": TEXT_COLOR,
    "SPINE_COLOR": TEXT_COLOR,
}


@contextmanager
def _scoped_style(overrides):
    orig = {k: getattr(style, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


def _unified_style():
    """Shared color contract for every render path (chrome + spines white)."""
    return _scoped_style(_UNIFIED_TEXT_STYLE)


def _static_layout_style():
    """Static-PNG-only typography/inset tuning for the bordered figure-1 grid."""
    return _scoped_style(STATIC_LAYOUT_STYLE)


@contextmanager
def _cell_border_chrome():
    """Spines off so the gray library CELL BORDER is the single panel frame —
    figure 2.5's GIF border model. Every cell (plots + caption) reads with one
    identical gray border; no white inner spine box competes, and the caption
    carries no border of its own (the cell border frames it)."""
    orig = style.SPINE_COLOR
    try:
        style.SPINE_COLOR = "none"
        yield
    finally:
        style.SPINE_COLOR = orig


# ---------------------------------------------------------------------------
# Plottables
# ---------------------------------------------------------------------------
class _ZeroLine(Plottable):
    """Faint horizontal reference at y = 0 so above/below (the sign) reads."""

    def __init__(self, *, color, alpha=0.6, linewidth=ZERO_LINE_WIDTH, zorder=1):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.linewidth = linewidth

    def draw(self, ax):
        ax.axhline(0.0, color=self.color, alpha=self.alpha,
                   linewidth=self.linewidth, zorder=self.zorder)


class _Stems(Plottable):
    """Stem plot drawn with explicit line + marker artists (no autoscale)."""

    def __init__(self, x, y, *, color, alpha, linewidth=SAMPLE_STEM_LINEWIDTH,
                 markersize=SAMPLE_MARKERSIZE, zorder=3):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.linewidth = linewidth
        self.markersize = markersize

    def draw(self, ax):
        for xi, yi in zip(self.x, self.y):
            ax.plot([xi, xi], [0.0, yi], color=self.color, alpha=self.alpha,
                    linewidth=self.linewidth, solid_capstyle="round",
                    zorder=self.zorder)
        ax.plot(self.x, self.y, linestyle="none", marker="o",
                markersize=self.markersize, color=self.color,
                alpha=self.alpha, zorder=self.zorder)


class _SineGhost(Plottable):
    """A smooth continuous trace of a pure sine, so the reference reads
    unmistakably as a *sine wave* even though the dot product only ever touches
    the sampled stem values."""

    def __init__(self, freq_hz, *, color, alpha=SINE_GHOST_ALPHA,
                 linewidth=SINE_GHOST_LINEWIDTH, zorder=1):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.freq_hz = float(freq_hz)
        self.linewidth = linewidth

    def draw(self, ax):
        idx = np.linspace(0.0, N - 1, 600)
        t = idx / SAMPLE_RATE_HZ
        y = np.sin(2.0 * np.pi * self.freq_hz * t)
        ax.plot(idx, y, color=self.color, alpha=self.alpha,
                linewidth=self.linewidth, zorder=self.zorder)


class _SignalGhost(Plottable):
    """A smooth continuous trace of the composite signal, the orange counterpart
    to the purple reference sine — so both operands read as continuous waveforms
    behind the sampled stems the dot product actually multiplies."""

    def __init__(self, *, color, alpha=SINE_GHOST_ALPHA,
                 linewidth=SINE_GHOST_LINEWIDTH, zorder=1):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.linewidth = linewidth

    def draw(self, ax):
        idx = np.linspace(0.0, N - 1, 600)
        t = idx / SAMPLE_RATE_HZ
        y = (TONE_A_AMP * np.sin(2.0 * np.pi * TONE_A_HZ * t)
             + TONE_B_AMP * np.sin(2.0 * np.pi * TONE_B_HZ * t))
        ax.plot(idx, y, color=self.color, alpha=self.alpha,
                linewidth=self.linewidth, zorder=self.zorder)


class _Cursor(Plottable):
    """A faint vertical highlight on the pair being multiplied this frame."""

    def __init__(self, x, *, color=CURSOR_COLOR, alpha=CURSOR_ALPHA, zorder=0):
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.x = float(x)

    def draw(self, ax):
        ax.axvspan(self.x - 0.5, self.x + 0.5, color=self.color,
                   alpha=self.alpha, linewidth=0.0, zorder=self.zorder)


class _SignStrip(Plottable):
    """The +/−/0 verdict for each multiplied pair, centred on the zero line. A
    background box punches each glyph out of any stem passing through zero."""

    def __init__(self, x, kinds, *, y=0.0, fontsize=SIGN_FONTSIZE,
                 color=TEXT_COLOR, zorder=6):
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.x = np.asarray(x, dtype=float)
        self.kinds = np.asarray(kinds, dtype=int)
        self.y = y
        self.fontsize = fontsize

    def draw(self, ax):
        glyph = {1: "+", -1: "−", 0: "0"}
        bbox = dict(facecolor=style.BG_COLOR, edgecolor="none",
                    boxstyle="square,pad=0.12")
        for xi, k in zip(self.x, self.kinds):
            ax.text(float(xi), self.y, glyph[int(k)],
                    color=self.color, fontsize=self.fontsize, fontweight="bold",
                    ha="center", va="center", zorder=self.zorder, bbox=bbox)


class _FreqLabel(Plottable):
    """A 'reference = k Hz' badge parked in the top-right of the signal panel,
    naming the frequency currently under test."""

    def __init__(self, freq_hz, *, color=style.SECONDARY_COLOR, zorder=8):
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.freq_hz = float(freq_hz)

    def draw(self, ax):
        bbox = dict(facecolor=style.BG_COLOR, edgecolor=self.color,
                    linewidth=1.6, boxstyle="round,pad=0.3")
        ax.text(0.985, 0.93, f"reference = {self.freq_hz:.0f} Hz",
                transform=ax.transAxes, color=self.color,
                fontsize=FREQ_LABEL_FONTSIZE, fontweight="bold",
                ha="right", va="top", zorder=self.zorder, bbox=bbox)


class _CoeffLabel(Plottable):
    """The coefficient the current reference just earned, floated above its
    spectrum bar so the running measurement reads as a number too."""

    def __init__(self, freq_hz, value, *, color=TEXT_COLOR, zorder=8):
        super().__init__(color=color, alpha=1.0, zorder=zorder)
        self.freq_hz = float(freq_hz)
        self.value = float(value)

    def draw(self, ax):
        # Directly under the stem, mirrored across zero: a positive stem points
        # up so its running-sum number sits underneath (below the zero line); a
        # negative stem points down so the number sits above. A value that rounds
        # to zero reads as a clean "0.00" underneath — never a signed "-0.00".
        if self.value > 0.005:
            text, y, va = f"{self.value:.2f}", -SPEC_LABEL_GAP, "top"
        elif self.value < -0.005:
            text, y, va = f"{self.value:.2f}", SPEC_LABEL_GAP, "bottom"
        else:
            text, y, va = "0.00", -SPEC_LABEL_GAP, "top"
        ax.text(self.freq_hz, y, text,
                color=self.color, fontsize=FREQ_LABEL_FONTSIZE,
                fontweight="bold", ha="center", va=va, zorder=self.zorder)


# ---------------------------------------------------------------------------
# Frame content — frame k (0..len-1) measures reference SWEEP_HZ[k].
# ---------------------------------------------------------------------------
def _signal_base() -> list:
    """Static background of the top panel: the zero line plus a faint full-length
    ghost of the signal stems, so the layout is fixed and the reader sees what is
    coming. The bright processed stems and the reference are drawn per frame."""
    return [
        _ZeroLine(color=style.DROPLINE_COLOR),
        _SignalGhost(color=style.PRIMARY_COLOR),
        _Stems(N_IDX, SIGNAL, color=style.PRIMARY_COLOR, alpha=GHOST_ALPHA,
               zorder=2),
    ]


def _signal_frame(k: int) -> list:
    """Top panel for global frame k: the current reference sine plus the pairs
    multiplied so far, lit up one at a time as the inner clock advances."""
    fi, count = _decode(k)
    freq = SWEEP_HZ[fi]
    ref = _sine(freq)
    items: list = [
        _SineGhost(freq, color=style.SECONDARY_COLOR),
        _Stems(N_IDX, ref, color=style.SECONDARY_COLOR, alpha=GHOST_ALPHA,
               zorder=2),
        _FreqLabel(freq),
    ]
    if count > 0:
        idx = slice(0, count)
        items.append(_Cursor(N_IDX[count - 1]))
        items.append(_Stems(N_IDX[idx], SIGNAL[idx], color=style.PRIMARY_COLOR,
                            alpha=BRIGHT_ALPHA, zorder=3))
        items.append(_Stems(N_IDX[idx], ref[idx], color=style.SECONDARY_COLOR,
                            alpha=BRIGHT_ALPHA, zorder=4))
        items.append(_SignStrip(N_IDX[idx], KIND[freq][idx]))
    return items


def _spectrum_base() -> list:
    return [_ZeroLine(color=style.DROPLINE_COLOR)]


def _spectrum_frame(k: int) -> list:
    """Bottom panel for global frame k: locked-in coefficients for every finished
    frequency, plus the current frequency's bar growing with the running partial
    sum as each pair is multiplied. Every coefficient is neutral white — the
    result is a magnitude, not an operand; the active bar reads by weight, not
    colour."""
    fi, count = _decode(k)
    items: list = []

    done = SWEEP_HZ[:fi]
    if done:
        items.append(_Stems(
            np.array(done, dtype=float),
            np.array([FINAL[f] for f in done], dtype=float),
            color=style.NEUTRAL_COLOR, alpha=SPEC_ALPHA,
            linewidth=SPEC_STEM_LW, markersize=SPEC_MARKERSIZE, zorder=3))

    cur = SWEEP_HZ[fi]
    value = float(RUNNING[cur][count - 1]) if count > 0 else 0.0
    items.append(_Stems(
        np.array([cur], dtype=float), np.array([value], dtype=float),
        color=style.NEUTRAL_COLOR, alpha=SPEC_ALPHA, linewidth=SPEC_STEM_LW,
        markersize=SPEC_MARKERSIZE, zorder=5))
    items.append(_CoeffLabel(cur, value))
    return items


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _dynamic_signal_panel() -> DynamicTimeSeriesPanel:
    return DynamicTimeSeriesPanel(
        units=(WIDTH_UNITS, 1),
        frame_fn=_signal_frame,
        num_frames=TOTAL_FRAMES,
        interval_ms=INTERVAL_MS,
        base_plottables=_signal_base(),
        x_label="sample n",
        y_label="signal , sine",
        xticks=X_TICKS,
        yticks=SIGNAL_YTICKS,
        xlim=X_LIM,
        ylim=SIGNAL_YLIM,
        show_xticklabels=True,
        show_yticklabels=True,
    )


def _dynamic_spectrum_panel() -> DynamicTimeSeriesPanel:
    return DynamicTimeSeriesPanel(
        units=(WIDTH_UNITS, 1),
        frame_fn=_spectrum_frame,
        num_frames=TOTAL_FRAMES,
        interval_ms=INTERVAL_MS,
        base_plottables=_spectrum_base(),
        x_label="reference frequency (Hz)",
        y_label="coeff",
        xticks=SPEC_XTICKS,
        yticks=SPEC_YTICKS,
        xlim=SPEC_XLIM,
        ylim=SPEC_YLIM,
        show_xticklabels=True,
        show_yticklabels=True,
    )


def _static_signal_panel(highlight_hz: float) -> TimeSeriesPanel:
    panel = TimeSeriesPanel(
        units=(WIDTH_UNITS, 1),
        x_label="sample n",
        y_label="signal , sine",
        xticks=X_TICKS,
        yticks=SIGNAL_YTICKS,
        xlim=X_LIM,
        ylim=SIGNAL_YLIM,
        show_xticklabels=True,
        show_yticklabels=True,
    )
    for p in _signal_base():
        panel.add(p)
    panel.add(_SineGhost(highlight_hz, color=style.SECONDARY_COLOR))
    panel.add(_Stems(N_IDX, _sine(highlight_hz), color=style.SECONDARY_COLOR,
                     alpha=BRIGHT_ALPHA, zorder=4))
    panel.add(_FreqLabel(highlight_hz))
    return panel


def _static_spectrum_panel(highlight_hz: float) -> TimeSeriesPanel:
    panel = TimeSeriesPanel(
        units=(WIDTH_UNITS, 1),
        x_label="reference frequency (Hz)",
        y_label="coeff",
        xticks=SPEC_XTICKS,
        yticks=SPEC_YTICKS,
        xlim=SPEC_XLIM,
        ylim=SPEC_YLIM,
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(_ZeroLine(color=style.DROPLINE_COLOR))
    freqs = np.array(SWEEP_HZ, dtype=float)
    coeffs = np.array([COEFFS[f] for f in SWEEP_HZ], dtype=float)
    panel.add(_Stems(freqs, coeffs, color=style.NEUTRAL_COLOR, alpha=SPEC_ALPHA,
                     linewidth=SPEC_STEM_LW, markersize=SPEC_MARKERSIZE, zorder=3))
    # Uniform stems — label only the two responders' coefficients.
    for f in (TONE_A_HZ, TONE_B_HZ):
        panel.add(_CoeffLabel(f, COEFFS[f]))
    return panel


# ---------------------------------------------------------------------------
# Composition + three-mode contract
# ---------------------------------------------------------------------------
def _side_text_panel(text: str) -> TextPanel:
    """Right-hand caption beside a plot (STATIC PNG). A short descriptive
    sentence, pre-broken into balanced lines and centred (horizontally +
    vertically) in its square cell so it reads as a label paired with the plot
    to its left. The cell is already framed by ``show_cell_borders``, so no
    border is drawn here."""
    return TextPanel(
        text,
        units=TEXT_UNITS,
        font_size=style.DEFAULT_SUBTITLE_FONT_SIZE,
        min_font_size=12,
        color=TEXT_COLOR,
        fontweight="bold",
        # Simple centered ax.text path: ha/va default to "center", text is
        # pre-broken with \n, auto_shrink scales the whole block to fit. justify
        # / top_anchor are left off on purpose — they route to the wrap pipeline
        # which only left-aligns or block-justifies, never true-centres.
        ha="center",
        va="center",
        auto_shrink=True,
        show_ghost_border=False,
        facecolor="none",
    )


def _gif_text_panel(text: str) -> TextPanel:
    """Right-hand caption for the GIF — figure 2.5's caption treatment exactly.
    Top-anchored, left-aligned text that draws NO border of its own: the gray
    library cell border (show_cell_borders=True, spines off via
    _cell_border_chrome) is the single frame, identical to every plot cell. The
    text is passed UNBROKEN so the wrap pipeline word-wraps it to the cell width,
    inset by content_margin_frac for the same breathing room the plots have."""
    return TextPanel(
        text,
        units=TEXT_UNITS,
        font_size=style.DEFAULT_SUBTITLE_FONT_SIZE,
        min_font_size=12.0,
        color=TEXT_COLOR,
        fontweight="normal",
        auto_shrink=True,
        # Wrap pipeline: top-anchored + left-aligned, laid out inside the
        # content-margin inset from the cell border. justify=False keeps a
        # natural left rag rather than block-justified gaps.
        justify=False,
        top_anchor=True,
        cell_padding_frac=0.0,
        content_margin_frac=0.06,
        # No inner box — the gray cell border is the ONE and only frame, exactly
        # as figure 2.5's text panel relies on the cell border.
        show_ghost_border=False,
        facecolor="none",
    )


def _compose(signal_panel, spectrum_panel, *, unit_inches=None, dpi=None,
             unit_height_inches=None, hold_ticks=None,
             row_heights=None, hspace=None, show_cell_borders=False,
             gif_caption=False) -> Figure:
    # Both paths frame every cell with the gray library cell border
    # (show_cell_borders=True). Only the caption text style differs:
    #   Static PNG  — small CENTRED captions, pre-broken into balanced lines.
    #   GIF         — figure 2.5's caption: top-anchored, left-aligned, the
    #                 UNBROKEN sentence word-wrapped, no border of its own.
    if gif_caption:
        top_text = _gif_text_panel(TOP_SUBTITLE)
        spec_text = _gif_text_panel(SPEC_SUBTITLE)
    else:
        top_text = _side_text_panel(TOP_SUBTITLE_LINES)
        spec_text = _side_text_panel(SPEC_SUBTITLE_LINES)
    rows = [
        [SuptitlePanel(TITLE, units=(TOTAL_WIDTH_UNITS, 1))],
        [signal_panel, top_text],
        [spectrum_panel, spec_text],
    ]
    if row_heights is None:
        row_heights = [0.3, 1.5, 1.5]
    kwargs = dict(
        rows=rows,
        row_heights=row_heights,
        wspace=COL_WSPACE,
        hspace=ROW_HSPACE if hspace is None else hspace,
        unit_inches=unit_inches,
        dpi=dpi,
        # Figure-1 treatment (static PNG only): every cell framed so the plots
        # and their right-hand caption boxes read as a bordered grid.
        show_cell_borders=show_cell_borders,
    )
    if unit_height_inches is not None:
        kwargs["unit_height_inches"] = unit_height_inches
    if hold_ticks is not None:
        kwargs["hold_ticks"] = hold_ticks
    return Figure.compose(**kwargs)


def _build_dynamic_figure(unit_inches=None, dpi=None,
                          unit_height_inches=None) -> Figure:
    # Figure 2.5's GIF border model: every cell framed by the gray library cell
    # border, caption top-anchored/left-aligned with no border of its own.
    return _compose(_dynamic_signal_panel(), _dynamic_spectrum_panel(),
                    unit_inches=unit_inches, dpi=dpi,
                    unit_height_inches=unit_height_inches,
                    hold_ticks=END_HOLD_TICKS,
                    show_cell_borders=True, gif_caption=True)


def _build_static_figure(unit_inches=None, dpi=None,
                         highlight_hz=STATIC_HIGHLIGHT_HZ) -> Figure:
    return _compose(_static_signal_panel(highlight_hz),
                    _static_spectrum_panel(highlight_hz),
                    unit_inches=unit_inches, dpi=dpi,
                    row_heights=STATIC_ROW_HEIGHTS, hspace=STATIC_ROW_HSPACE,
                    show_cell_borders=True)


def _bolden_spines(fig: Figure) -> None:
    for ax in fig._mpl_fig.axes:
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(SPINE_LINEWIDTH)


def _inset_ticks(fig: Figure) -> None:
    """Point every tick mark INWARD (figure 2.5's GIF treatment). StaticPanel
    draws ticks "inout", so the outward half pokes past the invisible spine and
    crosses the gray cell border that frames each cell — the x marks straddle the
    border, the y marks straddle the left. With the cell border as the single
    frame, the ticks must live inside it. Retargeted after build (tick direction
    isn't a style knob), mirroring _bolden_spines."""
    for ax in fig._mpl_fig.axes:
        ax.tick_params(axis="both", which="both", direction="in")


def _default_output_dir() -> str:
    research_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(
        os.path.join(research_dir, "..", "assets", "images", "dsp", "figures",
                     "by_figure", "fig_2_6_sine_basis")
    )


def render(
    output_dir: Optional[str] = None,
    output_filename: str = "v1_final.png",
) -> str:
    """Render the completed-spectrum static PNG. Returns the absolute path."""
    if output_dir is None:
        output_dir = _default_output_dir()
    with _unified_style(), _static_layout_style():
        fig = _build_static_figure()
        fig.render()
        # NB: spines left at their default (thin) weight here — the bolded
        # cell border is the visible frame (figure-1 / template look), so a
        # heavy spine would compete with it and read as "the" panel border
        # with the axis labels stranded outside it.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def save_gif(
    output_dir: Optional[str] = None,
    output_filename: str = "v1.gif",
    fps: int = 2,
) -> str:
    """Render the sweep to a looping GIF (one frame per reference frequency)."""
    from matplotlib.animation import PillowWriter
    if output_dir is None:
        output_dir = _default_output_dir()
    # Figure 2.5's GIF border model: the gray library cell border frames every
    # cell, spines off (_cell_border_chrome) so no white box competes, caption
    # framed by the cell border not its own. No _bolden_spines — there are no
    # visible spines to bold; the cell border IS the frame.
    with nb_compact_style(), _unified_style(), _cell_border_chrome():
        fig = _build_dynamic_figure(unit_inches=2.5, unit_height_inches=1.7, dpi=90)
        fig.render()
        _inset_ticks(fig)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig._anim.save(output_path, writer=PillowWriter(fps=fps))
    fig.close()
    return os.path.abspath(output_path)


def show() -> Figure:
    """Build, render, and display the animated §2.6 sweep in a Jupyter cell."""
    import matplotlib.pyplot as plt
    with nb_compact_style(), _unified_style(), _cell_border_chrome():
        fig = _build_dynamic_figure(unit_inches=2.5, unit_height_inches=1.7, dpi=72)
        fig.render()
        _inset_ticks(fig)
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
    with _unified_style(), _cell_border_chrome():
        fig = _build_dynamic_figure()
        fig.render()
        _inset_ticks(fig)
    return fig


if __name__ == "__main__":
    print(render())
