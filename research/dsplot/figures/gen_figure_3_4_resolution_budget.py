"""Figure 3.4 - Resolution Budget (what each transform does with n cycles).

Both transforms take the same parameter, and spend it differently. Call it n,
the number of cycles of the low tone the analysis is allowed to see:

    STFT   n sets the window length, `nperseg` = n * (SR / LOW_HZ).
           That one window is then applied at EVERY frequency.
    CWT    n sets `num_cycles`, the cycle count of each Morlet in the bank.
           Each wavelet is n cycles long AT ITS OWN FREQUENCY, so its window
           in seconds shrinks as the frequency rises.

Sweeping n from 1 to 16 shows the consequence. Both transforms need a large n
to pin down the 200 Hz tone - that is not a Fourier weakness, it is the
uncertainty bound, and the wavelet obeys it too. The difference is what the
large n costs. The STFT pays for it at every frequency at once, so by n = 16 its
92.9 ms window has smeared the 2 kHz event into near-invisibility. The CWT pays
only where it must: 16 cycles at 2 kHz is 8 ms, so the event stays sharp while
the low tone gets its long window.

The same sweep runs against both members of the loud-short / quiet-long pair -
a single 0.5 ms cycle of 2 kHz at full amplitude, versus 5 ms of 2 kHz at a
tenth of it. Matched energy, opposite shapes. They stay distinguishable all the
way down the sweep (a vertical comet against a horizontal dash), which is worth
knowing: the pair does NOT collide, because the short event is ten times shorter
and therefore ten times wider in frequency, and both transforms read that width.
The construction that DOES collide is Figure 3.3, and it gets there by holding
the durations equal and varying only the arrangement in time.

Normalization
-------------
Every panel is referenced to the 200 Hz line of its own row and transform. The
low tone is identical in both signals and present at every n, so it is the one
honest anchor: within a row, brightness is comparable across all four panels,
and what the eye reads is the height of the 2 kHz event RELATIVE to the carrier.
A single global reference would instead fold in the raw gain change that comes
from integrating over a longer wavelet, which is not what this figure is about.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

from .. import (
    Figure,
    Heatmap,
    HeatmapPanel,
    SuptitlePanel,
    style,
)
from .gen_figure_1_stft_vs_cwt import (
    STACK_BAND_HEIGHT,
    STACK_CELL_VPAD_INCHES,
    STACK_GUTTER_INCHES,
    STACK_PLOT_ROW_HEIGHT,
    _contender_tight_style,
    _side_text_panel,
    _to_db,
)
from utilities import compute_full_cwt


SR = 44100
LOW_HZ = 200.0
HIGH_HZ = 2000.0                       # a decade above the low tone
LOW_PERIOD_SAMPLES = int(round(SR / LOW_HZ))    # 220 samples = 5 ms

# Built long so every window length in the sweep has room, displayed short so
# the 2 kHz event is legible. The largest window (n = 16) is 80 ms; the build
# gives it 400 ms to sit in.
BUILD_DURATION_S = 0.40
DISPLAY_DURATION_S = 0.060

# The loud-short / quiet-long pair. Amplitude x duration is matched (1.0 x 0.5 ms
# against 0.1 x 5 ms), so the two carry the same 2 kHz energy.
SHORT_EVENT_S = 1.0 / HIGH_HZ          # 0.5 ms - one cycle of the high tone
SHORT_EVENT_AMP = 1.0
LONG_EVENT_S = LOW_PERIOD_SAMPLES / SR  # 5 ms - one cycle of the low tone
LONG_EVENT_AMP = 0.1

N_CYCLES: tuple[int, ...] = (1, 2, 4, 8, 16)

CWT_ROOT_HZ = 150.0
CWT_NUM_OCTAVES = 6
DISPLAY_FREQ_LIM_HZ: tuple[float, float] = (150.0, 4000.0)
DISPLAY_FREQ_TICKS: tuple[int, ...] = (200, 2000)
DB_FLOOR = -40.0

# Every panel is resampled onto this many columns after cropping. The CWT's
# native column rate depends on num_cycles (it drives chunk_size), and the STFT's
# depends on the hop - without equalising, a panel could look smeared simply
# because it has fewer columns, which would be an artifact rather than a result.
DISPLAY_COLUMNS = 512
# Fixed STFT hop, independent of window length, for the same reason. The library
# helper hardwires noverlap = nperseg // 2, which at n = 16 would leave barely
# two columns inside the display window.
STFT_HOP_SAMPLES = LOW_PERIOD_SAMPLES // 4      # 55 samples = 1.25 ms

FIGURE_TITLE = "Figure 3.4 - Resolution Budget"
COLUMN_TITLES = ("Loud + Short: Fourier", "Loud + Short: Wavelet",
                 "Quiet + Long: Fourier", "Quiet + Long: Wavelet")

# Scaffolding prose - final copy is authored by hand.
ROW_CAPTIONS = {
    1: ("One cycle is nothing to go on. Fourier's 200 Hz row is a solid bar and "
        "the wavelet bank rings across every band. Neither can place a frequency "
        "yet - and both pin the event in time."),
    2: ("Two cycles. The 200 Hz line begins to gather in both, and the Fourier "
        "panels have already started spreading the 2 kHz event sideways as their "
        "window grows."),
    4: ("Four cycles. The low tone is nearly resolved and the wavelet's ringing "
        "is clearing out of the upper bands, while the Fourier event keeps "
        "widening in time."),
    8: ("Eight cycles - the split opens. Both place 200 Hz cleanly now, but the "
        "Fourier window is 40 ms at every frequency, so the 2 kHz event is "
        "fading into it."),
    16: ("Sixteen cycles. The 200 Hz line is razor sharp in both. The Fourier "
         "window is 80 ms everywhere and the event is gone; the wavelet spends "
         "only 8 ms up at 2 kHz."),
}
CAPTION_FONT_SIZE = 26

# Four panels wide plus the caption column. A panel is 2 units square, and the
# caption square is 2 wide x the 2-unit row height - the same square model as
# Figure 1's stacked layout, just with more columns in it.
PLOT_UNITS = (2, 1)
N_PLOT_COLUMNS = 4
TOTAL_WIDTH_UNITS = N_PLOT_COLUMNS * PLOT_UNITS[0] + 2
HEADER_ROW_HEIGHT = 0.55

# Figure 1 seats its y-axis chrome in an in-plot label strip
# (STACK_LABEL_PAD_INCHES = 2.6"), which works when a cell is 21" wide. Here a
# cell is 5.6" and the strip would eat a fifth of every panel - and it has to be
# applied to all four columns, not just the one that draws the axis, or the
# panels stop being the same width and stop being comparable. So this figure
# keeps the strip at zero and widens the FIGURE MARGIN instead: the "Hz" label
# and its tick numbers hang off column 0's left spine into the margin, outside
# every cell, and all four heatmaps fill their cells edge to edge.
LABEL_PAD_INCHES = 0.0

# Layered on top of CONTENDER_TIGHT_STYLE. Narrower cells want smaller axis
# chrome, and the margin has to hold the y-axis label plus its numbers.
RESOLUTION_STYLE = {
    "DEFAULT_TICK_LABEL_SIZE": 30,
    "DEFAULT_AXIS_LABEL_SIZE": 34,
    "DEFAULT_MARGIN_INCHES": 1.25,
    "DEFAULT_Y_AXIS_LABEL_INSET_INCHES": 0.95,
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.35,
}


@contextmanager
def _resolution_style():
    """CONTENDER_TIGHT_STYLE with this figure's narrower-cell adjustments."""
    with _contender_tight_style():
        orig = {k: getattr(style, k) for k in RESOLUTION_STYLE}
        try:
            for k, v in RESOLUTION_STYLE.items():
                setattr(style, k, v)
            yield
        finally:
            for k, v in orig.items():
                setattr(style, k, v)


def _build_signals() -> list[tuple[str, np.ndarray]]:
    """The loud-short / quiet-long pair on a common continuous low tone."""
    n = int(SR * BUILD_DURATION_S)
    t = np.arange(n) / SR
    low = np.sin(2.0 * np.pi * LOW_HZ * t)
    centre = int(round(BUILD_DURATION_S / 2 * SR))

    def event(duration_s: float, amp: float) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        m = int(round(duration_s * SR))
        i0 = centre - m // 2
        out[i0:i0 + m] = amp * np.sin(2.0 * np.pi * HIGH_HZ * (np.arange(m) / SR))
        return out

    return [
        ("loud_short", low + event(SHORT_EVENT_S, SHORT_EVENT_AMP)),
        ("quiet_long", low + event(LONG_EVENT_S, LONG_EVENT_AMP)),
    ]


def _stft_dense(signal: np.ndarray, log_freqs: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """STFT at a FIXED hop, resampled onto the CWT's log-spaced bin grid.

    Returns (column_times_s, magnitude). Local rather than the shared
    `_stft_on_log_bins` because that one ties the hop to the window length; here
    the window is the variable under study, so the hop has to be held still.
    """
    from scipy.signal import stft as scipy_stft
    nperseg = int(min(nperseg, len(signal)))
    noverlap = max(0, nperseg - STFT_HOP_SAMPLES)
    freqs, times, Zxx = scipy_stft(signal, fs=SR, nperseg=nperseg,
                                   noverlap=noverlap)
    mag = np.abs(Zxx)[1:]
    freqs = freqs[1:]
    out = np.empty((len(log_freqs), mag.shape[1]), dtype=np.float64)
    for j in range(mag.shape[1]):
        out[:, j] = np.interp(log_freqs, freqs, mag[:, j], left=0.0, right=0.0)
    return times, out


def _crop_and_resample(matrix: np.ndarray, t_start_s: float, t_end_s: float,
                       col_start_s: float, col_end_s: float) -> np.ndarray:
    """Crop a panel to the display window and force it onto DISPLAY_COLUMNS."""
    n_cols = matrix.shape[1]
    span = max(col_end_s - col_start_s, 1e-12)
    a = int(round((t_start_s - col_start_s) / span * n_cols))
    b = int(round((t_end_s - col_start_s) / span * n_cols))
    a = max(0, min(a, n_cols - 1))
    b = max(a + 1, min(b, n_cols))
    cropped = matrix[:, a:b]
    idx = np.linspace(0, cropped.shape[1] - 1, DISPLAY_COLUMNS)
    return cropped[:, np.rint(idx).astype(int)]


def _low_band_reference(matrix: np.ndarray, freqs: np.ndarray) -> float:
    """Peak level in a narrow band around the low tone - the per-row anchor."""
    band = (freqs > LOW_HZ * 0.85) & (freqs < LOW_HZ * 1.15)
    if not band.any():
        return float(matrix.max()) or 1.0
    return float(matrix[band].max()) or 1.0


def _prepare() -> dict:
    """Run every (n, signal) combination through both transforms."""
    signals = _build_signals()
    centre_s = BUILD_DURATION_S / 2
    t_start = centre_s - DISPLAY_DURATION_S / 2
    t_end = centre_s + DISPLAY_DURATION_S / 2

    panels: dict[tuple[int, str, str], np.ndarray] = {}
    freqs_out: np.ndarray | None = None

    for cycles in N_CYCLES:
        raw: dict[tuple[str, str], np.ndarray] = {}
        for name, sig in signals:
            cwt, freqs, s0, s1 = compute_full_cwt(
                sig, SR, root_note_hz=CWT_ROOT_HZ,
                num_octaves=CWT_NUM_OCTAVES, num_cycles=cycles,
            )
            freqs_out = freqs
            raw[(name, "cwt")] = _crop_and_resample(
                np.abs(cwt), t_start, t_end, s0 / SR, s1 / SR)

            times, mag = _stft_dense(sig, freqs, cycles * LOW_PERIOD_SAMPLES)
            raw[(name, "stft")] = _crop_and_resample(
                mag, t_start, t_end, float(times[0]), float(times[-1]))

        # One reference per transform per row, taken from the low tone - which
        # is identical in both signals, so the two columns of a pair stay
        # directly comparable.
        for kind in ("stft", "cwt"):
            ref = _low_band_reference(raw[(signals[0][0], kind)], freqs_out)
            for name, _sig in signals:
                panels[(cycles, name, kind)] = _to_db(
                    raw[(name, kind)], ref, DB_FLOOR)

    return {"freqs": freqs_out, "panels": panels}


def _slice_band(data: dict) -> dict:
    freqs = data["freqs"]
    lo, hi = DISPLAY_FREQ_LIM_HZ
    b0 = max(0, min(int(np.searchsorted(freqs, lo, "left")), len(freqs) - 1))
    b1 = max(b0 + 1, min(int(np.searchsorted(freqs, hi, "right")), len(freqs)))
    return {
        "freqs": freqs[b0:b1],
        "panels": {k: v[b0:b1, :] for k, v in data["panels"].items()},
    }


def _panel(matrix: np.ndarray, freqs: np.ndarray, *, show_yaxis: bool) -> HeatmapPanel:
    panel = HeatmapPanel(
        units=PLOT_UNITS,
        x_label=None,
        y_label="Hz" if show_yaxis else None,
        xticks=[],
        show_xticklabels=False,
        show_yticklabels=show_yaxis,
    )
    panel.add(Heatmap(
        matrix, duration_s=DISPLAY_DURATION_S, freqs=freqs, log_freq=True,
        tick_freqs=DISPLAY_FREQ_TICKS,
        extent=(0.0, DISPLAY_DURATION_S, 0.0, float(len(freqs))),
        vmin=DB_FLOOR, vmax=0.0,
    ))
    panel.show_yaxis = show_yaxis
    return panel


def _build_figure(*, data: dict | None = None, dpi: int = 150,
                  unit_inches: float | None = None,
                  unit_height_inches: float | None = None,
                  debug: bool = False) -> Figure:
    band = _slice_band(data if data is not None else _prepare())
    freqs = band["freqs"]
    panels = band["panels"]

    rows: list[list] = [[SuptitlePanel(FIGURE_TITLE,
                                       units=(TOTAL_WIDTH_UNITS, 1), font_size=44)]]
    row_heights: list[float] = [STACK_BAND_HEIGHT]

    # Column headers: one band cell per plot column, plus a blank over the
    # caption column so the row's widths sum to the figure width.
    rows.append([SuptitlePanel(t, units=PLOT_UNITS, font_size=26)
                 for t in COLUMN_TITLES]
                + [SuptitlePanel("", units=(2, 1), font_size=26)])
    row_heights.append(HEADER_ROW_HEIGHT)

    order = (("loud_short", "stft"), ("loud_short", "cwt"),
             ("quiet_long", "stft"), ("quiet_long", "cwt"))
    for cycles in N_CYCLES:
        row = []
        for i, (name, kind) in enumerate(order):
            # Only the leftmost panel carries the Hz axis; the four share one
            # frequency scale and repeating it four times is noise.
            p = _panel(panels[(cycles, name, kind)], freqs, show_yaxis=(i == 0))
            p.content_left_pad_inches = LABEL_PAD_INCHES
            p.fill_cell_vertical = True
            p.fill_cell_pad_inches = STACK_CELL_VPAD_INCHES
            row.append(p)
        rows.append(row + [_side_text_panel(
            f"n = {cycles}", ROW_CAPTIONS[cycles], font_size=CAPTION_FONT_SIZE)])
        row_heights.append(STACK_PLOT_ROW_HEIGHT)

    u_w = unit_inches if unit_inches is not None else _unit_inches()
    u_h = unit_height_inches if unit_height_inches is not None else _unit_inches()
    return Figure.compose(
        rows=rows,
        row_heights=row_heights,
        hspace=STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT),
        wspace=STACK_GUTTER_INCHES / u_w,
        dpi=dpi,
        unit_inches=u_w,
        unit_height_inches=u_h,
        show_cell_borders=True,
        debug_guides=debug,
    )


CANVAS_WIDTH_INCHES = 28.0
# Everything the grid columns do NOT get: two margins plus the four column
# gutters. The gutters are passed to compose() as `wspace = gutter / unit`, i.e.
# a fraction of the axes width, so their inch total holds steady as the unit
# changes - which makes the overhead a constant and the width solvable in one
# step. Measured from a render; recheck if the margin or gutter constants move.
LAYOUT_OVERHEAD_INCHES = 5.2


def _unit_inches() -> float:
    """One square, sized so the figure lands on the house 28in canvas width.

    The margin and the column gutters are part of that width, so
    LAYOUT_OVERHEAD_INCHES comes off the top before the remainder is divided
    among the columns - otherwise widening the margin to seat the y-axis chrome
    would push the canvas past the house size.
    """
    return (CANVAS_WIDTH_INCHES - LAYOUT_OVERHEAD_INCHES) / TOTAL_WIDTH_UNITS


def render(output_dir: str = "assets/images/dsp/figures/by_figure/fig_3_4_resolution_budget",
           output_filename: str = "fig_3_4_resolution_budget_v1.png") -> str:
    """Build, render, and save at production DPI. Returns the absolute path."""
    with _resolution_style():
        fig = _build_figure()
        fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show(debug: bool = False) -> Figure:
    """Notebook-tuned render for inline display in dsp.ipynb."""
    with _resolution_style():
        fig = _build_figure(dpi=60, unit_inches=1.3, unit_height_inches=1.3,
                            debug=debug)
        fig.render()
    return fig
