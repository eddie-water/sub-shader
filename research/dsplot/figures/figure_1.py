"""Figure 1 — Fourier vs Wavelet Analysis motivator (dsplot rebuild).

Same content as the old `motivator.py::section1`, rebuilt with the dsplot
Panel/Plottable framework. Three vertically stacked panels share a 0–2 s
time axis:

  Row 1 — chirp time-series (TimeSeriesPanel + TimeSeries)
  Row 2 — STFT magnitude (HeatmapPanel + Heatmap, log-spaced bins)
  Row 3 — CWT magnitude  (HeatmapPanel + Heatmap, log-spaced bins)

Data prep mirrors the old motivator: 2.36 s chirp spline → compute_full_cwt
trims the chunk-boundary edge-effect regions (~175 ms each side) →
zero-crossing snap removes the boundary slab in the time-series panel →
~2.0 s visible window with a clean final x-tick at 2.0 s.

STFT is computed on linear-spaced scipy bins and then resampled onto the
CWT's log-spaced frequency grid so both spectrogram panels share the same
y-axis scale (and `Heatmap(log_freq=True, tick_freqs=...)` places ticks
identically on both).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import (
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    TextPanel,
    TimeSeries,
    TimeSeriesPanel,
    style,
)
from utilities import compute_full_cwt
from utilities.dsp_helpers import build_waypoint_chirp, build_click_plus_tone, build_low_vibrato


# ============================================================
# Chirp design (mirrors motivator.py::section1 — DO NOT CHANGE without
# re-tuning duration_s / trim accounting; see comments below).
# ============================================================
SR = 44100
DURATION_S = 2.36
WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.00, 50.0),
    (0.05, 22.0),      # sub-trim — bleeds 22 Hz CWT energy into the left edge
    (0.18, 800.0),     # peak 1
    (0.32, 60.0),      # deep dip — the only intentional STFT-smearing trough
    (0.55, 12000.0),   # peak 2
    (0.75, 2000.0),    # modest trough (above STFT smear threshold)
    (1.00, 25000.0),   # tail-trim — clean off-screen ascent
)
DISPLAY_FREQ_LIM_HZ = (20.0, 21500.0)
DISPLAY_FREQ_TICKS = (200, 2000, 20000)
XTICKS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Row labels live in a TextPanel column on the left of each row. Newlines are
# explicit so the wrap point is intentional (matplotlib's auto-wrap inside an
# axes box is unreliable).
ROW1_LABEL = "Audio Signal\n(Chirp Sweep + Clicks)"
ROW2_LABEL = "Fourier Analysis\n(STFT)"
ROW3_LABEL = "Wavelet Analysis\n(CWT)"
# Square 1-unit label cells — TextPanel auto-shrinks LABEL_FONT_SIZE to fit
# the longest label string.
LABEL_PANEL_UNITS = (1, 1)
LABEL_FONT_SIZE = 18                # uniform across all row labels (auto_shrink off)
# Local row-gutter override (template default = 2.0"). Plots get more
# vertical breathing room so x/y tick labels and axis-labels don't crowd
# adjacent panels.
ROW_GUTTER_INCHES = 1.0
# Suptitle gets a full 1.4" top reserve (matches earlier passes).
TOP_RESERVE_INCHES = 1.4
# Bottom reserve hosts the figure_number + figure_caption band. 2.5" gives
# breathing room: x-label is at ~(reserve - 0.8) from figure bottom, then
# `Figure 1` identifier well below that, then 3-line caption below the
# identifier — all visually separated.
BOTTOM_RESERVE_INCHES = 2.5
# Caption is multi-line so the long text fits within the figure's grid
# width at the figure_caption font size. Splits chosen at natural breaks.
FIGURE_CAPTION_PLACEHOLDER = (
    "A chirp that sweeps from _ down to _ back up to _ in 2 seconds — at the 1 s mark we inject\n"
    "5 broadband clicks spaced 20 ms apart — highlights the time resolution advantages of the\n"
    "wavelet analysis over fourier."
)

# ============================================================
# Hero — bouncy chirp (200 → 60 → 1.8k → 70 → 20k Hz) + broadband click
# at midpoint. Same STFT/CWT analysis as v8 — only the signal varies.
#
# Story: starts at 200 Hz where STFT resolves fine, dips to 60/70 Hz where
# STFT smears but CWT tracks the contour, peaks at 1.8 kHz, ends smooth at
# 20 kHz. A 5 ms broadband click at t=0.5·duration adds an abrupt transient
# that STFT averages away (window ≫ click) and CWT shows as a vertical
# streak — showcases contour tracking AND transient localization in one
# figure.
# ============================================================
HERO_DURATION_S = 2.36  # build duration; ~2.0 visible after v8-style CWT trim + ZC snap
HERO_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.00, 5500.0),     # boundary: trim brings visible-start down to ~1.9 kHz
    (0.55, 180.0),      # mid-signal dip — spline undershoots to ~98 Hz visible (32 ms CWT lag)
    (0.85, 13000.0),    # intermediate ascent anchor — controls climb rate so the visible right edge actually reaches ~20 kHz
    (1.00, 27000.0),    # tail-trim: overshoot display ceiling, the steepest spline portion gets eaten by CWT trim, leaving visible right edge at ~21 kHz
)
HERO_CLICK_T_FRAC = 0.525   # places cluster centre near the 1.0 s tick (trim offset ≈ 0.233 s)
HERO_CLICK_DURATION_S = 0.008      # 8 ms FWHM per click — narrow envelope keeps each burst's spectrum flat
HERO_CLICK_AMP = 7.0               # well above chirp peak so each click reads bright in CWT
HERO_CHIRP_AMP = 2.0               # multiplier applied to chirp post-build; click normalization uses the pre-amp peak so the click absolute level is unchanged when this knob is raised
HERO_CLICK_COUNT = 5               # N broadband bursts in the cluster — STFT smears into one blob, CWT resolves each
HERO_CLICK_SPACING_S = 0.020       # 20 ms between cluster members
HERO_MIRROR_PAD_S = 0.20           # symmetric mirror pad applied to BOTH STFT & CWT input
HERO_PANEL_UNITS = (4, 1)          # wider panels → low-f oscillations remain visible
HERO_DISPLAY_FREQ_LIM_HZ: tuple[float, float] = (20.0, 21500.0)
HERO_DISPLAY_FREQ_TICKS: tuple[int, ...] = (20, 200, 2000, 20000)

# ============================================================
# Anti-hero — on hold; see project_figure_1.md memory for design state.
# ============================================================
ANTIHERO_DURATION_S = 1.5
ANTIHERO_CARRIER_HZ = 60.0
ANTIHERO_DEPTH_HZ = 20.0
ANTIHERO_MOD_HZ = 8.0


def _prepare() -> dict:
    """Build chirp, run CWT, trim + zero-crossing snap. Returns a bundle."""
    signal, inst_freq, t = build_waypoint_chirp(
        SR, DURATION_S, WAYPOINTS, clip_to_waypoints=False
    )
    f_lo = min(f for _, f in WAYPOINTS)
    disp_lo, disp_hi = DISPLAY_FREQ_LIM_HZ

    cwt_root_hz = min(f_lo, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    # Trim + re-zero — discards CWT edge-effect regions; all three panels
    # share x ∈ [0, trimmed_duration] with no edge gaps.
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    # Zero-crossing snap — removes the boundary slab that fill_between
    # otherwise draws at t=0 (because trim leaves signal[0] at arbitrary
    # phase). Shift is ≤ half a cycle, imperceptible against the 2.0 s panel.
    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _stft_on_log_bins(signal: np.ndarray, sr: int, log_freqs: np.ndarray) -> np.ndarray:
    """Compute STFT magnitude and resample onto the supplied log-spaced bin grid.

    Lets the STFT spectrogram render through `Heatmap(log_freq=True)` with
    the SAME y-axis as the CWT panel (otherwise scipy's linear-spaced bins
    would force a log-yscale dance that imshow can't do cleanly).

    Uses scipy.signal.stft's default windowing/boundary behaviour — apples-
    to-apples with the v8 chirp path so any STFT-vs-CWT comparison rests on
    identical analysis settings; only the signal varies.
    """
    from scipy.signal import stft as scipy_stft
    n = len(signal)
    nperseg = min(1024, max(64, 1 << int(math.log2(max(64, n // 4)))))
    noverlap = nperseg // 2
    f_lin, _t, Zxx = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag_lin = np.abs(Zxx)
    # Drop DC bin so log interp is well-defined.
    f_lin = f_lin[1:]
    mag_lin = mag_lin[1:]

    mag_log = np.empty((len(log_freqs), mag_lin.shape[1]), dtype=mag_lin.dtype)
    for j in range(mag_lin.shape[1]):
        mag_log[:, j] = np.interp(log_freqs, f_lin, mag_lin[:, j],
                                  left=0.0, right=0.0)
    return mag_log


def _auto_xticks(duration_s: float) -> list[float]:
    """Generate 0.0, 0.5, 1.0, … xticks up to duration_s."""
    return [round(x, 1) for x in np.arange(0.0, duration_s + 0.01, 0.5).tolist()
            if x <= duration_s + 0.01]


def _mirror_pad_for_analysis(signal: np.ndarray, pad_samples: int) -> np.ndarray:
    """Symmetrically reflect both ends of `signal` to provide out-of-window context.

    Applied identically to both STFT and CWT inputs so edge artifacts and the
    CWT's intrinsic low-f time-support delay are pushed outside the visible
    window without introducing asymmetric treatment between the two transforms.
    Uses numpy reflect mode (value at boundary repeats so reflection is smooth).
    """
    if pad_samples <= 0 or len(signal) == 0:
        return signal
    pad_samples = min(pad_samples, len(signal) - 1)
    return np.pad(signal, pad_samples, mode="reflect")


def _stft_visible_window(signal: np.ndarray, sr: int, log_freqs: np.ndarray,
                         t_start_s: float, t_end_s: float) -> np.ndarray:
    """Compute STFT on `signal` (the full padded buffer) and slice time bins to
    the visible window [t_start_s, t_end_s] before resampling onto `log_freqs`.

    Same STFT settings as `_stft_on_log_bins` — apples-to-apples with the
    visible-signal STFT path; only the analysed buffer is wider.
    """
    from scipy.signal import stft as scipy_stft
    n = len(signal)
    nperseg = min(1024, max(64, 1 << int(math.log2(max(64, n // 4)))))
    noverlap = nperseg // 2
    f_lin, t_stft, Zxx = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag_lin = np.abs(Zxx)
    f_lin = f_lin[1:]
    mag_lin = mag_lin[1:]

    bin_lo = int(np.searchsorted(t_stft, t_start_s, side="left"))
    bin_hi = int(np.searchsorted(t_stft, t_end_s, side="right"))
    bin_lo = max(0, min(bin_lo, mag_lin.shape[1] - 1))
    bin_hi = max(bin_lo + 1, min(bin_hi, mag_lin.shape[1]))
    mag_lin = mag_lin[:, bin_lo:bin_hi]

    mag_log = np.empty((len(log_freqs), mag_lin.shape[1]), dtype=mag_lin.dtype)
    for j in range(mag_lin.shape[1]):
        mag_log[:, j] = np.interp(log_freqs, f_lin, mag_lin[:, j],
                                  left=0.0, right=0.0)
    return mag_log


def _build_3row_figure(
    data: dict,
    *,
    xticks: list[float],
    suptitle: str | None = None,
    figure_number: str | None = None,
    figure_caption: str | None = None,
    display_freq_lim_hz: tuple[float, float] | None = None,
    display_freq_ticks: tuple[int, ...] | None = None,
    panel_units: tuple[int, int] = (3, 1),
    dpi: int | None = None,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
) -> Figure:
    """Shared 3-row Fourier-vs-Wavelet layout used by v8 chirp + hero + anti-hero.

    Renders the time-series + STFT + CWT story with consistent visual chrome:
    orange inst-freq overlay on all 3 rows, log-spaced frequency grid via
    `_stft_on_log_bins`, twin-axis Hz labels on row 1 that rhyme with the
    spectrogram bin positions on rows 2 and 3.

    STFT/CWT analysis settings are held identical across callers (see
    `_stft_on_log_bins` and the v8 `compute_full_cwt` defaults) so figures
    differ only by their input signal — apples-to-apples comparison.

    If ``data["stft_mag_log"]`` is present (caller pre-computed STFT, e.g.
    for mirror-padded analysis), it is used directly. Otherwise STFT is
    computed here from the visible signal.

    ``display_freq_lim_hz`` (when provided) slices the CWT/STFT data and
    freq grid to the visible band so log-spaced panels fill instead of
    leaving empty rows. ``panel_units`` controls the panel aspect ratio.
    """
    duration_s = data["duration_s"]
    cwt_freqs = data["cwt_freqs"]
    cwt_data = data["cwt_data"]
    if "stft_mag_log" in data:
        stft_mag_log = data["stft_mag_log"]
    else:
        stft_mag_log = _stft_on_log_bins(data["signal"], SR, cwt_freqs)

    if display_freq_ticks is None:
        display_freq_ticks = DISPLAY_FREQ_TICKS

    if display_freq_lim_hz is not None:
        disp_lo, disp_hi = display_freq_lim_hz
        bin_lo = int(np.searchsorted(cwt_freqs, disp_lo, side="left"))
        bin_hi = int(np.searchsorted(cwt_freqs, disp_hi, side="right"))
        bin_lo = max(0, min(bin_lo, len(cwt_freqs) - 1))
        bin_hi = max(bin_lo + 1, min(bin_hi, len(cwt_freqs)))
        cwt_freqs = cwt_freqs[bin_lo:bin_hi]
        cwt_data = cwt_data[bin_lo:bin_hi, :]
        stft_mag_log = stft_mag_log[bin_lo:bin_hi, :]

    log_f0 = float(np.log2(cwt_freqs[0]))
    log_step = float(np.log2(cwt_freqs[1] / cwt_freqs[0]))
    freq_to_bin = lambda f: (np.log2(f) - log_f0) / log_step

    t_axis = np.arange(len(data["signal"])) / SR
    inst_freq_bins = np.interp(
        data["inst_freq"], cwt_freqs, np.arange(len(cwt_freqs))
    )

    twin_ytick_positions = [float(freq_to_bin(f)) for f in display_freq_ticks]
    twin_ytick_labels = [
        f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}'
        for f in display_freq_ticks
    ]
    twin_ylim = (0.0, float(len(cwt_freqs)))

    # HeatmapPanel.render() only overrides StaticPanel's (-1.25, 1.25)
    # default xlim when extent is set explicitly.
    spec_extent = (0.0, duration_s, 0.0, float(len(cwt_freqs)))

    row1 = TimeSeriesPanel(
        units=panel_units,
        x_label=None,
        xticks=xticks,
        twin_y=True,
        twin_y_label="Hz",
        twin_y_side="left",
        twin_yticks=twin_ytick_positions,
        twin_ytick_labels=twin_ytick_labels,
        twin_ylim=twin_ylim,
    )
    row1.add(TimeSeries(data["signal"], SR, color=style.TICK_LABEL_COLOR))
    row1.add_twin(Line(
        t_axis,
        inst_freq_bins,
        color=style.BG_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 5.5,
        alpha=1.0,
    ))
    row1.add_twin(Line(
        t_axis,
        inst_freq_bins,
        color=style.INST_FREQ_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 3.0,
        alpha=style.INST_FREQ_ALPHA,
    ))

    row2 = HeatmapPanel(
        units=panel_units,
        x_label=None,
        y_label="Hz",
        xticks=xticks,
    )
    row2.add(Heatmap(
        stft_mag_log,
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=display_freq_ticks,
        extent=spec_extent,
    ))

    row3 = HeatmapPanel(
        units=panel_units,
        x_label="s",
        y_label="Hz",
        xticks=xticks,
    )
    row3.add(Heatmap(
        cwt_data,
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=display_freq_ticks,
        extent=spec_extent,
    ))

    # Uniform fixed font size across all row labels — TextPanel.auto_shrink
    # would otherwise scale each panel's text independently (longer text
    # shrinks more), producing visibly different sizes per row. With
    # auto_shrink=False every row renders at the same nominal LABEL_FONT_SIZE.
    label_kwargs = dict(
        units=LABEL_PANEL_UNITS,
        font_size=LABEL_FONT_SIZE,
        auto_shrink=False,
    )
    label1 = TextPanel(ROW1_LABEL, **label_kwargs)
    label2 = TextPanel(ROW2_LABEL, **label_kwargs)
    label3 = TextPanel(ROW3_LABEL, **label_kwargs)

    # Resolve the hspace override against the unit HEIGHT we're about to use —
    # ROW_GUTTER_INCHES is in absolute inches and gridspec hspace is a
    # fraction of unit_height.
    effective_unit_height_inches = (
        unit_height_inches if unit_height_inches is not None
        else (unit_inches if unit_inches is not None
              else style.DEFAULT_PANEL_UNIT_INCHES)
    )
    hspace_override = ROW_GUTTER_INCHES / effective_unit_height_inches
    return Figure.compose(
        rows=[[label1, row1], [label2, row2], [label3, row3]],
        suptitle=suptitle,
        figure_number=figure_number,
        figure_caption=figure_caption,
        hspace=hspace_override,
        dpi=dpi,
        unit_inches=unit_inches,
        unit_height_inches=unit_height_inches,
        top_reserve_inches=TOP_RESERVE_INCHES,
        bottom_reserve_inches=BOTTOM_RESERVE_INCHES,
        # Same lego-kitchen-sink chrome as figure 2.4.1 — subtle gray rect
        # around each cell makes the layout structure visually explicit.
        show_cell_borders=True,
    )


def build_figure() -> Figure:
    return _build_3row_figure(
        _prepare(),
        xticks=XTICKS,
        suptitle="Fourier vs Wavelet Analysis",
    )


def render(output_dir: str = "assets/images/dsp/figures/figure_1",
           output_filename: str = "fourier_vs_wavelet.png") -> str:
    """Build, render, save. Returns absolute output path."""
    fig = build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show() -> Figure:
    """Build, render, and display in a notebook cell."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig


# ============================================================
# Hero + Anti-hero data prep and render functions.
# These extend figure_1.py without touching the locked v8 pipeline above.
# ============================================================

def _prepare_hero() -> dict:
    """Hero pipeline: bouncy chirp + broadband click. Mirrors v8 `_prepare()`
    pattern — only the signal differs.

    1. Build chirp from HERO_WAYPOINTS (200 → 60 → 1.8k → 70 → 20k Hz) over
       HERO_DURATION_S (2.36 s build → ~2.0 s visible after trim + ZC snap).
    2. Add a Gaussian-windowed broadband white-noise click at midpoint.
    3. CWT via compute_full_cwt; trim + zero-crossing snap (same as v8).
       STFT is computed downstream in `_build_3row_figure` on the same
       visible signal — apples-to-apples with v8 and with whatever future
       anti-hero we land on.
    """
    chirp_signal, inst_freq, t_chirp = build_waypoint_chirp(
        SR, HERO_DURATION_S, HERO_WAYPOINTS, clip_to_waypoints=False
    )

    n_build = len(chirp_signal)
    chirp_peak = float(np.max(np.abs(chirp_signal)))
    sigma = HERO_CLICK_DURATION_S / 2.3548200450309493  # FWHM → sigma
    cluster_center_t = HERO_CLICK_T_FRAC * HERO_DURATION_S
    cluster_half_span = (HERO_CLICK_COUNT - 1) * HERO_CLICK_SPACING_S / 2.0

    click_burst = np.zeros(n_build, dtype=np.float64)
    for i in range(HERO_CLICK_COUNT):
        click_t = cluster_center_t - cluster_half_span + i * HERO_CLICK_SPACING_S
        rng = np.random.default_rng(seed=i)
        noise = rng.standard_normal(n_build).astype(np.float64)
        envelope = np.exp(-((t_chirp - click_t) ** 2) / (2.0 * sigma ** 2))
        single = noise * envelope
        peak = float(np.max(np.abs(single)))
        if peak > 0.0:
            single = single * (HERO_CLICK_AMP * chirp_peak / peak)
        click_burst = click_burst + single
    signal = chirp_signal * HERO_CHIRP_AMP + click_burst

    disp_lo, disp_hi = HERO_DISPLAY_FREQ_LIM_HZ
    f_lo = min(f for _, f in HERO_WAYPOINTS)
    cwt_root_hz = min(f_lo, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    # v8 pattern: slice signal/inst_freq by [start:end] (sample-rate indices);
    # cwt_data is already trimmed to its reliable region by compute_full_cwt.
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t_chirp) > start_sample:
        t = t_chirp[start_sample:end_sample] - t_chirp[start_sample]
    else:
        t = t_chirp[:0]

    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            # v8 slices cwt_data by the same ZC indices; upper bound clips at
            # the hop-rate column count, which is the intended behaviour.
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _prepare_antihero() -> dict:
    """Build anti-hero signal (60 Hz ±20 Hz vibrato at 8 Hz), run CWT, trim + snap.

    Mirrors the structure of `_prepare()`. Returns the same bundle dict.
    CWT bank covers the full 20-21500 Hz range (same as v8/hero).
    Tighter display narrowing is a v2 task — see TODO below.
    """
    signal, inst_freq, t = build_low_vibrato(
        SR, ANTIHERO_DURATION_S,
        carrier_hz=ANTIHERO_CARRIER_HZ,
        depth_hz=ANTIHERO_DEPTH_HZ,
        mod_hz=ANTIHERO_MOD_HZ,
    )

    disp_lo, disp_hi = DISPLAY_FREQ_LIM_HZ
    cwt_root_hz = min(ANTIHERO_CARRIER_HZ - ANTIHERO_DEPTH_HZ, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)

    from research.utilities import compute_full_cwt
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _build_hero_figure(
    *,
    dpi: int = 250,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
) -> Figure:
    data = _prepare_hero()
    return _build_3row_figure(
        data,
        xticks=_auto_xticks(data["duration_s"]),
        suptitle="Fourier vs Wavelet Decomposition",
        figure_number="Figure 1",
        figure_caption=FIGURE_CAPTION_PLACEHOLDER,
        display_freq_lim_hz=HERO_DISPLAY_FREQ_LIM_HZ,
        display_freq_ticks=HERO_DISPLAY_FREQ_TICKS,
        panel_units=HERO_PANEL_UNITS,
        dpi=dpi,
        unit_inches=unit_inches,
        unit_height_inches=unit_height_inches,
    )


def _build_antihero_figure() -> Figure:
    # On hold per user direction — defaults match v8 (apples-to-apples).
    data = _prepare_antihero()
    return _build_3row_figure(
        data,
        xticks=_auto_xticks(data["duration_s"]),
        suptitle="Anti-hero — TBD",
    )


def render_hero(
    output_dir: str = "assets/images/dsp/figures/figure_1",
    output_filename: str = "hero_click_plus_tone_v1.png",
) -> str:
    """Build, render, and save the hero figure. Returns absolute output path."""
    fig = _build_hero_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def render_antihero(
    output_dir: str = "assets/images/dsp/figures/figure_1",
    output_filename: str = "antihero_low_vibrato_v1.png",
) -> str:
    """Build, render, and save the anti-hero figure. Returns absolute output path."""
    fig = _build_antihero_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def render_all() -> tuple[str, str, str]:
    """Render the v8 chirp + hero + anti-hero into assets/images/dsp/figures/figure_1/."""
    chirp_path = render()
    hero_path = render_hero()
    antihero_path = render_antihero()
    return chirp_path, hero_path, antihero_path


def show_hero() -> Figure:
    """Build, render, and display the hero figure in a notebook cell.

    Notebook-tuned size: ``unit_inches=2.5`` (cell width), ``unit_height_inches
    =3.75`` (1.5× taller cells), ``dpi=60``. Keeps the figure fitted to a
    typical Jupyter cell while giving each panel room for non-crowded x/y
    tick + axis labels. The static PNG path via ``render_hero()`` keeps the
    higher DPI for print/web.
    """
    import matplotlib.pyplot as plt
    fig = _build_hero_figure(dpi=60, unit_inches=2.5, unit_height_inches=3.75)
    fig.render()
    # Suppress the ipympl widget chrome ("Figure 1" / toolbar) — that header
    # is matplotlib canvas chrome, distinct from the figure-bottom
    # `Figure 1` identifier we render via figure_number.
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
