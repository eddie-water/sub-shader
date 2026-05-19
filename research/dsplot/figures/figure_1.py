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

from dsplot import (
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    TimeSeries,
    TimeSeriesPanel,
    style,
)
from utilities import compute_full_cwt
from utilities.dsp_helpers import build_waypoint_chirp


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

    # For each log_freq, interpolate magnitude along the freq axis for every
    # time bin. Loop avoids the memory hit of a full 2D interp grid.
    mag_log = np.empty((len(log_freqs), mag_lin.shape[1]), dtype=mag_lin.dtype)
    for j in range(mag_lin.shape[1]):
        mag_log[:, j] = np.interp(log_freqs, f_lin, mag_lin[:, j],
                                  left=0.0, right=0.0)
    return mag_log


def build_figure() -> Figure:
    data = _prepare()
    duration_s = data["duration_s"]
    cwt_freqs = data["cwt_freqs"]
    stft_mag_log = _stft_on_log_bins(data["signal"], SR, cwt_freqs)

    # Inst-freq overlay maps Hz → CWT bin position so the row-1 twin y-axis
    # rhymes with the row-2/row-3 spectrograms (same 200/2k/20k tick labels at
    # the same bin offsets). Mirrors legacy motivator.py::section1 lines 383-402.
    log_f0 = float(np.log2(cwt_freqs[0]))
    log_step = float(np.log2(cwt_freqs[1] / cwt_freqs[0]))
    freq_to_bin = lambda f: (np.log2(f) - log_f0) / log_step

    t_axis = np.arange(len(data["signal"])) / SR
    inst_freq_bins = np.interp(
        data["inst_freq"], cwt_freqs, np.arange(len(cwt_freqs))
    )

    twin_ytick_positions = [float(freq_to_bin(f)) for f in DISPLAY_FREQ_TICKS]
    twin_ytick_labels = [
        f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}'
        for f in DISPLAY_FREQ_TICKS
    ]
    twin_ylim = (0.0, float(len(cwt_freqs)))

    # Row 1 — chirp time-series + inst-freq overlay on a twin y-axis.
    # Twin y lives on the LEFT (visual rhyme with rows 2 & 3 below)
    # and carries the same 200/2k/20k bin-position labels as the
    # spectrogram panels, so the inst-freq curve reads as the same
    # "y = frequency" story across all three rows.
    row1 = TimeSeriesPanel(
        title="Time Series",
        units=(3, 1),
        x_label=None,
        xticks=XTICKS,
        twin_y=True,
        twin_y_label="f (Hz)",
        twin_y_side="left",
        twin_yticks=twin_ytick_positions,
        twin_ytick_labels=twin_ytick_labels,
        twin_ylim=twin_ylim,
    )
    row1.add(TimeSeries(data["signal"], SR, color=style.TICK_LABEL_COLOR))
    row1.add_twin(Line(
        t_axis,
        inst_freq_bins,
        color=style.INST_FREQ_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH,
        alpha=style.INST_FREQ_ALPHA,
    ))

    # Spectrogram extent — explicit, because HeatmapPanel.render() only
    # overrides StaticPanel's (-1.25, 1.25) default xlim when extent is set.
    spec_extent = (0.0, duration_s, 0.0, float(len(cwt_freqs)))

    # Row 2 — STFT magnitude on the shared log-spaced bin grid.
    row2 = HeatmapPanel(
        title="Fourier (STFT) Analysis",
        units=(3, 1),
        x_label=None,
        y_label="f (Hz)",
        xticks=XTICKS,
    )
    row2.add(Heatmap(
        stft_mag_log,
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=DISPLAY_FREQ_TICKS,
        extent=spec_extent,
    ))
    row2.add(Line(
        t_axis,
        inst_freq_bins,
        color=style.INST_FREQ_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH,
        alpha=style.INST_FREQ_ALPHA,
    ))

    # Row 3 — CWT magnitude.
    row3 = HeatmapPanel(
        title="Wavelet Analysis",
        units=(3, 1),
        x_label="t (s)",
        y_label="f (Hz)",
        xticks=XTICKS,
    )
    row3.add(Heatmap(
        data["cwt_data"],
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=DISPLAY_FREQ_TICKS,
        extent=spec_extent,
    ))
    row3.add(Line(
        t_axis,
        inst_freq_bins,
        color=style.INST_FREQ_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH,
        alpha=style.INST_FREQ_ALPHA,
    ))

    return Figure.compose(
        rows=[[row1], [row2], [row3]],
        suptitle="Fourier vs Wavelet Analysis",
    )


def render(output_dir: str = "assets/images/generated",
           output_filename: str = "figure_1_fourier_vs_wavelet.png") -> str:
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
