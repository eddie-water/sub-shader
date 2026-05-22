"""dsp_alignment_diagnostic.png renderer — exposes freq-dependent CWT shift.

Multi-tone burst signal (80/250/600 Hz Gaussian-windowed sines, all
centered at the same instant). A correctly-aligned CWT would show three
bright spots at the SAME displayed x; with the current `conv_tf[:, :input_n]`
trim each row is shifted right by half_width(f), so the spots fan out
— most-shifted at low freq, least at high freq. The figure stacks the
input time series above the CWT magnitude spectrogram with a shared
"true burst time" annotation line so the misalignment reads at a glance.

Per D-01, this figure module is consumer code and may import canonical
research utilities (compute_full_cwt, plot_time_series, plot_cwt_spectrogram).
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from .. import style
from utilities import (
    compute_full_cwt,
    create_grid_scaffold,
    plot_cwt_spectrogram,
    plot_time_series,
)


SR = 44100
DURATION_S = 0.4
BURST_T0 = 0.18
BURST_SIGMA = 0.012
BURST_FREQS = (80.0, 250.0, 600.0)
F_LO = 50.0
F_HI = 1000.0

# Cyan plays the "ground truth time" role here — high-contrast against the
# inferno spectrogram so the misalignment is unmistakable.
TRUE_BURST_COLOR = style.HIGHLIGHT_COLOR


def render(
    output_dir: str,
    output_filename: str = "dsp_alignment_diagnostic.png",
) -> str:
    """Render the CWT alignment-diagnostic figure. Returns absolute path."""
    n = int(SR * DURATION_S)
    t = np.arange(n) / SR

    signal = np.zeros(n, dtype=np.float64)
    for f in BURST_FREQS:
        envelope = np.exp(-((t - BURST_T0) / BURST_SIGMA) ** 2)
        signal += envelope * np.sin(2 * np.pi * f * t)
    signal /= np.abs(signal).max()

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR,
        root_note_hz=F_LO,
        num_octaves=max(1, math.ceil(math.log2(F_HI / F_LO))),
    )

    sig_disp = signal[start_sample:end_sample]
    duration_disp = (end_sample - start_sample) / SR
    burst_t_disp = BURST_T0 - start_sample / SR

    fig, axes = create_grid_scaffold(n_rows=2, n_cols=1, figsize=(20.0, 10.0))

    plot_time_series(axes[0][0], sig_disp, SR)
    axes[0][0].axvline(burst_t_disp, color=TRUE_BURST_COLOR, linewidth=1.5,
                       linestyle="--", alpha=0.85)

    plot_cwt_spectrogram(axes[1][0], cwt_data, duration_disp, cwt_freqs)
    axes[1][0].axvline(burst_t_disp, color=TRUE_BURST_COLOR, linewidth=1.5,
                       linestyle="--", alpha=0.85,
                       label=f"true burst time = {BURST_T0 * 1000:.0f} ms")
    for f in BURST_FREQS:
        bin_idx = float(np.interp(f, cwt_freqs, np.arange(len(cwt_freqs))))
        axes[1][0].axhline(bin_idx, color=TRUE_BURST_COLOR, linewidth=0.8,
                           linestyle=":", alpha=0.55)
    axes[1][0].legend(loc="upper right", facecolor=style.BG_COLOR,
                      edgecolor=style.SPINE_COLOR, labelcolor="white")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, facecolor=fig.get_facecolor(),
                dpi=style.DEFAULT_DPI,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return os.path.abspath(output_path)
