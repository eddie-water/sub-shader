"""Log-frequency tick computation for CWT-style heatmaps.

Maps a sorted log-spaced frequency array to bin positions on a heatmap y-axis.
Pure numpy helper — no external project imports.

Label formatting: frequencies >= 1000 Hz render as "{khz}k" (e.g. "2k", "20k");
frequencies < 1000 Hz render as the integer Hz value (e.g. "20", "200").
"""
from __future__ import annotations

import numpy as np


def compute_freq_yticks(
    freqs: np.ndarray,
    tick_freqs: tuple[float, ...] = (20, 200, 2000, 20000),
) -> tuple[list[float], list[str]]:
    """Map a log-spaced frequency array to bin positions for a heatmap y-axis.

    Each tick frequency in `tick_freqs` is included in the output only if it
    lies within `[freqs[0], freqs[-1]]`. Bin positions are interpolated against
    `np.arange(len(freqs))` so they line up with the imshow extent
    `[0, duration_s, 0, len(freqs)]` used by `Heatmap`.

    Returns
    -------
    (ytick_bins, ytick_labels)
        ytick_bins  — float positions in bin space (suitable for ax.set_yticks)
        ytick_labels — string labels formatted "{khz}k" or "{int(hz)}"
    """
    freq_min, freq_max = freqs[0], freqs[-1]
    n = len(freqs)
    ytick_bins: list[float] = []
    ytick_labels: list[str] = []
    for f in tick_freqs:
        if freq_min <= f <= freq_max:
            ytick_bins.append(float(np.interp(f, freqs, np.arange(n))))
            ytick_labels.append(f"{f/1000:.0f}k" if f >= 1000 else f"{int(f)}")
    return ytick_bins, ytick_labels
