"""Heatmap Plottable — 2D array rendered via imshow.

Drives the CWT / STFT spectrogram panel used by the motivator and alignment
diagnostic figures. When `log_freq=True` and `freqs` is provided, y-tick
labels are placed at the standard log-octave grid (20/200/2k/20k by default)
via `freq_axis.compute_freq_yticks`.

vmax resolution priority (highest first):
  1. explicit `vmax` kwarg
  2. `vmax_percentile` kwarg → np.percentile(data, vmax_percentile)
  3. style.DEFAULT_HEATMAP_VMAX_PERCENTILE → np.percentile(data, default)

cmap defaults to `style.DEFAULT_HEATMAP_CMAP` when unset (lazy lookup per
D-05 — runtime reassignment between construction and draw is observable).
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from ..freq_axis import compute_freq_yticks
from .base import Plottable


class Heatmap(Plottable):
    """2D array rendered via imshow.

    Default extent matches `Heatmap(cwt_data, duration_s, freqs)` — the
    image spans `[0, duration_s]` on x and `[0, len(freqs)]` on y so that
    log-frequency tick bins computed by `freq_axis.compute_freq_yticks`
    line up directly.

    When `log_freq=True` AND `freqs` is provided, `tick_freqs` are placed
    at the bin positions returned by `compute_freq_yticks(freqs, tick_freqs)`.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        duration_s: float | None = None,
        freqs: np.ndarray | None = None,
        log_freq: bool = False,
        tick_freqs: tuple[float, ...] = (20, 200, 2000, 20000),
        cmap: str | None = None,
        vmin: float = 0.0,
        vmax: float | None = None,
        vmax_percentile: float | None = None,
        alpha: float = 1.0,
        origin: str = "lower",
        aspect: str = "auto",
        extent: tuple[float, float, float, float] | None = None,
        zorder: int = 1,
    ) -> None:
        super().__init__(
            color=None,
            linewidth=None,
            alpha=alpha,
            linestyle="-",
            label=None,
            zorder=zorder,
        )
        self.data = np.asarray(data)
        self.duration_s = duration_s
        self.freqs = np.asarray(freqs) if freqs is not None else None
        self.log_freq = bool(log_freq)
        self.tick_freqs = tuple(tick_freqs)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = vmax
        self.vmax_percentile = vmax_percentile
        self.origin = origin
        self.aspect = aspect
        self.extent = extent

    def draw(self, ax: Axes) -> None:
        cmap = self.cmap if self.cmap is not None else style.DEFAULT_HEATMAP_CMAP

        if self.vmax is not None:
            vmax = float(self.vmax)
        else:
            percentile = (
                self.vmax_percentile
                if self.vmax_percentile is not None
                else style.DEFAULT_HEATMAP_VMAX_PERCENTILE
            )
            vmax = float(np.percentile(self.data, percentile))

        if self.extent is not None:
            extent = list(self.extent)
        elif self.duration_s is not None and self.freqs is not None:
            extent = [0.0, float(self.duration_s), 0.0, float(len(self.freqs))]
        else:
            h, w = self.data.shape
            extent = [0.0, float(w), 0.0, float(h)]

        ax.imshow(
            self.data,
            cmap=cmap,
            aspect=self.aspect,
            origin=self.origin,
            extent=extent,
            vmin=self.vmin,
            vmax=vmax,
            alpha=self.alpha,
            zorder=self.zorder,
        )

        if self.log_freq and self.freqs is not None:
            ytick_bins, ytick_labels = compute_freq_yticks(self.freqs, self.tick_freqs)
            ax.set_yticks(ytick_bins)
            ax.set_yticklabels(ytick_labels)

        ax.tick_params(
            labelsize=style.DEFAULT_TICK_LABEL_SIZE,
            colors=style.TICK_LABEL_COLOR,
        )
