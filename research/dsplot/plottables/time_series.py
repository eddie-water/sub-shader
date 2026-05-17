"""TimeSeries Plottable — 1D signal vs time via fill_between.

A bare TimeSeries renders in the NEUTRAL palette slot because the dsplot
palette reserves PRIMARY (vector a) and SECONDARY (vector b) for vector
identity. A figure that wants its TimeSeries branded (e.g. a chirp tinted
orange to match a downstream Vector) passes `color=` explicitly.

None-valued color resolves against `dsplot.style.NEUTRAL_COLOR` at draw()
time (lazy lookup per D-05), so runtime style reassignment between
construction and draw is observable.
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .base import Plottable


class TimeSeries(Plottable):
    """1D signal plotted vs time using fill_between.

    `signal` is rendered against a time axis built from `sample_rate`. The
    y-axis is auto-padded around the signal's peak amplitude by
    `ylim_padding`. Panel background and tick chrome inherit from
    `dsplot.style` so a TimeSeries always looks consistent across figures.

    The default color is `style.NEUTRAL_COLOR` (the palette's non-identity
    slot) — a bare TimeSeries is generic data display, not a vector identity.
    Pass `color=` to brand a specific timeseries.
    """

    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        *,
        color: str | None = None,
        alpha: float = 0.75,
        ylim_padding: float = 1.1,
        zorder: int = 2,
    ) -> None:
        super().__init__(
            color=color,
            linewidth=None,
            alpha=alpha,
            linestyle="-",
            label=None,
            zorder=zorder,
        )
        self.signal = np.asarray(signal)
        self.sample_rate = float(sample_rate)
        self.ylim_padding = float(ylim_padding)

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.NEUTRAL_COLOR

        t = np.arange(len(self.signal)) / self.sample_rate
        peak = max(float(np.abs(self.signal).max()), 1e-6) if len(self.signal) else 1e-6

        ax.fill_between(
            t,
            self.signal,
            0,
            color=color,
            alpha=self.alpha,
            zorder=self.zorder,
        )
        ax.set_xlim(0, t[-1] if len(t) > 0 else 0)
        ax.set_ylim(-peak * self.ylim_padding, peak * self.ylim_padding)
        ax.set_facecolor(style.BG_COLOR)
        ax.tick_params(
            labelsize=style.DEFAULT_TICK_LABEL_SIZE,
            colors=style.TICK_LABEL_COLOR,
        )
