"""Stem Plottable — discrete-time array view (vertical lines + markers).

Renders y[n] vs x[n] as a classical "stem" plot: a vertical line from a
baseline to each sample with a marker at the tip. Useful for showing the
discrete-sample interpretation of a continuous signal — pair a Stem panel
underneath a TimeSeries panel of the same source signal to make the
sampling step visually explicit.

None-valued color resolves against `dsplot.style.PRIMARY_COLOR` at draw()
time; None-valued linewidth resolves against
`dsplot.style.DEFAULT_VECTOR_LINEWIDTH` (lazy lookup per D-05).
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .base import Plottable


class Stem(Plottable):
    """Stem plot of y vs x. Each (x, y) becomes a vertical line from
    `baseline` to y with a marker at the tip.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str | None = None,
        linewidth: float | None = None,
        alpha: float = 1.0,
        baseline: float = 0.0,
        marker: str = "o",
        markersize: float = 6.0,
        label: str | None = None,
        zorder: int = 3,
    ) -> None:
        super().__init__(
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle="-",
            label=label,
            zorder=zorder,
        )
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.baseline = float(baseline)
        self.marker = marker
        self.markersize = float(markersize)

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.PRIMARY_COLOR
        linewidth = (
            self.linewidth
            if self.linewidth is not None
            else style.DEFAULT_VECTOR_LINEWIDTH
        )
        markerline, stemlines, baseline_artist = ax.stem(
            self.x, self.y,
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            bottom=self.baseline,
            label=self.label,
        )
        markerline.set_color(color)
        markerline.set_markerfacecolor(color)
        markerline.set_markeredgecolor(color)
        markerline.set_markersize(self.markersize)
        markerline.set_alpha(self.alpha)
        markerline.set_marker(self.marker)
        markerline.set_zorder(self.zorder + 1)

        stemlines.set_color(color)
        stemlines.set_linewidth(linewidth)
        stemlines.set_alpha(self.alpha)
        stemlines.set_zorder(self.zorder)

        # Force axis limits to fit the data — StaticPanel.render() sets a
        # default vector lim of ±4 which would clip the stems out of view
        # for typical signal amplitudes.
        if len(self.x) > 0:
            x_min, x_max = float(self.x.min()), float(self.x.max())
            ax.set_xlim(x_min, x_max)
        if len(self.y) > 0:
            y_lo = min(float(self.y.min()), self.baseline)
            y_hi = max(float(self.y.max()), self.baseline)
            span = max(y_hi - y_lo, 1e-6)
            ax.set_ylim(y_lo - span * 0.1, y_hi + span * 0.1)
