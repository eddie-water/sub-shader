"""Line Plottable — generic 1D line plot (x, y) for overlays.

Distinct from `TimeSeries`, which is a single-signal fill_between against a
sample-rate-derived time axis. `Line` takes explicit x and y arrays so it can
overlay any curve (e.g. an instantaneous-frequency track on a twin y-axis)
without baking in a time-domain assumption.

None-valued color resolves against `dsplot.style.NEUTRAL_COLOR` at draw() time;
None-valued linewidth resolves against `dsplot.style.DEFAULT_DROPLINE_LINEWIDTH`
(lazy lookup per D-05) so runtime style reassignment between construction and
draw is observable. Default zorder=3 so a Line drawn over a TimeSeries
(zorder=2 default) lands on top.
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .base import Plottable


class Line(Plottable):
    """1D line plot of y vs x onto a matplotlib Axes.

    The Line draws onto whatever Axes the owning Panel passes in — it does NOT
    set xlim / ylim / facecolor, so it composes cleanly onto a twin y-axis
    without stomping the primary axis's limits.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str | None = None,
        linewidth: float | None = None,
        alpha: float = 1.0,
        linestyle: str = "-",
        label: str | None = None,
        zorder: int = 3,
    ) -> None:
        super().__init__(
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.NEUTRAL_COLOR
        linewidth = (
            self.linewidth
            if self.linewidth is not None
            else style.DEFAULT_DROPLINE_LINEWIDTH
        )
        ax.plot(
            self.x,
            self.y,
            color=color,
            linewidth=linewidth,
            alpha=self.alpha,
            linestyle=self.linestyle,
            label=self.label,
            zorder=self.zorder,
        )
