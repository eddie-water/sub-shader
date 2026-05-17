"""Dropline Plottable — dashed perpendicular indicator between two points.

Used by the projection-reconstruction figure to close parallelograms and
by the alignment diagnostic for burst-time markers. All style knobs
(color / alpha / linewidth / linestyle) default to the dsplot dropline
constants and resolve lazily at draw() time per D-05.
"""
from __future__ import annotations

from matplotlib.axes import Axes

from .. import style
from .base import Plottable


class Dropline(Plottable):
    """Dashed perpendicular line from ``start`` to ``end``.

    All four style knobs (color / alpha / linewidth / linestyle) resolve
    against ``dsplot.style.DROPLINE_COLOR`` and ``DEFAULT_DROPLINE_*`` at
    draw() time when left as None.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str | None = None,
        alpha: float | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        zorder: int = 1,
    ) -> None:
        # NB: Plottable base has fixed defaults for alpha (1.0) and linestyle ("-"),
        # which we override here. We store the raw None on the instance and resolve
        # at draw() time instead — keeps the lazy-lookup contract per D-05.
        super().__init__(
            color=color,
            linewidth=linewidth,
            alpha=1.0,
            linestyle="-",
            label=None,
            zorder=zorder,
        )
        self.start = tuple(start)
        self.end = tuple(end)
        # Shadow the base attrs so draw() can detect None vs explicit override.
        self._raw_alpha = alpha
        self._raw_linestyle = linestyle

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.DROPLINE_COLOR
        alpha = (
            self._raw_alpha if self._raw_alpha is not None
            else style.DEFAULT_DROPLINE_ALPHA
        )
        linewidth = (
            self.linewidth if self.linewidth is not None
            else style.DEFAULT_DROPLINE_LINEWIDTH
        )
        linestyle = (
            self._raw_linestyle if self._raw_linestyle is not None
            else style.DEFAULT_DROPLINE_LINESTYLE
        )

        ax.plot(
            [self.start[0], self.end[0]],
            [self.start[1], self.end[1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=self.zorder,
        )
