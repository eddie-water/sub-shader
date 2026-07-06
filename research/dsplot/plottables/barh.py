"""Barh Plottable — horizontal bars (grouped rows or one stacked row).

The library's first bar vocabulary, added for the timing report figures. A
single ``Barh`` draws one ``ax.barh`` call, so it covers both shapes the timing
charts need:

    grouped   — one bar per category:  ys=[0,1,2], widths=[v0,v1,v2]
    stacked   — segments in one row:    ys=[0,0,0], widths=segs, lefts=cumsum

Colors may be one shared color or a per-bar list. Value labels and in-bar text
are NOT drawn here — compose them with the existing ``Annotation`` plottable, so
bars stay a pure geometry primitive (mirrors how ``Stem``/``Line`` stay pure).

None-valued color resolves against ``style.PRIMARY_COLOR`` at draw() time;
None-valued height/edge against the ``style.DEFAULT_BAR_*`` constants (lazy, D-05).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .base import Plottable


class Barh(Plottable):
    """Horizontal bars: each (y, width) is a bar from ``left`` to ``left+width``."""

    def __init__(
        self,
        ys: Sequence[float],
        widths: Sequence[float],
        *,
        lefts: Optional[Sequence[float]] = None,
        colors: Optional[Sequence[str]] = None,
        color: Optional[str] = None,
        height: Optional[float] = None,
        edgecolor: Optional[str] = None,
        edgewidth: Optional[float] = None,
        alpha: float = 1.0,
        zorder: int = 3,
    ) -> None:
        super().__init__(color=color, alpha=alpha, zorder=zorder)
        self.ys = np.asarray(ys, dtype=float)
        self.widths = np.asarray(widths, dtype=float)
        self.lefts = None if lefts is None else np.asarray(lefts, dtype=float)
        # Per-bar colors (list) win over the single fallback color.
        self.colors = list(colors) if colors is not None else None
        self.height = height
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.PRIMARY_COLOR
        fill: Union[str, list] = self.colors if self.colors is not None else color
        height = self.height if self.height is not None else style.DEFAULT_BAR_HEIGHT
        edgecolor = self.edgecolor if self.edgecolor is not None else style.BG_COLOR
        edgewidth = (
            self.edgewidth if self.edgewidth is not None
            else style.DEFAULT_BAR_EDGE_LINEWIDTH
        )
        left = self.lefts if self.lefts is not None else 0.0
        ax.barh(
            self.ys, self.widths,
            left=left, height=height,
            color=fill, edgecolor=edgecolor, linewidth=edgewidth,
            alpha=self.alpha, zorder=self.zorder,
        )
