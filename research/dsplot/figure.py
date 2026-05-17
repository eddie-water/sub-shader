"""Figure orchestrator — composes Panels into one matplotlib Figure.

Layout defaults (figsize, hspace, wspace, dpi) resolve LAZILY against
`dsplot.style.DEFAULT_*` at construction time (per D-05) — reassigning a
style constant between `Figure(...)` calls is observable through subsequent
gridspec configuration. The Figure NEVER hardcodes layout numerics.

Per D-04, `add_panel` accepts a `projection` kwarg so 3D cells (Axes3D) can
be composed via the same flow as 2D cells:
    fig = Figure(n_rows=1, n_cols=2)
    fig.add_panel(panel_3d, row=0, col=0, projection="3d")
    fig.add_panel(panel_2d, row=0, col=1)
    fig.render()
    fig.savefig("mixed.png")

A panel that accepts a 3D Axes is expected to do its own 2D-vs-3D branching
(or be a subclass — e.g. a future StaticPanel3D for the 3D foundation figure).
The Figure itself only wires the Axes; it does not adapt panel chrome to
dimensionality.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from . import style
from .panels.base import Panel


class Figure:
    """Compose Panels into one matplotlib Figure with a gridspec layout."""

    def __init__(
        self,
        *,
        n_rows: int = 1,
        n_cols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        suptitle: Optional[str] = None,
        hspace: Optional[float] = None,
        wspace: Optional[float] = None,
        width_ratios: Optional[List[float]] = None,
        height_ratios: Optional[List[float]] = None,
        dpi: Optional[int] = None,
    ) -> None:
        if figsize is None:
            figsize = (
                style.DEFAULT_PANEL_SIZE_INCHES * n_cols,
                style.DEFAULT_PANEL_SIZE_INCHES * n_rows,
            )
        if hspace is None:
            hspace = style.DEFAULT_HSPACE
        if wspace is None:
            wspace = style.DEFAULT_WSPACE
        if dpi is None:
            dpi = style.DEFAULT_DPI

        self.n_rows = n_rows
        self.n_cols = n_cols
        self._mpl_fig = plt.figure(figsize=figsize, dpi=dpi)
        self._mpl_fig.patch.set_facecolor(style.BG_COLOR)
        self._gs = self._mpl_fig.add_gridspec(
            n_rows, n_cols,
            hspace=hspace, wspace=wspace,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        if suptitle is not None:
            self._mpl_fig.suptitle(
                suptitle,
                color=style.TICK_LABEL_COLOR,
                fontsize=style.DEFAULT_SUPTITLE_FONT_SIZE,
                fontweight="bold",
                y=0.975,
            )

        self.panels: List[Tuple[Panel, int, int, int, int, Optional[str]]] = []

    def add_panel(
        self,
        panel: Panel,
        *,
        row: int = 0,
        col: int = 0,
        rowspan: int = 1,
        colspan: int = 1,
        projection: Optional[str] = None,
    ) -> Panel:
        self.panels.append((panel, row, col, rowspan, colspan, projection))
        return panel

    def render(self) -> None:
        for panel, row, col, rowspan, colspan, projection in self.panels:
            cell = self._gs[row:row + rowspan, col:col + colspan]
            if projection is None:
                ax = self._mpl_fig.add_subplot(cell)
                ax.set_facecolor(style.BG_COLOR)
                for spine in ax.spines.values():
                    spine.set_edgecolor(style.SPINE_COLOR)
                    spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
            else:
                ax = self._mpl_fig.add_subplot(cell, projection=projection)
                ax.set_facecolor(style.BG_COLOR)
            panel.attach(ax)
            panel.render()

    def savefig(self, path: str, **kwargs) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        bbox_inches = kwargs.pop("bbox_inches", "tight")
        pad_inches = kwargs.pop("pad_inches", 0.15)
        self._mpl_fig.savefig(
            path,
            facecolor=self._mpl_fig.get_facecolor(),
            dpi=self._mpl_fig.get_dpi(),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **kwargs,
        )
        return os.path.abspath(path)

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close(self._mpl_fig)
