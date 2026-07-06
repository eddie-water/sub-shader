"""BarPanel — horizontal bar-chart panel (additive; timing report figures).

A ``StaticPanel`` subclass for horizontal bars. Unlike the vector panels it does
NOT draw an origin crosshair or lock equal aspect — it mirrors ``HeatmapPanel``'s
"wide cell, data-driven limits, external axis labels" treatment, but the cell is
filled with bars (``Barh`` plottables) instead of an image. The value axis (x)
may be linear or log; the category axis (y) carries named tick labels.

It owns only axis chrome (scale, limits, category labels, x-axis decoration); the
bars themselves are ``Barh`` plottables and any value / in-bar text is composed
from ``Annotation`` plottables — so the existing plottable vocabulary draws onto
it exactly as it does any other panel. Nothing in the base library calls this
class, so it cannot change how existing figures render.
"""
from __future__ import annotations

from typing import ClassVar, Optional, Sequence, Tuple

from matplotlib.ticker import NullLocator

from .. import style
from .static_panel import StaticPanel


class BarPanel(StaticPanel):
    default_units: ClassVar[Tuple[int, int]] = (3, 1)

    def __init__(
        self,
        *,
        x_label: Optional[str] = None,
        xticks: Optional[Sequence[float]] = None,
        xticklabels: Optional[Sequence[str]] = None,
        category_labels: Optional[Sequence[str]] = None,
        xscale: str = "linear",
        show_xticklabels: bool = True,
        invert_y: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.x_label = x_label
        self.xticks = xticks
        self.xticklabels = xticklabels
        self.category_labels = category_labels
        self.xscale = xscale
        self.show_xticklabels = show_xticklabels
        # First category on top reads as a ranked list (fastest/biggest first),
        # the convention every timing chart uses.
        self.invert_y = invert_y

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("BarPanel.render() called before attach()")
        ax = self.ax

        ax.set_facecolor(style.BG_COLOR)
        if self.show_border:
            for spine in ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Scale BEFORE drawing so log-spaced bars lay out correctly.
        if self.xscale == "log":
            ax.set_xscale("log")

        for plottable in self._plottables:
            plottable.draw(ax)

        ax.set_aspect("auto")
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)

        # Category (y) tick labels — one per bar row. When absent (names are
        # drawn inside the plot instead), strip the y axis entirely.
        if self.category_labels is not None:
            ax.set_yticks(list(range(len(self.category_labels))))
            ax.set_yticklabels(list(self.category_labels))
            if self.invert_y:
                ax.invert_yaxis()
        else:
            ax.set_yticks([])

        # x-axis label + ticks through the shared heatmap helper so the chrome
        # (tick direction, font, label inset) matches every other panel kind.
        from .heatmap_panel import _apply_axis_decoration
        _apply_axis_decoration(
            ax,
            x_label=self.x_label, y_label=None,
            xticks=self.xticks, yticks=None,
            show_xticklabels=self.show_xticklabels, show_yticklabels=True,
        )
        # Custom x tick labels (e.g. "1 ms" … "100 s" on a log axis); kill the
        # log minor ticks so only the labeled decades show.
        if self.xticklabels is not None and self.xticks is not None:
            ax.set_xticks(list(self.xticks))
            ax.set_xticklabels(list(self.xticklabels))
        if self.xscale == "log":
            ax.xaxis.set_minor_locator(NullLocator())

        # Category labels read as chrome (gray, no tick marks).
        ax.tick_params(
            axis="y", which="both",
            colors=style.TICK_LABEL_COLOR, labelcolor=style.TICK_LABEL_COLOR,
            labelsize=style.DEFAULT_TICK_LABEL_SIZE, length=0,
        )

        self._render_chrome_titles()
