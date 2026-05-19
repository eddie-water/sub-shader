"""TimeSeriesPanel — semantic StaticPanel subclass for wide time-axis content.

`default_units = (3, 1)` so wide signal displays compose naturally in a
Figure.compose row. Overrides `render()` to release the square aspect lock
StaticPanel inherits from setup_vector_axes and applies optional axis
decoration (labels + ticks at cell edges).

Optional twin-axis support: pass `twin_y=True` to overlay a second y-axis
sharing the primary x-axis (e.g. an instantaneous-frequency curve over a
chirp time series). Twin axis side (`left` or `right`), label, ticks, tick
labels, and ylim are independently configurable. Use `add_twin(plottable)`
to attach plottables that draw onto the twin axis.
"""
from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Tuple

from matplotlib.axes import Axes

from .. import style
from .heatmap_panel import _apply_axis_decoration
from .static_panel import StaticPanel


class TimeSeriesPanel(StaticPanel):
    default_units: ClassVar[Tuple[int, int]] = (3, 1)

    def __init__(
        self,
        *,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        xticks: Optional[Sequence[float]] = None,
        yticks: Optional[Sequence[float]] = None,
        twin_y: bool = False,
        twin_y_label: Optional[str] = None,
        twin_yticks: Optional[Sequence[float]] = None,
        twin_ytick_labels: Optional[Sequence[str]] = None,
        twin_ylim: Optional[Tuple[float, float]] = None,
        twin_y_side: str = "right",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if twin_y_side not in ("left", "right"):
            raise ValueError(
                f"twin_y_side must be 'left' or 'right', got {twin_y_side!r}"
            )
        self.x_label = x_label
        self.y_label = y_label
        self.xticks = xticks
        self.yticks = yticks
        self.twin_y = twin_y
        self.twin_y_label = twin_y_label
        self.twin_yticks = twin_yticks
        self.twin_ytick_labels = twin_ytick_labels
        self.twin_ylim = twin_ylim
        self.twin_y_side = twin_y_side
        self.ax_twin: Optional[Axes] = None
        self._twin_plottables: List = []

    def add_twin(self, plottable) -> None:
        """Add a plottable to the twin y-axis.

        Drawn after the primary plottables in render(). Raises RuntimeError if
        twin_y is False so a misconfigured caller fails fast at the call site
        instead of silently dropping the overlay at render time.
        """
        if not self.twin_y:
            raise RuntimeError(
                "add_twin() called on TimeSeriesPanel constructed with twin_y=False"
            )
        self._twin_plottables.append(plottable)

    def render(self) -> None:
        super().render()
        if self.ax is None:
            return
        self.ax.set_aspect("auto")
        _apply_axis_decoration(
            self.ax,
            x_label=self.x_label,
            y_label=self.y_label,
            xticks=self.xticks,
            yticks=self.yticks,
        )

        if not self.twin_y:
            return

        self.ax_twin = self.ax.twinx()

        if self.twin_y_side == "left":
            self.ax_twin.yaxis.tick_left()
            self.ax_twin.yaxis.set_label_position("left")
            self.ax_twin.tick_params(
                axis="y", which="both", right=False, labelright=False
            )
        else:
            self.ax_twin.tick_params(
                axis="y", which="both", left=False, labelleft=False
            )

        self.ax_twin.minorticks_off()
        self.ax_twin.patch.set_visible(False)

        for spine in self.ax_twin.spines.values():
            spine.set_edgecolor(style.SPINE_COLOR)
            spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)

        _apply_axis_decoration(
            self.ax_twin,
            x_label=None,
            y_label=self.twin_y_label,
            xticks=None,
            yticks=self.twin_yticks,
        )
        if self.twin_ytick_labels is not None:
            self.ax_twin.set_yticklabels(list(self.twin_ytick_labels))

        if self.twin_ylim is not None:
            self.ax_twin.set_ylim(*self.twin_ylim)

        for plottable in self._twin_plottables:
            plottable.draw(self.ax_twin)
