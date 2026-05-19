"""TimeSeriesPanel — semantic StaticPanel subclass for wide time-axis content.

`default_units = (3, 1)` so wide signal displays compose naturally in a
Figure.compose row. Overrides `render()` to release the square aspect lock
StaticPanel inherits from setup_vector_axes and applies optional axis
decoration (labels + ticks at cell edges).
"""
from __future__ import annotations

from typing import ClassVar, Optional, Sequence, Tuple

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.x_label = x_label
        self.y_label = y_label
        self.xticks = xticks
        self.yticks = yticks

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
