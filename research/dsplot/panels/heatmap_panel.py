"""HeatmapPanel — semantic StaticPanel subclass for 2D field displays.

Default (1, 1) is square; callers pass `units=(N, 1)` for wide spectrograms.
Overrides `render()` to drive the axes lim from the contained Heatmap's
extent so the field fills the cell. Optional `x_label`, `y_label`, `xticks`,
`yticks` kwargs populate axis labels and tick marks at the cell edges; tick
labels and axis labels land in the surrounding gutter.
"""
from __future__ import annotations

from typing import ClassVar, Optional, Sequence, Tuple

from .. import style
from ..plottables.heatmap import Heatmap
from .static_panel import StaticPanel


class HeatmapPanel(StaticPanel):
    default_units: ClassVar[Tuple[int, int]] = (1, 1)

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
        for plottable in self._plottables:
            if isinstance(plottable, Heatmap) and plottable.extent is not None:
                x0, x1, y0, y1 = plottable.extent
                self.ax.set_xlim(x0, x1)
                self.ax.set_ylim(y0, y1)
                self.ax.set_aspect("auto")
                break
        _apply_axis_decoration(
            self.ax,
            x_label=self.x_label,
            y_label=self.y_label,
            xticks=self.xticks,
            yticks=self.yticks,
        )


def _apply_axis_decoration(
    ax,
    *,
    x_label: Optional[str],
    y_label: Optional[str],
    xticks: Optional[Sequence[float]],
    yticks: Optional[Sequence[float]],
) -> None:
    tick_kwargs = dict(
        colors=style.TICK_LABEL_COLOR,
        labelsize=style.DEFAULT_TICK_LABEL_SIZE,
        length=style.DEFAULT_TICK_LENGTH,
        width=style.DEFAULT_TICK_WIDTH,
    )
    if xticks is not None:
        ax.set_xticks(list(xticks))
        ax.tick_params(axis="x", **tick_kwargs)
    if yticks is not None:
        ax.set_yticks(list(yticks))
        ax.tick_params(axis="y", **tick_kwargs)
    # Modest pad in points between tick labels and axis label — enough to
    # visually separate them without blowing up the gutter into a full
    # outer-margin unit. The outer perimeter handles the breathing-to-edge.
    labelpad_pts = 12.0
    if x_label is not None:
        ax.set_xlabel(
            x_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
            labelpad=labelpad_pts,
        )
    if y_label is not None:
        ax.set_ylabel(
            y_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
            labelpad=labelpad_pts,
        )
