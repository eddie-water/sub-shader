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
    # Axis labels are forced to sit at PAD/2 from the spine (midway between
    # the inner panel and the cell border) via set_label_coords, so the
    # label position is independent of tick label widths. labelpad is left
    # at matplotlib's default since set_label_coords overrides it anyway.
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    bbox = ax.get_position()
    axes_w_in = max(bbox.width * fig_w_in, 0.1)
    axes_h_in = max(bbox.height * fig_h_in, 0.1)
    half_pad_in = style.DEFAULT_GUTTER_INCHES / 4.0  # = PAD / 2

    if x_label is not None:
        ax.set_xlabel(
            x_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
        )
        x_offset_axes = half_pad_in / axes_h_in
        if ax.xaxis.get_label_position() == "top":
            ax.xaxis.set_label_coords(0.5, 1.0 + x_offset_axes)
        else:
            ax.xaxis.set_label_coords(0.5, -x_offset_axes)

    if y_label is not None:
        ax.set_ylabel(
            y_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
        )
        y_offset_axes = half_pad_in / axes_w_in
        if ax.yaxis.get_label_position() == "right":
            ax.yaxis.set_label_coords(1.0 + y_offset_axes, 0.5)
        else:
            ax.yaxis.set_label_coords(-y_offset_axes, 0.5)
