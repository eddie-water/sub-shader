"""HeatmapPanel — semantic StaticPanel subclass for 2D field displays.

Default (1, 1) is square; callers pass `units=(N, 1)` for wide spectrograms.
Overrides `render()` to drive the axes lim from the contained Heatmap's
extent so the field fills the cell. Optional `x_label`, `y_label`, `xticks`,
`yticks` kwargs populate axis labels and tick marks at the cell edges; tick
labels and axis labels land in the surrounding gutter.

**Line overlays on HeatmapPanel:** A `Line` plottable added via
`panel.add(Line(...))` draws onto the panel's primary axis, which is in
HEATMAP COORDINATE SPACE — i.e. x is duration (seconds, or whatever
`Heatmap.extent`'s x-range is) and y is BIN INDEX (0 to `len(freqs)`),
NOT Hz. Callers overlaying a frequency-domain curve (e.g. instantaneous
frequency in Hz) must pre-transform the y-values into bin-space before
constructing the Line, typically via
`np.interp(inst_freq_hz, freqs, np.arange(len(freqs)))`. HeatmapPanel does
NOT support a twin y-axis — overlays share the primary axis only. See
`dsplot/figures/gen_figure_1_stft_vs_cwt.py::_build_3row_figure` for the canonical pattern
(twin-axis Line on row 1's TimeSeriesPanel; primary-axis Line overlays
available on rows 2/3's HeatmapPanels via the bin-space transform).
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
        show_xticklabels: bool = True,
        show_yticklabels: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.x_label = x_label
        self.y_label = y_label
        self.xticks = xticks
        self.yticks = yticks
        self.show_xticklabels = show_xticklabels
        self.show_yticklabels = show_yticklabels

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
            show_xticklabels=self.show_xticklabels,
            show_yticklabels=self.show_yticklabels,
        )


def _apply_axis_decoration(
    ax,
    *,
    x_label: Optional[str],
    y_label: Optional[str],
    xticks: Optional[Sequence[float]],
    yticks: Optional[Sequence[float]],
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
) -> None:
    tick_kwargs = dict(
        colors=style.TICK_LABEL_COLOR,
        labelsize=style.DEFAULT_TICK_LABEL_SIZE,
        length=style.DEFAULT_TICK_LENGTH,
        width=style.DEFAULT_TICK_WIDTH,
        # DEFAULT_HEATMAP_TICK_DIRECTION ("out") → tick marks point AWAY from
        # the data (left for left-side y, right for right-side y, down for
        # bottom x). Keeps the plot area clean and makes the marks read as chrome.
        direction=style.DEFAULT_HEATMAP_TICK_DIRECTION,
    )
    if xticks is not None:
        ax.set_xticks(list(xticks))
        ax.tick_params(axis="x", **tick_kwargs)
    if yticks is not None:
        ax.set_yticks(list(yticks))
        ax.tick_params(axis="y", **tick_kwargs)
    # show_{x,y}ticklabels controls whether tick LABELS show (tick MARKS
    # are unaffected). Explicitly drive BOTH directions every call — mpl's
    # tick_params is sticky, so an earlier call that hid labels would
    # otherwise survive a later call that wants them visible.
    #
    # Critical: drive the side that ACTUALLY hosts the labels (left vs.
    # right for y, bottom vs. top for x). Forcing labelleft and labelright
    # both to False would clobber a twin y-axis whose labels live on the
    # right. yaxis.get_label_position() / xaxis.get_label_position() report
    # mpl's current side per axis.
    if ax.xaxis.get_label_position() == "top":
        ax.tick_params(axis="x", labeltop=show_xticklabels, labelbottom=False)
    else:
        ax.tick_params(axis="x", labelbottom=show_xticklabels, labeltop=False)
    if ax.yaxis.get_label_position() == "right":
        ax.tick_params(axis="y", labelright=show_yticklabels, labelleft=False)
    else:
        ax.tick_params(axis="y", labelleft=show_yticklabels, labelright=False)
    # Axis labels sit DEFAULT_{X,Y}_AXIS_LABEL_INSET_INCHES from the spine via
    # set_label_coords. The ONLY clamp is the figure-edge clamp — labels live
    # OUTSIDE the spine in the chrome zone, so an own-axes clamp wouldn't
    # protect anything useful and would break the style-template derivation
    # (tiny inner composite cells would silently use a different inset than
    # top-level panels). The chrome-zone clearance (0.15") leaves room for
    # the rotated label's visual half-width without it kissing the figure
    # edge.
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    bbox = ax.get_position()
    axes_w_in = max(bbox.width * fig_w_in, 0.1)
    axes_h_in = max(bbox.height * fig_h_in, 0.1)
    axes_x0_in = bbox.x0 * fig_w_in
    axes_x1_to_edge_in = (1.0 - bbox.x1) * fig_w_in
    axes_y0_in = bbox.y0 * fig_h_in
    axes_y1_to_edge_in = (1.0 - bbox.y1) * fig_h_in
    edge_clearance = style.DEFAULT_HEATMAP_AXIS_EDGE_CLEARANCE_INCHES
    x_inset_in = min(
        style.DEFAULT_X_AXIS_LABEL_INSET_INCHES,
        max(axes_y0_in - edge_clearance, 0.1)
        if ax.xaxis.get_label_position() != "top"
        else max(axes_y1_to_edge_in - edge_clearance, 0.1),
    )
    y_inset_in = min(
        style.DEFAULT_Y_AXIS_LABEL_INSET_INCHES,
        max(axes_x0_in - edge_clearance, 0.1)
        if ax.yaxis.get_label_position() != "right"
        else max(axes_x1_to_edge_in - edge_clearance, 0.1),
    )

    if x_label is not None:
        ax.set_xlabel(
            x_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
            labelpad=0,
        )
        # va='center' so the label CENTER aligns with set_label_coords. Default
        # va='top' would put the top edge at the position — that adds label
        # height to one side and breaks the spine↔cell-border centering.
        ax.xaxis.label.set_va("center")
        x_offset_axes = x_inset_in / axes_h_in
        if ax.xaxis.get_label_position() == "top":
            ax.xaxis.set_label_coords(0.5, 1.0 + x_offset_axes)
        else:
            ax.xaxis.set_label_coords(0.5, -x_offset_axes)

    if y_label is not None:
        ax.set_ylabel(
            y_label,
            color=style.TICK_LABEL_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
            labelpad=0,
        )
        # Same fix as x-label: default va='bottom' (which becomes the right
        # edge after rotation=90) shifts the label off-center by half its
        # rotated width. Force va='center' for spine↔cell-border centering.
        ax.yaxis.label.set_va("center")
        y_offset_axes = y_inset_in / axes_w_in
        if ax.yaxis.get_label_position() == "right":
            ax.yaxis.set_label_coords(1.0 + y_offset_axes, 0.5)
        else:
            ax.yaxis.set_label_coords(-y_offset_axes, 0.5)

    _pin_extreme_ticklabels(ax)


def _pin_extreme_ticklabels(ax) -> None:
    """Pin extreme tick labels to the spine edge so they don't project past it.

    matplotlib centers tick labels on their tick marks, so the topmost label
    extends ABOVE the top spine and the bottommost extends BELOW the bottom
    spine. In tightly-tiled lego-style layouts that projection crosses into
    adjacent cells' chrome zones, causing the "tick label sitting on the
    border" overlap. Pinning va='bottom' on the lowest tick label (and the
    ha/va equivalents for the other extremes) keeps every tick label
    contained inside the spine bbox so cells tile cleanly regardless of
    inter-row/column gutter size.

    Connected as a draw_event callback because matplotlib regenerates tick
    label defaults on autoscale/draw cycles — applying va/ha once doesn't
    survive interactive backends. Idempotent: tags the Axes so repeated
    setup calls don't accumulate duplicate listeners.
    """
    if getattr(ax, "_dsplot_tick_pinner_attached", False):
        return

    def _on_draw(_event):
        y_labels = [t for t in ax.get_yticklabels()
                    if t.get_visible() and t.get_text()]
        if len(y_labels) >= 2:
            y_labels[0].set_va("bottom")
            y_labels[-1].set_va("top")
        elif len(y_labels) == 1:
            y_labels[0].set_va("center")
        x_labels = [t for t in ax.get_xticklabels()
                    if t.get_visible() and t.get_text()]
        if len(x_labels) >= 2:
            x_labels[0].set_ha("left")
            x_labels[-1].set_ha("right")
        elif len(x_labels) == 1:
            x_labels[0].set_ha("center")

    ax.figure.canvas.mpl_connect("draw_event", _on_draw)
    ax._dsplot_tick_pinner_attached = True
