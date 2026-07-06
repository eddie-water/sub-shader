"""AccumulatorStrip — one heavy stem + a big readout on a zero baseline.

A tall narrow strip that collapses a quantity into a SINGLE bar rising (positive)
or falling (negative) from a zero line to ``value``, with ``value`` printed big on
the empty side of the baseline (under an upward stem, above a downward one) so the
number never sits on the bar. Lifted from figure 2.5's "Sum strip" (``_AccumStem``
+ ``_SumReadout`` + its zero line) into the library so figures 2.4.2 / 2.4.3 / 2.5
/ 2.6 share one accumulator vocabulary.

Drop it into any panel whose y-scale spans the value range (the strip draws no
ticks of its own — the readout carries magnitude, the zero line carries sign):

    panel = TimeSeriesPanel(xticks=[], yticks=[], xlim=(-1, 1), ylim=(-12, 12))
    panel.add(AccumulatorStrip(dot_product))

Style knobs left None resolve lazily against ``dsplot.style.*`` at draw() time
(per D-05), so a global/profile reassignment between construction and draw is
observed.
"""
from __future__ import annotations

from typing import Optional

from matplotlib.axes import Axes

from .. import style
from .base import Plottable


# How far into the empty well (as a fraction of |value|) the readout chip sits.
# 0.5 centers it in the well opposite the bar given the conventional ±1.4×value
# y-range — clear of the zero line, clear of the bar, clear of the cell edge.
_READOUT_WELL_FRACTION = 0.5


class AccumulatorStrip(Plottable):
    """One stem from y=0 to ``value`` at ``x`` plus a ``{value:+d}`` readout.

    ``value``       the quantity to display (sign sets the stem direction).
    ``x``           strip centerline in data coords (default 0.0).
    ``show_zero_line`` draw the faint y=0 baseline (default True).
    ``decimals``    readout precision; 0 (default) prints a signed integer.
    Remaining knobs (color/linewidth/markersize/font/zero-line) default to the
    ``style.DEFAULT_ACCUM_*`` constants when left None.
    """

    def __init__(
        self,
        value: float,
        *,
        x: float = 0.0,
        color: Optional[str] = None,
        alpha: float = 0.9,
        linewidth: Optional[float] = None,
        markersize: Optional[float] = None,
        readout_color: Optional[str] = None,
        readout_font_size: Optional[float] = None,
        decimals: int = 0,
        show_zero_line: bool = True,
        zero_line_color: Optional[str] = None,
        zero_line_alpha: Optional[float] = None,
        zero_line_width: Optional[float] = None,
        vertical_spine: bool = False,
        readout_origin_offset: Optional[float] = None,
        zorder: int = 3,
    ) -> None:
        super().__init__(color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        self.value = float(value)
        self.x = float(x)
        self.markersize = markersize
        self.readout_color = readout_color
        self.readout_font_size = readout_font_size
        self.decimals = decimals
        self.show_zero_line = show_zero_line
        # Zero baseline ("spine") styling. Left None, color/alpha/width resolve to
        # the faint DEFAULT_ACCUM_ZERO_LINE_* gauge look; pass the axis-decoration
        # values to make the baseline read as a bright crosshair spine matching the
        # vector panels next to it.
        self.zero_line_color = zero_line_color
        self.zero_line_alpha = zero_line_alpha
        self.zero_line_width = zero_line_width
        # vertical_spine draws the baseline as a VERTICAL axis line through the
        # strip center (a number-line the bar rides) instead of the horizontal
        # y=0 baseline — the gauge then reads as "a value marked on a vertical
        # axis". readout_origin_offset, when set, parks the readout a FIXED number
        # of data units from the origin (in the empty well opposite the bar)
        # instead of scaling its position with the bar's height.
        self.vertical_spine = vertical_spine
        self.readout_origin_offset = readout_origin_offset

    def draw(self, ax: Axes) -> None:
        stem_color = self.color or style.DEFAULT_ACCUM_STEM_COLOR
        linewidth = self.linewidth or style.DEFAULT_ACCUM_STEM_LINEWIDTH
        markersize = self.markersize or style.DEFAULT_ACCUM_MARKERSIZE
        readout_color = self.readout_color or style.DEFAULT_ACCUM_READOUT_COLOR
        readout_size = self.readout_font_size or style.DEFAULT_ACCUM_READOUT_FONT_SIZE

        if self.show_zero_line or self.vertical_spine:
            zero_color = self.zero_line_color or style.DROPLINE_COLOR
            zero_alpha = (
                self.zero_line_alpha if self.zero_line_alpha is not None
                else style.DEFAULT_ACCUM_ZERO_LINE_ALPHA
            )
            zero_width = (
                self.zero_line_width if self.zero_line_width is not None
                else style.DEFAULT_ACCUM_ZERO_LINE_WIDTH
            )
            if self.vertical_spine:
                # Origin crosshair: a vertical axis down the strip center (the bar
                # rides it) PLUS a horizontal axis through y=0, matching the vector
                # panels' two-axis look.
                ax.axvline(
                    self.x, color=zero_color, alpha=zero_alpha,
                    linewidth=zero_width, zorder=1,
                )
                ax.axhline(
                    0.0, color=zero_color, alpha=zero_alpha,
                    linewidth=zero_width, zorder=1,
                )
            else:
                ax.axhline(
                    0.0, color=zero_color,
                    alpha=zero_alpha,
                    linewidth=zero_width,
                    zorder=1,
                )

        # ONE artist (line + tip marker via markevery=[1]) so the circle shares
        # the stem's exact coordinate and can't round off-centre; snap=False
        # stops the stroke pixel-snapping away from it.
        stem, = ax.plot(
            [self.x, self.x], [0.0, self.value],
            color=stem_color, alpha=self.alpha, linewidth=linewidth,
            solid_capstyle="round", marker="o", markevery=[1],
            markersize=markersize, zorder=self.zorder,
        )
        stem.set_snap(False)

        # Readout sits centered in the EMPTY half of the strip — the well OPPOSITE
        # the bar (below the baseline for an upward stem, above for a downward one).
        # Placed in DATA coords at a fraction of the value's magnitude into that
        # well (not a few points off the baseline) so the number stands cleanly
        # clear of the zero line / bar base instead of crowding the origin. A solid
        # black chip frames it so the running total reads as a discrete value.
        text = f"{self.value:+.{self.decimals}f}" if self.decimals else f"{round(self.value):+d}"
        if self.readout_origin_offset is not None:
            # Fixed distance from the origin, in the empty well opposite the bar
            # (below for an upward/zero bar, above for a downward one).
            offset = abs(self.readout_origin_offset)
            readout_y = -offset if self.value >= 0 else offset
        else:
            readout_y = -self.value * _READOUT_WELL_FRACTION
        ax.annotate(
            text,
            xy=(self.x, readout_y),
            ha="center", va="center",
            color=readout_color, fontsize=readout_size, fontweight="bold",
            zorder=self.zorder + 4,
            bbox=dict(facecolor="black", edgecolor="none",
                      boxstyle="square,pad=0.45"),
        )
