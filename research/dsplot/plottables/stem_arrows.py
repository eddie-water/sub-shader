"""StemArrows Plottable — discrete values rendered as colored arrows.

A "stem plot" whose stems are full :class:`Vector` arrows rather than the
classic line+marker: each value ``v_i`` becomes a vertical arrow from
``(x_i, baseline)`` up (positive) or down (negative) to ``(x_i, baseline + v_i)``.
Because each stem is a real Vector, the bars read with the SAME arrow
vocabulary (head shape, linewidth, color) as every other panel in a figure —
so a "components as bars" panel sits beside "components as vectors" panels
without a visual register change.

Each entry keeps its own color (e.g. a-components orange, b-components
purple) so a grouped layout (a-family left, b-family right) reads at a
glance. Optional per-stem labels sit just past the baseline on the opposite
side of the arrow, so a downward (negative) stem labels above the axis and an
upward stem labels below it.

None-valued ``linewidth`` resolves against
``style.DEFAULT_VECTOR_BOLD_LINEWIDTH`` at draw() time (lazy per D-05). Unlike
:class:`Stem`, StemArrows NEVER touches axis limits — it draws inside the
host Panel's existing line-axis chrome (a fixed ±lim vector cell).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from matplotlib.axes import Axes

from .. import style
from .base import Plottable
from .vector import Vector


class StemArrows(Plottable):
    """Discrete values drawn as vertical Vector arrows at evenly-spaced x-slots.

    ``values``, ``colors`` (and optional ``labels``) are parallel sequences —
    one entry per stem. ``x_positions`` overrides the default symmetric
    even spacing across ``x_span``; pass explicit slots to GROUP entries
    (e.g. a-family negative x, b-family positive x).
    """

    def __init__(
        self,
        values: Sequence[float],
        *,
        colors: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        x_positions: Optional[Sequence[float]] = None,
        x_span: float = 3.0,
        baseline: float = 0.0,
        linewidth: float | None = None,
        label_pad: float = 0.45,
        label_size: float | None = None,
        alpha: float = 1.0,
        zorder: int = 3,
    ) -> None:
        super().__init__(linewidth=linewidth, alpha=alpha, zorder=zorder)
        self.values = [float(v) for v in values]
        self.colors = list(colors)
        if len(self.colors) != len(self.values):
            raise ValueError("StemArrows: colors must match values length")
        self.labels = list(labels) if labels is not None else None
        if self.labels is not None and len(self.labels) != len(self.values):
            raise ValueError("StemArrows: labels must match values length")
        self.x_positions = (
            list(x_positions) if x_positions is not None else None
        )
        if self.x_positions is not None and \
                len(self.x_positions) != len(self.values):
            raise ValueError("StemArrows: x_positions must match values length")
        self.x_span = float(x_span)
        self.baseline = float(baseline)
        self.label_pad = float(label_pad)
        self.label_size = label_size

    def _slots(self) -> List[float]:
        if self.x_positions is not None:
            return self.x_positions
        n = len(self.values)
        if n == 1:
            return [0.0]
        step = (2.0 * self.x_span) / (n - 1)
        return [-self.x_span + i * step for i in range(n)]

    def draw(self, ax: Axes) -> None:
        linewidth = (
            self.linewidth if self.linewidth is not None
            else style.DEFAULT_VECTOR_BOLD_LINEWIDTH
        )
        label_size = (
            self.label_size if self.label_size is not None
            else style.DEFAULT_LABEL_FONT_SIZE
        )
        slots = self._slots()

        for idx, (value, color, x) in enumerate(
                zip(self.values, self.colors, slots)):
            # Each stem is a real Vector arrow so the head/linewidth match the
            # rest of the figure. A ~zero value draws nothing (no degenerate
            # arrowhead) but still gets its label below the baseline.
            if abs(value) > 1e-9:
                Vector(
                    (0.0, value),
                    origin=(x, self.baseline),
                    color=color,
                    linewidth=linewidth,
                    alpha=self.alpha,
                    zorder=self.zorder,
                ).draw(ax)

            if self.labels is not None:
                text = self.labels[idx]
                # Label opposite the arrow direction: upward stem labels below
                # the baseline, downward stem labels above it, so the glyph
                # never collides with the arrowhead.
                ly = self.baseline - self.label_pad if value >= 0 \
                    else self.baseline + self.label_pad
                va = "top" if value >= 0 else "bottom"
                ax.text(
                    x, ly, text,
                    color=color, alpha=self.alpha,
                    fontsize=label_size, fontweight="bold",
                    ha="center", va=va,
                    zorder=self.zorder + 1,
                )
