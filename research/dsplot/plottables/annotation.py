"""Annotation Plottable — 2D text with optional arrow callout.

Supports data-coord placement (default) and axes-relative placement
(transform="axes") via matplotlib's transAxes. The axes-relative path
enables panel-relative captions (e.g., "a · b = +1.0" below a vector
panel) without coupling the figure to specific data limits.

None-valued color and fontsize resolve against dsplot.style at draw()
time (lazy lookup per D-05).
"""
from __future__ import annotations

from matplotlib.axes import Axes

from .. import style
from .base import Plottable


_VALID_TRANSFORMS = ("data", "axes")


class Annotation(Plottable):
    """Text at (x, y) with optional arrow callout to a target point.

    transform="data" (default) places text at xy in data coords;
    transform="axes" places text at xy in axes-relative coords (0,0 = lower
    left of axes, 1,1 = upper right). The arrow callout (if any) uses data
    coords regardless of transform — arrows always point to a data location.
    """

    def __init__(self,
                 text: str,
                 xy: tuple[float, float],
                 *,
                 arrow_to: tuple[float, float] | None = None,
                 color: str | None = None,
                 fontsize: float | None = None,
                 fontweight: str = "normal",
                 ha: str = "center",
                 va: str = "center",
                 transform: str = "data",
                 alpha: float = 1.0,
                 zorder: int = 5) -> None:
        if transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}, got {transform!r}"
            )
        super().__init__(
            color=color, linewidth=None, alpha=alpha,
            linestyle="-", label=None, zorder=zorder,
        )
        self.text = text
        self.xy = tuple(xy)
        self.arrow_to = tuple(arrow_to) if arrow_to is not None else None
        self.fontsize = fontsize
        self.fontweight = fontweight
        self.ha = ha
        self.va = va
        self.transform = transform

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.TICK_LABEL_COLOR
        fontsize = (
            self.fontsize if self.fontsize is not None
            else style.DEFAULT_LABEL_FONT_SIZE
        )

        if self.arrow_to is not None:
            ax.annotate(
                self.text,
                xy=self.arrow_to, xytext=self.xy,
                arrowprops=dict(arrowstyle="->", color=color, alpha=self.alpha),
                color=color, fontsize=fontsize, fontweight=self.fontweight,
                ha=self.ha, va=self.va, zorder=self.zorder,
            )
            return

        # Axes3D requires text2D (or text with 3 coords). Detect via the
        # presence of get_zlim — Annotation is 2D semantically but the
        # axes-relative form is still useful for 3D legends.
        is_3d = hasattr(ax, "get_zlim")

        if self.transform == "axes":
            if is_3d:
                ax.text2D(
                    self.xy[0], self.xy[1], self.text,
                    transform=ax.transAxes,
                    color=color, fontsize=fontsize, fontweight=self.fontweight,
                    ha=self.ha, va=self.va,
                    alpha=self.alpha, zorder=self.zorder,
                )
            else:
                ax.text(
                    self.xy[0], self.xy[1], self.text,
                    color=color, fontsize=fontsize, fontweight=self.fontweight,
                    ha=self.ha, va=self.va, transform=ax.transAxes,
                    alpha=self.alpha, zorder=self.zorder,
                )
        else:
            ax.text(
                self.xy[0], self.xy[1], self.text,
                color=color, fontsize=fontsize, fontweight=self.fontweight,
                ha=self.ha, va=self.va,
                alpha=self.alpha, zorder=self.zorder,
            )
