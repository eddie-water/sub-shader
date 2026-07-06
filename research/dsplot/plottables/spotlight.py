"""Spotlight Plottable — highlight overlay for emphasizing a value or region.

Three rendering modes:

  - ``"rectangle"`` — axvspan(x_range) and/or axhspan(y_range). Either or
    both may be supplied; passing neither raises ValueError on draw. Used
    by the alignment diagnostic to mark burst-time bands.
  - ``"scatter"`` — single emphasized point at ``xy`` (large dot, white
    edge). Used to mark a peak or sample-of-interest.
  - ``"glow"`` — alpha-graded Circle at ``xy`` with configurable
    ``radius``. Useful for animation focus where the spotlight moves.

Default color is ``style.TERTIARY_COLOR`` (gold — high visibility against
the dark background). Style knobs resolve against ``dsplot.style.*`` at
draw() time per D-05.

Validation runs at ``draw()``, not in ``__init__`` — matches the rest of
the library's defer-failure-to-render contract.
"""
from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import Circle

from .. import style
from .base import Plottable


_VALID_MODES = ("rectangle", "scatter", "glow")


class Spotlight(Plottable):
    """Highlight overlay for emphasizing a value or region.

    Parameters
    ----------
    mode : {"rectangle", "scatter", "glow"}
        Selects the rendering shape; see module docstring for semantics.
    x_range, y_range : tuple[float, float] | None
        Used only by ``mode="rectangle"``. At least one must be set;
        passing both renders two crossed spans.
    xy : tuple[float, float] | None
        Used by ``mode="scatter"`` and ``mode="glow"`` for the point.
    radius : float
        Used by ``mode="glow"`` for the Circle radius.
    color, alpha, linewidth, linestyle : style knobs
        Color defaults to ``style.TERTIARY_COLOR`` at draw() time.
    """

    def __init__(
        self,
        *,
        mode: str = "rectangle",
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        xy: tuple[float, float] | None = None,
        radius: float = 0.05,
        color: str | None = None,
        alpha: float = 0.35,
        linewidth: float = 1.5,
        linestyle: str = "--",
        zorder: int = 6,
    ) -> None:
        super().__init__(
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=None,
            zorder=zorder,
        )
        self.mode = mode
        self.x_range = tuple(x_range) if x_range is not None else None
        self.y_range = tuple(y_range) if y_range is not None else None
        self.xy = tuple(xy) if xy is not None else None
        self.radius = float(radius)

    def draw(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.TERTIARY_COLOR

        if self.mode == "rectangle":
            self._draw_rectangle(ax, color)
        elif self.mode == "scatter":
            self._draw_scatter(ax, color)
        elif self.mode == "glow":
            self._draw_glow(ax, color)
        else:
            raise ValueError(
                f"Spotlight.mode must be one of {set(_VALID_MODES)}, "
                f"got {self.mode!r}"
            )

    def _draw_rectangle(self, ax: Axes, color: str) -> None:
        if self.x_range is None and self.y_range is None:
            raise ValueError(
                "Spotlight(mode='rectangle') requires at least one of "
                "x_range or y_range"
            )
        if self.x_range is not None:
            ax.axvspan(
                self.x_range[0], self.x_range[1],
                alpha=self.alpha,
                color=color,
                linewidth=self.linewidth,
                linestyle=self.linestyle,
                zorder=self.zorder,
            )
        if self.y_range is not None:
            ax.axhspan(
                self.y_range[0], self.y_range[1],
                alpha=self.alpha,
                color=color,
                linewidth=self.linewidth,
                linestyle=self.linestyle,
                zorder=self.zorder,
            )

    def _draw_scatter(self, ax: Axes, color: str) -> None:
        if self.xy is None:
            raise ValueError("Spotlight(mode='scatter') requires xy")
        ax.scatter(
            [self.xy[0]], [self.xy[1]],
            s=style.DEFAULT_SPOTLIGHT_SIZE,
            color=color,
            alpha=self.alpha,
            edgecolors=style.DEFAULT_SPOTLIGHT_EDGE_COLOR,
            linewidths=self.linewidth,
            zorder=self.zorder,
        )

    def _draw_glow(self, ax: Axes, color: str) -> None:
        if self.xy is None:
            raise ValueError("Spotlight(mode='glow') requires xy")
        circle = Circle(
            self.xy,
            self.radius,
            color=color,
            alpha=self.alpha,
            zorder=self.zorder,
            fill=True,
        )
        ax.add_patch(circle)
