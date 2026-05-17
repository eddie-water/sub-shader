"""Vector Plottable — polymorphic on tuple length (D-02).

A 2-tuple input renders a 2D arrow on a regular matplotlib Axes via
FancyArrowPatch (sharp head, configurable head dimensions). A 3-tuple
input renders a 3D arrow on an Axes3D via ax.plot + scatter tip + 3D text.

Asymmetric dispatch (D-06):
  - 3-tuple drawn onto a 2D Axes → TypeError (no defensible 3D→2D collapse).
  - 2-tuple drawn onto an Axes3D → implicitly extended to (x, y, 0) and
    rendered flat in the z=0 plane (defensible projection default).

None-valued style knobs resolve against dsplot.style.* at draw() time, so
runtime reassignment of style constants between construction and draw is
observable.
"""
from __future__ import annotations

import math

from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from .. import style
from .base import Plottable


def _is_axes_3d(ax) -> bool:
    return hasattr(ax, "get_zlim")


def _arrowstyle(head_length: float, head_width: float) -> str:
    return f"-|>,head_length={head_length},head_width={head_width}"


class Vector(Plottable):
    """Single arrow from origin to origin + vec.

    Dimensionality inferred from len(vec):
      - 2 → 2D arrow on a regular mpl Axes (FancyArrowPatch shaft + head).
      - 3 → 3D arrow on an Axes3D (ax.plot + scatter tip + 3D text label).

    Drawing a 2-tuple onto an Axes3D extends the vector to (x, y, 0) and
    renders flat. Drawing a 3-tuple onto a 2D Axes raises TypeError.
    """

    def __init__(self,
                 vec,
                 *,
                 origin=None,
                 color: str | None = None,
                 linewidth: float | None = None,
                 alpha: float = 1.0,
                 linestyle: str = "-",
                 label: str | None = None,
                 label_offset=None,
                 show_tip: bool = True,
                 zorder: int = 2) -> None:
        vec_t = tuple(vec)
        if len(vec_t) not in (2, 3):
            raise ValueError(
                f"Vector requires a 2-tuple or 3-tuple, got {len(vec_t)}-tuple"
            )
        super().__init__(
            color=color, linewidth=linewidth, alpha=alpha,
            linestyle=linestyle, label=label, zorder=zorder,
        )
        self.vec = vec_t
        self.origin = tuple(origin) if origin is not None else (0.0,) * len(vec_t)
        self.label_offset = label_offset
        self.show_tip = show_tip

    def draw(self, ax: Axes) -> None:
        if len(self.vec) == 3:
            if not _is_axes_3d(ax):
                raise TypeError(
                    "3D Vector requires an Axes3D (projection='3d')"
                )
            self._draw_3d(ax, vec=self.vec, origin=self.origin)
            return

        if _is_axes_3d(ax):
            self._draw_3d(
                ax,
                vec=(*self.vec, 0.0),
                origin=(*self.origin, 0.0),
            )
            return

        self._draw_2d(ax)

    def _draw_2d(self, ax: Axes) -> None:
        color = self.color if self.color is not None else style.PRIMARY_COLOR
        linewidth = (
            self.linewidth if self.linewidth is not None
            else style.DEFAULT_VECTOR_LINEWIDTH
        )
        label_offset = (
            self.label_offset if self.label_offset is not None
            else style.DEFAULT_VECTOR_LABEL_OFFSET
        )

        ox, oy = self.origin
        vx, vy = self.vec
        tip = (ox + vx, oy + vy)

        arrowstyle = _arrowstyle(
            head_length=style.DEFAULT_ARROW_HEAD_LENGTH,
            head_width=style.DEFAULT_ARROW_HEAD_WIDTH,
        )

        if self.linestyle == "-":
            patch = FancyArrowPatch(
                (ox, oy), tip,
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                alpha=self.alpha,
                mutation_scale=style.DEFAULT_ARROW_MUTATION,
                shrinkA=0, shrinkB=0,
                zorder=self.zorder,
            )
            ax.add_patch(patch)
        else:
            ax.plot(
                [ox, tip[0]], [oy, tip[1]],
                color=color,
                linewidth=linewidth,
                linestyle=self.linestyle,
                alpha=self.alpha,
                zorder=self.zorder,
            )
            norm = math.hypot(vx, vy)
            if norm > 1e-12:
                head_back = (
                    tip[0] - (vx / norm) * 1e-3,
                    tip[1] - (vy / norm) * 1e-3,
                )
            else:
                head_back = (ox, oy)
            head_patch = FancyArrowPatch(
                head_back, tip,
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                alpha=self.alpha,
                mutation_scale=style.DEFAULT_ARROW_MUTATION,
                shrinkA=0, shrinkB=0,
                zorder=self.zorder,
            )
            ax.add_patch(head_patch)

        if self.label is not None:
            if isinstance(label_offset, tuple):
                dx, dy = label_offset
                lx, ly = tip[0] + dx, tip[1] + dy
            else:
                norm = math.hypot(vx, vy)
                if norm > 1e-12:
                    lx = tip[0] + (vx / norm) * float(label_offset)
                    ly = tip[1] + (vy / norm) * float(label_offset)
                else:
                    lx, ly = tip
            ax.text(
                lx, ly, self.label,
                color=color, alpha=self.alpha,
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                fontweight="bold",
                ha="center", va="center",
                zorder=self.zorder + 1,
            )

    def _draw_3d(self, ax, *, vec, origin) -> None:
        color = self.color if self.color is not None else style.NEUTRAL_COLOR
        linewidth = (
            self.linewidth if self.linewidth is not None
            else style.DEFAULT_VECTOR_BOLD_LINEWIDTH + 0.6
        )

        ox, oy, oz = origin
        vx, vy, vz = vec
        tip = (ox + vx, oy + vy, oz + vz)

        ax.plot(
            [ox, tip[0]], [oy, tip[1]], [oz, tip[2]],
            color=color,
            linewidth=linewidth,
            linestyle=self.linestyle,
            solid_capstyle="round",
            alpha=self.alpha,
            zorder=self.zorder,
        )

        if self.show_tip:
            ax.scatter(
                [tip[0]], [tip[1]], [tip[2]],
                color=color, s=80,
                zorder=self.zorder + 1,
                depthshade=False,
            )

        if self.label is not None:
            xlim = ax.get_xlim()
            lim_span = xlim[1] - xlim[0]
            label_pad = lim_span * 0.02
            ax.text(
                tip[0] + label_pad, tip[1] + label_pad, tip[2] + label_pad,
                self.label,
                color=color, fontweight="bold",
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
            )
