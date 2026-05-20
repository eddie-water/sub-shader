"""VectorComponents — x/y decomposition of a 2D vector (2D only).

Renders the tip-to-tail reconstruction of a vector as two dashed component
arrows, plus optional droplines from the tip perpendicular to each axis.
"""
from __future__ import annotations

from matplotlib.axes import Axes

from .. import style
from .base import Plottable
from .vector import Vector


class VectorComponents(Plottable):
    """X/Y decomposition: dashed component arrows from origin + droplines from tip.

    `first_axis="x"` renders the x-component from origin, then the y-component
    from the x-component's tip. `first_axis="y"` does the reverse. Either order
    terminates at the same tip — the decomposition is commutative, which the
    figure puts on display.
    """

    def __init__(self,
                 vec,
                 *,
                 first_axis: str = "x",
                 from_origin: bool = False,
                 show_droplines: bool = True,
                 component_color: str | None = None,
                 dropline_color: str | None = None,
                 label_x: str | None = None,
                 label_y: str | None = None,
                 alpha: float = 0.95,
                 zorder: int = 2) -> None:
        vec_t = tuple(vec)
        if len(vec_t) != 2:
            raise ValueError(
                "VectorComponents is 2D only — pass a 2-tuple"
            )
        if first_axis not in ("x", "y"):
            raise ValueError(
                f"first_axis must be 'x' or 'y', got {first_axis!r}"
            )
        super().__init__(
            color=component_color,
            linewidth=None,
            alpha=alpha,
            linestyle="--",
            label=None,
            zorder=zorder,
        )
        self.vec = vec_t
        self.first_axis = first_axis
        self.from_origin = from_origin
        self.show_droplines = show_droplines
        self.component_color = component_color
        self.dropline_color = dropline_color
        self.label_x = label_x
        self.label_y = label_y

    def draw(self, ax: Axes) -> None:
        vx, vy = self.vec
        component_color = (
            self.component_color if self.component_color is not None
            else style.NEUTRAL_COLOR
        )
        dropline_color = (
            self.dropline_color if self.dropline_color is not None
            else style.DROPLINE_COLOR
        )

        x_comp = (vx, 0.0)
        y_comp = (0.0, vy)
        if self.from_origin:
            # Both components emanate from origin, landing ON the axes.
            first_vec, first_origin = x_comp, (0.0, 0.0)
            second_vec, second_origin = y_comp, (0.0, 0.0)
        elif self.first_axis == "x":
            first_vec, first_origin = x_comp, (0.0, 0.0)
            second_vec, second_origin = y_comp, (vx, 0.0)
        else:
            first_vec, first_origin = y_comp, (0.0, 0.0)
            second_vec, second_origin = x_comp, (0.0, vy)

        Vector(
            first_vec,
            origin=first_origin,
            color=component_color,
            linestyle="--",
            alpha=self.alpha,
            zorder=self.zorder,
        ).draw(ax)
        Vector(
            second_vec,
            origin=second_origin,
            color=component_color,
            linestyle="--",
            alpha=self.alpha,
            zorder=self.zorder,
        ).draw(ax)

        if self.show_droplines:
            # Droplines are subordinate to the bold vector + dashed components,
            # so they're thinner and a darker gray than the components — closer
            # to the panel background so they read as a quiet "construction
            # line" not a primary decoration.
            dropline_kwargs = dict(
                color=style.SPINE_COLOR,
                alpha=self.alpha,
                linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
                linestyle="--",
                zorder=self.zorder - 1,
            )
            ax.plot([vx, vx], [vy, 0.0], **dropline_kwargs)
            ax.plot([vx, 0.0], [vy, vy], **dropline_kwargs)

        if self.label_x is not None:
            ax.text(
                vx / 2.0, -0.10, self.label_x,
                color=component_color,
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                fontweight="bold",
                ha="center", va="top",
                zorder=self.zorder + 1,
            )
        if self.label_y is not None:
            # When from_origin, the y-component sits ON the y-axis (x=0), so
            # the label anchors to the LEFT of that vertical line. In the
            # staircase mode, it anchors to the RIGHT of the y-component
            # (which lives at x=vx in that mode).
            if self.from_origin:
                label_x = -0.20
                ha = "right"
            else:
                label_x = vx + 0.08
                ha = "left"
            ax.text(
                label_x, vy / 2.0, self.label_y,
                color=component_color,
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                fontweight="bold",
                ha=ha, va="center",
                zorder=self.zorder + 1,
            )
