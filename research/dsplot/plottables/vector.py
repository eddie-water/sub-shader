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
from mpl_toolkits.mplot3d import proj3d

from .. import style
from .base import Plottable


# Fraction of a 3D vector's length used as the SOLID head stub on a dashed 3D
# arrow (dashed shaft + solid head, mirroring the 2D split). The head itself is
# a FancyArrowPatch sized in display points, not by this ratio.
_ARROW_HEAD_RATIO_3D = 0.15


class _Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch whose endpoints live in data 3-space.

    Gives a 3D arrow the SAME crisp filled ``-|>`` head as the 2D vectors
    (matched head_length/head_width/mutation_scale) instead of ``ax.quiver``'s
    crude line-cone, which renders as asymmetric split barbs once the vector is
    large. ``do_3d_projection`` is called by mpl during draw: it projects the
    two 3D endpoints to 2D and feeds them to the underlying FancyArrowPatch, and
    returns the min camera-z for depth sorting against the rest of the scene.
    """

    def __init__(self, xyz_tail, xyz_tip, **kwargs) -> None:
        super().__init__((0, 0), (0, 0), **kwargs)
        self._xyz_tail = xyz_tail
        self._xyz_tip = xyz_tip

    def do_3d_projection(self, renderer=None) -> float:
        (x0, y0, z0), (x1, y1, z1) = self._xyz_tail, self._xyz_tip
        xs, ys, zs = proj3d.proj_transform(
            (x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(min(zs))


def _is_axes_3d(ax) -> bool:
    return hasattr(ax, "get_zlim")


def _arrowstyle(head_length: float, head_width: float) -> str:
    return f"-|>,head_length={head_length},head_width={head_width}"


def _head_length_data(ax: Axes, tail, tip, norm: float) -> float:
    """Length of the `-|>` arrowhead in DATA units along the tail→tip direction.

    The head is sized in points (``head_length × mutation_scale``); converting
    to data units lets a dashed vector seat a solid head on a stub that is
    guaranteed longer than the head, so matplotlib never scales the head down.
    Returns 0.0 if the transform isn't usable yet (caller falls back to a
    single patch).
    """
    if norm < 1e-12:
        return 0.0
    try:
        p0 = ax.transData.transform(tail)
        p1 = ax.transData.transform(tip)
    except Exception:
        return 0.0
    pix = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if pix < 1e-9:
        return 0.0
    pts = pix * 72.0 / ax.figure.dpi          # arrow length in points
    head_pts = style.DEFAULT_ARROW_HEAD_LENGTH * style.DEFAULT_ARROW_MUTATION
    return norm * head_pts / pts


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
                 show_arrowhead: bool = False,
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
        self.show_arrowhead = show_arrowhead

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

        def _add_head_patch(tail, head_tip, *, linestyle):
            patch = FancyArrowPatch(
                tail, head_tip,
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=self.alpha,
                mutation_scale=style.DEFAULT_ARROW_MUTATION,
                shrinkA=0, shrinkB=0,
                joinstyle="miter",
                capstyle="butt",
                zorder=self.zorder,
            )
            ax.add_patch(patch)

        norm = math.hypot(vx, vy)
        head_len = _head_length_data(ax, (ox, oy), tip, norm)

        # Solid vectors (and arrows too short to split) are a single
        # FancyArrowPatch: the shaft and the full-size filled `-|>` head in one
        # path. For a DASHED vector that single path would dash the HEAD outline
        # too — a thick dashed edge eats visible chunks out of the small head
        # silhouette ("chewed arrowhead"). So a dashed vector is split: a dashed
        # Line2D shaft plus a SOLID head drawn over a stub long enough that
        # matplotlib doesn't scale the head down (`head_len * 1.25`). The head
        # therefore stays the same size as a solid vector's, with a clean edge.
        if self.linestyle == "-" or norm < 1e-9 or head_len <= 0.0 \
                or norm <= head_len * 1.25:
            _add_head_patch((ox, oy), tip, linestyle=self.linestyle)
        else:
            ux, uy = vx / norm, vy / norm
            stub = head_len * 1.25
            stub_tail = (tip[0] - ux * stub, tip[1] - uy * stub)
            ax.plot(
                [ox, tip[0]], [oy, tip[1]],
                color=color,
                linewidth=linewidth,
                linestyle=self.linestyle,
                alpha=self.alpha,
                dash_capstyle="butt",
                solid_capstyle="butt",
                zorder=self.zorder,
            )
            _add_head_patch(stub_tail, tip, linestyle="-")

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
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)

        arrowstyle = _arrowstyle(
            head_length=style.DEFAULT_ARROW_HEAD_LENGTH,
            head_width=style.DEFAULT_ARROW_HEAD_WIDTH,
        )

        def _add_head_arrow(tail, head_tip, *, linestyle):
            arrow = _Arrow3D(
                tail, head_tip,
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=self.alpha,
                mutation_scale=style.DEFAULT_ARROW_MUTATION,
                shrinkA=0, shrinkB=0,
                joinstyle="miter",
                capstyle="butt",
                zorder=self.zorder,
            )
            ax.add_artist(arrow)

        if self.show_arrowhead and norm > 1e-9 and self.linestyle == "-":
            # Solid arrow: one FancyArrowPatch-in-3D — shaft + crisp filled head,
            # matching the 2D vectors (no crude quiver line-cone).
            _add_head_arrow(origin, tip, linestyle="-")
        elif self.show_arrowhead and norm > 1e-9:
            # Dashed arrow: dashed shaft (ax.plot) + a SOLID head stub. Dashing
            # the head outline would "chew" its silhouette, so the head is a solid
            # crisp stub at the tip — the 3D analog of the 2D split.
            ux, uy, uz = vx / norm, vy / norm, vz / norm
            stub = norm * _ARROW_HEAD_RATIO_3D * 1.6
            stub_tail = (tip[0] - ux * stub,
                         tip[1] - uy * stub,
                         tip[2] - uz * stub)
            ax.plot(
                [ox, tip[0]], [oy, tip[1]], [oz, tip[2]],
                color=color,
                linewidth=linewidth,
                linestyle=self.linestyle,
                alpha=self.alpha,
                zorder=self.zorder,
            )
            _add_head_arrow(stub_tail, tip, linestyle="-")
        else:
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
