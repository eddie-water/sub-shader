"""StaticPanel3D — single-state Panel for Axes3D cells.

3D cells need different chrome from 2D cells:
  - mpl's default 3D axis chrome (panes, gridlines, ticks, bounding box) is
    hidden via ``ax.set_axis_off()``.
  - Three neutral axis spines through the origin are drawn manually using
    the polymorphic Vector Plottable (3-tuples per LOCKED D-02) so depth
    sorting plays nicely with the rest of the scene.
  - ``computed_zorder = False`` so the Panel's plottable insertion order
    (and per-Plottable ``zorder`` kwargs) controls draw order — without
    this, mpl's depth sort would hide the diagonal vector behind segments
    whose projection bounds overlap.

Plottables added via ``.add()`` are still drawn via the inherited render
loop. A polymorphic Vector with a 3-tuple just works (D-02); a 2-tuple
silently extends to ``(x, y, 0)`` (D-06).

The 3D chrome (scene config + spines/ticks/labels/border) lives in
module-level functions so DynamicPanel3D can reuse it without duplicating
StaticPanel3D — both panels host the same Axes3D look.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .. import style
from ..plottables.vector import Vector
from .base import Panel


def apply_3d_scene(ax, *, lim_3d: float, view_init: Tuple[float, float],
                   box_zoom: float = 1.55) -> None:
    """Configure an Axes3D as a dark, chrome-less cube viewed from view_init.

    Hides mpl's default 3D chrome, sets symmetric ±lim_3d limits on all three
    axes, applies the camera angle, and zooms the cube outward so it fills the
    cell. ``computed_zorder = False`` hands draw order to per-Plottable zorder.

    ``box_zoom`` magnifies the cube within the axes bbox. The legacy default
    (1.55) fills the cell, but at that magnification a vector pointing toward a
    cell edge can overflow the axes rectangle and have its arrowhead clipped —
    lower it (toward ~1.3) for figures whose vectors reach the cell border.
    """
    ax.set_axis_off()
    ax.set_facecolor(style.BG_COLOR)
    ax.computed_zorder = False

    ax.set_xlim(-lim_3d, lim_3d)
    ax.set_ylim(-lim_3d, lim_3d)
    ax.set_zlim(-lim_3d, lim_3d)
    ax.view_init(elev=view_init[0], azim=view_init[1])
    ax.set_box_aspect((1, 1, 1), zoom=box_zoom)


def draw_3d_floor_grid(ax, *, lim_3d: float) -> None:
    """A light grid on the z=0 (xy) plane only — the scene 'floor'.

    set_axis_off() kills mpl's native 3D grid/panes, so the floor reference is
    drawn manually: integer lines parallel to x and y, at z=0, spanning the cube.
    Only the floor (not the back/side panes) so the grid grounds the vectors
    without boxing the scene in three planes. Faint (DEFAULT_AXIS_GRID_ALPHA_3D)
    and zorder 0 so it sits under everything.
    """
    import math
    lo, hi = -lim_3d, lim_3d
    ticks = [float(i) for i in range(math.ceil(lo), math.floor(hi) + 1)]
    grid_kwargs = dict(
        color=style.DEFAULT_AXIS_GRID_COLOR,
        alpha=style.DEFAULT_AXIS_GRID_ALPHA_3D,
        linewidth=style.DEFAULT_AXIS_GRID_LINEWIDTH,
        zorder=0,
    )
    for x in ticks:
        ax.plot([x, x], [lo, hi], [0.0, 0.0], **grid_kwargs)
    for y in ticks:
        ax.plot([lo, hi], [y, y], [0.0, 0.0], **grid_kwargs)


def draw_3d_spines(ax, *, lim_3d: float, spine_extension: float,
                   alpha: float = 0.55) -> None:
    """Three neutral axis spines through the origin, both directions.

    Uses the polymorphic Vector Plottable with 3-tuples (D-02). Each spine is
    rendered as two Vectors (one positive direction, one negative) so the origin
    sits in the middle. Spine length = ``lim_3d * spine_extension`` so spines
    can reach past the visible cube into the panel border area. ``alpha`` is the
    spine opacity — the legacy default (0.55) reads faint; bump it toward 1.0
    for crisper, higher-contrast spines.
    """
    lim = lim_3d * spine_extension
    spine_kwargs = dict(
        color=style.SPINE_COLOR,
        linewidth=style.DEFAULT_SPINE_3D_LINEWIDTH,
        alpha=alpha,
        show_tip=False,
    )
    Vector((lim, 0.0, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
    Vector((-lim, 0.0, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
    Vector((0.0, lim, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
    Vector((0.0, -lim, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
    Vector((0.0, 0.0, lim), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
    Vector((0.0, 0.0, -lim), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)


def draw_3d_spine_ticks(ax, *, lim_3d: float, show_labels: bool = True) -> None:
    """Small tick crosses (+ optional numeric labels) at integer positions.

    Uses short orthogonal Vectors as tick crosses (same polymorphic-Vector
    approach as the spines themselves) so depth sort plays nicely. When
    ``show_labels`` is False the tick crosses are kept but the numeric labels
    are dropped — declutters a busy animated scene.
    """
    positions = [
        float(i) for i in range(-int(lim_3d), int(lim_3d) + 1)
        if i != 0 and abs(i) <= lim_3d
    ]
    tick_len = lim_3d * 0.05
    tick_kwargs = dict(
        color=style.SPINE_COLOR,
        linewidth=style.DEFAULT_SPINE_3D_LINEWIDTH * 0.6,
        alpha=0.85,
        show_tip=False,
    )
    label_kwargs = dict(
        color=style.TICK_LABEL_COLOR,
        fontsize=style.DEFAULT_TICK_LABEL_SIZE,
    )
    for p in positions:
        # x-axis tick cross (+ numeric label below)
        Vector((0.0, tick_len, 0.0), origin=(p, -tick_len / 2.0, 0.0),
               **tick_kwargs).draw(ax)
        # y-axis tick cross (+ numeric label left)
        Vector((tick_len, 0.0, 0.0), origin=(-tick_len / 2.0, p, 0.0),
               **tick_kwargs).draw(ax)
        # z-axis tick cross (+ numeric label left)
        Vector((tick_len, 0.0, 0.0), origin=(-tick_len / 2.0, 0.0, p),
               **tick_kwargs).draw(ax)
        if show_labels:
            ax.text(p, -tick_len * 1.8, 0.0, f"{int(p)}",
                    ha="center", va="top", **label_kwargs)
            ax.text(-tick_len * 1.8, p, 0.0, f"{int(p)}",
                    ha="right", va="center", **label_kwargs)
            ax.text(-tick_len * 1.8, 0.0, p, f"{int(p)}",
                    ha="right", va="center", **label_kwargs)


def draw_3d_axis_labels(
    ax,
    *,
    lim_3d: float,
    spine_extension: float,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
    anchor_z: float | None = None,
) -> None:
    """x / y / z labels SKEWERED by their respective spines.

    Each label is anchored in 3D space at a point along the spine and centered
    horizontally + vertically so the spine line (origin → spine tip → label)
    passes through the geometric center of the text.

    Anchor radii (data coords along each spine) default to the legacy values:
    x + z anchor just inside the cube edge (``lim_3d * 0.95``) and y is pushed
    further radially (``lim_3d * spine_extension * 1.15``). Pass explicit
    ``anchor_x/y/z`` to override — e.g. push x/z out to the border and pull y
    back in.
    """
    label_kwargs_3d = dict(
        color=style.TICK_LABEL_COLOR,
        fontsize=style.DEFAULT_AXIS_LABEL_SIZE + 2,
        fontweight="bold",
        fontstyle="italic",
    )
    ax_x = anchor_x if anchor_x is not None else lim_3d * 0.95
    ax_z = anchor_z if anchor_z is not None else lim_3d * 0.95
    ax_y = anchor_y if anchor_y is not None else lim_3d * spine_extension * 1.15
    ax.text(ax_x, 0.0, 0.0, "x",
            ha="center", va="center", **label_kwargs_3d)
    ax.text(0.0, ax_y, 0.0, "y",
            ha="center", va="center", **label_kwargs_3d)
    ax.text(0.0, 0.0, ax_z, "z",
            ha="center", va="center", **label_kwargs_3d)


def render_3d_chrome(
    ax,
    *,
    lim_3d: float,
    view_init: Tuple[float, float],
    spine_extension: float,
    show_spines: bool = True,
    show_spine_ticks: bool = True,
    show_spine_tick_labels: bool = True,
    label_anchors: Optional[Tuple[float, float, float]] = None,
    spine_alpha: float = 0.55,
    box_zoom: float = 1.55,
    show_floor_grid: bool = True,
) -> None:
    """Apply the full Axes3D chrome: scene config, then spines/ticks/labels.

    ``show_spine_tick_labels=False`` keeps the tick crosses but drops the
    numeric labels. ``label_anchors`` (x, y, z) overrides the per-axis label
    radii along each spine. ``spine_alpha`` sets the spine opacity. ``box_zoom``
    magnifies the cube (lower it if edge-pointing vectors get arrowhead-clipped).
    ``show_floor_grid`` lays a light grid on the z=0 plane for ground reference.
    Does NOT draw the figure-coord border or chrome titles — callers handle those.
    """
    apply_3d_scene(ax, lim_3d=lim_3d, view_init=view_init, box_zoom=box_zoom)
    if show_floor_grid:
        draw_3d_floor_grid(ax, lim_3d=lim_3d)
    if show_spines:
        draw_3d_spines(ax, lim_3d=lim_3d, spine_extension=spine_extension,
                       alpha=spine_alpha)
        if show_spine_ticks:
            draw_3d_spine_ticks(ax, lim_3d=lim_3d,
                                show_labels=show_spine_tick_labels)
        ax_x, ax_y, ax_z = label_anchors if label_anchors else (None, None, None)
        draw_3d_axis_labels(ax, lim_3d=lim_3d, spine_extension=spine_extension,
                            anchor_x=ax_x, anchor_y=ax_y, anchor_z=ax_z)


def draw_3d_border(ax) -> None:
    """Neutral rectangle border around the 3D cell, in figure coords.

    ``set_axis_off()`` hides mpl's 3D pane edges, so a 2D Rectangle in figure
    coords gives the same border treatment as 2D panels' spine-based border.
    Added to the figure (not the axes) so it lands on top of the 3D scene.
    """
    import matplotlib.patches as mpatches
    bbox = ax.get_position()
    border = mpatches.Rectangle(
        (bbox.x0, bbox.y0), bbox.width, bbox.height,
        fill=False,
        edgecolor=style.SPINE_COLOR,
        linewidth=style.DEFAULT_SPINE_LINEWIDTH,
        transform=ax.figure.transFigure,
        clip_on=False,
        zorder=100,
    )
    ax.figure.add_artist(border)


class StaticPanel3D(Panel):
    """Single-state Panel for an Axes3D cell.

    Constructor kwargs:
      - ``lim_3d``: symmetric ±lim_3d on x, y, z.
      - ``view_init``: (elev, azim) tuple — defaults match the legacy
        ``_plot_vector_projection_3d`` perspective.
      - ``title`` / ``subtitle``: rendered via ``ax.set_title`` /
        ``ax.text2D`` respectively when set.
      - ``show_spines``: when True (default), draws three neutral axis
        spines through the origin and labels (x, y, z) at the positive tips.

    Plottables added via ``.add()`` are drawn after the chrome.
    """

    def __init__(
        self,
        *,
        units: Optional[Tuple[int, int]] = None,
        lim_3d: float = 1.0,
        view_init: Tuple[float, float] = (30.0, -60.0),
        title: str | None = None,
        subtitle: str | None = None,
        caption: str | None = None,
        show_spines: bool = True,
        show_spine_ticks: bool = True,
        show_border: bool = True,
        spine_extension: float = 1.0,
    ) -> None:
        super().__init__(units=units)
        self.lim_3d = float(lim_3d)
        self.view_init = (float(view_init[0]), float(view_init[1]))
        self.title = title
        self.subtitle = subtitle
        self.caption = caption
        self.show_spines = show_spines
        self.show_spine_ticks = show_spine_ticks
        self.show_border = show_border
        # Multiplier on lim_3d for the visible spine length and label position.
        # 1.0 = spines end at the visible cube faces (±lim_3d). >1 lets spines
        # extend past the cube so they reach the panel border instead of the
        # cube interior, giving a more decisive "axes through the box" look.
        self.spine_extension = float(spine_extension)

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("StaticPanel3D.render() called before attach()")

        ax = self.ax
        render_3d_chrome(
            ax,
            lim_3d=self.lim_3d,
            view_init=self.view_init,
            spine_extension=self.spine_extension,
            show_spines=self.show_spines,
            show_spine_ticks=self.show_spine_ticks,
        )

        self._render_chrome_titles()

        for plottable in self._plottables:
            plottable.draw(ax)

        # Border last so it lands on top of plottables.
        if self.show_border:
            draw_3d_border(ax)
