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
"""
from __future__ import annotations

from typing import Tuple

from .. import style
from ..plottables.vector import Vector
from .base import Panel


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
        lim_3d: float = 1.0,
        view_init: Tuple[float, float] = (30.0, -60.0),
        title: str | None = None,
        subtitle: str | None = None,
        show_spines: bool = True,
    ) -> None:
        super().__init__()
        self.lim_3d = float(lim_3d)
        self.view_init = (float(view_init[0]), float(view_init[1]))
        self.title = title
        self.subtitle = subtitle
        self.show_spines = show_spines

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("StaticPanel3D.render() called before attach()")

        ax = self.ax
        # The 2D chrome from setup_vector_axes is irrelevant in 3D — we
        # tear down mpl's default 3D chrome and rebuild only what we want.
        ax.set_axis_off()
        ax.set_facecolor(style.BG_COLOR)
        # Respect plottable zorder over mpl's depth sort.
        ax.computed_zorder = False

        lim = self.lim_3d
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=self.view_init[0], azim=self.view_init[1])

        if self.show_spines:
            self._draw_spines(ax)
            self._draw_axis_labels(ax)

        self._render_chrome_titles()

        for plottable in self._plottables:
            plottable.draw(ax)

    def _draw_spines(self, ax) -> None:
        """Three neutral axis spines through the origin, both directions.

        Uses the polymorphic Vector Plottable with 3-tuples (D-02). Each
        spine is rendered as two Vectors (one positive direction, one
        negative) so the origin sits in the middle.
        """
        lim = self.lim_3d
        spine_kwargs = dict(
            color=style.SPINE_COLOR,
            linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
            alpha=0.55,
            show_tip=False,
        )
        # +x and -x spines
        Vector((lim, 0.0, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
        Vector((-lim, 0.0, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
        # +y and -y spines
        Vector((0.0, lim, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
        Vector((0.0, -lim, 0.0), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
        # +z and -z spines
        Vector((0.0, 0.0, lim), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)
        Vector((0.0, 0.0, -lim), origin=(0.0, 0.0, 0.0), **spine_kwargs).draw(ax)

    def _draw_axis_labels(self, ax) -> None:
        """Italic x / y / z labels at the positive spine tips."""
        lim = self.lim_3d
        pad = lim * 0.05
        label_kwargs = dict(
            color=style.SPINE_COLOR,
            fontsize=style.DEFAULT_AXIS_LABEL_SIZE + 2,
            fontweight="bold",
            fontstyle="italic",
        )
        ax.text(lim + pad, 0.0, 0.0, "x", ha="left", va="center", **label_kwargs)
        ax.text(0.0, lim + pad, 0.0, "y", ha="left", va="center", **label_kwargs)
        ax.text(0.0, 0.0, lim + pad, "z", ha="center", va="bottom", **label_kwargs)
