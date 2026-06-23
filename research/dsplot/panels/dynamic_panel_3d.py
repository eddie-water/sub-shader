"""DynamicPanel3D — animated Panel for an Axes3D cell.

DynamicPanel is 2D-only: its ``_render_background`` calls ``setup_vector_axes``
and 2D spine/tick logic that breaks on an Axes3D. StaticPanel3D has the 3D
chrome but no animation. DynamicPanel3D bridges them: it subclasses
DynamicPanel — inheriting the frame model, the master-clock ``tick()`` hook,
``_managed_externally``, and the tracked-artist per-frame removal — and
overrides ONLY ``_render_background`` to install the shared 3D chrome
(``render_3d_chrome`` + ``draw_3d_border`` from static_panel_3d) instead of the
2D axis setup.

The per-frame artist removal in the inherited ``_animate`` works on Axes3D
because 3D Vector draws via ``ax.quiver``/``ax.plot`` + scatter tip + 3D text,
which land in ``ax.collections`` / ``ax.lines`` / ``ax.texts`` — the same
artist collections the id-diff snapshot tracks for 2D.

Because compose()'s master clock ticks every ``isinstance(p, DynamicPanel)``
panel and assigns ``projection="3d"`` to 3D panels, a DynamicPanel3D sits in a
mixed Figure in lockstep with 2D DynamicPanels off the same shared clock.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .dynamic_panel import DynamicPanel, Frame
from .static_panel_3d import draw_3d_border, render_3d_chrome


class DynamicPanel3D(DynamicPanel):
    """Multi-frame Panel rendered as a FuncAnimation loop on an Axes3D cell.

    Shares DynamicPanel's frame contract (``frames`` OR ``frame_fn`` +
    ``num_frames``) and the figure master-clock orchestration. 3D-specific
    chrome kwargs mirror StaticPanel3D (``lim_3d``, ``view_init``,
    ``spine_extension``, ``show_spines``, ``show_spine_ticks``).
    """

    def __init__(
        self,
        *,
        units: Optional[Tuple[int, int]] = None,
        frames: Optional[List[Frame]] = None,
        frame_fn: Optional[Callable[[int], Frame]] = None,
        num_frames: Optional[int] = None,
        interval_ms: int = 250,
        repeat: bool = True,
        base_plottables: Optional[List["Plottable"]] = None,  # noqa: F821
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        caption: Optional[str] = None,
        # 3D chrome (mirrors StaticPanel3D):
        lim_3d: float = 1.0,
        view_init: Tuple[float, float] = (30.0, -60.0),
        show_spines: bool = True,
        show_spine_ticks: bool = True,
        show_spine_tick_labels: bool = True,
        label_anchors: Optional[Tuple[float, float, float]] = None,
        spine_alpha: float = 0.55,
        box_zoom: float = 1.55,
        show_border: bool = True,
        spine_extension: float = 1.0,
        fill_cell: bool = False,
    ) -> None:
        super().__init__(
            units=units,
            frames=frames,
            frame_fn=frame_fn,
            num_frames=num_frames,
            interval_ms=interval_ms,
            repeat=repeat,
            base_plottables=base_plottables,
            title=title,
            subtitle=subtitle,
            caption=caption,
            show_border=show_border,
            fill_cell=fill_cell,
        )
        self.lim_3d = float(lim_3d)
        self.view_init = (float(view_init[0]), float(view_init[1]))
        self.show_spines = show_spines
        self.show_spine_ticks = show_spine_ticks
        self.show_spine_tick_labels = show_spine_tick_labels
        self.label_anchors = label_anchors
        self.spine_alpha = float(spine_alpha)
        self.box_zoom = float(box_zoom)
        self.spine_extension = float(spine_extension)

    def _render_background(self) -> None:
        """Install the Axes3D chrome ONCE; never re-drawn during animation.

        Mirrors StaticPanel3D.render() order: scene + spines/ticks/labels, then
        chrome titles, then persistent base_plottables, then the figure-coord
        border on top. Frame plottables are drawn separately by the inherited
        ``_animate`` and cleared each tick.
        """
        ax = self.ax
        render_3d_chrome(
            ax,
            lim_3d=self.lim_3d,
            view_init=self.view_init,
            spine_extension=self.spine_extension,
            show_spines=self.show_spines,
            show_spine_ticks=self.show_spine_ticks,
            show_spine_tick_labels=self.show_spine_tick_labels,
            label_anchors=self.label_anchors,
            spine_alpha=self.spine_alpha,
            box_zoom=self.box_zoom,
        )

        self._render_chrome_titles()

        for plottable in self.base_plottables:
            plottable.draw(ax)

        if self.show_border:
            draw_3d_border(ax)
