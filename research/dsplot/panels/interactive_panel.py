"""InteractivePanel — multi-frame Panel driven by matplotlib widget controls.

Frame model matches DynamicPanel: `frames: list[list[Plottable]]`. Stepping
happens through `matplotlib.widgets.Button` (Prev / Next, always present),
optionally a `matplotlib.widgets.Slider` (`slider=True`) and a
`matplotlib.widgets.CheckButtons` (`checkbox=("label", callback)`).

Controls render in inset axes anchored to the panel's `transAxes`, so they
sit directly below the panel they control. In multi-panel Figures, each
InteractivePanel's controls remain column-aligned under its own panel.

Backend behavior:
    Widgets are created regardless of backend. With an interactive backend
    (`%matplotlib widget`, `qt5`, etc.) pointer events fire the bound
    callbacks. With Agg (test / headless), the widgets exist and their
    callbacks can be invoked programmatically for testing.

Per-frame artist cleanup uses tracked-artist removal (NOT `ax.cla()`) so the
background, spines, axes setup, and base_plottables persist across stepping
— same pattern as DynamicPanel.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import matplotlib.widgets as mwidgets
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import style
from ..axes_setup import setup_vector_axes
from .base import Panel


Frame = List["Plottable"]  # noqa: F821 — forward reference for typing only


class InteractivePanel(Panel):
    """Multi-frame Panel driven by matplotlib widget controls.

    Always-on: Prev / Next buttons centered just below the panel.
    Optional: slider (`slider=True`) for direct frame index, checkbox
    (`checkbox=("label", callback)`) for plot-element toggling.
    """

    _base_bottom_pad: float = 0.18

    def __init__(
        self,
        *,
        units: Optional[Tuple[int, int]] = None,
        frames: List[Frame],
        base_plottables: Optional[List["Plottable"]] = None,  # noqa: F821
        slider: bool = False,
        checkbox: Optional[Tuple[str, Callable[[bool], None]]] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        lim: Optional[Union[float, Tuple[float, float]]] = None,
        axis_style: str = "line",
        axis_labels: bool = False,
        show_border: bool = True,
    ) -> None:
        super().__init__(units=units)
        if not frames or len(frames) < 1:
            raise ValueError(
                "InteractivePanel requires at least one frame in `frames`"
            )
        self.frames = frames
        self.base_plottables = base_plottables or []
        self.use_slider = slider
        self.checkbox_spec = checkbox

        self.title = title
        self.subtitle = subtitle
        self.lim = lim
        self.axis_style = axis_style
        self.axis_labels = axis_labels
        self.show_border = show_border

        self._current_frame = 0
        self._frame_artists: List = []
        self._widgets: dict = {}
        self._widget_axes: List = []

    def _scalar_lim(self) -> Optional[float]:
        if self.lim is None:
            return None
        if isinstance(self.lim, tuple):
            lo, hi = self.lim
            return max(abs(lo), abs(hi))
        return float(self.lim)

    def _render_background(self) -> None:
        ax = self.ax
        ax.set_facecolor(style.BG_COLOR)
        if self.show_border:
            for spine in ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        setup_vector_axes(
            ax,
            lim=self._scalar_lim(),
            panel_title=None,
            show_border=self.show_border,
            axis_style=self.axis_style,
            axis_labels=self.axis_labels,
        )

        self._render_chrome_titles()

        for plottable in self.base_plottables:
            plottable.draw(ax)

    def _set_frame(self, frame_idx: int) -> None:
        ax = self.ax
        for artist in self._frame_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._frame_artists = []

        before_patches = set(id(p) for p in ax.patches)
        before_lines = set(id(p) for p in ax.lines)
        before_collections = set(id(p) for p in ax.collections)
        before_texts = set(id(p) for p in ax.texts)
        before_images = set(id(p) for p in ax.images)

        for plottable in self.frames[frame_idx]:
            plottable.draw(ax)

        for collection_name, before_ids in (
            ("patches", before_patches),
            ("lines", before_lines),
            ("collections", before_collections),
            ("texts", before_texts),
            ("images", before_images),
        ):
            for artist in getattr(ax, collection_name):
                if id(artist) not in before_ids:
                    self._frame_artists.append(artist)

        self._current_frame = frame_idx

    def _step(self, delta: int) -> None:
        new_idx = max(0, min(len(self.frames) - 1, self._current_frame + delta))
        if new_idx == self._current_frame:
            return
        self._set_frame(new_idx)
        self._request_redraw()

    def _jump_to(self, frame_idx: int) -> None:
        idx = int(max(0, min(len(self.frames) - 1, frame_idx)))
        if idx == self._current_frame:
            return
        self._set_frame(idx)
        self._request_redraw()

    def _request_redraw(self) -> None:
        if self.ax is None or self.ax.figure is None:
            return
        try:
            self.ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def _anchored_axes(
        self, x: float, y: float, w: float, h: float,
    ):
        """Create an inset axes positioned in the panel's transAxes-relative
        coordinates. `(x, y)` is the lower-left corner; `(w, h)` the size.
        Negative y values place the inset below the panel."""
        widget_ax = inset_axes(
            self.ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(x, y, w, h),
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )
        widget_ax.set_facecolor(style.BG_COLOR)
        for spine in widget_ax.spines.values():
            spine.set_visible(False)
        widget_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self._widget_axes.append(widget_ax)
        return widget_ax

    def _style_button(self, btn: "mwidgets.Button") -> None:
        btn.color = style.SPINE_COLOR
        btn.hovercolor = style.TICK_LABEL_COLOR
        btn.ax.set_facecolor(style.SPINE_COLOR)
        btn.label.set_color(style.NEUTRAL_COLOR)
        btn.label.set_fontsize(style.DEFAULT_AXIS_LABEL_SIZE)
        btn.label.set_fontweight("bold")

    def _style_slider(self, slider: "mwidgets.Slider") -> None:
        slider.ax.set_facecolor(style.SPINE_COLOR)
        slider.poly.set_color(style.PRIMARY_COLOR)
        slider.label.set_color(style.NEUTRAL_COLOR)
        slider.valtext.set_color(style.NEUTRAL_COLOR)

    def _style_checkbox(self, check: "mwidgets.CheckButtons") -> None:
        check.ax.set_facecolor(style.BG_COLOR)
        for rect in getattr(check, "rectangles", []):
            rect.set_facecolor(style.SPINE_COLOR)
            rect.set_edgecolor(style.NEUTRAL_COLOR)
        for lines in getattr(check, "lines", []):
            for line in lines:
                line.set_color(style.PRIMARY_COLOR)
        for label in check.labels:
            label.set_color(style.NEUTRAL_COLOR)

    def _render_controls(self) -> None:
        """Place mpl widgets (prev/next + optional slider/checkbox) anchored
        below the panel. Layout is top-to-bottom: buttons row, slider row,
        checkbox row. All controls styled to match the dark theme."""
        button_w = 0.16
        button_h = 0.08
        button_gap = 0.04
        button_y = -0.18
        pair_w = 2 * button_w + button_gap
        button_x_start = 0.5 - pair_w / 2

        prev_ax = self._anchored_axes(button_x_start, button_y, button_w, button_h)
        next_ax = self._anchored_axes(
            button_x_start + button_w + button_gap, button_y, button_w, button_h,
        )

        prev_btn = mwidgets.Button(prev_ax, "Prev", color=style.SPINE_COLOR, hovercolor=style.TICK_LABEL_COLOR)
        next_btn = mwidgets.Button(next_ax, "Next", color=style.SPINE_COLOR, hovercolor=style.TICK_LABEL_COLOR)
        self._style_button(prev_btn)
        self._style_button(next_btn)
        prev_btn.on_clicked(lambda _evt: self._step(-1))
        next_btn.on_clicked(lambda _evt: self._step(+1))
        self._widgets["prev"] = prev_btn
        self._widgets["next"] = next_btn

        cursor_y = button_y - 0.04

        if self.use_slider:
            slider_h = 0.05
            slider_w = 0.70
            slider_ax = self._anchored_axes(
                0.5 - slider_w / 2, cursor_y - slider_h, slider_w, slider_h,
            )
            slider = mwidgets.Slider(
                slider_ax,
                label="frame",
                valmin=0,
                valmax=len(self.frames) - 1,
                valinit=0,
                valstep=1,
                color=style.PRIMARY_COLOR,
            )
            self._style_slider(slider)
            slider.on_changed(lambda val: self._jump_to(int(val)))
            self._widgets["slider"] = slider
            cursor_y -= slider_h + 0.04

        if self.checkbox_spec is not None:
            label, callback = self.checkbox_spec
            cb_h = 0.07
            cb_w = 0.40
            cb_ax = self._anchored_axes(
                0.5 - cb_w / 2, cursor_y - cb_h, cb_w, cb_h,
            )
            check = mwidgets.CheckButtons(cb_ax, [label], [False])
            self._style_checkbox(check)

            def _on_check(_label: str) -> None:
                callback(bool(check.get_status()[0]))

            check.on_clicked(_on_check)
            self._widgets["checkbox"] = check

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("InteractivePanel.render() called before attach()")

        self._render_background()
        self._frame_artists = []
        self._current_frame = 0
        self._set_frame(0)

        self._render_controls()
