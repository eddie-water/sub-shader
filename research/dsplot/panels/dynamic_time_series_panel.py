"""DynamicTimeSeriesPanel — a DynamicPanel rendered on TIME-SERIES axes.

DynamicPanel sets its axes up through ``setup_vector_axes`` (square aspect,
symmetric ±lim, no ticks) — right for vector figures, wrong for a stem plot
over a sample index. This subclass swaps only the background pass: it pins an
explicit ``xlim``/``ylim``, lays out ``xticks``/``yticks`` with the shared
``_apply_axis_decoration`` chrome (exactly like ``TimeSeriesPanel``), and then
locks autoscale OFF so the per-frame artists drawn by the inherited
``_animate`` loop can never nudge the limits frame-to-frame.

Because it subclasses ``DynamicPanel``, ``Figure`` discovers it through the
same ``isinstance(panel, DynamicPanel)`` checks that drive the master clock —
no figure.py changes are needed to animate one of these in a composed grid.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

from .. import style
from .dynamic_panel import DynamicPanel, Frame
from .heatmap_panel import _apply_axis_decoration


class DynamicTimeSeriesPanel(DynamicPanel):
    """A multi-frame Panel on time-series (non-square) axes.

    Shares DynamicPanel's frame model (``frames`` or ``frame_fn`` +
    ``num_frames``), per-frame artist cleanup, and figure-master-clock
    integration. Only the one-time background setup differs.
    """

    default_units = (2, 1)

    def __init__(
        self,
        *,
        units: Optional[Tuple[int, int]] = None,
        frames: Optional[List[Frame]] = None,
        frame_fn: Optional[Callable[[int], Frame]] = None,
        num_frames: Optional[int] = None,
        interval_ms: int = 250,
        repeat: bool = True,
        base_plottables: Optional[List] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        caption: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        xticks: Optional[Sequence[float]] = None,
        yticks: Optional[Sequence[float]] = None,
        show_xticklabels: bool = True,
        show_yticklabels: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        show_border: bool = True,
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
        )
        self.x_label = x_label
        self.y_label = y_label
        self.xticks = xticks
        self.yticks = yticks
        self.show_xticklabels = show_xticklabels
        self.show_yticklabels = show_yticklabels
        self.xlim = xlim
        self.ylim = ylim

    def _render_background(self) -> None:
        """Time-series background: explicit limits + tick chrome, autoscale off.

        Drawn once before any frame; never re-drawn during animation.
        """
        ax = self.ax
        ax.set_facecolor(style.BG_COLOR)
        if self.show_border:
            for spine in ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax.set_aspect("auto")
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)

        _apply_axis_decoration(
            ax,
            x_label=self.x_label,
            y_label=self.y_label,
            xticks=self.xticks,
            yticks=self.yticks,
            show_xticklabels=self.show_xticklabels,
            show_yticklabels=self.show_yticklabels,
        )

        # Re-pin AFTER decoration (mirrors TimeSeriesPanel.render) and freeze
        # autoscale so per-frame Stem/bar artists can't shift the limits and
        # make the plot jitter as samples stream in.
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        ax.set_autoscale_on(False)

        self._render_chrome_titles()

        for plottable in self.base_plottables:
            plottable.draw(ax)
