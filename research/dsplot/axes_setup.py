"""Reusable axes-setup helper for 2D vector panels.

Configures a matplotlib Axes with square aspect ratio, symmetric ±lim limits,
no ticks, and optional origin-crosshair (`axis_style="line"`) or +x/+y arrow
axes (`axis_style="arrow"`). Optional in-axes "x" / "y" labels are placed near
the corresponding axis tips. Optional panel title + result-text annotations
mirror the legacy `setup_vector_axes` in `research.utilities.plotting`.

All `None`-valued kwargs resolve LAZILY against `dsplot.style.DEFAULT_*` at
call time (per D-05) — reassigning a style constant between this call and a
later one observes the new value. Concrete fallback values for `axis_alpha`
and `axis_linewidth` stay inline because they're legacy crosshair-weight
constants; figures wanting different values pass them per-call.
"""
from __future__ import annotations

from typing import Optional, Union

from matplotlib.axes import Axes

from . import style


def setup_vector_axes(
    ax: Axes,
    *,
    lim: Optional[float] = None,
    panel_title: Optional[str] = None,
    result_text: Optional[str] = None,
    show_border: bool = True,
    axis_style: str = "line",
    axis_labels: bool = False,
    x_color: Optional[str] = None,
    y_color: Optional[str] = None,
    axis_alpha: Optional[float] = None,
    axis_linewidth: Optional[float] = None,
) -> None:
    if lim is None:
        lim = style.DEFAULT_VECTOR_LIM
    if x_color is None:
        x_color = style.SPINE_COLOR
    if y_color is None:
        y_color = style.SPINE_COLOR
    if axis_alpha is None:
        axis_alpha = 0.85
    if axis_linewidth is None:
        axis_linewidth = 1.8

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    if not show_border:
        for spine in ax.spines.values():
            spine.set_visible(False)

    inset = style.DEFAULT_AXIS_ARROW_INSET
    if axis_style == "arrow":
        for (tail, head), color in (
            (((-lim + inset, 0.0), (lim - inset, 0.0)), x_color),
            (((0.0, -lim + inset), (0.0, lim - inset)), y_color),
        ):
            ax.annotate(
                "", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="<|-|>",
                                color=color,
                                alpha=axis_alpha,
                                linewidth=max(axis_linewidth, 0.9),
                                mutation_scale=14,
                                shrinkA=0, shrinkB=0),
                zorder=0,
            )
    elif axis_style == "line":
        ax.axhline(0, color=x_color,
                   alpha=axis_alpha, linewidth=axis_linewidth, zorder=0)
        ax.axvline(0, color=y_color,
                   alpha=axis_alpha, linewidth=axis_linewidth, zorder=0)
    elif axis_style == "none":
        pass
    else:
        raise ValueError(
            f"axis_style must be 'line', 'arrow', or 'none' (got {axis_style!r})"
        )

    if axis_labels:
        # Place labels ALONG the spine (inset from the panel border) rather
        # than at the spine tips, so they read as "axis labels next to the
        # axis" instead of decorations sitting on the panel edge.
        offset = style.DEFAULT_AXIS_LABEL_OFFSET
        label_pos_along = lim * 0.92
        ax.text(label_pos_along, -offset, "x",
                color=style.TICK_LABEL_COLOR, alpha=0.95,
                fontsize=style.DEFAULT_AXIS_LABEL_SIZE + 2,
                fontweight="bold",
                ha="center", va="top", style="italic")
        ax.text(offset, label_pos_along, "y",
                color=style.TICK_LABEL_COLOR, alpha=0.95,
                fontsize=style.DEFAULT_AXIS_LABEL_SIZE + 2,
                fontweight="bold",
                ha="left", va="center", style="italic")

    if panel_title is not None:
        ax.set_title(panel_title, color=style.TICK_LABEL_COLOR,
                     fontsize=style.DEFAULT_SUBTITLE_FONT_SIZE,
                     loc="center", pad=8)
    if result_text is not None:
        ax.text(0.5, -0.08, result_text, transform=ax.transAxes,
                ha="center", va="top",
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                color=style.TICK_LABEL_COLOR,
                family="monospace")
