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

from typing import Optional, Tuple, Union

from matplotlib.axes import Axes

from . import style


def _resolve_lo_hi(
    explicit: Optional[Tuple[float, float]],
    lim: Union[float, Tuple[float, float], None],
) -> Tuple[float, float]:
    """Resolve a per-axis (lo, hi) from an explicit xlim/ylim or the shared `lim`.

    `explicit` (an xlim/ylim tuple) wins; otherwise `lim` is interpreted as a
    (lo, hi) tuple if given that way, else as a symmetric scalar (-lim, lim).
    Keeping the resolution in one place lets x and y share the exact fallback
    logic so an equal-range pair preserves the honest equal-aspect geometry.
    """
    if explicit is not None:
        return float(explicit[0]), float(explicit[1])
    if isinstance(lim, tuple):
        return float(lim[0]), float(lim[1])
    if lim is None:
        lim = style.DEFAULT_VECTOR_LIM
    return -float(lim), float(lim)


def setup_vector_axes(
    ax: Axes,
    *,
    lim: Union[float, Tuple[float, float], None] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    panel_title: Optional[str] = None,
    result_text: Optional[str] = None,
    show_border: bool = True,
    axis_style: str = "line",
    axis_labels: bool = False,
    axis_label_size: Optional[float] = None,
    x_color: Optional[str] = None,
    y_color: Optional[str] = None,
    axis_alpha: Optional[float] = None,
    axis_linewidth: Optional[float] = None,
) -> None:
    # Per-axis limits: `lim` (float → symmetric, or a (lo, hi) tuple applied to
    # both axes) is the simple knob; explicit `xlim` / `ylim` override per axis.
    # An asymmetric pair shifts the origin off-center so a vector that lives in
    # one quadrant fills the cell instead of wasting the opposite quadrants —
    # while EQUAL ranges on both axes keep the equal-aspect geometry honest.
    x_lo, x_hi = _resolve_lo_hi(xlim, lim)
    y_lo, y_hi = _resolve_lo_hi(ylim, lim)
    if x_color is None:
        x_color = style.SPINE_COLOR
    if y_color is None:
        y_color = style.SPINE_COLOR
    if axis_alpha is None:
        axis_alpha = style.DEFAULT_AXIS_DECORATION_ALPHA
    if axis_linewidth is None:
        axis_linewidth = style.DEFAULT_AXIS_DECORATION_LINEWIDTH

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    if not show_border:
        for spine in ax.spines.values():
            spine.set_visible(False)

    inset = style.DEFAULT_AXIS_ARROW_INSET
    if axis_style == "arrow":
        for (tail, head), color in (
            (((x_lo + inset, 0.0), (x_hi - inset, 0.0)), x_color),
            (((0.0, y_lo + inset), (0.0, y_hi - inset)), y_color),
        ):
            ax.annotate(
                "", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="<|-|>",
                                color=color,
                                alpha=axis_alpha,
                                linewidth=max(axis_linewidth,
                                              style.DEFAULT_AXIS_ARROW_MIN_LINEWIDTH),
                                mutation_scale=style.DEFAULT_AXIS_ARROW_MUTATION,
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
        # Position each axis label a fixed fraction along its axis' POSITIVE
        # extent (x_hi / y_hi) so an asymmetric range still parks "x"/"y" near
        # the visible axis tips rather than off-screen.
        x_label_pos = x_hi * style.DEFAULT_AXIS_LABEL_POS_FRAC
        y_label_pos = y_hi * style.DEFAULT_AXIS_LABEL_POS_FRAC
        label_size = (
            axis_label_size if axis_label_size is not None
            else style.DEFAULT_AXIS_LABEL_SIZE + style.DEFAULT_AXIS_LABEL_SIZE_BUMP
        )
        ax.text(x_label_pos, -offset, "x",
                color=style.TICK_LABEL_COLOR, alpha=style.DEFAULT_AXIS_LABEL_ALPHA,
                fontsize=label_size,
                fontweight=style.DEFAULT_AXIS_LABEL_FONT_WEIGHT,
                ha="center", va="top", style=style.DEFAULT_AXIS_LABEL_FONT_STYLE)
        ax.text(offset, y_label_pos, "y",
                color=style.TICK_LABEL_COLOR, alpha=style.DEFAULT_AXIS_LABEL_ALPHA,
                fontsize=label_size,
                fontweight=style.DEFAULT_AXIS_LABEL_FONT_WEIGHT,
                ha="left", va="center", style=style.DEFAULT_AXIS_LABEL_FONT_STYLE)

    if panel_title is not None:
        ax.set_title(panel_title, color=style.TICK_LABEL_COLOR,
                     fontsize=style.DEFAULT_SUBTITLE_FONT_SIZE,
                     loc="center", pad=8)
    if result_text is not None:
        ax.text(0.5, style.DEFAULT_RESULT_TEXT_Y, result_text, transform=ax.transAxes,
                ha="center", va="top",
                fontsize=style.DEFAULT_LABEL_FONT_SIZE,
                color=style.TICK_LABEL_COLOR,
                family=style.DEFAULT_MONO_FONT_FAMILY)
