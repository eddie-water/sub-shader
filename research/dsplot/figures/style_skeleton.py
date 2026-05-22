"""dsplot canonical style skeleton — kitchen-sink reference figure.

Every panel type and plottable exercised in a single 2x3 Figure; every
styling value flows from `dsplot.style.*` — zero literal style knobs.
"""
from __future__ import annotations

import os

import numpy as np

from .. import (
    Annotation,
    Dropline,
    DynamicPanel,
    Figure,
    Heatmap,
    InteractivePanel,
    Spotlight,
    StaticPanel,
    StaticPanel3D,
    TimeSeries,
    Vector,
    VectorComponents,
    style,
)


# Dummy content constants (data, not style) — only literals allowed.
A2 = (2.0, 3.0)
A3 = (2.0, 3.0, 1.0)
LIM_2D = 4.0                 # symmetric ±4 fits A2 with headroom for label offset
LIM_3D = 4.0                 # symmetric ±4 fits A3 in all three axes
SAMPLE_RATE = 1000.0         # Hz
DURATION = 1.0               # s
SIGNAL_FREQ = 2.0            # Hz — ~2 cycles over DURATION
GRID_N = 32                  # heatmap grid size
BUILDUP_FRAMES = 5           # DynamicPanel frame count
SLIDER_FRAMES = 5            # InteractivePanel sweep length


def _build_sinusoid() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * SIGNAL_FREQ * t)
    return t, signal


def _build_gaussian(sigma: float) -> np.ndarray:
    coords = np.linspace(-1, 1, GRID_N)
    X, Y = np.meshgrid(coords, coords)
    return np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))


def _panel_static_vector_arrow_axes() -> StaticPanel:
    panel = StaticPanel(
        title="Vectors + Components",
        lim=LIM_2D,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )
    panel.add(
        VectorComponents(
            A2,
            first_axis="x",
            show_droplines=False,
            component_color=style.NEUTRAL_COLOR,
        )
    )
    panel.add(
        VectorComponents(
            A2,
            first_axis="y",
            show_droplines=False,
            component_color=style.NEUTRAL_COLOR,
        )
    )
    panel.add(
        Vector(
            A2,
            color=style.PRIMARY_COLOR,
            label="a",
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    return panel


def _panel_static_timeseries() -> StaticPanel:
    panel = StaticPanel(
        title="Time Series + Spotlight",
        axis_style="line",
        show_border=True,
    )
    t, signal = _build_sinusoid()
    peak_idx = int(np.argmax(signal))
    t_peak = float(t[peak_idx])
    y_peak = float(signal[peak_idx])
    panel.add(TimeSeries(signal, SAMPLE_RATE, color=style.PRIMARY_COLOR))
    panel.add(Dropline(start=(t_peak, y_peak), end=(t_peak, 0.0)))
    panel.add(Spotlight(mode="scatter", xy=(t_peak, y_peak)))
    panel.add(
        Annotation(
            "peak",
            xy=(t_peak + 0.06, y_peak + 0.18),
            arrow_to=(t_peak, y_peak),
            color=style.HIGHLIGHT_COLOR,
            fontsize=style.DEFAULT_ANNOTATION_FONT_SIZE,
            fontweight="bold",
        )
    )
    return panel


def _heatmap_in_panel_coords(sigma: float) -> Heatmap:
    return Heatmap(
        _build_gaussian(sigma=sigma),
        extent=(-1.0, 1.0, -1.0, 1.0),
        aspect="equal",
        vmax_percentile=100.0,
    )


def _panel_static_heatmap() -> StaticPanel:
    panel = StaticPanel(
        title="Heatmap + Annotation",
        show_border=True,
    )
    panel.add(_heatmap_in_panel_coords(sigma=0.45))
    panel.add(
        Annotation(
            "centroid",
            xy=(0.5, 0.94),
            transform="axes",
            color=style.HIGHLIGHT_COLOR,
            fontsize=style.DEFAULT_ANNOTATION_FONT_SIZE,
            fontweight="bold",
        )
    )
    return panel


def _panel_static_3d_vector() -> StaticPanel3D:
    panel = StaticPanel3D(
        lim_3d=LIM_3D,
        title="3D Vector",
    )
    ax_val, ay_val, az_val = A3
    panel.add(
        Vector(
            A3,
            color=style.PRIMARY_COLOR,
            label="a",
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )
    # Dashed projection rays — Dropline is 2D-only; in 3D use Vector with show_tip=False.
    panel.add(
        Vector(
            (0.0, 0.0, -az_val),
            origin=A3,
            color=style.DROPLINE_COLOR,
            linestyle="--",
            show_tip=False,
            alpha=style.DEFAULT_DROPLINE_ALPHA,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
        )
    )
    panel.add(
        Vector(
            (-ax_val, 0.0, 0.0),
            origin=(ax_val, ay_val, 0.0),
            color=style.DROPLINE_COLOR,
            linestyle="--",
            show_tip=False,
            alpha=style.DEFAULT_DROPLINE_ALPHA,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
        )
    )
    panel.add(
        Vector(
            (0.0, -ay_val, 0.0),
            origin=(ax_val, ay_val, 0.0),
            color=style.DROPLINE_COLOR,
            linestyle="--",
            show_tip=False,
            alpha=style.DEFAULT_DROPLINE_ALPHA,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
        )
    )
    return panel


def _buildup_frames() -> list[list]:
    _, signal = _build_sinusoid()
    n = len(signal)
    frames: list[list] = []
    for k in range(BUILDUP_FRAMES):
        cutoff = int((k + 1) / BUILDUP_FRAMES * n)
        window = np.zeros(n)
        window[:cutoff] = signal[:cutoff]
        frames.append(
            [TimeSeries(window, SAMPLE_RATE, color=style.PRIMARY_COLOR)]
        )
    return frames


def _panel_dynamic_buildup() -> DynamicPanel:
    return DynamicPanel(
        frames=_buildup_frames(),
        interval_ms=400,
        title="Animated Buildup",
        subtitle="cumulative sinusoid",
        axis_style="line",
        show_border=True,
    )


def _panel_static_buildup_final() -> StaticPanel:
    panel = StaticPanel(
        title="Animated Buildup",
        subtitle="final frame",
        axis_style="line",
        show_border=True,
    )
    for plottable in _buildup_frames()[-1]:
        panel.add(plottable)
    return panel


def _sigma_sweep_frames() -> list[list]:
    sigmas = np.linspace(0.20, 0.60, SLIDER_FRAMES)
    return [[_heatmap_in_panel_coords(sigma=float(sigma))] for sigma in sigmas]


def _panel_interactive_sigma_sweep() -> InteractivePanel:
    return InteractivePanel(
        frames=_sigma_sweep_frames(),
        slider=True,
        title="Sigma Sweep",
        subtitle="interactive Heatmap",
        show_border=True,
    )


def _panel_static_sigma_sweep_final() -> StaticPanel:
    panel = StaticPanel(
        title="Sigma Sweep",
        subtitle="final frame",
        show_border=True,
    )
    for plottable in _sigma_sweep_frames()[-1]:
        panel.add(plottable)
    return panel


def build_figure(*, static_export: bool = False) -> Figure:
    """Build the canonical 2x3 kitchen-sink Figure (un-rendered).

    static_export=True swaps the DynamicPanel + InteractivePanel for
    StaticPanels showing their final frame, so the saved PNG is free of
    animation flicker and slider UI chrome.
    """
    fig = Figure(
        n_rows=2,
        n_cols=3,
        width_ratios=[1, 2, 1],
        suptitle="dsplot — Canonical Style Skeleton",
    )
    fig.add_panel(_panel_static_vector_arrow_axes(), row=0, col=0)
    fig.add_panel(_panel_static_timeseries(),        row=0, col=1)
    fig.add_panel(_panel_static_heatmap(),           row=0, col=2)
    fig.add_panel(_panel_static_3d_vector(),         row=1, col=0, projection="3d")
    if static_export:
        fig.add_panel(_panel_static_buildup_final(),     row=1, col=1)
        fig.add_panel(_panel_static_sigma_sweep_final(), row=1, col=2)
    else:
        fig.add_panel(_panel_dynamic_buildup(),          row=1, col=1)
        fig.add_panel(_panel_interactive_sigma_sweep(),  row=1, col=2)
    return fig


def overlay_layout_guides(fig: Figure) -> None:
    """Draw dashed guide lines at every layout boundary and label each with
    the controlling ``dsplot.style.*`` constant. Call after ``fig.render()``.

    Turns the skeleton into a self-documenting blueprint: every gap,
    margin, font size, and panel-internal chrome line is labeled with
    the constant or matplotlib knob that sets it.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    mpl_fig = fig._mpl_fig
    gs = fig._gs
    # All boundaries derive from style.DEFAULT_MARGIN_INCHES via
    # Figure.render() — a single inch value converted to fractions per
    # figsize. Recompute the fractions here so guide lines land exactly
    # at the resolved subplots_adjust boundaries.
    fig_w, fig_h = mpl_fig.get_size_inches()
    margin_inches = style.DEFAULT_MARGIN_INCHES
    left_pad = margin_inches / fig_w
    right_pad = 1.0 - margin_inches / fig_w
    bottom_pad = margin_inches / fig_h
    suptitle_h_inches = (style.DEFAULT_SUPTITLE_FONT_SIZE / 72.0) * 1.4
    suptitle_y = 1.0 - margin_inches / fig_h
    top_pad = 1.0 - (2.0 * margin_inches + suptitle_h_inches) / fig_h

    guide_kwargs = dict(
        color=style.HIGHLIGHT_COLOR,
        alpha=0.55,
        linewidth=0.8,
        linestyle="--",
        transform=mpl_fig.transFigure,
        figure=mpl_fig,
    )
    label_kwargs = dict(
        color=style.HIGHLIGHT_COLOR,
        fontsize=style.DEFAULT_ANNOTATION_FONT_SIZE,
        fontweight="bold",
        family="monospace",
        transform=mpl_fig.transFigure,
    )

    # --- Figure perimeter (PNG edge after savefig pad_inches crop) ---
    mpl_fig.add_artist(mpatches.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        fill=False,
        edgecolor=style.HIGHLIGHT_COLOR,
        linewidth=0.8,
        linestyle="--",
        alpha=0.45,
        transform=mpl_fig.transFigure,
        figure=mpl_fig,
    ))
    mpl_fig.text(
        0.5, 0.992, "figure perimeter — savefig saves at figsize × dpi exactly",
        ha="center", va="top",
        **{k: v for k, v in label_kwargs.items() if k not in ("ha", "va")},
    )

    # --- Horizontal margin lines ---
    text_label_kw = {k: v for k, v in label_kwargs.items() if k not in ("ha", "va")}
    for y, lbl in (
        (suptitle_y, f"suptitle top = 1 - margin/h = {suptitle_y:.3f}"),
        (top_pad,    f"top_pad = {top_pad:.3f} (panel-top)"),
        (bottom_pad, f"bottom margin = margin/h = {bottom_pad:.3f}"),
    ):
        mpl_fig.add_artist(mlines.Line2D([0.0, 1.0], [y, y], **guide_kwargs))
        mpl_fig.text(
            left_pad + 0.005, y + 0.005, lbl,
            ha="left", va="bottom",
            **text_label_kw,
        )

    # --- Vertical margin lines ---
    for x, lbl, anchor_ha in (
        (left_pad,  f"left margin = margin/w = {left_pad:.3f}",   "left"),
        (right_pad, f"right margin = 1 - margin/w = {right_pad:.3f}", "right"),
    ):
        mpl_fig.add_artist(mlines.Line2D([x, x], [0.0, 1.0], **guide_kwargs))
        x_offset = 0.005 if anchor_ha == "left" else -0.005
        mpl_fig.text(
            x + x_offset, top_pad - 0.005, lbl,
            ha=anchor_ha, va="top",
            **text_label_kw,
        )

    # --- HSPACE (inter-row gap) ---
    row_bounds = [gs[r, 0].get_position(mpl_fig) for r in range(gs.nrows)]
    if len(row_bounds) >= 2:
        gap_y = (row_bounds[0].y0 + row_bounds[1].y1) / 2.0
        mpl_fig.add_artist(mlines.Line2D([left_pad, right_pad], [gap_y, gap_y], **guide_kwargs))
        mpl_fig.text(
            left_pad + 0.005, gap_y + 0.005,
            f"DEFAULT_HSPACE = {style.DEFAULT_HSPACE}",
            **label_kwargs,
        )

    # --- WSPACE markers (between every pair of columns) ---
    col_bounds = [gs[0, c].get_position(mpl_fig) for c in range(gs.ncols)]
    for c in range(len(col_bounds) - 1):
        gap_x = (col_bounds[c].x1 + col_bounds[c + 1].x0) / 2.0
        mpl_fig.add_artist(mlines.Line2D(
            [gap_x, gap_x], [bottom_pad, top_pad], **guide_kwargs,
        ))
        mpl_fig.text(
            gap_x + 0.003, bottom_pad + 0.010,
            f"DEFAULT_WSPACE = {style.DEFAULT_WSPACE}",
            **label_kwargs,
        )

    # --- Per-panel chrome (title-Y, subtitle-Y) — sampled from row-0 col-0
    # which uses the canonical StaticPanel chrome. ---
    if row_bounds:
        bb = gs[0, 0].get_position(mpl_fig)
        # Title sits at axes-relative y=DEFAULT_TITLE_Y above the axes box,
        # so figure-y = bb.y1 + (DEFAULT_TITLE_Y - 1.0) * bb.height
        title_fig_y = bb.y1 + (style.DEFAULT_TITLE_Y - 1.0) * bb.height
        mpl_fig.add_artist(mlines.Line2D(
            [bb.x0, bb.x1], [title_fig_y, title_fig_y], **guide_kwargs,
        ))
        mpl_fig.text(
            bb.x0 - 0.003, title_fig_y,
            f"DEFAULT_TITLE_Y = {style.DEFAULT_TITLE_Y} | DEFAULT_TITLE_FONT_SIZE = {style.DEFAULT_TITLE_FONT_SIZE}",
            ha="right", va="center",
            **{k: v for k, v in label_kwargs.items() if k not in ("ha", "va")},
        )
        # Subtitle sits at axes-relative y=DEFAULT_SUBTITLE_Y (negative,
        # below the axes box).
        bb2 = gs[1, 1].get_position(mpl_fig)
        subtitle_fig_y = bb2.y0 + style.DEFAULT_SUBTITLE_Y * bb2.height
        mpl_fig.add_artist(mlines.Line2D(
            [bb2.x0, bb2.x1], [subtitle_fig_y, subtitle_fig_y], **guide_kwargs,
        ))
        mpl_fig.text(
            bb2.x1 + 0.003, subtitle_fig_y,
            f"DEFAULT_SUBTITLE_Y = {style.DEFAULT_SUBTITLE_Y} | DEFAULT_SUBTITLE_FONT_SIZE = {style.DEFAULT_SUBTITLE_FONT_SIZE}",
            ha="left", va="center",
            **{k: v for k, v in label_kwargs.items() if k not in ("ha", "va")},
        )

    # --- Constants legend (bottom-left, below bottom margin) ---
    legend_lines = [
        f"DEFAULT_LABEL_FONT_SIZE       = {style.DEFAULT_LABEL_FONT_SIZE}",
        f"DEFAULT_ANNOTATION_FONT_SIZE  = {style.DEFAULT_ANNOTATION_FONT_SIZE}",
        f"DEFAULT_VECTOR_LINEWIDTH      = {style.DEFAULT_VECTOR_LINEWIDTH}",
        f"DEFAULT_VECTOR_BOLD_LINEWIDTH = {style.DEFAULT_VECTOR_BOLD_LINEWIDTH}",
        f"DEFAULT_DPI                   = {style.DEFAULT_DPI}",
        f"DEFAULT_HEATMAP_CMAP          = {style.DEFAULT_HEATMAP_CMAP!r}",
        f"PRIMARY_COLOR    = {style.PRIMARY_COLOR}",
        f"HIGHLIGHT_COLOR  = {style.HIGHLIGHT_COLOR}",
        f"NEUTRAL_COLOR    = {style.NEUTRAL_COLOR}",
        f"BG_COLOR         = {style.BG_COLOR}",
    ]
    mpl_fig.text(
        0.005, 0.005, "\n".join(legend_lines),
        color=style.TICK_LABEL_COLOR,
        fontsize=style.DEFAULT_ANNOTATION_FONT_SIZE - 1,
        family="monospace",
        ha="left", va="bottom",
        transform=mpl_fig.transFigure,
    )


def show() -> "Figure":
    """Build, render, and display the style skeleton figure."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
