"""dsplot canonical style skeleton — kitchen-sink reference figure.

Every panel type and plottable exercised in a single 2x3 Figure; every
styling value flows from `dsplot.style.*` — zero literal style knobs.
"""
from __future__ import annotations

import os

import numpy as np

from dsplot import (
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
    ax_val, ay_val = A2
    panel.add(Dropline(start=(ax_val, ay_val), end=(ax_val, 0.0)))
    panel.add(Dropline(start=(ax_val, ay_val), end=(0.0, ay_val)))
    panel.add(
        VectorComponents(
            A2,
            first_axis="x",
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
    panel.add(
        Annotation(
            "a",
            xy=(ax_val + 0.30, ay_val / 2.0),
            color=style.PRIMARY_COLOR,
            fontsize=style.DEFAULT_LABEL_FONT_SIZE,
            fontweight="bold",
            ha="left",
            va="center",
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
            xy=(t_peak, y_peak),
            arrow_to=(t_peak, y_peak),
            color=style.HIGHLIGHT_COLOR,
            fontsize=style.DEFAULT_LABEL_FONT_SIZE,
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
            xy=(0.5, 0.92),
            transform="axes",
            color=style.HIGHLIGHT_COLOR,
            fontsize=style.DEFAULT_LABEL_FONT_SIZE,
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


def show() -> "Figure":
    """Build, render, and display the style skeleton figure."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
