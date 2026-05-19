"""lego_demo — exercises Figure.compose with the panel-unit OOP layer.

Two 5-unit rows: row 1 = StaticPanel + TimeSeriesPanel(3) + HeatmapPanel;
row 2 = StaticPanel3D + CompositePanel(3, 3-row stack) + HeatmapPanel.
"""
from __future__ import annotations

import os

import numpy as np

from dsplot import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    StaticPanel,
    StaticPanel3D,
    TimeSeries,
    TimeSeriesPanel,
    Vector,
    style,
)


# Dummy content constants (data, not style).
SAMPLE_RATE = 1000.0
DURATION = 1.0
SIGNAL_FREQ = 3.0
GRID_N = 32
LIM_3D = 1.0
A3 = (0.6, 0.6, 0.6)


def _build_sine() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    sin = np.sin(2 * np.pi * SIGNAL_FREQ * t)
    return t, sin


def _build_gaussian(sigma: float) -> np.ndarray:
    coords = np.linspace(-1, 1, GRID_N)
    X, Y = np.meshgrid(coords, coords)
    return np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))


def build_figure() -> Figure:
    """Compose the lego_demo Figure: two 5-unit rows.

    Row 1 — three side-by-side panels (Vector, Signal, Field).
    Row 2 — StaticPanel3D, a 3-row CompositePanel (Field / Signal / Field),
            and a trailing HeatmapPanel.
    """
    _, sin = _build_sine()
    gaussian = _build_gaussian(0.4)

    # --- Row 1: 1 + 3 + 1 = 5 units ---
    static_vec = StaticPanel(title="Vector")
    static_vec.add(Vector(A3[:2], color=style.PRIMARY_COLOR))

    ts_panel = TimeSeriesPanel(title="Signal")
    ts_panel.add(TimeSeries(sin, SAMPLE_RATE, color=style.PRIMARY_COLOR))

    field_panel = HeatmapPanel(title="Field")
    field_panel.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal")
    )

    row1 = [static_vec, ts_panel, field_panel]

    # --- Row 2: 1 + 3 + 1 = 5 units ---
    panel_3d = StaticPanel3D(title="3D", lim_3d=LIM_3D)
    panel_3d.add(
        Vector(
            A3,
            color=style.PRIMARY_COLOR,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        )
    )

    # Composite: 3-row stack inside a 3-wide outer cell. Each inner row spans
    # the full 3-unit width (units=(3, 1)) so the row widths agree.
    inner_top = HeatmapPanel(units=(3, 1))
    inner_top.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="auto")
    )
    inner_mid = TimeSeriesPanel(units=(3, 1))
    inner_mid.add(TimeSeries(sin, SAMPLE_RATE, color=style.PRIMARY_COLOR))
    inner_bot = HeatmapPanel(units=(3, 1))
    inner_bot.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="auto")
    )
    composite = CompositePanel(
        rows=[[inner_top], [inner_mid], [inner_bot]],
        units=(3, 1),
    )

    trailing_field = HeatmapPanel(title="Field")
    trailing_field.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal")
    )

    row2 = [panel_3d, composite, trailing_field]

    return Figure.compose(
        rows=[row1, row2],
        suptitle="dsplot — Lego Composition Demo",
    )


def show() -> Figure:
    """Build, render, and display the lego_demo figure in a notebook cell."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
