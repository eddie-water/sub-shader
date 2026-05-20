"""lego_demo — exercises Figure.compose with the panel-unit OOP layer.

Two 4-unit rows: row 1 = StaticPanel + HeatmapPanel(2) + HeatmapPanel;
row 2 = StaticPanel3D + CompositePanel(2, 3-row stack) + HeatmapPanel.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import chirp

from dsplot import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    StaticPanel,
    StaticPanel3D,
    TimeSeries,
    TimeSeriesPanel,
    Vector,
    VectorComponents,
    style,
)


# Dummy content constants (data, not style).
SAMPLE_RATE = 1000.0
DURATION = 1.0
SIGNAL_FREQ = 3.0
GRID_N = 64
# 2D vector tip is (3, 3) in a (-4, 4) box; 3D version is (3, 3, 4) in a
# slightly larger (-5, 5) box so the z tip clears the cube face.
A2 = (3.0, 3.0)
A3 = (3.0, 3.0, 2.0)
LIM_3D = 5.0

# Extreme log chirp: 2 Hz → 20 kHz (factor of 10,000). Plotted on a linear
# y-axis the curve is nearly flat for the first ~75% and snaps upward in
# the final quartile — the visual "hockey stick" the user wants.
CHIRP_DURATION_S = 1.0
CHIRP_F0_HZ = 2.0
CHIRP_F1_HZ = 20_000.0
CHIRP_SR_HZ = 48_000.0


def _build_chirp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logarithmic chirp 10 Hz → 100 Hz over CHIRP_DURATION_S.

    Returns (t, signal, instantaneous_frequency).
    """
    n = int(CHIRP_SR_HZ * CHIRP_DURATION_S)
    t = np.arange(n) / CHIRP_SR_HZ
    sig = chirp(t, f0=CHIRP_F0_HZ, f1=CHIRP_F1_HZ,
                t1=CHIRP_DURATION_S, method="logarithmic").astype(np.float64)
    # For a log chirp, inst. freq follows f(t) = f0 * (f1/f0)^(t/t1).
    inst_f = CHIRP_F0_HZ * (CHIRP_F1_HZ / CHIRP_F0_HZ) ** (t / CHIRP_DURATION_S)
    return t, sig, inst_f


def _build_sine() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    sin = np.sin(2 * np.pi * SIGNAL_FREQ * t)
    return t, sin


def _build_gaf(n: int = GRID_N, base_freq_hz: float = 4.0) -> np.ndarray:
    """Gramian Angular Field of a single-frequency sine — produces the woven
    diagonal-band pattern seen in the pyts plot_single_gaf example.

    The raw GAF lands in [-1, 1] with peaks at +1 and troughs at -1; under the
    inferno cmap troughs map to near-black and the weave between peaks gets
    lost. Remapping to a brighter floor lifts the troughs into the visible
    cmap range while preserving the relative pattern.
    """
    t = np.linspace(0, 1, n)
    sig = np.sin(2 * np.pi * base_freq_hz * t)
    phi = np.arccos(np.clip(sig, -1.0, 1.0))
    raw = np.cos(phi[:, None] + phi[None, :])
    return (raw + 1.0) / 2.0 * 0.9 + 0.1


def _build_gaussian(sigma: float) -> np.ndarray:
    coords = np.linspace(-1, 1, GRID_N)
    X, Y = np.meshgrid(coords, coords)
    return np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))


def build_figure() -> Figure:
    """Compose the lego_demo Figure: two 4-unit rows.

    Row 1 — Vector, 2-unit Heatmap, Field.
    Row 2 — StaticPanel3D, a 3-row CompositePanel (Field / Signal / Field)
            occupying 2 units, and a trailing HeatmapPanel.
    """
    _, sin = _build_sine()
    chirp_t, chirp_sig, chirp_inst_f = _build_chirp()
    gaf_field = _build_gaf()
    gaussian = _build_gaussian(0.4)

    # --- Row 1: 1 + 2 + 1 = 4 units ---
    static_vec = StaticPanel(
        title="2D Vector Plot",
        axis_labels=True,
        show_ticks=True,
        show_grid=True,
    )
    static_vec.add(VectorComponents(
        A2,
        from_origin=True,
        component_color=style.DROPLINE_COLOR,
        dropline_color=style.DROPLINE_COLOR,
        label_x="aₓ",
        label_y="aᵧ",
    ))
    static_vec.add(Vector(
        A2,
        color=style.PRIMARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        label="a",
        zorder=4,
    ))

    # Inst-freq twin axis: log range spanning the chirp; inst_f is plotted
    # twice — a wider near-black backing line first, then the orange line on
    # top — so the orange has a soft dark border that lifts it off the gray
    # waveform underneath.
    wide_signal = TimeSeriesPanel(
        title="Time Series + Linear Plot",
        units=(2, 1),
        x_label="time (s)",
        y_label="amplitude",
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        twin_y=True,
        twin_y_label="f (Hz)",
        twin_yticks=[CHIRP_F0_HZ, 200.0, CHIRP_F1_HZ],
        twin_ytick_labels=["2", "200", "20k"],
        twin_ylim=(CHIRP_F0_HZ, CHIRP_F1_HZ),
        twin_y_side="right",
    )
    wide_signal.add(
        TimeSeries(chirp_sig, CHIRP_SR_HZ, color=style.NEUTRAL_COLOR)
    )
    wide_signal.add_twin(
        Line(
            chirp_t, chirp_inst_f,
            color="#000000",
            linewidth=style.INST_FREQ_LINEWIDTH + 3.5,
            alpha=0.95,
        )
    )
    wide_signal.add_twin(
        Line(
            chirp_t, chirp_inst_f,
            color=style.PRIMARY_COLOR,
            linewidth=style.INST_FREQ_LINEWIDTH + 1.0,
            alpha=style.INST_FREQ_ALPHA,
        )
    )

    field_panel = HeatmapPanel(
        title="Heatmap 1",
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    field_panel.add(
        Heatmap(gaf_field, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal")
    )

    row1 = [static_vec, wide_signal, field_panel]

    # --- Row 2: 1 + 2 + 1 = 4 units ---
    panel_3d = StaticPanel3D(
        title="3D Vector Plot",
        lim_3d=LIM_3D,
        spine_extension=1.7,
    )
    # Dashed component staircase (x → y → z) projecting the vector onto each
    # axis. Matches the 2D vector's dashed component style: same color
    # (DROPLINE_COLOR), same dash linestyle, same vector linewidth + alpha.
    component_kwargs = dict(
        color=style.DROPLINE_COLOR,
        linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
        alpha=0.95,
        linestyle="--",
        show_tip=False,
        zorder=2,
    )
    panel_3d.add(Vector(
        (A3[0], 0.0, 0.0), origin=(0.0, 0.0, 0.0), **component_kwargs
    ))
    panel_3d.add(Vector(
        (0.0, A3[1], 0.0), origin=(A3[0], 0.0, 0.0), **component_kwargs
    ))
    panel_3d.add(Vector(
        (0.0, 0.0, A3[2]), origin=(A3[0], A3[1], 0.0), **component_kwargs
    ))
    panel_3d.add(
        Vector(
            A3,
            color=style.PRIMARY_COLOR,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
            show_arrowhead=True,
            zorder=5,
        )
    )

    # Composite: 3-row stack inside a 2-wide outer cell. Each inner row spans
    # the full 2-unit width so row widths agree.
    inner_top = HeatmapPanel(
        units=(2, 1),
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    inner_top.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="auto")
    )
    inner_mid = TimeSeriesPanel(
        units=(2, 1),
        x_label="time (s)",
        y_label="amplitude",
        xticks=[0.0, 0.5, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    inner_mid.add(TimeSeries(sin, SAMPLE_RATE, color=style.PRIMARY_COLOR))
    inner_bot = HeatmapPanel(
        units=(2, 1),
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    inner_bot.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="auto")
    )
    composite = CompositePanel(
        rows=[[inner_top], [inner_mid], [inner_bot]],
        units=(2, 1),
        title="Composite Plots",
        share_x=True,
    )

    trailing_field = HeatmapPanel(
        title="Heatmap 2",
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    trailing_field.add(
        Heatmap(gaussian, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal")
    )

    row2 = [panel_3d, composite, trailing_field]

    return Figure.compose(
        rows=[row1, row2],
        suptitle="dsplot — Lego Composition Demo",
        show_cell_borders=True,
    )


def show() -> Figure:
    """Build, render, and display the lego_demo figure in a notebook cell."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
