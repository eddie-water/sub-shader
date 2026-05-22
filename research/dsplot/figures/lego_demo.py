"""lego_demo — exercises Figure.compose with the panel-unit OOP layer.

Two 4-unit rows: row 1 = StaticPanel + HeatmapPanel(2) + HeatmapPanel;
row 2 = StaticPanel3D + CompositePanel(2, 3-row stack) + HeatmapPanel.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import chirp

from .. import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    StaticPanel,
    StaticPanel3D,
    Stem,
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

# Aggressive iterated-log chirp: 2 Hz → 100 Hz over 1 s. The inst freq
# follows f0 * (f1/f0)^((t/t1)^N) with N >> 1, so the curve stays near
# f0 for most of the duration and snaps up to f1 only in the final
# fraction — a much more aggressive ramp than a plain logarithmic chirp.
CHIRP_DURATION_S = 1.0
CHIRP_F0_HZ = 2.0
CHIRP_F1_HZ = 100.0
CHIRP_SR_HZ = 8_000.0
CHIRP_RAMP_POWER = 2.5


def _build_chirp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterated-log chirp: inst freq follows f0 * (f1/f0)^((t/t1)^N) with
    N = CHIRP_RAMP_POWER. For N > 1 the curve hugs f0 for most of the
    duration and snaps up to f1 only in the final fraction — far more
    aggressive than scipy.signal.chirp's plain 'logarithmic' method.

    Signal is reconstructed by numerically integrating inst_f to a phase
    track and taking cos(phase).
    """
    n = int(CHIRP_SR_HZ * CHIRP_DURATION_S)
    t = np.arange(n) / CHIRP_SR_HZ
    tau = (t / CHIRP_DURATION_S) ** CHIRP_RAMP_POWER
    inst_f = CHIRP_F0_HZ * (CHIRP_F1_HZ / CHIRP_F0_HZ) ** tau
    phase = 2.0 * np.pi * np.cumsum(inst_f) / CHIRP_SR_HZ
    sig = np.cos(phase).astype(np.float64)
    return t, sig, inst_f


def _build_sine() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    sin = np.sin(2 * np.pi * SIGNAL_FREQ * t)
    return t, sin


def _build_gaf(n: int = GRID_N, base_freq_hz: float = 2.0) -> np.ndarray:
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
        y_label_side="right",
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        twin_y=True,
        twin_y_label="f (Hz)",
        twin_yticks=[CHIRP_F0_HZ, 10.0, CHIRP_F1_HZ],
        twin_ytick_labels=["2", "10", "100"],
        twin_ylim=(CHIRP_F0_HZ, CHIRP_F1_HZ),
        twin_y_side="left",
    )
    wide_signal.add(
        TimeSeries(chirp_sig, CHIRP_SR_HZ, color=style.NEUTRAL_COLOR, alpha=0.75)
    )
    wide_signal.add_twin(
        Line(
            chirp_t, chirp_inst_f,
            color=style.BG_COLOR,
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

    # Composite: 4-row foursome inside a 2-wide outer cell, stacked with no
    # gaps (hspace=0). Each inner row is ~1/4 the height of a top-level
    # panel cell. Pattern: continuous sine A → its discrete stem samples
    # → continuous sine B at 2×F in the secondary color → its discrete
    # stem samples. The stem rows sample the curve directly above them so
    # the analog-to-discrete relationship is visually exact.
    sin_freq_a = SIGNAL_FREQ            # 3 Hz, primary
    sin_freq_b = SIGNAL_FREQ * 2.0      # 6 Hz, secondary
    t_dense = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    sin_a_dense = np.sin(2 * np.pi * sin_freq_a * t_dense)
    sin_b_dense = np.sin(2 * np.pi * sin_freq_b * t_dense)
    stem_n = 32
    stem_t = np.linspace(0.0, DURATION, stem_n, endpoint=False)
    stem_a = np.sin(2 * np.pi * sin_freq_a * stem_t)
    stem_b = np.sin(2 * np.pi * sin_freq_b * stem_t)

    inner_sine_a = TimeSeriesPanel(units=(2, 1), yticks=[-1.0, 0.0, 1.0])
    inner_sine_a.add(
        TimeSeries(sin_a_dense, SAMPLE_RATE, color=style.PRIMARY_COLOR)
    )
    inner_stem_a = TimeSeriesPanel(units=(2, 1), yticks=[-1.0, 0.0, 1.0])
    inner_stem_a.add(Stem(stem_t, stem_a, color=style.PRIMARY_COLOR))
    inner_sine_b = TimeSeriesPanel(units=(2, 1), yticks=[-1.0, 0.0, 1.0])
    inner_sine_b.add(
        TimeSeries(sin_b_dense, SAMPLE_RATE, color=style.SECONDARY_COLOR)
    )
    inner_stem_b = TimeSeriesPanel(
        units=(2, 1),
        x_label="time (s)",
        xticks=[0.0, 0.5, 1.0],
        yticks=[-1.0, 0.0, 1.0],
    )
    inner_stem_b.add(Stem(stem_t, stem_b, color=style.SECONDARY_COLOR))
    composite = CompositePanel(
        rows=[
            [inner_sine_a],
            [inner_stem_a],
            [inner_sine_b],
            [inner_stem_b],
        ],
        units=(2, 1),
        title="Composite Plots",
        share_x=True,
        hspace=0.0,
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
        suptitle="dsplot — Sample Template",
        show_cell_borders=True,
    )


def show() -> Figure:
    """Build, render, and display the lego_demo figure in a notebook cell."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
