"""sample_template — kitchen-sink dsplot showcase (v3).

Exercises Panel kinds and Plottables in a four-row composition. Visual style
derives from the v52 lego_demo language via dsplot.style — no hardcoded
style literals.

Three-mode contract (the canonical convention every figure module should
implement):
    render(output_dir, output_filename) -> str   # production PNG on disk
    show() -> Figure                              # notebook inline display
    embed(target) -> Figure                       # caller-provided container

Iterate the visual style by bumping output_filename: v1.png -> v2.png -> vN.png.

v3 Layout (4 columns wide)
--------------------------
Row 0  SuptitlePanel(4,1)
Row 1  StaticPanel(1,1) 2D | StaticPanel3D(1,1) 3D | HeatmapPanel(1,1) GAF | HeatmapPanel(1,1) centroid
Row 2  TimeSeriesPanel(4,1) — chirp + twin inst-freq Line
Row 3  CompositePanel(2,1) — 4 stacked stem rows  |  CompositePanel(1,1) — title + jargon body  |  StaticPanel(1,1) — vector projection

Coverage
--------
Panel kinds   : StaticPanel · StaticPanel3D · TimeSeriesPanel · HeatmapPanel · CompositePanel · TextPanel · SuptitlePanel
Plottables    : Vector · VectorComponents · Heatmap · TimeSeries · Line · Stem

DynamicPanel / InteractivePanel and the decoration plottables (Annotation,
Dropline, Spotlight) are out of scope for v3 — they re-enter in a later
revision once the v3 visual language is locked.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import matplotlib as mpl
import numpy as np

from .. import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    StaticPanel,
    StaticPanel3D,
    Stem,
    SuptitlePanel,
    TextPanel,
    TimeSeries,
    TimeSeriesPanel,
    Vector,
    VectorComponents,
    nb_compact_style,
    style,
)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
ROW_WIDTH_UNITS = 4
LIM_2D = 6.0
LIM_3D = 5.0
GRID_N = 64

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------
A2 = (3.0, 4.0)
A3 = (3.0, 3.0, 2.0)

CHIRP_SR_HZ = 8_000.0
CHIRP_DURATION_S = 1.0
CHIRP_F0_HZ = 20.0
CHIRP_F1_HZ = 100.0
CHIRP_RAMP_POWER = 2.0
# Breathing-room factor for the chirp amp ylim. Main amp is plotted in [-1, 1]
# but the spine extends past the signal so it doesn't kiss the borders. The
# twin (inst-freq) ylim scales identically so a y-position on the twin always
# corresponds to the same y-position on the main — i.e. amp=-1 line coincides
# with twin tick at 20 Hz, amp=+1 line coincides with 100 Hz tick.
CHIRP_AMP_LIM = 1.05

STEM_N = 100
STEM_SINE_HZ_LOW = 5.0
STEM_SINE_HZ_HIGH = 10.0
STEM_YLIM = (-2.0, 2.0)

PROJ_A = (3.0, 4.0)
PROJ_B = (5.0, 1.5)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def _build_chirp_placeholder() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterated-log chirp (signal, inst_freq_hz, t) at CHIRP_SR_HZ."""
    n = int(CHIRP_SR_HZ * CHIRP_DURATION_S)
    t = np.arange(n) / CHIRP_SR_HZ
    tau = (t / CHIRP_DURATION_S) ** CHIRP_RAMP_POWER
    inst_f = CHIRP_F0_HZ * (CHIRP_F1_HZ / CHIRP_F0_HZ) ** tau
    phase = 2.0 * np.pi * np.cumsum(inst_f) / CHIRP_SR_HZ
    sig = np.cos(phase).astype(np.float64)
    return sig, inst_f, t


def _build_gaussian(sigma: float = 0.35) -> np.ndarray:
    """Centroid heatmap — single peak with Gaussian falloff, centered."""
    coords = np.linspace(-1.0, 1.0, GRID_N)
    X, Y = np.meshgrid(coords, coords)
    return np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))


def _build_gaf(n: int = GRID_N, base_freq_hz: float = 4.0) -> np.ndarray:
    """Gramian Angular Field weave — diagonal-band interference pattern.

    Raw cos-sum is in [-1, 1]; remap to [0, 1] then apply a <1 power so
    midrange lifts brighter (most of the field reads as purple → orange)
    while the trough still bottoms out at 0 (black) and the peak stays at 1
    (yellow). Power < 1 makes the black-to-yellow transition QUICK — narrow
    black bands around the troughs, then a fast climb through the inferno
    palette to bright peaks.
    """
    t = np.linspace(0, 1, n)
    sig = np.sin(2.0 * np.pi * base_freq_hz * t)
    phi = np.arccos(np.clip(sig, -1.0, 1.0))
    raw = np.cos(phi[:, None] + phi[None, :])
    return ((raw + 1.0) / 2.0) ** 0.7


# ---------------------------------------------------------------------------
# Row 1 — four 1×1 panels
# ---------------------------------------------------------------------------
def _r1_vector_2d() -> StaticPanel:
    panel = StaticPanel(
        title="2D Vector",
        units=(1, 1),
        lim=LIM_2D,
        axis_labels=False,
        show_ticks=True,
        show_tick_labels=True,
        show_grid=True,
        x_label="x",
        y_label="y",
        tick_positions=(-4.0, -2.0, 0.0, 2.0, 4.0),
    )
    panel.add(VectorComponents(
        A2,
        from_origin=True,
        component_color=style.DROPLINE_COLOR,
        dropline_color=style.DROPLINE_COLOR,
        label_x="aₓ",
        label_y="aᵧ",
    ))
    panel.add(Vector(
        A2,
        color=style.PRIMARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        label="a",
        zorder=4,
    ))
    return panel


def _r1_vector_3d() -> StaticPanel3D:
    panel = StaticPanel3D(
        units=(1, 1),
        title="3D Vector",
        lim_3d=LIM_3D,
        spine_extension=1.7,
        show_spine_ticks=False,
    )
    component_kwargs = dict(
        color=style.DROPLINE_COLOR,
        linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
        alpha=style.DEFAULT_DROPLINE_ALPHA,
        linestyle="--",
        show_tip=False,
        zorder=2,
    )
    panel.add(Vector((A3[0], 0.0, 0.0), origin=(0.0, 0.0, 0.0), **component_kwargs))
    panel.add(Vector((0.0, A3[1], 0.0), origin=(A3[0], 0.0, 0.0), **component_kwargs))
    panel.add(Vector((0.0, 0.0, A3[2]), origin=(A3[0], A3[1], 0.0), **component_kwargs))
    panel.add(Vector(
        A3,
        color=style.PRIMARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        show_arrowhead=True,
        zorder=5,
    ))
    return panel


def _r1_heatmap_weave() -> HeatmapPanel:
    panel = HeatmapPanel(
        units=(1, 1),
        title="Heatmap - Weave Filter",
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(Heatmap(
        _build_gaf(),
        extent=(-1.0, 1.0, -1.0, 1.0),
        aspect="equal",
    ))
    return panel


def _r1_heatmap_centroid() -> HeatmapPanel:
    panel = HeatmapPanel(
        units=(1, 1),
        title="Heatmap - Centroid",
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(Heatmap(
        _build_gaussian(),
        extent=(-1.0, 1.0, -1.0, 1.0),
        aspect="equal",
    ))
    return panel


# ---------------------------------------------------------------------------
# Row 2 — full-width chirp with twin inst-freq Line overlay
# ---------------------------------------------------------------------------
def _r2_chirp_with_inst_freq() -> TimeSeriesPanel:
    sig, inst_f, t = _build_chirp_placeholder()
    panel = TimeSeriesPanel(
        units=(ROW_WIDTH_UNITS, 1),
        title="Chirp + Twin Inst-Freq Line",
        x_label="s",
        y_label="amp",
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        yticks=[-1.0, 1.0],
        ylim=(-CHIRP_AMP_LIM, CHIRP_AMP_LIM),
        show_xticklabels=True,
        show_yticklabels=True,
        twin_y=True,
        twin_y_side="right",
        twin_y_label="hz",
        twin_yticks=[CHIRP_F0_HZ, CHIRP_F1_HZ],
        twin_ylim=(
            (CHIRP_F0_HZ + CHIRP_F1_HZ) / 2.0
            - (CHIRP_F1_HZ - CHIRP_F0_HZ) / 2.0 * CHIRP_AMP_LIM,
            (CHIRP_F0_HZ + CHIRP_F1_HZ) / 2.0
            + (CHIRP_F1_HZ - CHIRP_F0_HZ) / 2.0 * CHIRP_AMP_LIM,
        ),
    )
    panel.add(TimeSeries(sig, CHIRP_SR_HZ, color=style.NEUTRAL_COLOR, alpha=0.75))
    panel.add_twin(Line(
        t, inst_f,
        color=style.PRIMARY_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 1.0,
        alpha=style.INST_FREQ_ALPHA,
    ))
    return panel


# ---------------------------------------------------------------------------
# Row 3 — stems composite | jargon panel | vector projection
# ---------------------------------------------------------------------------
def _stem_row_panel(
    t: np.ndarray,
    y: np.ndarray,
    *,
    color: str | None = None,
    color_per_sample: Optional[np.ndarray] = None,
    show_xticks: bool = False,
) -> TimeSeriesPanel:
    """Inner stem row for the composite.

    Y axis stripped entirely (no ticks, no label) so the rows read as a
    pure waveform stack — the wave shape itself identifies the row. Only the
    bottom row carries x-axis decoration (share_x strips it elsewhere) and
    its last x-tick sits at STEM_N (= the right edge of the plot).

    `color`: uniform Stem color (mutually exclusive with color_per_sample).
    `color_per_sample`: per-sample hex colors — emits N single-sample Stems
    so the row reads as a colormap-encoded magnitude scan (e.g. inferno over
    |y|).
    """
    panel = TimeSeriesPanel(
        units=(2, 1),
        x_label="n" if show_xticks else None,
        y_label=None,
        xticks=[0, STEM_N // 4, STEM_N // 2, 3 * STEM_N // 4, STEM_N] if show_xticks else [],
        yticks=[],
        show_xticklabels=show_xticks,
        show_yticklabels=False,
        ylim=STEM_YLIM,
        xlim=(0.0, float(STEM_N)),
    )
    if color_per_sample is not None:
        for i in range(len(t)):
            panel.add(Stem(
                np.array([float(t[i])]),
                np.array([float(y[i])]),
                color=str(color_per_sample[i]),
            ))
    else:
        panel.add(Stem(t, y, color=color))
    return panel


def _r3_stem_quartet() -> CompositePanel:
    n = STEM_N
    t = np.arange(n, dtype=np.float64)

    # Row 1 — square wave (unitary ±1, NEUTRAL_COLOR / white)
    square = np.where(t < n // 2, 1.0, -1.0)

    # Rows 2 & 3 — 5 Hz and 10 Hz sines across n=100 samples (sample rate
    # implicit = n Hz → freq cycles per second = freq cycles per n samples).
    sin_low = np.sin(2.0 * np.pi * STEM_SINE_HZ_LOW * t / n)
    sin_high = np.sin(2.0 * np.pi * STEM_SINE_HZ_HIGH * t / n)

    # Row 4 — triangle wave: -1 at t=0, +1 at midpoint, -1 at end.
    # Per-sample inferno colormap encodes magnitude so the row reads as a
    # heat-scan: dim at the zero-crossings, brightest at the apex/troughs.
    triangle = 1.0 - 4.0 * np.abs(t / (n - 1) - 0.5)
    mag = np.abs(triangle)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    cmap = mpl.colormaps["inferno"]
    triangle_colors = np.array([mpl.colors.to_hex(cmap(m)) for m in mag_norm])

    return CompositePanel(
        units=(2, 1),
        title="Stem Quartet",
        rows=[
            [_stem_row_panel(t, square, color=style.NEUTRAL_COLOR)],
            [_stem_row_panel(t, sin_low, color=style.SECONDARY_COLOR)],
            [_stem_row_panel(t, sin_high, color=style.PRIMARY_COLOR)],
            [_stem_row_panel(t, triangle, color_per_sample=triangle_colors, show_xticks=True)],
        ],
        share_x=True,
    )


JARGON_TITLE = "Text Panel"
JARGON_BODY = (
    "There are symphonies everywhere for those with the eyes to see them\n"
    "\n"
    "Sub Shader - a real-time feature extraction signal processing pipeline accelerated by parallel computing\n"
    "\n"
    "Or in plain english, an audio visualizer that allows you to see what the original signal sounds like"
)


def _r3_jargon_panel() -> CompositePanel:
    """Text Panel — title sits in the composite's chrome zone (same y as
    every other panel's title), body fills the one and only inner cell.

    Body uses justify mode: each line distributes its words across the cell
    width (Microsoft-Word-style "Justify"). Font size is pinned explicitly to
    DEFAULT_TITLE_FONT_SIZE so the caption type matches the panel title
    typography. show_ghost_border draws a faint rectangle at the same content
    rect the justify pipeline targets — the caption's borders.
    """
    body = TextPanel(
        JARGON_BODY,
        units=(1, 1),
        # Decoupled from DEFAULT_TITLE_FONT_SIZE — caption type should sit
        # well below the title visually, and auto_shrink needs headroom
        # between font_size (start) and min_font_size (floor) to actually
        # fit a multi-line body into the cell.
        font_size=18,
        min_font_size=10,
        color=style.TICK_LABEL_COLOR,
        fontweight="bold",
        auto_shrink=True,
        cell_padding_frac=0.0,
        justify=False,
        show_ghost_border=True,
        top_anchor=True,
    )
    return CompositePanel(
        units=(1, 1),
        title=JARGON_TITLE,
        rows=[[body]],
    )


def _r3_vector_projection() -> StaticPanel:
    panel = StaticPanel(
        units=(1, 1),
        title="Projection (a on b)",
        lim=LIM_2D,
        # axis_labels=False → suppress the on-plot italic "x"/"y" near the
        # spine tips; external x_label/y_label render outside the spine in
        # the chrome zone (same path as Heatmap/TimeSeries panels).
        axis_labels=False,
        show_ticks=True,
        show_tick_labels=True,
        show_grid=True,
        x_label="x",
        y_label="y",
        tick_positions=(-4.0, -2.0, 0.0, 2.0, 4.0),
    )
    a = PROJ_A
    b = PROJ_B
    dot = a[0] * b[0] + a[1] * b[1]
    b_sq = b[0] * b[0] + b[1] * b[1]
    scalar = dot / b_sq
    proj = (scalar * b[0], scalar * b[1])

    # b — purple, drawn first so a overlays on top
    panel.add(Vector(
        b,
        color=style.SECONDARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        label="b",
        zorder=2,
    ))
    # a — orange, the projected vector
    panel.add(Vector(
        a,
        color=style.PRIMARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
        label="a",
        zorder=3,
    ))
    # proj_b(a) — orange parallel-to-b component, half-alpha so b shows through
    panel.add(Vector(
        proj,
        color=style.PRIMARY_COLOR,
        linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
        alpha=0.55,
        show_tip=False,
        zorder=4,
    ))
    # perpendicular drop from a's tip to proj's tip — dashed, neutral
    panel.add(Vector(
        (a[0] - proj[0], a[1] - proj[1]),
        origin=proj,
        color=style.DROPLINE_COLOR,
        linewidth=style.DEFAULT_VECTOR_LINEWIDTH,
        linestyle="--",
        show_tip=False,
        alpha=style.DEFAULT_DROPLINE_ALPHA,
        zorder=2,
    ))
    return panel


# ---------------------------------------------------------------------------
# Figure composition + three-mode contract
# ---------------------------------------------------------------------------
def _build_figure(
    unit_inches: Optional[float] = None,
    dpi: Optional[int] = None,
    debug: bool = False,
) -> Figure:
    """Compose the kitchen-sink Figure (un-rendered).

    Four rows × ROW_WIDTH_UNITS columns. Row 0 is the SuptitlePanel band at
    quarter height; rows 1–3 host the panel/plottable coverage.

    debug=True enables the figure-level layout guides AND mutates every
    panel before render to force-show its tick labels + populate empty tick
    lists, so the user can spot any hidden chrome that would overlap.
    """
    row0 = [SuptitlePanel("dsplot - Sample Template Showcase", units=(ROW_WIDTH_UNITS, 1))]
    row1 = [
        _r1_vector_2d(),
        _r1_vector_3d(),
        _r1_heatmap_weave(),
        _r1_heatmap_centroid(),
    ]
    row2 = [
        _r2_chirp_with_inst_freq(),
    ]
    row3 = [
        _r3_stem_quartet(),
        _r3_jargon_panel(),
        _r3_vector_projection(),
    ]
    if debug:
        for row in (row1, row2, row3):
            for panel in row:
                _debug_unhide(panel)
    return Figure.compose(
        rows=[row0, row1, row2, row3],
        row_heights=[0.25, 1.0, 1.0, 1.0],
        unit_inches=unit_inches,
        dpi=dpi,
        show_cell_borders=True,
        debug_guides=debug,
    )


def _debug_unhide(panel) -> None:
    """Force-enable every flag that suppresses tick labels and populate any
    empty tick lists with sensible defaults so the figure renders with
    maximum chrome visible.
    """
    if hasattr(panel, "show_xticklabels"):
        panel.show_xticklabels = True
    if hasattr(panel, "show_yticklabels"):
        panel.show_yticklabels = True
    if hasattr(panel, "show_tick_labels"):
        panel.show_tick_labels = True
    # Stem rows have xticks=[]/yticks=[] when suppressed. Populate defaults
    # so debug mode renders something at every spine.
    if hasattr(panel, "xticks") and panel.xticks is not None and len(panel.xticks) == 0:
        panel.xticks = [0, STEM_N // 4, STEM_N // 2, 3 * STEM_N // 4, STEM_N]
    if hasattr(panel, "yticks") and panel.yticks is not None and len(panel.yticks) == 0:
        panel.yticks = [-1.0, 1.0]
    # Recurse into CompositePanel rows.
    if hasattr(panel, "rows"):
        for r in panel.rows:
            for child in r:
                _debug_unhide(child)


def render(
    output_dir: str = "assets/images/dsp/figures/sample_template",
    output_filename: str = "v3.png",
    debug: bool = False,
) -> str:
    """Build, render, and save the figure. Returns absolute output path.

    debug=True enables the template/layout debug overlay: figure-level
    guide lines for margins, gutters, title bands, axes bboxes, and axis-
    label inset positions, plus force-shown tick labels on every panel
    (with empty tick lists populated by sensible defaults) so hidden
    chrome that might overlap surfaces visibly.
    """
    fig = _build_figure(debug=debug)
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show() -> Figure:
    """Notebook-tuned rendering for dsp.ipynb inline display.

    Smaller unit_inches / dpi so the kitchen-sink figure fits inside a Jupyter
    cell. A nb-compact style profile shrinks chrome (titles, axis labels,
    tick labels, suptitle) so the chrome:cell ratio matches what the
    production render at unit_inches=4.0 looks like. Suppresses ipympl
    widget chrome on the returned Figure before plt.show() so the canvas
    reads as pure figure content.
    """
    import matplotlib.pyplot as plt
    with nb_compact_style():
        fig = _build_figure(unit_inches=2.5, dpi=60)
        fig.render()
    canvas = fig._mpl_fig.canvas
    for attr in ("header_visible", "toolbar_visible", "footer_visible"):
        try:
            setattr(canvas, attr, False)
        except Exception:
            pass
    try:
        canvas.manager.set_window_title("")
    except Exception:
        pass
    plt.show()
    return fig


def embed(target: object | None = None) -> Figure:
    """Drop into a caller-provided matplotlib container.

    v3 behaviour:
      - ``target is None``: behave like show() WITHOUT chrome suppression
        (returns the Figure for the caller to display however they like).
      - ``target: matplotlib.figure.Figure``: NotImplementedError — re-hosting
        the kitchen-sink layout onto a caller-provided Figure is reserved
        for a later revision.
      - ``target: matplotlib.axes.Axes``: NotImplementedError — Axes-targeting
        is reserved for simpler single-panel figure modules.
    """
    import matplotlib.axes
    import matplotlib.figure as mpl_figure
    if target is None:
        with nb_compact_style():
            fig = _build_figure(unit_inches=2.5, dpi=60)
            fig.render()
        return fig
    if isinstance(target, mpl_figure.Figure):
        raise NotImplementedError(
            "sample_template.embed(target: Figure) is reserved for a later "
            "revision — the kitchen-sink layout cannot currently re-host "
            "onto a caller Figure cleanly. Use render() or show()."
        )
    if isinstance(target, matplotlib.axes.Axes):
        raise NotImplementedError(
            "sample_template.embed(target: Axes) is not supported — "
            "Axes-targeting is reserved for simpler single-panel figure "
            "modules. Use render() or show()."
        )
    raise TypeError(
        "sample_template.embed(target=...) expects None, matplotlib Figure, "
        f"or matplotlib Axes; got {type(target).__name__}"
    )


if __name__ == "__main__":
    print(render())
