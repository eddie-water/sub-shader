"""Surface demo - audio time series beside a 3D CWT "topographic" surface.

Single-row dsplot Figure: left `TimeSeriesPanel` shows the raw waveform,
right `StaticPanel3D` hosts a `SurfaceHeatmap` render of the same signal's
CWT magnitude. Reuses Figure 1's locked HERO pipeline
(`gen_figure_1_stft_vs_cwt._prepare_hero()`) - the click-plus-tone signal
has amplitude structure that actually reads in a waveform panel (the plain
waypoint chirp is full-scale everywhere and fills the panel solid). No new
chirp math, no new import worldview (per D-01: figures/ is consumer code and
may reach into sibling figure modules; the sys.path shim + relative-import
pattern is set up once in `dsplot/__init__.py`).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

from .. import Figure, StaticPanel3D, SurfaceHeatmap, TimeSeries, TimeSeriesPanel, style
from . import gen_figure_1_stft_vs_cwt as fig1

TS_PANEL_UNITS = (2, 1)
SURFACE_PANEL_UNITS = (2, 1)
LIM_3D = 1.0
VIEW_INIT = (32.0, -63.0)


def _build_figure() -> Figure:
    data = fig1._prepare_hero()
    xticks = fig1._auto_xticks(data["duration_s"])

    ts_panel = TimeSeriesPanel(
        units=TS_PANEL_UNITS,
        title="Audio Signal",
        x_label="s",
        xticks=xticks,
    )
    ts_panel.add(TimeSeries(data["signal"], fig1.SR,
                            color=style.TICK_LABEL_COLOR))

    # Spines/tick-crosses off: the cube axes are time/freq/magnitude here, so
    # mpl-style x/y/z spine labels through the terrain would mislabel the
    # scene. The floor grid (drawn by the panel chrome) keeps ground reference.
    # Borderless: fill_cell expands the axes AFTER the panel draws its
    # figure-coord border, so a border here would sit at the stale (pre-fill)
    # rect with the cube floor spilling past it.
    surface_panel = StaticPanel3D(
        units=SURFACE_PANEL_UNITS,
        title="Wavelet Analysis",
        lim_3d=LIM_3D,
        view_init=VIEW_INIT,
        show_spines=False,
        show_border=False,
    )
    # Figure._apply_fill_cell expands any panel with a truthy fill_cell attr;
    # 3D cells need it to fill their cell (see project dsplot 3D-fill notes).
    surface_panel.fill_cell = True
    surface_panel.add(SurfaceHeatmap(
        data["cwt_data"],
        duration_s=data["duration_s"],
        freqs=data["cwt_freqs"],
        normalize_to=LIM_3D,
        ccount=110,
    ))

    return Figure.compose(rows=[[ts_panel, surface_panel]])


# Camera-angle exploration sheet. (elev, azim) per cell:
#   - (90, -90) is true top-down: +time right, +freq up - reads as the 2D
#     scalogram.
#   - azim=-90 keeps the time axis EXACTLY horizontal on screen; elev alone
#     tilts the surface away from the page (waterfall family).
#   - Swinging azim off -90 angles the freq axis out of vertical (~the azim
#     delta), at the cost of slightly tilting the time axis too - mpl's
#     projection can't angle freq while keeping time strictly horizontal.
VIEW_VARIANTS: tuple[tuple[float, float, str], ...] = (
    (90.0, -90.0, "top-down (2D scalogram)"),
    (60.0, -90.0, "time locked horizontal"),
    (30.0, -90.0, "time locked horizontal"),
    (30.0, -75.0, "freq swung 15deg"),
    (30.0, -60.0, "freq swung 30deg"),
    (30.0, -45.0, "freq swung 45deg"),
)


# README-band proportions: figure-1's spectrogram rows are ~3:1, so each
# variant gets a (3, 1) cell and stretch_fill anamorphically stretches the
# projection to the wide cell (Axes3D otherwise re-squares its position box
# every draw). Scene box stays cube-ish; the CELL aspect supplies the 3:1.
VARIANT_CELL_UNITS = (3, 1)
VARIANT_BOX_ASPECT = (1.0, 1.0, 0.6)
VARIANT_BOX_ZOOM = 1.9


def _build_variants_figure() -> Figure:
    data = fig1._prepare_hero()

    # No subtitles: in a multi-row grid the below-axes subtitle band lands
    # right above the NEXT row's titles and reads as a mislabel.
    def surface_cell(elev: float, azim: float, note: str) -> StaticPanel3D:
        panel = StaticPanel3D(
            units=VARIANT_CELL_UNITS,
            title=f"elev {elev:.0f} / azim {azim:.0f}",
            lim_3d=LIM_3D,
            view_init=(elev, azim),
            show_spines=False,
            show_border=False,
            box_zoom=VARIANT_BOX_ZOOM,
            box_aspect=VARIANT_BOX_ASPECT,
            stretch_fill=True,
        )
        panel.fill_cell = True
        panel.add(SurfaceHeatmap(
            data["cwt_data"],
            duration_s=data["duration_s"],
            freqs=data["cwt_freqs"],
            normalize_to=LIM_3D,
            ccount=240,
        ))
        return panel

    cells = [surface_cell(elev, azim, note) for elev, azim, note in VIEW_VARIANTS]
    return Figure.compose(rows=[[cell] for cell in cells])


# STFT-vs-CWT surface comparison: left Fourier, right Wavelet, SAME signal,
# camera, mesh, and gamma - only the transform differs (apples-to-apples).
# gamma < 1 lifts the broadband click cluster into visible relief next to the
# vmax-clipped tonal ridge. No fill_cell: the gridspec margins/gutter stay,
# giving the clean centered composition; stretch_fill still lets each scene
# span its own (wide) cell rect.
COMPARE_VIEW = (80.0, -75.0)
COMPARE_CELL_UNITS = (2, 1)
COMPARE_BOX_ZOOM = 1.25
COMPARE_GAMMA = 0.5


@contextmanager
def _white_chrome():
    """Render with white title/tick text instead of the default gray.

    dsplot Plottables/Panels resolve style constants lazily at render time
    (D-05), so swapping the module constant for the duration of a render is
    the sanctioned per-figure override pattern (mirrors fig1's overrides).
    """
    prev = style.TICK_LABEL_COLOR
    style.TICK_LABEL_COLOR = "#FFFFFF"
    try:
        yield
    finally:
        style.TICK_LABEL_COLOR = prev


def _build_stft_vs_cwt_figure() -> Figure:
    data = fig1._prepare_hero()
    stft_mag = fig1._stft_on_log_bins(data["signal"], fig1.SR, data["cwt_freqs"])

    def surface_cell(title: str, mag: "np.ndarray") -> StaticPanel3D:  # noqa: F821
        panel = StaticPanel3D(
            units=COMPARE_CELL_UNITS,
            title=title,
            lim_3d=LIM_3D,
            view_init=COMPARE_VIEW,
            show_spines=False,
            show_border=False,
            box_zoom=COMPARE_BOX_ZOOM,
            box_aspect=VARIANT_BOX_ASPECT,
            stretch_fill=True,
            show_floor_grid=False,
        )
        panel.add(SurfaceHeatmap(
            mag,
            duration_s=data["duration_s"],
            freqs=data["cwt_freqs"],
            normalize_to=LIM_3D,
            z_base=-LIM_3D,
            gamma=COMPARE_GAMMA,
            mesh_upsample=(4.0, 2.0),
            ccount=200,
        ))
        return panel

    return Figure.compose(rows=[[
        surface_cell("Fourier Analysis", stft_mag),
        surface_cell("Wavelet Analysis", data["cwt_data"]),
    ]])


def render_stft_vs_cwt(
    output_dir: str = "assets/images/figures",
    output_filename: str = "surface_stft_vs_cwt_v9.png",
) -> str:
    """Build, render, save the side-by-side comparison. Returns absolute path."""
    with _white_chrome():
        fig = _build_stft_vs_cwt_figure()
        fig.render()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path)
    return os.path.abspath(output_path)


def render_view_variants(
    output_dir: str = "assets/images/figures",
    output_filename: str = "surface_view_variants_v6.png",
) -> str:
    """Build, render, save the camera-angle sheet. Returns absolute path."""
    fig = _build_variants_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def render(
    output_dir: str = "assets/images/figures",
    output_filename: str = "surface_demo_v3.png",
) -> str:
    """Build, render, save. Returns absolute output path."""
    fig = _build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(f"  Saved -> {render()}")
