"""gen_fbands_comb - frequency-band comb filters, STFT vs CWT (README first pass).

Comb-filter response banks for the two transforms:
    STFT - constant-width teeth (fixed delta-f)
    CWT  - proportional teeth (bandwidth = +/- % of center f)

Renderers:
    render(variant, density)  - two stacked full-width panels (one variant)
    render_matrix()           - 3 density levels x (linear | log) x (STFT, CWT)
                                in one 12-panel figure

Density levels decimate the tooth count so combs stay resolvable at README
scale; "full" is the honest bank (STFT bin width 44100/8192 ~= 5.4 Hz,
chromatic CWT at 12 teeth/octave).

Both axis variants plot x as a normalized 0-1 position on a linear panel and
relabel the ticks after render - dsplot panels stay untouched (no xscale hook
needed). Normalizing also keeps the x/y data-range ratio near 1 so
StaticPanel's square-aspect pass doesn't collapse the axes box while panel
titles compute their chrome offset (a raw 20-20k Hz x-range pushes titles
off-canvas).

Each tooth is sampled on its own local segment (fixed point count across its
support, uniform in the *axis* coordinate) instead of one shared frequency
grid - a shared grid undersamples 5 Hz teeth at the top of a log axis and
renders moire ragging instead of a clean envelope.
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import matplotlib as mpl
import numpy as np

from .. import (
    Figure,
    Line,
    Stem,
    SuptitlePanel,
    TextPanel,
    TimeSeriesPanel,
    style,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
ROW_WIDTH_UNITS = 4
Y_LIM = (0.0, 1.08)
SEG_POINTS = 96
SEG_HALF_WIDTH_SIGMAS = 4.5

# Matrix figure: tighter unitary pad (dsplot default 1.5" leaves too much
# dead space across 7 rows) + a larger, denser canvas so the full-density
# rows survive zooming. Applied via the sanctioned Mode-1 global override
# (style_override_demo) with restore in finally.
MATRIX_PAD_INCHES = 0.8
MATRIX_UNIT_INCHES = 7.0
MATRIX_DPI = 200
MATRIX_ROW_HEIGHT = {"a_only": 0.5, "white": 0.55, "chromatic": 0.85}
# Typography: column headers match panel titles (DEFAULT_TITLE_FONT_SIZE);
# the suptitle sits one step above for hierarchy.
MATRIX_HEADER_FONT_SIZE = style.DEFAULT_TITLE_FONT_SIZE
MATRIX_SUPTITLE_FONT_SIZE = 58

# ---------------------------------------------------------------------------
# Band constants
# ---------------------------------------------------------------------------
F_MIN_HZ = 20.0
F_MAX_HZ = 20_000.0

# Each level is a musical subset of SubShader's actual chromatic bank
# (A0 root, 12 notes/octave, 10 octaves - see WaveletConfig):
#   a_only    - just the A notes, one band per octave
#   white     - the white keys (naturals), 7 bands per octave
#   chromatic - the full bank, every semitone
LEVELS = ("a_only", "white", "chromatic")

# STFT: constant-width teeth paired with each level. "chromatic" pairs with
# the real bin width (44100 / 8192); the coarser levels widen it so
# individual teeth stay resolvable.
STFT_DF_HZ = {
    "a_only": 430.0,
    "white": 43.0,
    "chromatic": 44_100.0 / 8_192.0,
}

CHROMATIC_ROOT_HZ = 27.5
CHROMATIC_NOTES_PER_OCTAVE = 12
CHROMATIC_NUM_OCTAVES = 10
# Names from the A0 root upward.
NOTE_NAMES = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")
LEVEL_NOTE_NAMES = {
    "a_only": ("A",),
    "white": ("A", "B", "C", "D", "E", "F", "G"),
    "chromatic": NOTE_NAMES,
}
# Effective teeth/octave per level - sets each level's tooth width. White
# keys are unevenly spaced (whole/half steps); 7/octave is the average.
LEVEL_TEETH_PER_OCTAVE = {"a_only": 1.0, "white": 7.0, "chromatic": 12.0}


def _chromatic_scale() -> tuple[np.ndarray, list[str]]:
    k = np.arange(CHROMATIC_NOTES_PER_OCTAVE * CHROMATIC_NUM_OCTAVES)
    freqs = CHROMATIC_ROOT_HZ * 2.0 ** (k / CHROMATIC_NOTES_PER_OCTAVE)
    in_range = (freqs >= F_MIN_HZ) & (freqs <= F_MAX_HZ)
    names = [NOTE_NAMES[i % CHROMATIC_NOTES_PER_OCTAVE] for i in k[in_range]]
    return freqs[in_range], names


def _chromatic_notes() -> np.ndarray:
    return _chromatic_scale()[0]


def _cwt_teeth_per_octave(density: str) -> float:
    return LEVEL_TEETH_PER_OCTAVE[density]


def _cwt_bw_frac(density: str) -> float:
    """Fractional bandwidth: FWHM = frac * fc, i.e. each band spreads roughly
    +/- (frac/2 * 100)% around its center."""
    return 2.0 ** (1.0 / _cwt_teeth_per_octave(density)) - 1.0


# Highlighted reference tooth (nearest center to this f in each bank).
REF_TOOTH_HZ = 1_000.0

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

CMAP_NAME = "inferno"
CMAP_LO, CMAP_HI = 0.18, 0.92

TOOTH_ALPHA = 0.85
REF_LINEWIDTH_BOOST = 1.5

# Chromatic note guides - one faint vertical at every pitch the pipeline
# measures. Wavelet teeth sit ON them; Fourier bins ignore them.
NOTE_LINE_ALPHA = 0.18
NOTE_LINE_WIDTH = 1.2


# ---------------------------------------------------------------------------
# Band builders
# ---------------------------------------------------------------------------
def _stft_centers(density: str) -> np.ndarray:
    df = STFT_DF_HZ[density]
    return np.arange(df, F_MAX_HZ + df / 2.0, df)


def _cwt_centers(density: str) -> np.ndarray:
    freqs, names = _chromatic_scale()
    keep = LEVEL_NOTE_NAMES[density]
    return np.array([f for f, n in zip(freqs, names) if n in keep])


def _tooth_color(fc: float) -> str:
    span = np.log10(F_MAX_HZ) - np.log10(F_MIN_HZ)
    m = (np.log10(fc) - np.log10(F_MIN_HZ)) / span
    cmap = mpl.colormaps[CMAP_NAME]
    return mpl.colors.to_hex(cmap(CMAP_LO + (CMAP_HI - CMAP_LO) * float(np.clip(m, 0.0, 1.0))))


# ---------------------------------------------------------------------------
# Axis mapping (frequency Hz <-> normalized 0-1 x position)
# ---------------------------------------------------------------------------
def _to_axis_x(f: np.ndarray, variant: str) -> np.ndarray:
    if variant == "log":
        lo, hi = np.log10(F_MIN_HZ), np.log10(F_MAX_HZ)
        return (np.log10(f) - lo) / (hi - lo)
    return (f - F_MIN_HZ) / (F_MAX_HZ - F_MIN_HZ)


def _from_axis_x(u: np.ndarray, variant: str) -> np.ndarray:
    if variant == "log":
        lo, hi = np.log10(F_MIN_HZ), np.log10(F_MAX_HZ)
        return 10.0 ** (lo + u * (hi - lo))
    return F_MIN_HZ + u * (F_MAX_HZ - F_MIN_HZ)


def _axis_ticks(variant: str) -> tuple[list[float], list[str]]:
    if variant == "log":
        tick_hz = [20.0, 200.0, 2_000.0, 20_000.0]
        labels = ["20", "200", "2k", "20k"]
    else:
        tick_hz = [F_MIN_HZ, 5_000.0, 10_000.0, 15_000.0, 20_000.0]
        labels = ["20", "5k", "10k", "15k", "20k"]
    ticks = [float(u) for u in _to_axis_x(np.array(tick_hz), variant)]
    return ticks, labels


def _tooth_xy(fc: float, fwhm_hz: float, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Local segment for one Gaussian tooth, uniform in the axis coordinate."""
    sigma = fwhm_hz / FWHM_TO_SIGMA
    f_lo = max(fc - SEG_HALF_WIDTH_SIGMAS * sigma, F_MIN_HZ)
    f_hi = min(fc + SEG_HALF_WIDTH_SIGMAS * sigma, F_MAX_HZ)
    u = np.linspace(
        float(_to_axis_x(np.array([f_lo]), variant)[0]),
        float(_to_axis_x(np.array([f_hi]), variant)[0]),
        SEG_POINTS,
    )
    f = _from_axis_x(u, variant)
    y = np.exp(-0.5 * ((f - fc) / sigma) ** 2)
    return u, y


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _bank(kind: str, density: str) -> tuple[np.ndarray, np.ndarray]:
    """(centers, fwhm) arrays for one transform at one density."""
    if kind == "stft":
        centers = _stft_centers(density)
        fwhm = np.full_like(centers, STFT_DF_HZ[density])
    else:
        centers = _cwt_centers(density)
        fwhm = _cwt_bw_frac(density) * centers
    return centers, fwhm


def _bank_label(kind: str, density: str, n: int) -> str:
    if kind == "stft":
        df = STFT_DF_HZ[density]
        width = f"{df:.0f} Hz" if df >= 10 else f"{df:.1f} Hz"
        return f"Fourier - {n:,} bands, each {width} wide"
    cwt_labels = {
        "a_only": f"Wavelet - just the A notes ({n} bands)",
        "white": f"Wavelet - the white keys ({n} bands)",
        "chromatic": f"Wavelet - full chromatic ({n} bands)",
    }
    return cwt_labels[density]


def _comb_panel(
    kind: str,
    variant: str,
    *,
    density: str = "decimated",
    units: tuple[int, int] = (ROW_WIDTH_UNITS, 1),
    title: str | None = None,
    show_xticks: bool = True,
    show_yticks: bool = True,
    y_label: str | None = "response",
) -> TimeSeriesPanel:
    xticks, _ = _axis_ticks(variant)
    centers, fwhm = _bank(kind, density)
    if title is None:
        title = _bank_label(kind, density, len(centers))

    panel = TimeSeriesPanel(
        units=units,
        title=title,
        x_label=None,
        y_label=y_label if show_yticks else None,
        xticks=xticks,
        yticks=[0.0, 1.0] if show_yticks else [],
        show_xticklabels=show_xticks,
        show_yticklabels=show_yticks,
        xlim=(0.0, 1.0),
        ylim=Y_LIM,
    )

    # Chromatic note guides under everything.
    for note in _chromatic_notes():
        un = float(_to_axis_x(np.array([note]), variant)[0])
        panel.add(Line(
            np.array([un, un]), np.array([Y_LIM[0], Y_LIM[1]]),
            color=style.NEUTRAL_COLOR,
            linewidth=NOTE_LINE_WIDTH,
            alpha=NOTE_LINE_ALPHA,
            zorder=1,
        ))

    ref_idx = int(np.argmin(np.abs(centers - REF_TOOTH_HZ)))
    for i, (fc, w) in enumerate(zip(centers, fwhm)):
        if i == ref_idx:
            continue
        x, y = _tooth_xy(float(fc), float(w), variant)
        panel.add(Line(
            x, y,
            color=_tooth_color(float(fc)),
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
            alpha=TOOTH_ALPHA,
        ))
    # Reference tooth on top - same f in every panel so widths compare directly.
    x, y = _tooth_xy(float(centers[ref_idx]), float(fwhm[ref_idx]), variant)
    panel.add(Line(
        x, y,
        color=style.NEUTRAL_COLOR,
        linewidth=style.DEFAULT_DROPLINE_LINEWIDTH + REF_LINEWIDTH_BOOST,
        alpha=1.0,
        zorder=5,
    ))
    return panel


def _relabel_freq_ticks(panels_by_variant: list[tuple[TimeSeriesPanel, str]]) -> None:
    """Post-render pass: swap normalized tick positions for Hz labels."""
    for panel, variant in panels_by_variant:
        if panel.ax is None:
            continue
        _, labels = _axis_ticks(variant)
        panel.ax.set_xticklabels(labels if panel.show_xticklabels else [])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _build_figure(variant: str, density: str = "decimated") -> Figure:
    suptitle = SuptitlePanel(
        f"Frequency Bands - STFT vs CWT    [{variant} frequency axis]",
        units=(ROW_WIDTH_UNITS, 1),
    )
    top = _comb_panel("stft", variant, density=density, show_xticks=False)
    bottom = _comb_panel("cwt", variant, density=density, show_xticks=True)
    fig = Figure.compose(
        rows=[[suptitle], [top], [bottom]],
        row_heights=[0.25, 1.0, 1.0],
    )
    fig.render()
    _relabel_freq_ticks([(top, variant), (bottom, variant)])
    return fig


def _column_header(text: str) -> TextPanel:
    return TextPanel(
        text,
        units=(ROW_WIDTH_UNITS // 2, 1),
        font_size=MATRIX_HEADER_FONT_SIZE,
        color=style.TICK_LABEL_COLOR,
        fontweight="bold",
        cell_padding_frac=0.0,
    )


def _build_matrix_figure() -> Figure:
    """3 density levels; per level STFT row + CWT row; linear | log columns."""
    suptitle = SuptitlePanel(
        "Frequency Bands - Fourier vs Wavelet",
        units=(ROW_WIDTH_UNITS, 1),
        font_size=MATRIX_SUPTITLE_FONT_SIZE,
    )
    header_row = [
        _column_header("linear frequency axis"),
        _column_header("log frequency axis"),
    ]
    half = (ROW_WIDTH_UNITS // 2, 1)
    rows: list[list] = [[suptitle], header_row]
    row_heights = [0.2, 0.12]
    relabel: list[tuple[TimeSeriesPanel, str]] = []
    for density in LEVELS:
        for kind in ("stft", "cwt"):
            is_bottom = kind == "cwt"
            row = []
            for variant in ("linear", "log"):
                panel = _comb_panel(
                    kind, variant,
                    density=density,
                    units=half,
                    show_xticks=is_bottom,
                    show_yticks=(variant == "linear"),
                    y_label=None,
                )
                relabel.append((panel, variant))
                row.append(panel)
            rows.append(row)
            row_heights.append(MATRIX_ROW_HEIGHT[density])
    fig = Figure.compose(
        rows=rows,
        row_heights=row_heights,
        unit_inches=MATRIX_UNIT_INCHES,
        dpi=MATRIX_DPI,
    )
    fig.render()
    _relabel_freq_ticks(relabel)
    return fig


def render(
    variant: str = "log",
    output_dir: str = "assets/images/claude/plots",
    output_filename: str | None = None,
    density: str = "chromatic",
) -> str:
    if variant not in ("log", "linear"):
        raise ValueError(f"variant must be 'log' or 'linear', got {variant!r}")
    if density not in LEVELS:
        raise ValueError(f"density must be one of {LEVELS}, got {density!r}")
    if output_filename is None:
        output_filename = f"fbands_combs_stft_vs_cwt_v3_{variant}_{density}.png"
    fig = _build_figure(variant, density)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Sampling-density vs bandwidth demo (2x2)
# ---------------------------------------------------------------------------
# Two tones 1.5 semitones apart, measured by four banks over 500-2000 Hz.
# Rows: band density (smoothness of the measurement). Columns: band width
# (actual resolution). Density never separates the tones; width does.
DEMO_F_LO_HZ = 500.0
DEMO_F_HI_HZ = 2_000.0
DEMO_TONE_SPLIT_SEMITONES = 1.5
DEMO_TONES_HZ = (
    1_000.0 / 2.0 ** (DEMO_TONE_SPLIT_SEMITONES / 2.0 / 12.0),
    1_000.0 * 2.0 ** (DEMO_TONE_SPLIT_SEMITONES / 2.0 / 12.0),
)
DEMO_DENSITIES_PER_OCTAVE = (12.0, 96.0)
DEMO_WIDTHS_FRAC = (0.20, 0.03)  # FWHM as fraction of center: +/-10% | +/-1.5%


def _demo_axis_u(f: np.ndarray) -> np.ndarray:
    return np.log2(np.asarray(f, dtype=np.float64) / DEMO_F_LO_HZ) / np.log2(DEMO_F_HI_HZ / DEMO_F_LO_HZ)


def _demo_centers(per_octave: float) -> np.ndarray:
    n_octaves = np.log2(DEMO_F_HI_HZ / DEMO_F_LO_HZ)
    k = np.arange(int(np.floor(n_octaves * per_octave)) + 1)
    return DEMO_F_LO_HZ * 2.0 ** (k / per_octave)


def _demo_bank_response(centers: np.ndarray, width_frac: float) -> np.ndarray:
    sigma = width_frac * centers / FWHM_TO_SIGMA
    r = np.zeros_like(centers)
    for tone in DEMO_TONES_HZ:
        r += np.exp(-0.5 * ((tone - centers) / sigma) ** 2)
    return r / r.max()


def _demo_panel(per_octave: float, width_frac: float, *, show_xticks: bool,
                show_yticks: bool) -> TimeSeriesPanel:
    centers = _demo_centers(per_octave)
    response = _demo_bank_response(centers, width_frac)
    u = _demo_axis_u(centers)
    pct = width_frac / 2.0 * 100.0
    tick_hz = [500.0, 1_000.0, 2_000.0]
    panel = TimeSeriesPanel(
        units=(ROW_WIDTH_UNITS // 2, 1),
        title=f"{per_octave:.0f} bands/octave · each ±{pct:.1f}% wide",
        x_label=None,
        y_label=None,
        xticks=[float(t) for t in _demo_axis_u(np.array(tick_hz))],
        yticks=[0.0, 1.0] if show_yticks else [],
        show_xticklabels=show_xticks,
        show_yticklabels=show_yticks,
        xlim=(0.0, 1.0),
        ylim=Y_LIM,
    )
    # True tone locations - dashed white verticals under everything.
    for tone in DEMO_TONES_HZ:
        ut = float(_demo_axis_u(np.array([tone]))[0])
        panel.add(Line(
            np.array([ut, ut]), np.array([0.0, 1.0]),
            color=style.NEUTRAL_COLOR,
            linewidth=style.DEFAULT_DROPLINE_LINEWIDTH,
            linestyle="--",
            alpha=0.55,
            zorder=2,
        ))
    # One stem per band = what the bank actually measures.
    for uc, fc, rc in zip(u, centers, response):
        panel.add(Stem(
            np.array([float(uc)]), np.array([float(rc)]),
            color=_tooth_color(float(fc)),
        ))
    return panel


def _build_sampling_demo_figure() -> Figure:
    suptitle = SuptitlePanel(
        f"Two tones {DEMO_TONE_SPLIT_SEMITONES:g} semitones apart - what each bank measures",
        units=(ROW_WIDTH_UNITS, 1),
    )
    rows: list[list] = [[suptitle]]
    row_heights = [0.2]
    for i, per_octave in enumerate(DEMO_DENSITIES_PER_OCTAVE):
        is_bottom = i == len(DEMO_DENSITIES_PER_OCTAVE) - 1
        rows.append([
            _demo_panel(per_octave, width_frac,
                        show_xticks=is_bottom, show_yticks=(j == 0))
            for j, width_frac in enumerate(DEMO_WIDTHS_FRAC)
        ])
        row_heights.append(0.6)
    fig = Figure.compose(
        rows=rows,
        row_heights=row_heights,
        unit_inches=MATRIX_UNIT_INCHES,
        dpi=MATRIX_DPI,
    )
    fig.render()
    labels = ["500", "1k", "2k"]
    for row in rows[1:]:
        for panel in row:
            if panel.ax is not None:
                panel.ax.set_xticklabels(labels if panel.show_xticklabels else [])
    return fig


def render_sampling_demo(
    output_dir: str = "assets/images/claude/plots",
    output_filename: str = "fbands_density_vs_width_v1.png",
) -> str:
    saved = (
        style.DEFAULT_PAD_INCHES,
        style.DEFAULT_MARGIN_INCHES,
        style.DEFAULT_GUTTER_INCHES,
        style.DEFAULT_COLUMN_GUTTER_INCHES,
    )
    style.DEFAULT_PAD_INCHES = MATRIX_PAD_INCHES
    style.DEFAULT_MARGIN_INCHES = MATRIX_PAD_INCHES
    style.DEFAULT_GUTTER_INCHES = 2.0 * MATRIX_PAD_INCHES
    style.DEFAULT_COLUMN_GUTTER_INCHES = 2.0 * MATRIX_PAD_INCHES
    try:
        fig = _build_sampling_demo_figure()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path)
    finally:
        (
            style.DEFAULT_PAD_INCHES,
            style.DEFAULT_MARGIN_INCHES,
            style.DEFAULT_GUTTER_INCHES,
            style.DEFAULT_COLUMN_GUTTER_INCHES,
        ) = saved
    return os.path.abspath(output_path)


def render_matrix(
    output_dir: str = "assets/images/claude/plots",
    output_filename: str = "fbands_combs_matrix_v2.png",
) -> str:
    saved = (
        style.DEFAULT_PAD_INCHES,
        style.DEFAULT_MARGIN_INCHES,
        style.DEFAULT_GUTTER_INCHES,
        style.DEFAULT_COLUMN_GUTTER_INCHES,
    )
    style.DEFAULT_PAD_INCHES = MATRIX_PAD_INCHES
    style.DEFAULT_MARGIN_INCHES = MATRIX_PAD_INCHES
    style.DEFAULT_GUTTER_INCHES = 2.0 * MATRIX_PAD_INCHES
    style.DEFAULT_COLUMN_GUTTER_INCHES = 2.0 * MATRIX_PAD_INCHES
    try:
        fig = _build_matrix_figure()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path)
    finally:
        (
            style.DEFAULT_PAD_INCHES,
            style.DEFAULT_MARGIN_INCHES,
            style.DEFAULT_GUTTER_INCHES,
            style.DEFAULT_COLUMN_GUTTER_INCHES,
        ) = saved
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render_matrix())
