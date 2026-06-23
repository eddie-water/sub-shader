"""optichrome_showcase — the Optichrome colormaps, side by side.

Optichrome is Felipe Pantone's saturated full-spectrum palette. One grid,
four rows, one variant per row; three columns hold the same three fields
(weave · centroid · CWT) so a row compares one colormap across plot kinds
and a column compares all four colormaps on one plot.

    Row 0  SuptitlePanel(3,1)
    Row 1  optichrome         Weave | Centroid | CWT  (sequential sweep)
    Row 2  optichrome_cyclic  Weave | Centroid | CWT  (wrap-around)
    Row 3  optichrome_polar   Weave | Centroid | CWT  (diverging / white-center)
    Row 4  optichrome_pixel   Weave | Centroid | CWT  (posterized steps)

Colormaps come from `dsplot.colormaps` (registered on import). The fields
are synthetic stand-ins — a Gramian-angular-field weave, a Gaussian
centroid, and a log-frequency chirp ridge that reads like a CWT magnitude
spectrogram — so the figure is self-contained (no GPU / audio needed).

Three-mode contract: render() / show() / embed(). Iterate the look by
bumping output_filename: v1.png -> v2.png -> vN.png.
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

import numpy as np

from .. import (
    Figure,
    Heatmap,
    HeatmapPanel,
    SuptitlePanel,
    colormaps,
    nb_compact_style,
)


# ---------------------------------------------------------------------------
# Layout / data constants
# ---------------------------------------------------------------------------
ROW_WIDTH_UNITS = 3
GRID_N = 96

# The Optichrome aesthetic is posterized — a finite palette in hard-edged
# blocks, not a smooth ramp. Every variant is quantized to BANDS discrete
# colors so the fields read as the blocky artwork rather than a gradient.
BANDS = 8

CWT_N_FREQS = 96
CWT_N_TIME = 256

# Short display labels for the four variants, in registry order.
VARIANT_LABELS = {
    "optichrome":        "Optichrome",
    "optichrome_cyclic": "Cyclic",
    "optichrome_polar":  "Polar",
    "optichrome_pixel":  "Pixel",
}


# ---------------------------------------------------------------------------
# Data builders — synthetic stand-ins for the real fields
# ---------------------------------------------------------------------------
def _build_weave(n: int = GRID_N, base_freq_hz: float = 4.0) -> np.ndarray:
    """Gramian Angular Field weave — diagonal-band interference, in [0, 1]."""
    t = np.linspace(0, 1, n)
    sig = np.sin(2.0 * np.pi * base_freq_hz * t)
    phi = np.arccos(np.clip(sig, -1.0, 1.0))
    raw = np.cos(phi[:, None] + phi[None, :])
    return (raw + 1.0) / 2.0


def _build_centroid(n: int = GRID_N, sigma: float = 0.35) -> np.ndarray:
    """Single Gaussian peak with radial falloff, centered, in [0, 1]."""
    coords = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(coords, coords)
    return np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))


def _build_cwt(n_f: int = CWT_N_FREQS, n_t: int = CWT_N_TIME) -> np.ndarray:
    """Log-frequency chirp ridge magnitude field — reads like a CWT.

    A primary ridge sweeps low→high in bin space (a log chirp is ~linear in
    log-frequency bins), with a Gaussian cross-frequency spread, a quieter
    octave-up harmonic, and a slow amplitude breathe over time. Normalized
    to [0, 1].
    """
    t = np.linspace(0.0, 1.0, n_t)
    fbins = np.arange(n_f)[:, None]

    ridge = 0.12 * n_f + 0.74 * n_f * t ** 1.5
    spread = 0.045 * n_f
    field = np.exp(-((fbins - ridge[None, :]) ** 2) / (2.0 * spread ** 2))

    harmonic = ridge + 0.18 * n_f
    field += 0.45 * np.exp(-((fbins - harmonic[None, :]) ** 2) / (2.0 * (1.3 * spread) ** 2))

    breathe = 0.6 + 0.4 * np.sin(2.0 * np.pi * t) ** 2
    field *= breathe[None, :]

    return field / field.max()


# Build the three fields once; every variant row reuses them.
_WEAVE = _build_weave()
_CENTROID = _build_centroid()
_CWT = _build_cwt()


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def _square_panel(title: str, data: np.ndarray, cmap) -> HeatmapPanel:
    panel = HeatmapPanel(
        units=(1, 1),
        title=title,
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(Heatmap(data, extent=(-1.0, 1.0, -1.0, 1.0), cmap=cmap))
    return panel


def _cwt_panel(title: str, cmap) -> HeatmapPanel:
    n_f = _CWT.shape[0]
    panel = HeatmapPanel(
        units=(1, 1),
        title=title,
        x_label="t (s)",
        y_label="f bin",
        xticks=[0.0, 0.5, 1.0],
        yticks=[0.0, n_f / 2.0, float(n_f)],
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(Heatmap(_CWT, extent=(0.0, 1.0, 0.0, float(n_f)), cmap=cmap))
    return panel


def _variant_row(variant: str) -> list[HeatmapPanel]:
    """Weave · Centroid · CWT, all drawn with `variant` quantized to BANDS."""
    label = VARIANT_LABELS.get(variant, variant)
    cmap = colormaps.quantize(variant, BANDS)
    return [
        _square_panel(f"{label} · Weave", _WEAVE, cmap),
        _square_panel(f"{label} · Centroid", _CENTROID, cmap),
        _cwt_panel(f"{label} · CWT", cmap),
    ]


# ---------------------------------------------------------------------------
# Figure composition + three-mode contract
# ---------------------------------------------------------------------------
def _build_figure(
    unit_inches: Optional[float] = None,
    dpi: Optional[int] = None,
) -> Figure:
    """Compose the 4-variant × 3-field grid (un-rendered)."""
    colormaps.register()
    row0 = [SuptitlePanel(
        f"Optichrome Colormaps — 4 Variants · {BANDS} bands",
        units=(ROW_WIDTH_UNITS, 1),
    )]
    variant_rows = [_variant_row(v) for v in colormaps.OPTICHROME_VARIANTS]
    return Figure.compose(
        rows=[row0, *variant_rows],
        row_heights=[0.25, 1.0, 1.0, 1.0, 1.0],
        unit_inches=unit_inches,
        dpi=dpi,
        show_cell_borders=True,
    )


def render(
    output_dir: str = "assets/images/dsp/figures/optichrome_showcase",
    output_filename: str = "v1.png",
) -> str:
    """Build, render, and save the figure. Returns absolute output path."""
    fig = _build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show() -> Figure:
    """Notebook-tuned inline render (smaller unit_inches / dpi + compact chrome)."""
    import matplotlib.pyplot as plt
    with nb_compact_style():
        fig = _build_figure(unit_inches=2.2, dpi=60)
        fig.render()
    canvas = fig._mpl_fig.canvas
    for attr in ("header_visible", "toolbar_visible", "footer_visible"):
        try:
            setattr(canvas, attr, False)
        except Exception:
            pass
    plt.show()
    return fig


def embed(target: object | None = None) -> Figure:
    """Drop into a caller-provided container (None → compact Figure, like show)."""
    import matplotlib.axes
    import matplotlib.figure as mpl_figure
    if target is None:
        with nb_compact_style():
            fig = _build_figure(unit_inches=2.2, dpi=60)
            fig.render()
        return fig
    if isinstance(target, (mpl_figure.Figure, matplotlib.axes.Axes)):
        raise NotImplementedError(
            "optichrome_showcase.embed(target=...) onto a caller Figure/Axes "
            "is reserved for a later revision. Use render() or show()."
        )
    raise TypeError(
        "optichrome_showcase.embed(target=...) expects None, matplotlib "
        f"Figure, or matplotlib Axes; got {type(target).__name__}"
    )


if __name__ == "__main__":
    print(render())
