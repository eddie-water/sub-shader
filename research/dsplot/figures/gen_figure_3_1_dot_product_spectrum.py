"""gen_figure_3_1_dot_product_spectrum — §3.1 of DSP.md.

The dot product as a frequency-measurement operation. A two-tone signal is
swept against a family of sine references; each dot product becomes one bar
of a spectrum. The signal carries 1.0 of 2 Hz and 0.5 of 5 Hz, and the
spectrum reads back exactly those amplitudes at exactly those bins — the
essence of the Fourier transform.

Numbers mirror research/dot.py (the console sandbox this figure grew out of):
a 1 s window pins the bins to the integers (spacing = 1/duration = 1 Hz) and
makes the integer-Hz sines orthogonal, so each measurement is independent.

Three-mode contract (see dsplot/figures/__init__.py):
    render(output_dir, output_filename) -> str   # production PNG on disk
    show() -> Figure                              # notebook inline display
    embed(target=None) -> Figure                  # caller-provided container
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
    Stem,
    SuptitlePanel,
    TimeSeries,
    TimeSeriesPanel,
    nb_compact_style,
    style,
)


# ---------------------------------------------------------------------------
# Signal / measurement constants (mirror research/dot.py)
# ---------------------------------------------------------------------------
DURATION_S = 1.0
SAMPLE_RATE_HZ = 64.0
N = int(DURATION_S * SAMPLE_RATE_HZ)

TONE_A_HZ, TONE_A_AMP = 2.0, 1.0
TONE_B_HZ, TONE_B_AMP = 5.0, 0.5

SWEEP_HZ = list(range(1, 11))
ROW_WIDTH_UNITS = 4
SIGNAL_AMP_LIM = 1.65          # breathing room around the |signal| peak (~1.5)
SPECTRUM_YLIM = (0.0, 1.15)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def _sine(freq_hz: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(N) / SAMPLE_RATE_HZ
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def _example_signal() -> np.ndarray:
    return _sine(TONE_A_HZ, TONE_A_AMP) + _sine(TONE_B_HZ, TONE_B_AMP)


def _coefficient(signal: np.ndarray, freq_hz: float) -> float:
    """Normalized dot product: a unit sine at freq_hz scores 1.0."""
    return float(np.dot(signal, _sine(freq_hz))) / (N / 2.0)


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.array(SWEEP_HZ, dtype=np.float64)
    coeffs = np.array([_coefficient(signal, f) for f in SWEEP_HZ], dtype=np.float64)
    return freqs, coeffs


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _signal_panel() -> TimeSeriesPanel:
    signal = _example_signal()
    panel = TimeSeriesPanel(
        units=(ROW_WIDTH_UNITS, 1),
        title="Signal   1.0·sin(2 Hz) + 0.5·sin(5 Hz)",
        x_label="s",
        y_label="amp",
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        ylim=(-SIGNAL_AMP_LIM, SIGNAL_AMP_LIM),
        show_xticklabels=True,
        show_yticklabels=True,
    )
    panel.add(TimeSeries(signal, SAMPLE_RATE_HZ, color=style.NEUTRAL_COLOR, alpha=0.9))
    return panel


def _spectrum_panel() -> TimeSeriesPanel:
    freqs, coeffs = _spectrum(_example_signal())
    panel = TimeSeriesPanel(
        units=(ROW_WIDTH_UNITS, 1),
        title="Dot product against each sine reference  →  spectrum",
        x_label="reference frequency (Hz)",
        y_label="coeff",
        xticks=SWEEP_HZ,
        yticks=[0.0, 0.5, 1.0],
        xlim=(SWEEP_HZ[0] - 0.6, SWEEP_HZ[-1] + 0.6),
        ylim=SPECTRUM_YLIM,
        show_xticklabels=True,
        show_yticklabels=True,
    )
    # Every bin as a faint stem so the empties read as measured-and-zero, then
    # the two responding bins branded on top (2 Hz orange, 5 Hz purple).
    panel.add(Stem(
        freqs, coeffs,
        color=style.DROPLINE_COLOR,
        alpha=0.45,
        markersize=4.0,
        zorder=2,
    ))
    for freq_hz, brand in ((TONE_A_HZ, style.PRIMARY_COLOR), (TONE_B_HZ, style.SECONDARY_COLOR)):
        panel.add(Stem(
            np.array([freq_hz]),
            np.array([_coefficient(_example_signal(), freq_hz)]),
            color=brand,
            linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH,
            markersize=8.0,
            zorder=4,
        ))
    return panel


# ---------------------------------------------------------------------------
# Composition + three-mode contract
# ---------------------------------------------------------------------------
def _build_figure(
    unit_inches: Optional[float] = None,
    dpi: Optional[int] = None,
) -> Figure:
    row0 = [SuptitlePanel("Measuring Frequency with the Dot Product", units=(ROW_WIDTH_UNITS, 1))]
    row1 = [_signal_panel()]
    row2 = [_spectrum_panel()]
    return Figure.compose(
        rows=[row0, row1, row2],
        row_heights=[0.25, 1.0, 1.0],
        unit_inches=unit_inches,
        dpi=dpi,
    )


def _default_output_dir() -> str:
    """Repo-root-anchored output dir so render() lands in the same place
    regardless of the caller's cwd (research/ vs repo root)."""
    research_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(
        os.path.join(research_dir, "..", "assets", "images", "dsp", "figures", "dot_product_spectrum")
    )


def render(
    output_dir: Optional[str] = None,
    output_filename: str = "v1.png",
) -> str:
    """Build, render, and save the figure. Returns absolute output path."""
    if output_dir is None:
        output_dir = _default_output_dir()
    fig = _build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show() -> Figure:
    """Notebook-tuned rendering for dsp.ipynb inline display."""
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
    """Drop into a caller-provided container. target=None returns the Figure
    without chrome suppression; Figure/Axes targets are reserved for a later
    revision (multi-panel layout can't re-host cleanly yet).
    """
    import matplotlib.axes
    import matplotlib.figure as mpl_figure
    if target is None:
        with nb_compact_style():
            fig = _build_figure(unit_inches=2.5, dpi=60)
            fig.render()
        return fig
    if isinstance(target, (mpl_figure.Figure, matplotlib.axes.Axes)):
        raise NotImplementedError(
            "gen_figure_3_1 is multi-panel; re-hosting onto a caller Figure/Axes "
            "is reserved for a later revision. Use render() or show()."
        )
    raise TypeError(
        "embed(target=...) expects None, matplotlib Figure, or Axes; "
        f"got {type(target).__name__}"
    )


if __name__ == "__main__":
    print(render())
