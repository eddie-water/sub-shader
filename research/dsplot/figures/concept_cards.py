"""concept_cards - small single-topic figures for notes/concepts/*.md.

Each card illustrates exactly one concept from the knowledge base, using the
real pipeline parameters from subshader.config (chunk_size=16384, 44.1 kHz,
num_cycles=6, num_fwhm_cycles=3, A0 root, 12 notes/octave) and, where
possible, the actual WaveletKernel implementation - so the figures are
renders of the real code, not illustrations of it.

Cards:
    morlet_construction  - carrier sinusoid x Gaussian envelope = Morlet
    wavelet_scaling      - one Morlet shape at A1/A3/A5, stretched/compressed
    chromatic_scale      - the 116 exponentially-spaced analysis frequencies
    colormap_gamma       - the fragment shader's inferno colormap + gamma curves
    overlap_hop          - 16384-sample chunks advancing by an 8192-sample hop

Follows the module conventions of sample_template.py: sys.path shim for
direct execution, render(output_dir, output_filename) -> str per card,
render_all() for batch. Output goes to assets/images/notes/.
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import numpy as np

from .. import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    Stem,
    SuptitlePanel,
    TimeSeriesPanel,
    style,
)

from subshader.dsp.wavelet_kernel import WaveletKernel

# ---------------------------------------------------------------------------
# Real pipeline parameters (mirror subshader.config defaults - data, not style)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44_100.0
CHUNK_SIZE = 1 << 14          # 16384 samples
OVERLAP_FACTOR = 0.5
HOP_SIZE = int(CHUNK_SIZE * (1.0 - OVERLAP_FACTOR))   # 8192 samples
NUM_CYCLES = 6
NUM_FWHM_CYCLES = 3
ROOT_NOTE_HZ = 27.5           # A0
NOTES_PER_OCTAVE = 12
NUM_OCTAVES = 10
NYQUIST_HZ = SAMPLE_RATE / 2.0
GAMMA_DEFAULT = 0.5

# Local style derivations (figure-level, per D-05 local-override convention)
CURVE_LW = 6.0
CURVE_LW_BOLD = 8.0
GUIDE_LW = 3.0
OUTPUT_DIR = "assets/images/notes"


def _kernel(freq_hz: float) -> WaveletKernel:
    """Build a real pipeline wavelet kernel at the given center frequency."""
    return WaveletKernel(
        f=np.float64(freq_hz),
        sample_rate=SAMPLE_RATE,
        num_cycles=NUM_CYCLES,
        num_fwhm_cycles=NUM_FWHM_CYCLES,
        input_n=CHUNK_SIZE,
    )


def _render(fig: Figure, output_dir: str, output_filename: str) -> str:
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Card 1 - Morlet construction: sinusoid x Gaussian = Morlet
# ---------------------------------------------------------------------------
def build_morlet_construction() -> Figure:
    k = _kernel(55.0)  # A1 - 6 cycles at 55 Hz = 109 ms support, easy to read
    t_ms = k.time_t * 1000.0
    xlim = (-62.0, 62.0)
    ylim = (-1.3, 1.3)
    xticks = [-50.0, 0.0, 50.0]

    carrier = TimeSeriesPanel(
        title="Carrier Sinusoid",
        units=(1, 1),
        x_label="time (ms)",
        y_label="amplitude",
        xticks=xticks,
        yticks=[-1.0, 0.0, 1.0],
        xlim=xlim,
        ylim=ylim,
    )
    carrier.add(Line(t_ms, k.sin_t.real, color=style.PRIMARY_COLOR, linewidth=CURVE_LW))

    envelope = TimeSeriesPanel(
        title="× Gaussian Envelope",
        units=(1, 1),
        x_label="time (ms)",
        xticks=xticks,
        yticks=[0.0, 0.5, 1.0],
        xlim=xlim,
        ylim=ylim,
    )
    envelope.add(Line(t_ms, k.gauss_t.real, color=style.TERTIARY_COLOR, linewidth=CURVE_LW))

    morlet = TimeSeriesPanel(
        title="= Morlet Wavelet",
        units=(1, 1),
        x_label="time (ms)",
        xticks=xticks,
        yticks=[-1.0, 0.0, 1.0],
        xlim=xlim,
        ylim=ylim,
    )
    product = k.sin_t * k.gauss_t
    morlet.add(Line(t_ms, k.gauss_t.real, color=style.TERTIARY_COLOR,
                    linewidth=GUIDE_LW, linestyle="--", alpha=0.7))
    morlet.add(Line(t_ms, -k.gauss_t.real, color=style.TERTIARY_COLOR,
                    linewidth=GUIDE_LW, linestyle="--", alpha=0.7))
    morlet.add(Line(t_ms, product.imag, color=style.SECONDARY_COLOR,
                    linewidth=CURVE_LW, alpha=0.75))
    morlet.add(Line(t_ms, product.real, color=style.PRIMARY_COLOR,
                    linewidth=CURVE_LW))

    return Figure.compose(
        rows=[
            [SuptitlePanel("Morlet Wavelet - Sinusoid × Gaussian (55 Hz)", units=(3, 1))],
            [carrier, envelope, morlet],
        ],
        show_cell_borders=True,
    )


def render_morlet_construction(output_dir: str = OUTPUT_DIR,
                               output_filename: str = "concept_morlet_construction_v2.png") -> str:
    return _render(build_morlet_construction(), output_dir, output_filename)


# ---------------------------------------------------------------------------
# Card 2 - Mother wavelet scaling: same shape at A1 / A3 / A5
# ---------------------------------------------------------------------------
def build_wavelet_scaling() -> Figure:
    freqs = [(55.0, "A1", style.PRIMARY_COLOR),
             (220.0, "A3", style.TERTIARY_COLOR),
             (880.0, "A5", style.SECONDARY_COLOR)]
    xlim = (-62.0, 62.0)
    rows = []
    for i, (f, _name, color) in enumerate(freqs):
        k = _kernel(f)
        t_ms = k.time_t * 1000.0
        wave = k.kernel_t.real
        wave = wave / np.max(np.abs(wave))
        last = i == len(freqs) - 1
        panel = TimeSeriesPanel(
            units=(3, 1),
            x_label="time (ms)" if last else None,
            xticks=[-50.0, -25.0, 0.0, 25.0, 50.0] if last else None,
            yticks=[],
            xlim=xlim,
            ylim=(-1.35, 1.35),
            show_xticklabels=last,
            show_yticklabels=False,
        )
        panel.add(Line(t_ms, wave, color=color, linewidth=CURVE_LW))
        rows.append([panel])

    composite = CompositePanel(
        rows=rows,
        units=(3, 1),
        share_x=True,
    )
    return Figure.compose(
        rows=[
            [SuptitlePanel("Same Wavelet at 55 / 220 / 880 Hz - Wide Low, Narrow High", units=(3, 1))],
            [composite],
        ],
        show_cell_borders=True,
    )


def render_wavelet_scaling(output_dir: str = OUTPUT_DIR,
                           output_filename: str = "concept_wavelet_scaling_v3.png") -> str:
    return _render(build_wavelet_scaling(), output_dir, output_filename)


# ---------------------------------------------------------------------------
# Card 3 - Chromatic frequency scale: 116 exponential frequencies below Nyquist
# ---------------------------------------------------------------------------
def build_chromatic_scale() -> Figure:
    scale_factor = 2 ** (1 / NOTES_PER_OCTAVE)
    i = np.arange(0, NOTES_PER_OCTAVE * NUM_OCTAVES, dtype=np.float64)
    freqs = ROOT_NOTE_HZ * (scale_factor ** i)
    keep = freqs < NYQUIST_HZ
    i, freqs = i[keep], freqs[keep]

    octave_idx = np.arange(0, len(i), NOTES_PER_OCTAVE, dtype=int)
    freqs_khz = freqs / 1000.0
    nyquist_khz = NYQUIST_HZ / 1000.0

    panel = TimeSeriesPanel(
        title=f"{len(freqs)} semitones, A0 to 21.1 kHz - dashed = Nyquist",
        units=(3, 1),
        y_label="kHz",
        x_label="note index (12 per octave)",
        xticks=[float(x) for x in i[octave_idx]],
        yticks=[0.0, 20.0],
        xlim=(-3.0, 119.0),
        ylim=(-0.8, 24.0),
    )
    panel.add(Line([-3.0, 119.0], [nyquist_khz, nyquist_khz],
                   color=style.NEUTRAL_COLOR, linewidth=GUIDE_LW,
                   linestyle="--", alpha=0.6))
    panel.add(Stem(i[octave_idx], freqs_khz[octave_idx],
                   color=style.TERTIARY_COLOR, linewidth=GUIDE_LW,
                   alpha=0.85, marker="o"))
    panel.add(Line(i, freqs_khz, color=style.PRIMARY_COLOR, linewidth=CURVE_LW_BOLD))

    return Figure.compose(
        rows=[
            [SuptitlePanel("Chromatic Scale - Equal in Pitch, Exponential in Hz", units=(3, 1))],
            [panel],
        ],
        show_cell_borders=True,
    )


def render_chromatic_scale(output_dir: str = OUTPUT_DIR,
                           output_filename: str = "concept_chromatic_scale_v6.png") -> str:
    return _render(build_chromatic_scale(), output_dir, output_filename)


# ---------------------------------------------------------------------------
# Card 4 - Fragment shader colormap strip + gamma curves
# ---------------------------------------------------------------------------
def build_colormap_gamma() -> Figure:
    strip = np.tile(np.linspace(0.0, 1.0, 512), (48, 1))
    strip_panel = HeatmapPanel(
        title="inferno colormap - 0 (silent) to 1 (loudest)",
        units=(3, 1),
        xticks=[0.0, 0.5, 1.0],
        yticks=[],
    )
    strip_panel.add(Heatmap(strip, extent=(0.0, 1.0, 0.0, 1.0),
                            aspect="auto", cmap="inferno", vmax=1.0))

    x = np.linspace(0.0, 1.0, 512)
    curves = TimeSeriesPanel(
        title="gamma 0.5 (orange) · 1.0 (dashed) · 2.2 (purple)",
        units=(3, 1),
        xticks=[0.0, 0.5, 1.0],
        yticks=[0.0, 1.0],
        xlim=(-0.02, 1.02),
        ylim=(-0.04, 1.08),
    )
    curves.add(Line(x, x ** 1.0, color=style.NEUTRAL_COLOR,
                    linewidth=GUIDE_LW, linestyle="--", alpha=0.65))
    curves.add(Line(x, x ** 2.2, color=style.SECONDARY_COLOR,
                    linewidth=CURVE_LW, alpha=0.85))
    curves.add(Line(x, x ** GAMMA_DEFAULT, color=style.PRIMARY_COLOR,
                    linewidth=CURVE_LW_BOLD))

    return Figure.compose(
        rows=[
            [SuptitlePanel("Shader Color - Inferno Lookup + Gamma Correction", units=(3, 1))],
            [strip_panel],
            [curves],
        ],
        show_cell_borders=True,
    )


def render_colormap_gamma(output_dir: str = OUTPUT_DIR,
                          output_filename: str = "concept_colormap_gamma_v4.png") -> str:
    return _render(build_colormap_gamma(), output_dir, output_filename)


# ---------------------------------------------------------------------------
# Card 5 - Overlapping chunks and the hop between them
# ---------------------------------------------------------------------------
def build_overlap_hop() -> Figure:
    chunk_ms = CHUNK_SIZE / SAMPLE_RATE * 1000.0     # 371.5 ms
    hop_ms = HOP_SIZE / SAMPLE_RATE * 1000.0         # 185.8 ms
    n_chunks = 4
    total_ms = (n_chunks - 1) * hop_ms + chunk_ms

    panel = TimeSeriesPanel(
        title="371.5 ms window every 185.8 ms - orange = new audio per frame",
        units=(3, 1),
        x_label="time (ms)",
        xticks=[round(k * hop_ms) * 1.0 for k in range(n_chunks + 2)],
        yticks=[],
        xlim=(-18.0, total_ms + 18.0),
        ylim=(0.3, n_chunks + 0.7),
        show_yticklabels=False,
    )

    for k in range(n_chunks + 2):
        panel.add(Line([k * hop_ms, k * hop_ms], [0.3, n_chunks + 0.7],
                       color=style.DROPLINE_COLOR, linewidth=2.0,
                       linestyle="--", alpha=0.45, zorder=2))

    bar_lw = 16.0
    for k in range(n_chunks):
        y = float(n_chunks - k)
        start = k * hop_ms
        panel.add(Line([start, start + chunk_ms], [y, y],
                       color=style.NEUTRAL_COLOR, linewidth=bar_lw,
                       alpha=0.35, zorder=3))
        panel.add(Line([start + chunk_ms - hop_ms, start + chunk_ms], [y, y],
                       color=style.PRIMARY_COLOR, linewidth=bar_lw,
                       alpha=1.0, zorder=4))

    return Figure.compose(
        rows=[
            [SuptitlePanel("Audio Chunks Overlap 50% - the Hop Is What's New", units=(3, 1))],
            [panel],
        ],
        show_cell_borders=True,
    )


def render_overlap_hop(output_dir: str = OUTPUT_DIR,
                       output_filename: str = "concept_overlap_hop_v1b.png") -> str:
    return _render(build_overlap_hop(), output_dir, output_filename)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def render_all(output_dir: str = OUTPUT_DIR) -> list[str]:
    return [
        render_morlet_construction(output_dir),
        render_wavelet_scaling(output_dir),
        render_chromatic_scale(output_dir),
        render_colormap_gamma(output_dir),
        render_overlap_hop(output_dir),
    ]


if __name__ == "__main__":
    for path in render_all():
        print(path)
