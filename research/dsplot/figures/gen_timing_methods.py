"""Timing figure — Fourier vs Wavelet Performance (method comparison).

The sibling of Figure 1 ("Fourier vs Wavelet Analysis"): same dark optichrome
template — title band, white-bordered panel grid, a right-hand caption column —
but the panel is a horizontal bar chart (the new BarPanel + Barh vocabulary)
instead of a spectrogram. Four transforms timed on the same audio, log time axis,
each bar carrying its own time + frequency-resolution character; the caption
column holds the shared test parameters.

Data source: ``assets/timing/timing_methods.csv`` (written by
``research/timing_methods.py``). This module only renders.
"""
from __future__ import annotations

import csv
import os

from .. import (
    Figure,
    BarPanel,
    Barh,
    Annotation,
    CompositePanel,
    TextPanel,
    SuptitlePanel,
    style,
)

LABEL_PANEL_UNITS = (1, 1)
PANEL_UNITS = (3, 1)

# Log-time axis: one decade per tick, ms below a second, seconds above.
_TIME_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
_TIME_TICK_LABELS = ["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _latest_method_rows() -> list[dict]:
    path = os.path.join(_repo_root(), "assets", "timing", "timing_methods.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    latest_ts = rows[-1]["timestamp"]
    return [r for r in rows if r["timestamp"] == latest_ts]


def _method_label(r: dict) -> str:
    method, backend = r["method"], r["backend"]
    if method == "CWT":
        return "SubShader CuPy CWT" if backend == "GPU" else "SubShader NumPy CWT"
    return method  # "SciPy STFT", "PyWavelet CWT"


def _bar_color(r: dict) -> str:
    if r["method"] == "CWT" and r["backend"] == "GPU":
        return style.PRIMARY_COLOR          # the CWT we run live (orange)
    if r.get("res_kind") == "log":
        return style.SECONDARY_COLOR        # other constant-Q CWTs (purple)
    return style.TICK_LABEL_COLOR           # STFT, linear (gray)


def _time_text(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.1f} ms"


def _res_text(r: dict) -> str:
    if r.get("res_kind") == "log":
        return "Variable — 1 Semitone / Octave"
    parts = (r.get("native_res") or "").split()
    hz = parts[0] if parts else "?"
    return f"Fixed — {hz} Hz everywhere"


def _params_caption(rows: list[dict]) -> str:
    r = rows[0]

    def _num(key):
        try:
            return float(r.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    samples = int(_num("chunk_size"))
    sr = _num("sample_rate")
    lo, hi = _num("freq_lo"), _num("freq_hi")
    nfreq = int(_num("num_freqs"))
    hi_txt = f"{hi / 1000:.1f} kHz" if hi >= 1000 else f"{hi:g} Hz"
    lines = [f"Samples:  {samples:,}"]
    if sr:
        lines.append(f"Sample rate:  {sr / 1000:.1f} kHz")
    if lo and hi:
        lines.append(f"Range:  {lo:g} Hz – {hi_txt}")
    if nfreq:
        lines.append(f"Resolution:  12 / octave")
        lines.append(f"Frequencies:  {nfreq}")
    return "\n".join(lines)


def _caption_panel(title: str, caption: str) -> CompositePanel:
    """Right-hand caption column — same construction as Figure 1's row labels."""
    body = TextPanel(
        caption,
        units=LABEL_PANEL_UNITS,
        font_size=style.DEFAULT_SUBTITLE_FONT_SIZE,
        min_font_size=14,
        color=style.TICK_LABEL_COLOR,
        fontweight="bold",
        auto_shrink=True,
        cell_padding_frac=0.0,
        justify=False,
        show_ghost_border=True,
        top_anchor=True,
    )
    return CompositePanel(units=LABEL_PANEL_UNITS, title=title, rows=[[body]])


def build_figure() -> Figure:
    rows = _latest_method_rows()
    if not rows:
        raise RuntimeError("No method timing rows in assets/timing/timing_methods.csv")
    rows = sorted(rows, key=lambda r: float(r["mean_ms"]))  # fastest first (top)
    n = len(rows)
    ys = list(range(n))
    widths = [float(r["mean_ms"]) for r in rows]
    colors = [_bar_color(r) for r in rows]

    bar_panel = BarPanel(
        units=PANEL_UNITS,
        xscale="log",
        xticks=_TIME_TICKS,
        xticklabels=_TIME_TICK_LABELS,
        xlim=(1.0, 130_000.0),
        ylim=(n - 0.4, -0.6),  # fastest (index 0) at the top, room for top label
    )
    bar_panel.add(Barh(ys, widths, colors=colors))
    # Axis caption placed by hand (axes-relative) below the unit ticks — the
    # shared x-label inset would land it on the tick row in this short cell.
    bar_panel.add(Annotation(
        "Time (Log Scale)", (0.5, -0.17), transform="axes",
        color=style.TICK_LABEL_COLOR, fontsize=style.DEFAULT_AXIS_LABEL_SIZE,
        ha="center", va="center", zorder=6,
    ))

    # Names are too long for a left axis margin, so each bar is labeled in place:
    # "Name · time" on the gap just above the bar, resolution character inside.
    # White reads on every bar color and on the dark ground; left-aligned at the
    # axis floor so the labels form a clean column.
    for i, r in enumerate(rows):
        bar_panel.add(Annotation(
            f"{_method_label(r)}    {_time_text(float(r['mean_ms']))}",
            (1.25, i - 0.42),
            color=style.NEUTRAL_COLOR, fontsize=style.DEFAULT_BAR_VALUE_FONT_SIZE,
            fontweight="bold", ha="left", va="center", zorder=6,
        ))
        bar_panel.add(Annotation(
            _res_text(r), (1.25, i + 0.02),
            color=style.NEUTRAL_COLOR, fontsize=style.DEFAULT_BAR_INBAR_FONT_SIZE,
            ha="left", va="center", zorder=6,
        ))

    caption = _caption_panel("Test Setup", _params_caption(rows))

    total_w = PANEL_UNITS[0] + LABEL_PANEL_UNITS[0]
    return Figure.compose(
        rows=[
            [SuptitlePanel("Fourier vs Wavelet Performance", units=(total_w, 1))],
            [bar_panel, caption],
            [SuptitlePanel("Figure — SubShader Timing", units=(total_w, 1))],
        ],
        row_heights=[0.25, 1.0, 0.25],
        total_width_inches=style.FIGURE_WIDTH_INCHES,
        unit_height_inches=8.0,  # taller data cell so tick + axis-label chrome clears
        dpi=style.FIGURE_DPI,
        header_band_inches=style.HEADER_BAND_INCHES,
        show_cell_borders=True,
    )


def render(output_dir: str = "assets/images/dsp/figures/timing",
           output_filename: str = "timing_methods_dsplot.png") -> str:
    """Build, render, save. Returns absolute output path."""
    fig = build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render())
