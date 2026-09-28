"""Ledger evidence - Fix 3 (down-sample: index decimation → block max-pool).

The output-side half of a ledger entry: the SAME audio, the SAME CWT, the
same hop-center magnitudes, reduced to the display width both ways, plus
their difference. Three rows, one shared color scale:

    before  - index decimation (keep 1 column in 128) - what Fix 3 replaced
    after   - block max-pool (peak hold over each 128-column block)
    diff    - after − before: everything decimation dropped between its kept
              columns (never negative: a block's max ≥ any one column in it)

Frames are laid side by side exactly as the renderer scrolls them, so the
figure reads as the scalogram the user sees. Real pipeline objects
(``CpuCWT``, ``AudioReader``) - nothing simulated.

Chrome is figure 1's stacked-contender layout, reused piece for piece
(``_contender_tight_style``, ``_side_text_panel`` caption squares, the
STACK_* geometry, in-cell label strip, range-bar y-axis, cell borders) so
ledger figures read as the same family as the README/DSP figures. Iteration
outputs land in NEW ``ledger_fix3_downsample_v*.png`` files - never overwrite
an existing PNG.
"""
from __future__ import annotations

import os

import numpy as np

from subshader.audio.reader import AudioReader
from subshader.config import CWTConfig
from subshader.dsp.cwt import CpuCWT

from .. import Figure, HeatmapPanel, Heatmap, SuptitlePanel
from .gen_figure_1_stft_vs_cwt import (
    _auto_xticks, _contender_tight_style, _side_text_panel,
    _apply_hero_band_xaxis,
    CONTENDER_CAPTION_FONT_SIZE, HERO_LABEL_STRIP_IN, HERO_CELL_VPAD_IN,
    STACK_PLOT_UNITS, STACK_TEXT_UNITS, STACK_PLOT_ROW_HEIGHT,
    STACK_BAND_HEIGHT, STACK_W_UNIT_INCHES, STACK_UNIT_INCHES,
    STACK_GUTTER_INCHES,
)
from .gen_timing_gantt import _repo_root

AUDIO = "assets/audio/reference/beltran_sc_rip_4_bar.wav"
N_FRAMES = 24                 # ~4.5 s of the clip at 186 ms per frame
VMAX_PCT = 99.5
DISPLAY_FREQ_TICKS = (30, 300, 3000, 20000)   # chromatic bank spans 27.5 Hz - 21 kHz
TITLE = "Fix 3 - Down-sample: decimation → max-pool"

ROWS = (
    ("before", "index decimation - keep 1 column in 128, drop the rest"),
    ("after", "block max-pool - peak hold over every 128-column block"),
    ("diff", "after − before - what decimation dropped between its kept columns"),
)


def decimate(coefs: np.ndarray, target_width: int) -> np.ndarray:
    """The pre-Fix-3 down-sample, verbatim from the patch's removed lines."""
    _, num_samples = coefs.shape
    hop = num_samples / target_width
    indices = np.floor(np.arange(target_width) * hop).astype(int)
    indices = np.clip(indices, 0, num_samples - 1)
    return coefs[:, indices]


def hop_center_frames(n_frames: int):
    """(list of hop-center magnitude matrices, cwt, cfg) from the real pipeline."""
    cfg = CWTConfig(file_path=os.path.join(_repo_root(), AUDIO))
    reader = AudioReader(cfg)
    cfg.sample_rate = float(reader.sample_rate)
    cwt = CpuCWT(cfg)
    frames = []
    for _ in range(n_frames):
        chunk = reader.get_chunk()
        if chunk is None:
            break
        raw = cwt.transform(cwt.pre(chunk))
        mag = cwt._compute_mag(cwt._normalize_by_scale(raw))
        frames.append(cwt.extract_hop_center(cwt.discard_unreliable_coefs(mag)))
    return frames, cwt, cfg


def _plot_panel(data, duration_s, freqs, xticks, vmax, *, last: bool) -> HeatmapPanel:
    # Hero treatment: no per-axis unit labels - the footer corner names both
    # axes ("f vs t") once the hero band x-axis pass runs.
    # (The bottom row keeps x_label="s": the band pass locates it by that
    # label, then replaces it with the corner label.)
    panel = HeatmapPanel(units=STACK_PLOT_UNITS, xticks=xticks,
                         x_label="s" if last else None,
                         show_xticklabels=last)
    panel.add(Heatmap(data, duration_s=duration_s, freqs=freqs, log_freq=True,
                      tick_freqs=DISPLAY_FREQ_TICKS,
                      extent=(0.0, duration_s, 0.0, float(len(freqs))),
                      vmin=0.0, vmax=vmax))
    # Same in-cell treatment as figure 1's stacked hero: y labels live in a
    # strip inside the plot's own cell, the data axes fills the cell height,
    # range-bar y-axis.
    panel.content_left_pad_inches = HERO_LABEL_STRIP_IN
    panel.fill_cell_vertical = True
    panel.fill_cell_pad_inches = HERO_CELL_VPAD_IN
    panel.range_bar_yaxis = True
    return panel


OVERLAP_ROWS = (
    ("after", "block max-pool - Fix 3 as shipped: 128-column blocks, no overlap"),
    ("overlap", "overlapped max-pool - window 256, stride 128: each bin also sees its neighbor's columns"),
    ("diff", "overlap − after - the smoothing: every event bleeds one bin wider"),
)


def overlap_pool(pool: np.ndarray) -> np.ndarray:
    """Window-256 / stride-128 max-pool, from the shipped non-overlap pool.

    max over blocks i and i+1 == max over the overlapping 256-column window
    starting at block i. The last bin clamps (no wraparound).
    """
    shifted = np.concatenate([pool[:, 1:], pool[:, -1:]], axis=1)
    return np.maximum(pool, shifted)


def build_figure(n_frames: int = N_FRAMES) -> Figure:
    frames, cwt, cfg = hop_center_frames(n_frames)
    width = cwt.output_n
    before = np.hstack([decimate(f, width) for f in frames])
    after = np.hstack([cwt.downsample(f, width) for f in frames])
    diff = after - before
    duration_s = len(frames) * cfg.hop_size / cfg.sample_rate
    vmax = float(np.percentile(after, VMAX_PCT))
    xticks = _auto_xticks(duration_s)

    rows = []
    for (title, caption), data in zip(ROWS, (before, after, diff)):
        last = title == ROWS[-1][0]
        rows.append([
            _plot_panel(data, duration_s, cwt.freqs, xticks, vmax, last=last),
            _side_text_panel(title, caption, font_size=CONTENDER_CAPTION_FONT_SIZE),
        ])

    total_w = STACK_PLOT_UNITS[0] + STACK_TEXT_UNITS[0]
    rows.insert(0, [SuptitlePanel(TITLE, units=(total_w, 1), font_size=44)])
    rows.append([SuptitlePanel("", units=(total_w, 1), font_size=34)])
    row_heights = ([STACK_BAND_HEIGHT] + [STACK_PLOT_ROW_HEIGHT] * len(ROWS)
                   + [STACK_BAND_HEIGHT])
    u_w, u_h = STACK_W_UNIT_INCHES, STACK_UNIT_INCHES
    return Figure.compose(
        rows=rows, row_heights=row_heights,
        hspace=STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT),
        wspace=STACK_GUTTER_INCHES / u_w,
        dpi=150, unit_inches=u_w, unit_height_inches=u_h,
        show_cell_borders=True,
    )


def build_overlap_figure(n_frames: int = N_FRAMES) -> Figure:
    """Mock: would overlapping pool windows smooth the display? Three rows -
    the shipped max-pool, the 2x-overlap variant, and their difference."""
    frames, cwt, cfg = hop_center_frames(n_frames)
    width = cwt.output_n
    after = np.hstack([cwt.downsample(f, width) for f in frames])
    overlap = overlap_pool(after)
    diff = overlap - after
    duration_s = len(frames) * cfg.hop_size / cfg.sample_rate
    vmax = float(np.percentile(after, VMAX_PCT))
    xticks = _auto_xticks(duration_s)

    rows = []
    for (title, caption), data in zip(OVERLAP_ROWS, (after, overlap, diff)):
        last = title == OVERLAP_ROWS[-1][0]
        rows.append([
            _plot_panel(data, duration_s, cwt.freqs, xticks, vmax, last=last),
            _side_text_panel(title, caption, font_size=CONTENDER_CAPTION_FONT_SIZE),
        ])

    total_w = STACK_PLOT_UNITS[0] + STACK_TEXT_UNITS[0]
    rows.insert(0, [SuptitlePanel("Down-sample - overlapped max-pool mock",
                                  units=(total_w, 1), font_size=44)])
    rows.append([SuptitlePanel("", units=(total_w, 1), font_size=34)])
    row_heights = ([STACK_BAND_HEIGHT]
                   + [STACK_PLOT_ROW_HEIGHT] * len(OVERLAP_ROWS)
                   + [STACK_BAND_HEIGHT])
    u_w, u_h = STACK_W_UNIT_INCHES, STACK_UNIT_INCHES
    return Figure.compose(
        rows=rows, row_heights=row_heights,
        hspace=STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT),
        wspace=STACK_GUTTER_INCHES / u_w,
        dpi=150, unit_inches=u_w, unit_height_inches=u_h,
        show_cell_borders=True,
    )


def render(output_path: str | None = None, *,
           builder=build_figure) -> str:
    with _contender_tight_style():
        fig = builder()
        fig.render()
        # Same post-render passes as ``render_hero_stacked``: no physical tick
        # stubs, x-axis rebuilt as footer-band furniture with the "f vs t"
        # corner label. (No top-row clamp: this figure keeps its title band.)
        for mpl_ax in fig._mpl_fig.axes:
            mpl_ax.tick_params(axis="both", which="both", length=0)
        _apply_hero_band_xaxis(fig)
    path = output_path or os.path.join(_repo_root(), "assets", "timing",
                                       "ledger_fix3_downsample_v1.png")
    fig.savefig(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "overlap":
        print(render(args[1] if len(args) > 1 else None,
                     builder=build_overlap_figure))
    else:
        print(render(args[0] if args else None))
