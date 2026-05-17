"""dsp_motivator_v*.png renderers — 3-row signal/STFT/CWT motivator figures.

Each motivator is a vertical stack:
  row 0 — time-series + instantaneous-frequency overlay (twin y-axes)
  row 1 — STFT magnitude spectrogram (log-frequency y, time x)
  row 2 — CWT magnitude spectrogram (log-frequency y, time x)

Row 0 is the canonical TWIN-AXIS ESCAPE HATCH: the time-series and
inst-freq curves share an x-axis but live on independent y-scales, which
doesn't fit the single-Axes Plottable contract. The figure uses
matplotlib's `twinx()` directly, scoped to this module — no Plottable
abstraction.

Per D-01, this figure module is consumer code and may import canonical
research utilities (compute_full_cwt, plot_*_spectrogram, build_*_chirp).
Per D-05, motivator-specific layout constants derive locally from
dsplot.style.DEFAULT_* (no hardcoded layout numbers).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt

from dsplot import style

# Canonical figure utilities — allowed import per D-01 (figures/ is
# consumer code that may use research.utilities for raw signal/CWT math).
from utilities import (
    BouncingChirpConfig,
    WaypointChirpConfig,
    compute_full_cwt,
    create_grid_scaffold,
    plot_cwt_spectrogram,
    plot_inst_freq,
    plot_stft_spectrogram,
    plot_time_series,
)
from utilities.dsp_helpers import build_bouncing_chirp, build_waypoint_chirp


# ============================================================
# Local layout — derived from dsplot.style defaults per D-05.
# Override globally by reassigning dsplot.style.DEFAULT_*; override
# motivator-only by editing these module-level constants.
# ============================================================
LAYOUT_HSPACE       = style.DEFAULT_HSPACE         # row gap (axes-fraction)
LAYOUT_LABEL_RATIO  = style.DEFAULT_LABEL_RATIO    # left "label column" width
LAYOUT_MARGIN       = style.DEFAULT_PANEL_MARGIN   # outer figure margin
ROW_LABEL_FONT_SIZE = style.DEFAULT_ROW_LABEL_SIZE
AXIS_LABEL_FONT_SIZE = style.DEFAULT_AXIS_LABEL_SIZE
TICK_LABEL_FONT_SIZE = style.DEFAULT_TICK_LABEL_SIZE

DEFAULT_FIGSIZE = (18.0, 13.0)


@dataclass
class MotivatorConfig:
    """Per-version config for a motivator figure."""
    name: str
    chirp: "BouncingChirpConfig | WaypointChirpConfig"
    output_filename: str
    figsize: tuple[float, float] = DEFAULT_FIGSIZE


# ============================================================
# Canonical six versions — same parameters as research.dsp_figures.
# ============================================================
VERSIONS: tuple[MotivatorConfig, ...] = (
    MotivatorConfig(
        name="v4",
        chirp=BouncingChirpConfig(
            duration_s=0.5,
            f_decades=(100.0, 2000.0),
            bounces_per_decade=3,
        ),
        output_filename="dsp_motivator_v4_100-2000hz_0.5s.png",
    ),
    MotivatorConfig(
        name="v5",
        chirp=BouncingChirpConfig(
            duration_s=1.0,
            f_decades=(50.0, 5000.0),
            bounces_per_decade=3,
        ),
        output_filename="dsp_motivator_v5_50-5000hz_1.0s.png",
    ),
    MotivatorConfig(
        name="vw1_gentle",
        chirp=WaypointChirpConfig(
            duration_s=0.5,
            waypoints=(
                (0.00, 250.0),
                (0.22, 110.0),
                (0.50, 950.0),
                (0.72, 240.0),
                (1.00, 2000.0),
            ),
        ),
        output_filename="dsp_motivator_vw1_gentle_100-2000hz_0.5s.png",
    ),
    MotivatorConfig(
        name="vw2_moderate",
        chirp=WaypointChirpConfig(
            duration_s=0.5,
            waypoints=(
                (0.00, 350.0),
                (0.18, 105.0),
                (0.42, 1100.0),
                (0.65, 200.0),
                (0.85, 1500.0),
                (1.00, 2000.0),
            ),
        ),
        output_filename="dsp_motivator_vw2_moderate_100-2000hz_0.5s.png",
    ),
    MotivatorConfig(
        name="vw3_aggressive",
        chirp=WaypointChirpConfig(
            duration_s=0.5,
            waypoints=(
                (0.00, 400.0),
                (0.14, 105.0),
                (0.34, 1300.0),
                (0.55, 180.0),
                (0.78, 1700.0),
                (1.00, 2000.0),
            ),
        ),
        output_filename="dsp_motivator_vw3_aggressive_100-2000hz_0.5s.png",
    ),
    MotivatorConfig(
        name="vw4_aggressive",
        chirp=WaypointChirpConfig(
            duration_s=2.0,
            waypoints=(
                (0.00, 50.0),
                (0.07, 22.0),
                (0.18, 800.0),
                (0.30, 60.0),
                (0.42, 4000.0),
                (0.55, 250.0),
                (0.68, 12000.0),
                (0.78, 1500.0),
                (0.88, 14000.0),
                (0.95, 18000.0),
                (1.00, 20000.0),
            ),
            clip_to_waypoints=False,
        ),
        output_filename="dsp_motivator_vw4_aggressive_20-20000hz_2.0s.png",
    ),
)


def _build_signal(cfg: MotivatorConfig):
    """Build (signal, inst_freq, t, f_lo, f_hi) for either chirp flavor."""
    if isinstance(cfg.chirp, WaypointChirpConfig):
        signal, inst_freq, t = build_waypoint_chirp(
            cfg.chirp.sr,
            cfg.chirp.duration_s,
            cfg.chirp.waypoints,
            bc_type=cfg.chirp.bc_type,
            clip_to_waypoints=cfg.chirp.clip_to_waypoints,
        )
        wp_freqs = [f for _, f in cfg.chirp.waypoints]
        return signal, inst_freq, t, min(wp_freqs), max(wp_freqs)
    signal, inst_freq, t = build_bouncing_chirp(
        sr=cfg.chirp.sr,
        duration_s=cfg.chirp.duration_s,
        f_decades=list(cfg.chirp.f_decades),
        bounces_per_decade=cfg.chirp.bounces_per_decade,
        seed=cfg.chirp.seed,
    )
    return signal, inst_freq, t, cfg.chirp.f_decades[0], cfg.chirp.f_decades[-1]


def render_one(cfg: MotivatorConfig, output_dir: str) -> str:
    """Render a single 3-row motivator figure. Returns absolute output path."""
    signal, inst_freq, t, f_lo, f_hi = _build_signal(cfg)

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, cfg.chirp.sr,
        root_note_hz=f_lo,
        num_octaves=max(1, math.ceil(math.log2(f_hi / f_lo))),
    )

    # Slice companion arrays to the exact sample range the CWT covers,
    # then re-zero the time axis. All subplots now share x ∈ [0, duration_s].
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]
    duration_s = (end_sample - start_sample) / cfg.chirp.sr

    # Two-column grid: hidden left "label column" + data column for the
    # three panels. Row titles get overlaid via fig.text after layout
    # finalizes (so we can read the data-column position precisely).
    fig, axes = create_grid_scaffold(
        n_rows=3, n_cols=2,
        figsize=cfg.figsize,
        hspace=LAYOUT_HSPACE,
        wspace=0.0,
        width_ratios=[LAYOUT_LABEL_RATIO, 1.0],
    )
    for r in range(3):
        axes[r][0].axis("off")

    ax_top, ax_stft, ax_cwt = axes[0][1], axes[1][1], axes[2][1]

    # Row 0 — TWIN-AXIS ESCAPE HATCH: time-series + inst-freq on
    # independent y-scales sharing one x. Time-series y-axis hidden so
    # the inst-freq twin tells the whole "y = frequency" story.
    plot_time_series(ax_top, signal, cfg.chirp.sr)
    ax_top.set_yticks([])
    ax_top.spines['left'].set_visible(False)

    ax_top_twin = ax_top.twinx()
    plot_inst_freq(ax_top_twin, inst_freq, t, cwt_freqs)
    ax_top_twin.yaxis.tick_right()
    ax_top_twin.yaxis.set_label_position("right")
    ax_top_twin.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_top_twin.minorticks_off()
    ax_top_twin.patch.set_visible(False)
    for spine in ax_top_twin.spines.values():
        spine.set_edgecolor(style.SPINE_COLOR)
        spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)

    plot_stft_spectrogram(ax_stft, signal, cfg.chirp.sr,
                          freq_lim_hz=(cwt_freqs[0], cwt_freqs[-1]),
                          duration_s=duration_s)
    plot_cwt_spectrogram(ax_cwt, cwt_data, duration_s, cwt_freqs)

    # Move STFT and CWT y-axis ticks/labels to the right (visual rhyme
    # with row 0's right-side inst-freq ticks).
    for ax in (ax_stft, ax_cwt):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # Axis units: "f (Hz)" vertical on the right of every row, "t (s)"
    # under the bottom panel only.
    for ax in (ax_top_twin, ax_stft, ax_cwt):
        ax.set_ylabel("f (Hz)", fontsize=AXIS_LABEL_FONT_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=4,
                      rotation=270, va='center')
    ax_cwt.set_xlabel("t (s)", fontsize=AXIS_LABEL_FONT_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=4)

    for ax in (ax_top, ax_top_twin, ax_stft, ax_cwt):
        ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)

    # X-tick labels on bottom panel only.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top_twin.get_xticklabels(), visible=False)
    plt.setp(ax_stft.get_xticklabels(), visible=False)

    # Symmetric margins (in absolute inches): horizontal margin is scaled
    # by H/W so left/right padding equals top/bottom padding in inches.
    fig_w, fig_h = cfg.figsize
    v_margin = LAYOUT_MARGIN
    h_margin = v_margin * fig_h / fig_w
    fig.subplots_adjust(
        left=h_margin, right=1 - h_margin,
        top=1 - v_margin, bottom=v_margin,
        hspace=LAYOUT_HSPACE,
        wspace=0.0,
    )

    # Row titles in the hidden label column, vertically centered on each
    # data row. fig.canvas.draw() forces layout-resolution so .get_position()
    # returns the FINAL bbox (after subplots_adjust).
    fig.canvas.draw()
    label_col_pos = axes[0][0].get_position()
    label_x = (label_col_pos.x0 + label_col_pos.x1) / 2
    row_titles = (
        "Time Series + \nInst. Frequency",
        "Fourier (STFT) \nAnalysis",
        "Wavelet Analysis",
    )
    for r, title in enumerate(row_titles):
        row_pos = axes[r][1].get_position()
        label_y = row_pos.y0 + row_pos.height / 2
        fig.text(label_x, label_y, title,
                 ha="center", va="center",
                 fontsize=ROW_LABEL_FONT_SIZE,
                 color=style.TICK_LABEL_COLOR,
                 fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.output_filename)
    fig.savefig(output_path, facecolor=fig.get_facecolor(),
                dpi=style.DEFAULT_DPI)
    plt.close(fig)
    return os.path.abspath(output_path)


def render_all(output_dir: str, suffix: str = "") -> list[str]:
    """Render every canonical motivator version. Returns paths in render order.

    ``suffix`` is inserted before ``.png`` on every output filename so the
    new renders coexist with originals during visual diff.
    """
    paths: list[str] = []
    for cfg in VERSIONS:
        if suffix:
            stem, ext = os.path.splitext(cfg.output_filename)
            patched = MotivatorConfig(
                name=cfg.name,
                chirp=cfg.chirp,
                output_filename=f"{stem}{suffix}{ext}",
                figsize=cfg.figsize,
            )
        else:
            patched = cfg
        paths.append(render_one(patched, output_dir))
    return paths
