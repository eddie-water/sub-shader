"""
DSP.md figure generators.

Generators for figures embedded in src/subshader/dsp/DSP.md. Output PNGs land
in assets/images/generated/ alongside the README figures, prefixed with "dsp_"
for namespacing.

Each figure config is an explicit dataclass entry — easy to tweak parameters
and re-render any single version without touching others.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utilities import (
    BouncingChirpConfig,
    WaypointChirpConfig,
    IMAGES_DSP_DIR,
    IMAGES_GENERATED_DIR,
    compute_full_cwt,
    compute_freq_yticks,
    create_grid_scaffold,
    create_panel_row,
    plot_cwt_spectrogram,
    plot_inst_freq,
    plot_projection,
    plot_stft_spectrogram,
    plot_time_series,
    plot_vector,
    print_section_end,
    print_section_start,
    setup_vector_axes,
)
from utilities import style
from utilities.dsp_helpers import build_bouncing_chirp, build_waypoint_chirp


# =============================================================================
# §1 MOTIVATOR — 3-row vertical stack (signal-overlay / STFT / CWT) on a
# bouncing chirp
# =============================================================================

DEFAULT_MOTIVATOR_FIGSIZE = (18.0, 13.0)


@dataclass
class ChirpFigureConfig:
    """Per-version config for a §1 chirp motivator figure.

    `chirp` may be either a `BouncingChirpConfig` (procedural decade-bounce
    pattern) or a `WaypointChirpConfig` (explicit user-shaped curve).
    """
    name: str
    chirp: BouncingChirpConfig | WaypointChirpConfig
    output_filename: str
    figsize: tuple[float, float] = DEFAULT_MOTIVATOR_FIGSIZE


MOTIVATOR_VERSIONS = [
    ChirpFigureConfig(
        name="v4",
        chirp=BouncingChirpConfig(
            duration_s=0.5,
            f_decades=(100.0, 2000.0),
            bounces_per_decade=3,
        ),
        output_filename="dsp_motivator_v4_100-2000hz_0.5s.png",
    ),
    ChirpFigureConfig(
        name="v5",
        chirp=BouncingChirpConfig(
            duration_s=1.0,
            f_decades=(50.0, 5000.0),
            bounces_per_decade=3,
        ),
        output_filename="dsp_motivator_v5_50-5000hz_1.0s.png",
    ),

    # Waypoint variants over the v4 frame (100-2000 Hz, 0.5s):
    # deep dip near 100 → swing up to ~1k → dip back down → final rise to 2k.
    # Three slope-aggressiveness levels — tighter time fractions = steeper slope.
    ChirpFigureConfig(
        name="vw1_gentle",
        chirp=WaypointChirpConfig(
            duration_s=0.5,
            waypoints=(
                (0.00, 250.0),
                (0.22, 110.0),   # deep dip near low
                (0.50, 950.0),   # peak ~1k
                (0.72, 240.0),   # dip back down
                (1.00, 2000.0),  # final rise to ceiling
            ),
        ),
        output_filename="dsp_motivator_vw1_gentle_100-2000hz_0.5s.png",
    ),
    ChirpFigureConfig(
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
    ChirpFigureConfig(
        name="vw3_aggressive",
        chirp=WaypointChirpConfig(
            duration_s=0.5,
            waypoints=(
                (0.00, 400.0),
                (0.14, 105.0),   # tight dip → steep slope
                (0.34, 1300.0),  # tight peak ~1k+
                (0.55, 180.0),
                (0.78, 1700.0),
                (1.00, 2000.0),
            ),
        ),
        output_filename="dsp_motivator_vw3_aggressive_100-2000hz_0.5s.png",
    ),

    # Full-range chirp (20 Hz - 20 kHz). Same aggressive shape as before,
    # compressed so the visible window ends at exactly 1.0s with the curve
    # landing at 20 kHz. The spline runs to 1.4s but compute_full_cwt crops
    # ~0.4s off the edges, leaving 20 kHz visible at the right edge of the
    # displayed time axis. Descent waypoints past 20 kHz are cropped.
    ChirpFigureConfig(
        name="vw4_aggressive",
        chirp=WaypointChirpConfig(
            duration_s=2.0,
            waypoints=(
                (0.00, 50.0),
                (0.07, 22.0),       # deep dip near 20 Hz
                (0.18, 800.0),      # peak 1
                (0.30, 60.0),       # dip
                (0.42, 4000.0),     # peak 2
                (0.55, 250.0),      # dip
                (0.68, 12000.0),    # peak 3
                (0.78, 1500.0),     # dip
                (0.88, 14000.0),    # peak 4
                (0.95, 18000.0),    # near top
                (1.00, 20000.0),    # land at 20 kHz
            ),
            clip_to_waypoints=False,
        ),
        output_filename="dsp_motivator_vw4_aggressive_20-20000hz_2.0s.png",
    ),
]


def render_motivator(cfg: ChirpFigureConfig, output_dir: str) -> str:
    """Render a single 3-row motivator figure for one ChirpFigureConfig.

    Layout (top → bottom):
      1. time series + instantaneous-frequency overlay (twin y-axes, shared x)
      2. STFT magnitude spectrogram (log-frequency y, time x)
      3. CWT magnitude spectrogram (log-frequency y, time x)

    Returns the absolute output path.
    """
    if isinstance(cfg.chirp, WaypointChirpConfig):
        signal, inst_freq, t = build_waypoint_chirp(
            cfg.chirp.sr,
            cfg.chirp.duration_s,
            cfg.chirp.waypoints,
            bc_type=cfg.chirp.bc_type,
            clip_to_waypoints=cfg.chirp.clip_to_waypoints,
        )
        wp_freqs = [f for _, f in cfg.chirp.waypoints]
        f_lo, f_hi = min(wp_freqs), max(wp_freqs)
    else:
        signal, inst_freq, t = build_bouncing_chirp(
            sr=cfg.chirp.sr,
            duration_s=cfg.chirp.duration_s,
            f_decades=list(cfg.chirp.f_decades),
            bounces_per_decade=cfg.chirp.bounces_per_decade,
            seed=cfg.chirp.seed,
        )
        f_lo, f_hi = cfg.chirp.f_decades[0], cfg.chirp.f_decades[-1]

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, cfg.chirp.sr,
        root_note_hz=f_lo,
        num_octaves=max(1, math.ceil(math.log2(f_hi / f_lo))),
    )

    # Slice companion arrays to the exact sample range the CWT data covers,
    # then re-zero the time axis. All subplots now share x ∈ [0, duration_s].
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    t = t[start_sample:end_sample] - t[start_sample] if len(t) > start_sample else t[:0]
    duration_s = (end_sample - start_sample) / cfg.chirp.sr

    # Two-column layout: a hidden left "label column" reserved for row titles
    # (placed via fig.text after layout finalizes), and a data column for the
    # 3 panels. Mirrors the row-label pattern in research/comparison.py.
    fig, axes = create_grid_scaffold(
        n_rows=3, n_cols=2,
        figsize=cfg.figsize,
        hspace=style.MOTIVATOR_HSPACE,
        wspace=0.0,
        width_ratios=[style.MOTIVATOR_LABEL_RATIO, 1.0],
    )
    for r in range(3):
        axes[r][0].axis("off")  # label column — text overlaid below

    ax_top = axes[0][1]
    ax_stft = axes[1][1]
    ax_cwt = axes[2][1]

    # Row 0 — time series + inst-freq overlay. Time-series y-axis hidden so
    # the inst-freq twin's y-axis tells the whole story. Twin's ticks live
    # on the right (matches STFT/CWT below).
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
        spine.set_linewidth(style.SPINE_LINEWIDTH)

    plot_stft_spectrogram(ax_stft, signal, cfg.chirp.sr,
                          freq_lim_hz=(cwt_freqs[0], cwt_freqs[-1]),
                          duration_s=duration_s)
    plot_cwt_spectrogram(ax_cwt, cwt_data, duration_s, cwt_freqs)

    # Move STFT and CWT y-axis ticks/labels to the right.
    for ax in (ax_stft, ax_cwt):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # Axis units: "f (Hz)" vertical on the right of every row's y-axis,
    # "t (s)" under the bottom panel. Vertical orientation (rotation=270)
    # is matplotlib's convention for right-side ylabels and keeps the label
    # narrow enough to fit inside the right margin.
    for ax in (ax_top_twin, ax_stft, ax_cwt):
        ax.set_ylabel("f (Hz)", fontsize=style.MOTIVATOR_AXIS_LABEL_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=4, rotation=270,
                      va='center')
    ax_cwt.set_xlabel("t (s)", fontsize=style.MOTIVATOR_AXIS_LABEL_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=4)

    # Compact tick labels.
    for ax in (ax_top, ax_top_twin, ax_stft, ax_cwt):
        ax.tick_params(labelsize=style.MOTIVATOR_TICK_LABEL_SIZE)

    # X-tick labels on bottom panel only.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top_twin.get_xticklabels(), visible=False)
    plt.setp(ax_stft.get_xticklabels(), visible=False)

    # Symmetric margins (in absolute inches): horizontal margin is scaled by
    # H/W so left/right padding equals top/bottom padding in inches. hspace
    # picked so the inter-panel gap matches that absolute margin too.
    fig_w, fig_h = cfg.figsize
    v_margin = style.MOTIVATOR_MARGIN
    h_margin = v_margin * fig_h / fig_w
    fig.subplots_adjust(
        left=h_margin, right=1 - h_margin,
        top=1 - v_margin, bottom=v_margin,
        hspace=style.MOTIVATOR_HSPACE,
        wspace=0.0,
    )

    # Place row titles in the (hidden) label column. Centered between the
    # figure left edge and the data column's left edge.
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
                 fontsize=style.MOTIVATOR_ROW_LABEL_SIZE,
                 color=style.TICK_LABEL_COLOR,
                 fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.output_filename)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), dpi=style.DEFAULT_DPI)
    plt.close(fig)
    return output_path


def generate_motivator_versions(versions=None, output_dir=None) -> list:
    """Render all motivator versions. Returns list of output paths."""
    if versions is None:
        versions = MOTIVATOR_VERSIONS
    if output_dir is None:
        output_dir = IMAGES_GENERATED_DIR

    print_section_start(f"Generating {len(versions)} motivator versions")
    paths = []
    for cfg in versions:
        path = render_motivator(cfg, output_dir)
        print(f"  Saved -> {path}")
        paths.append(path)
    print_section_end()
    return paths


# =============================================================================
# ALIGNMENT DIAGNOSTIC — exposes per-frequency time shift in CWT.transform()
# =============================================================================

def generate_alignment_diagnostic(output_dir: str = None,
                                   output_filename: str = "dsp_alignment_diagnostic.png") -> str:
    """Render a figure that surfaces freq-dependent time shift in CWT output.

    Multi-tone burst signal: three Gaussian-windowed sines (80/250/600 Hz), all
    centered at the same instant. A correctly-aligned CWT would show three
    bright spots at the SAME displayed x. With the current trim
    (`conv_tf[:, :input_n]`), each row is shifted right by half_width(f),
    so the spots fan out — most-shifted at low freq, least at high freq.
    """
    if output_dir is None:
        output_dir = IMAGES_GENERATED_DIR

    sr = 44100
    duration_s = 0.4
    n = int(sr * duration_s)
    t = np.arange(n) / sr

    burst_t0 = 0.18
    burst_sigma = 0.012
    burst_freqs = [80.0, 250.0, 600.0]
    signal = np.zeros(n, dtype=np.float64)
    for f in burst_freqs:
        envelope = np.exp(-((t - burst_t0) / burst_sigma) ** 2)
        signal += envelope * np.sin(2 * np.pi * f * t)
    signal /= np.abs(signal).max()

    f_lo, f_hi = 50.0, 1000.0
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, sr,
        root_note_hz=f_lo,
        num_octaves=max(1, math.ceil(math.log2(f_hi / f_lo))),
    )

    sig_disp = signal[start_sample:end_sample]
    duration_disp = (end_sample - start_sample) / sr
    burst_t_disp = burst_t0 - start_sample / sr

    fig, axes = create_grid_scaffold(n_rows=2, n_cols=1, figsize=(20.0, 10.0))

    plot_time_series(axes[0][0], sig_disp, sr)
    axes[0][0].axvline(burst_t_disp, color="cyan", linewidth=1.5,
                       linestyle="--", alpha=0.85)

    plot_cwt_spectrogram(axes[1][0], cwt_data, duration_disp, cwt_freqs)
    axes[1][0].axvline(burst_t_disp, color="cyan", linewidth=1.5,
                       linestyle="--", alpha=0.85,
                       label=f"true burst time = {burst_t0*1000:.0f} ms")
    for f in burst_freqs:
        bin_idx = float(np.interp(f, cwt_freqs, np.arange(len(cwt_freqs))))
        axes[1][0].axhline(bin_idx, color="cyan", linewidth=0.8,
                           linestyle=":", alpha=0.55)
    axes[1][0].legend(loc="upper right", facecolor=style.BG_COLOR,
                      edgecolor=style.SPINE_COLOR, labelcolor="white")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), dpi=style.DEFAULT_DPI,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


# =============================================================================
# TOP-LEVEL ENTRY POINT
# =============================================================================

def generate_all_dsp_figures() -> None:
    """Generate all DSP.md figures."""
    generate_motivator_versions()
    print_section_start("Generating CWT alignment diagnostic")
    path = generate_alignment_diagnostic()
    print(f"  Saved -> {path}")
    print_section_end()
    generate_foundations_figures()


# =============================================================================
# §2 FOUNDATION FIGURES — vector arithmetic primitives that motivate the
# inner-product → projection → basis-function arc. All four panels share the
# same square-aspect look defined by style.VECTOR_*.
# =============================================================================

def _save(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=style.DEFAULT_DPI,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _plot_vector_xy_projection(output_dir: str = None,
                                filename: str = "vector_xy_projection.png") -> str:
    """§2.4.1 — Simple projection onto the x and y axes.

    Single panel: vector **a** drawn upper-right, with its x-component (aₓ) and
    y-component (aᵧ) drawn as orange arrows along each axis, plus dashed
    droplines from a's tip down/across to make the right-angle decomposition
    visible. Sets up the "components are projections along reference
    directions" framing that 2.4.2 generalizes.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=1,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.05,
    )
    ax = axes[0]
    setup_vector_axes(ax, lim=1.15)

    a = (0.80, 0.65)

    # Droplines from a's tip down to each axis (dashed, drawn first so they
    # sit behind the projection arrows).
    ax.plot([a[0], a[0]], [a[1], 0],
            color=style.VECTOR_DROPLINE_COLOR,
            alpha=style.VECTOR_DROPLINE_ALPHA,
            linewidth=1.2, linestyle="--", zorder=1)
    ax.plot([a[0], 0], [a[1], a[1]],
            color=style.VECTOR_DROPLINE_COLOR,
            alpha=style.VECTOR_DROPLINE_ALPHA,
            linewidth=1.2, linestyle="--", zorder=1)

    # x and y projection arrows — labels placed manually so they sit
    # outside the active diagram area instead of overlapping the original.
    plot_vector(ax, (a[0], 0.0), color=style.VECTOR_PROJ_COLOR,
                alpha=0.95, zorder=2)
    plot_vector(ax, (0.0, a[1]), color=style.VECTOR_PROJ_COLOR,
                alpha=0.95, zorder=2)
    ax.text(a[0] / 2, -0.10, "aₓ",
            color=style.VECTOR_PROJ_COLOR, fontweight="bold",
            fontsize=style.VECTOR_LABEL_FONT_SIZE,
            ha="center", va="top")
    ax.text(-0.10, a[1] / 2, "aᵧ",
            color=style.VECTOR_PROJ_COLOR, fontweight="bold",
            fontsize=style.VECTOR_LABEL_FONT_SIZE,
            ha="right", va="center")

    # Original vector on top
    plot_vector(ax, a, color=style.VECTOR_A_COLOR, label="a",
                alpha=1.0, zorder=3)

    return _save(fig, output_dir, filename)


def _plot_vector_xy_reconstruction(output_dir: str = None,
                                    filename: str = "vector_xy_reconstruction.png") -> str:
    """§2.4.1 — Components recombine in any order to reconstruct **a**.

    Two panels: a unit vector at angle pi/6 decomposed tip-to-tail in opposite
    orders (x then y vs y then x). Bold a + dashed components in the same
    color show the chain converges on the same tip either way.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=2,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.05,
    )

    a = (math.cos(math.pi / 6), math.sin(math.pi / 6))
    ax_comp = (a[0], 0.0)
    ay_comp = (0.0, a[1])
    vec_color = style.VECTOR_PROJ_COLOR

    for ax, order in zip(axes, ("xy", "yx")):
        setup_vector_axes(ax, lim=1.15,
                          show_border=False,
                          axis_style="arrow",
                          axis_labels=True)

        plot_vector(ax, a, color=vec_color, label="a",
                    alpha=1.0, zorder=4,
                    linewidth=style.VECTOR_BOLD_LINEWIDTH)

        if order == "xy":
            plot_vector(ax, ax_comp, color=vec_color,
                        alpha=0.95, zorder=3, linestyle="--")
            plot_vector(ax, ay_comp, origin=ax_comp,
                        color=vec_color, alpha=0.95, zorder=3, linestyle="--")
            ax.text(a[0] / 2, -0.10, "aₓ",
                    color=vec_color, fontweight="bold",
                    fontsize=style.VECTOR_LABEL_FONT_SIZE,
                    ha="center", va="top")
            ax.text(a[0] + 0.08, a[1] / 2, "aᵧ",
                    color=vec_color, fontweight="bold",
                    fontsize=style.VECTOR_LABEL_FONT_SIZE,
                    ha="left", va="center")
        else:
            plot_vector(ax, ay_comp, color=vec_color,
                        alpha=0.95, zorder=3, linestyle="--")
            plot_vector(ax, ax_comp, origin=ay_comp,
                        color=vec_color, alpha=0.95, zorder=3, linestyle="--")
            ax.text(-0.10, a[1] / 2, "aᵧ",
                    color=vec_color, fontweight="bold",
                    fontsize=style.VECTOR_LABEL_FONT_SIZE,
                    ha="right", va="center")
            ax.text(a[0] / 2, a[1] + 0.08, "aₓ",
                    color=vec_color, fontweight="bold",
                    fontsize=style.VECTOR_LABEL_FONT_SIZE,
                    ha="center", va="bottom")

    return _save(fig, output_dir, filename)


def _plot_vector_basics(output_dir: str = None,
                        filename: str = "vector_basics.png") -> str:
    """§2.4.1 — Several arrows of varying magnitude and direction.

    Single panel; no dot-product math, no labels — just visual proof that a
    vector is "an arrow with magnitude (length) and direction (angle)".
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=1,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.05,
    )
    ax = axes[0]
    setup_vector_axes(ax, lim=1.15)

    # Five arrows: short/long, varied angles. Hand-picked so they radiate from
    # the origin without overlapping or running off the panel.
    samples = [
        ((0.95, 0.55),  style.VECTOR_A_COLOR),
        ((-0.70, 0.80), style.VECTOR_B_COLOR),
        ((-0.85, -0.30), style.VECTOR_PROJ_COLOR),
        ((0.30, -0.90), style.GRID_WAVEFORM_COLOR),
        ((0.50, 0.18),  style.VECTOR_NEUTRAL_COLOR),
    ]
    for vec, color in samples:
        plot_vector(ax, vec, color=color, alpha=0.95)
    return _save(fig, output_dir, filename)


def _plot_dot_product_geometry(output_dir: str = None,
                               filename: str = "dot_product_geometry.png") -> str:
    """§2.4.1 — Four canonical angle cases: parallel-same, parallel-opposite,
    perpendicular, oblique.

    Each panel shows two unit-ish vectors a, b plus the sign-of-result
    annotation underneath, so the reader sees angle → sign at a glance. The
    oblique panel (added 2026-05) absorbs the standalone vector_similarity
    figure: same in-between case, fewer separate images.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(n_panels=4)

    # Panel 1 — parallel, same direction (positive)
    setup_vector_axes(
        axes[0],
        panel_title="Parallel, same direction",
        result_text="a · b  >  0",
    )
    plot_vector(axes[0], (0.90, 0.45), color=style.VECTOR_A_COLOR, label="a")
    plot_vector(axes[0], (0.50, 0.25), color=style.VECTOR_B_COLOR, label="b",
                origin=(0.0, -0.05))

    # Panel 2 — parallel, opposite direction (negative)
    setup_vector_axes(
        axes[1],
        panel_title="Parallel, opposite direction",
        result_text="a · b  <  0",
    )
    plot_vector(axes[1], (0.90, 0.45), color=style.VECTOR_A_COLOR, label="a")
    plot_vector(axes[1], (-0.55, -0.275), color=style.VECTOR_B_COLOR, label="b",
                origin=(0.0, 0.0))

    # Panel 3 — perpendicular (zero)
    setup_vector_axes(
        axes[2],
        panel_title="Perpendicular",
        result_text="a · b  =  0",
    )
    plot_vector(axes[2], (0.90, 0.45), color=style.VECTOR_A_COLOR, label="a")
    # b is rotated +90° from a → (-0.45, 0.90) is perpendicular
    plot_vector(axes[2], (-0.45, 0.90), color=style.VECTOR_B_COLOR, label="b")

    # Panel 4 — oblique (partial)
    setup_vector_axes(
        axes[3],
        panel_title="Oblique",
        result_text="a · b  >  0  (partial)",
    )
    plot_vector(axes[3], (0.95, 0.20), color=style.VECTOR_A_COLOR, label="a")
    plot_vector(axes[3], (0.55, 0.75), color=style.VECTOR_B_COLOR, label="b")

    return _save(fig, output_dir, filename)


def _plot_vector_similarity(output_dir: str = None,
                            filename: str = "vector_similarity.png") -> str:
    """§2.4.1 — Two oblique pairs: one acute (kind of similar), one obtuse
    (not so similar). Shows that similarity varies smoothly with angle, not
    just at the three canonical cases.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(n_panels=2)

    # Panel 1 — small angle → kind of similar (positive but not maxed)
    setup_vector_axes(
        axes[0],
        panel_title="Acute angle: kind of similar",
        result_text="a · b  >  0  (small)",
    )
    plot_vector(axes[0], (0.95, 0.20), color=style.VECTOR_A_COLOR, label="a")
    plot_vector(axes[0], (0.55, 0.75), color=style.VECTOR_B_COLOR, label="b")

    # Panel 2 — wide angle (obtuse) → not so similar (negative)
    setup_vector_axes(
        axes[1],
        panel_title="Obtuse angle: not so similar",
        result_text="a · b  <  0  (small)",
    )
    plot_vector(axes[1], (0.95, 0.20), color=style.VECTOR_A_COLOR, label="a")
    plot_vector(axes[1], (-0.55, 0.75), color=style.VECTOR_B_COLOR, label="b")

    return _save(fig, output_dir, filename)


def _plot_vector_projection(output_dir: str = None,
                            filename: str = "vector_projection.png") -> str:
    """§2.4.2 — Vector projection ("shadow") in two regimes: a long shadow
    (b largely lies along a → high similarity) vs a short shadow (b barely
    aligns with a → low similarity). The dropline makes the right-angle
    decomposition visible.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(n_panels=2)

    # Panel 1 — b largely projects onto a (long shadow)
    setup_vector_axes(
        axes[0],
        panel_title="Large projection",
        result_text="long shadow → similar",
    )
    a1 = (0.95, 0.20)
    b1 = (0.80, 0.55)
    plot_projection(axes[0], a1, b1)

    # Panel 2 — b barely projects onto a (short shadow)
    setup_vector_axes(
        axes[1],
        panel_title="Small projection",
        result_text="short shadow → not similar",
    )
    a2 = (0.95, 0.20)
    b2 = (0.10, 0.95)
    plot_projection(axes[1], a2, b2)

    return _save(fig, output_dir, filename)


def _plot_projection_reference_directions(
        output_dir: str = None,
        filename: str = "projection_reference_directions.png") -> str:
    """§2.4.1 — Projection onto a reference direction (3 panels juxtaposed).

    Panel 1 ("onto x and y axes"):
        Vector a projected onto the x and y axes — the axes are just a
        convenient pair of reference directions, and a's components along
        them ARE the projections (aₓ along x, aᵧ along y). Establishes
        that "projection" is the same operation whether the reference is
        a coordinate axis or another vector.

    Panel 2 ("a onto b"):
        Vector a projected onto vector b. The shadow lands along b's
        direction; its length is the dot-product magnitude.

    Panel 3 ("b onto a"):
        Reverses panel 2 — vector b projected onto vector a. The shadow
        looks visually different (different length, different direction)
        but the dot product comes out IDENTICAL: a·b = b·a = 1.00. This
        is the symmetry of the dot product made visible.

    Color palette (sourced from style.PALETTE_*):
      - Vector a   → PALETTE_PRIMARY   (orange) — stable identity in every panel
      - Vector b   → PALETTE_SECONDARY (purple) — stable identity in every panel
      - Components (panel 1)            → neutral off-white (shadows)
      - Projection arrows (panels 2-3)  → neutral off-white (shadows)
      - Spines / droplines / labels     → neutral grey (scaffolding)

    The pedagogical contract for color:
      - PRIMARY/SECONDARY paint vector IDENTITY: a is always orange, b is
        always purple, regardless of which direction the projection runs.
      - Anything that's a *projection result* (shadow) is neutral, so the
        viewer's eye sees "real thing" vs "shadow of a real thing" without
        having to read color labels.
      - The third palette color (TERTIARY gold) does NOT appear here — it's
        reserved for the third dimension and surfaces only in the 3D figure.

    Math sits in the markdown beneath the figure as LaTeX blocks (so \\vec{a}
    renders correctly via the markdown viewer's MathJax/KaTeX), not inside
    the figure as monospace text. Numbers chosen so the dot product lands
    on a clean integer:
      a = (1.0, 0.5),  b = (0.6, 0.8)
      a · b = (1.0)(0.6) + (0.5)(0.8) = 0.6 + 0.4 = 1.00
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=3,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.25,
        height=style.VECTOR_FIGSIZE_HEIGHT,
    )

    # Spines + axis crosshairs are neutral; role colors are reserved for the
    # vectors themselves. Per-axis x_color / y_color overrides therefore both
    # resolve to the neutral spine color.
    axis_kwargs = dict(
        x_color=_DIM_SPINE,
        y_color=_DIM_SPINE,
        axis_alpha=0.55,
        axis_linewidth=1.2,
    )

    # Same a, b vectors used across all 3 panels so the pedagogy is uniform:
    # the components shown in panel 1 are exactly the numbers that appear
    # inside the dot product expressions in panels 2 & 3.
    a = (1.0, 0.5)
    b = (0.6, 0.8)

    # Color tokens for this figure (2D — only PRIMARY + SECONDARY surface):
    a_color = _DIM_X_COLOR    # primary  — vector a, in every panel
    b_color = _DIM_Y_COLOR    # secondary — vector b, in every panel

    # ----- Panel 1: a projected onto x and y axes -----
    # Math annotations live in the markdown beneath the figure as LaTeX.
    # Panel intent: a is the subject (primary color); its components on the
    # axes are SHADOWS — neutral, just like the projection arrows in P2/P3.
    setup_vector_axes(axes[0], lim=1.25,
                      panel_title="onto x and y axes",
                      **axis_kwargs)

    axes[0].plot([a[0], a[0]], [a[1], 0],
                 color=style.VECTOR_DROPLINE_COLOR,
                 alpha=style.VECTOR_DROPLINE_ALPHA,
                 linewidth=1.2, linestyle="--", zorder=1)
    axes[0].plot([a[0], 0], [a[1], a[1]],
                 color=style.VECTOR_DROPLINE_COLOR,
                 alpha=style.VECTOR_DROPLINE_ALPHA,
                 linewidth=1.2, linestyle="--", zorder=1)

    # x and y component arrows: neutral shadows.
    plot_vector(axes[0], (a[0], 0.0),
                color=_DIM_NEUTRAL, alpha=0.9, zorder=2)
    plot_vector(axes[0], (0.0, a[1]),
                color=_DIM_NEUTRAL, alpha=0.9, zorder=2)
    axes[0].text(a[0] / 2, -0.10, "aₓ",
                 color=_DIM_NEUTRAL, fontweight="bold",
                 fontsize=style.VECTOR_LABEL_FONT_SIZE,
                 ha="center", va="top")
    axes[0].text(-0.10, a[1] / 2, "aᵧ",
                 color=_DIM_NEUTRAL, fontweight="bold",
                 fontsize=style.VECTOR_LABEL_FONT_SIZE,
                 ha="right", va="center")
    # Vector a in PRIMARY orange — the subject of this panel.
    plot_vector(axes[0], a, color=a_color, label="a",
                alpha=1.0, zorder=3,
                linewidth=style.VECTOR_BOLD_LINEWIDTH)

    # ----- Panels 2 & 3: same a, b; reference flips -----
    # Math goes in the markdown beneath the figure (LaTeX block).
    # Vector identity stays stable across panels: a is always primary orange,
    # b is always secondary purple. The shadow (the projection result) is
    # neutral in both panels — projections are shadows, shadows are neutral.
    def _draw_proj(ax, *, source, target,
                   source_color, target_color,
                   source_label, target_label):
        src = np.asarray(source, dtype=float)
        tgt = np.asarray(target, dtype=float)
        scale = float(np.dot(tgt, src)) / float(np.dot(tgt, tgt))
        foot = scale * tgt

        plot_vector(ax, target, color=target_color, label=target_label,
                    alpha=1.0, zorder=2,
                    linewidth=style.VECTOR_BOLD_LINEWIDTH)
        plot_vector(ax, source, color=source_color, label=source_label,
                    alpha=1.0, zorder=3,
                    linewidth=style.VECTOR_BOLD_LINEWIDTH)
        plot_vector(ax, tuple(foot), origin=(0.0, -0.04),
                    color=_DIM_NEUTRAL,
                    linewidth=style.VECTOR_LINEWIDTH + 0.4,
                    alpha=0.9, zorder=4)
        ax.plot([foot[0], src[0]], [foot[1], src[1]],
                color=style.VECTOR_DROPLINE_COLOR,
                alpha=style.VECTOR_DROPLINE_ALPHA,
                linewidth=1.2, linestyle="--", zorder=1)

    setup_vector_axes(axes[1], lim=1.25,
                      panel_title="a onto b",
                      **axis_kwargs)
    _draw_proj(axes[1], source=a, target=b,
               source_color=a_color, target_color=b_color,
               source_label="a", target_label="b")

    setup_vector_axes(axes[2], lim=1.25,
                      panel_title="b onto a",
                      **axis_kwargs)
    _draw_proj(axes[2], source=b, target=a,
               source_color=b_color, target_color=a_color,
               source_label="b", target_label="a")

    return _save(fig, output_dir, filename)


# Per-dimension palette aliases — pull from the centralized PALETTE_* constants
# in style.py so the 2D and 3D figures share one source of truth. Spines and
# decorators stay neutral; the role colors only paint pedagogically meaningful
# elements (vectors in 2D, dimension components in 3D).
_DIM_NEUTRAL = style.VECTOR_NEUTRAL_COLOR
_DIM_SPINE   = style.VECTOR_AXIS_COLOR        # spines, droplines, axis labels
_DIM_X_COLOR = style.PALETTE_PRIMARY          # vector a (2D) | x dim (3D)
_DIM_Y_COLOR = style.PALETTE_SECONDARY        # vector b (2D) | y dim (3D)
_DIM_Z_COLOR = style.PALETTE_TERTIARY         # 3rd dim only, surfaces in 3D


def _plot_vector_projection_3d(
        output_dir: str = None,
        filename: str = "vector_projection_3d.png") -> str:
    """§2.4.2 — 3D vector decomposed into x/y/z components, with two
    reconstruction paths in different orders both terminating at the same tip.

    Showcases two ideas in one image:
      1. Projection onto reference directions extends from 2D (axes) to 3D
         (still axes) with no change in operation.
      2. The component arrows can be added in any order and still reach the
         original vector — superposition is order-independent. Shown via two
         dashed reconstruction paths (x→y→z and z→y→x) whose segments are
         colored by dimension, so the viewer sees the SAME components
         rearranged in a different sequence.

    Color palette (sourced from style.PALETTE_*):
      - x-component segments  → PALETTE_PRIMARY   (orange) — same color as
                                                              vector a in 2D
      - y-component segments  → PALETTE_SECONDARY (purple) — same color as
                                                              vector b in 2D
      - z-component segments  → PALETTE_TERTIARY  (gold)   — surfaces ONLY
                                                              when a third
                                                              dimension exists
      - Spines (x, y, z lines through origin)  → neutral grey
      - Spine labels (x, y, z text)            → neutral grey
      - Vector a itself                        → neutral off-white

    The pedagogical contract for color:
      - Role colors paint MEANING (vector identity in 2D, dimension identity
        in 3D); they never paint scaffolding (spines, decorators).
      - The third color (gold) is reserved for the third dimension and only
        appears in 3D figures — its arrival signals "we just added a
        dimension," reinforcing the §2.4.1 → §2.4.2 transition.

    Visual style:
      - All default 3D axis machinery (panes, ticks, bounding box) is hidden
        via set_axis_off(). Three neutral axis SPINES are drawn manually
        through the origin, extending in BOTH directions (negative + positive)
        so the origin sits centered in the plot.
      - View angle (elev=38, azim=-55) keeps z roughly vertical and splays
        x and y forward toward the viewer, matching the reference perspective
        in assets/images/claude/3d_vector plot.png. Avoids matplotlib 3.10's
        positive-azim projection collapse.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig = plt.figure(
        figsize=(style.VECTOR_FIGSIZE_PER_PANEL * 1.6,
                 style.VECTOR_FIGSIZE_HEIGHT * 1.4),
        dpi=style.DEFAULT_DPI,
    )
    fig.patch.set_facecolor(style.BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(style.BG_COLOR)
    # Honour our own zorder; otherwise mpl's depth-sort can hide the
    # diagonal a-line behind segments with overlapping projection bounds.
    ax.computed_zorder = False
    # Hide every piece of default 3D axis chrome (panes, gridlines, ticks,
    # bounding-box edges). The spine look is rebuilt manually below.
    ax.set_axis_off()

    # Same vector a = (1.0, 0.5) used in the 2D projection figure
    # (projection_reference_directions.png), now extended into the third
    # dimension with z = 0.4 — "the same vector, plus depth out of the page."
    # Reusing the 2D vector keeps the §2.4.1 → §2.4.2 transition visually
    # continuous: the reader sees a familiar arrow gain a third component
    # rather than a brand-new vector with new numbers to track.
    a = np.array([1.0, 0.5, 0.4])

    # Symmetric lims matched to the 2D foundation figures' VECTOR_DEFAULT_LIM
    # (= 1.25) so the 3D plot reads at the same scale as the 2D panels it
    # extends. Origin sits centered in the plot; spines extend equally in
    # the negative and positive directions through (0, 0, 0).
    lim = style.VECTOR_DEFAULT_LIM
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # ----- Three muted axis spines through origin (both directions) -----
    spine_lw = 2.4
    ax.plot([-lim, lim], [0, 0], [0, 0],
            color=_DIM_SPINE, linewidth=spine_lw, zorder=1)
    ax.plot([0, 0], [-lim, lim], [0, 0],
            color=_DIM_SPINE, linewidth=spine_lw, zorder=1)
    ax.plot([0, 0], [0, 0], [-lim, lim],
            color=_DIM_SPINE, linewidth=spine_lw, zorder=1)

    # Spine labels at the positive ends, in the matching MUTED hue.
    label_size = style.VECTOR_LABEL_FONT_SIZE + 2
    ax.text(lim + 0.06, 0, 0, "x",
            color=_DIM_SPINE, fontsize=label_size,
            fontweight="bold", fontstyle="italic",
            ha="left", va="center")
    ax.text(0, lim + 0.06, 0, "y",
            color=_DIM_SPINE, fontsize=label_size,
            fontweight="bold", fontstyle="italic",
            ha="left", va="center")
    ax.text(0, 0, lim + 0.07, "z",
            color=_DIM_SPINE, fontsize=label_size,
            fontweight="bold", fontstyle="italic",
            ha="center", va="bottom")

    # ----- Reconstruction paths — SPOTLIGHT segments per dimension -----
    # Path 1: x → y → z (along the bottom-front edges of the parallelepiped)
    p1 = [
        np.array([0, 0, 0]),
        np.array([a[0], 0, 0]),
        np.array([a[0], a[1], 0]),
        np.array([a[0], a[1], a[2]]),
    ]
    p1_segment_colors = [_DIM_X_COLOR,
                         _DIM_Y_COLOR,
                         _DIM_Z_COLOR]
    for (p, q), seg_color in zip(zip(p1, p1[1:]), p1_segment_colors):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=seg_color, linewidth=2.4,
                linestyle="--", alpha=0.95, zorder=3)

    # Path 2: z → y → x (along the back-top edges of the parallelepiped)
    p2 = [
        np.array([0, 0, 0]),
        np.array([0, 0, a[2]]),
        np.array([0, a[1], a[2]]),
        np.array([a[0], a[1], a[2]]),
    ]
    p2_segment_colors = [_DIM_Z_COLOR,
                         _DIM_Y_COLOR,
                         _DIM_X_COLOR]
    for (p, q), seg_color in zip(zip(p2, p2[1:]), p2_segment_colors):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=seg_color, linewidth=2.4,
                linestyle="--", alpha=0.95, zorder=3)

    # ----- Vector a — neutral off-white, single solid line -----
    # ax.quiver in 3D under-renders thin lines on dark backgrounds, so use
    # ax.plot for the spine and a scatter dot for the tip "head".
    ax.plot([0, a[0]], [0, a[1]], [0, a[2]],
            color=_DIM_NEUTRAL,
            linewidth=style.VECTOR_BOLD_LINEWIDTH + 0.6,
            solid_capstyle="round",
            zorder=5)
    ax.scatter([a[0]], [a[1]], [a[2]],
               color=_DIM_NEUTRAL,
               s=80, zorder=6, depthshade=False)
    ax.text(a[0] + 0.05, a[1] + 0.05, a[2] + 0.04, "a",
            color=_DIM_NEUTRAL, fontweight="bold",
            fontsize=style.VECTOR_LABEL_FONT_SIZE)

    # ----- Order legend — top-left, neutral text since paths share colors -----
    ax.text2D(0.02, 0.97, "Path 1:  x → y → z",
              transform=ax.transAxes,
              color=style.VECTOR_PANEL_RESULT_COLOR,
              fontsize=13, family="monospace", va="top")
    ax.text2D(0.02, 0.92, "Path 2:  z → y → x",
              transform=ax.transAxes,
              color=style.VECTOR_PANEL_RESULT_COLOR,
              fontsize=13, family="monospace", va="top")

    # elev=38 + azim=-55 keeps z near vertical with x/y splayed forward,
    # matching the reference image's perspective. Positive azim (e.g. 35)
    # triggers a matplotlib 3.10 projection collapse — keep azim negative.
    ax.view_init(elev=38, azim=-55)

    return _save(fig, output_dir, filename)


def generate_foundations_figures(output_dir: str = None) -> list:
    """Render all §2 foundation figures. Returns list of output paths.

    Lineup as of 2026-05 §2.4 consolidation:
      - vector_basics.png ............... §2.4.1 beat 1 (vector = arrow)
      - projection_reference_directions . §2.4.1 beats 2-4 (axes / a→b / b→a)
      - dot_product_geometry.png (4 pan) . §2.4.1 beat 5 (angle → sign)
      - vector_projection_3d.png ........ §2.4.2 (3D + superposition)
      - vector_xy_reconstruction.png .... §2.6 (basis-function recombination)

    Helpers retained but not dispatched here (available for future re-use):
      _plot_vector_xy_projection (replaced by panel 1 of the 3-panel figure),
      _plot_vector_similarity (absorbed by the oblique panel of the 4-panel
      sign-cases figure), _plot_vector_projection (long/short shadow — its
      message is now carried by the 4-panel figure's parallel + oblique cases).
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    print_section_start(f"Generating §2 foundation figures -> {output_dir}/")
    paths = [
        _plot_vector_basics(output_dir),
        _plot_projection_reference_directions(output_dir),
        _plot_dot_product_geometry(output_dir),
        _plot_vector_projection_3d(output_dir),
        _plot_vector_xy_reconstruction(output_dir),
    ]
    for p in paths:
        print(f"  Saved -> {p}")
    print_section_end()
    return paths


if __name__ == "__main__":
    generate_all_dsp_figures()
