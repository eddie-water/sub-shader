"""DEPRECATED — DSP.md figure generators moved to ``research/dsplot/figures/``.

Backwards-compatibility shim. Importing this module emits a DeprecationWarning.

Migration map:
  A, A_PRIME, B, A_Z, FOUND_LIM  -> dsplot.figures.foundation_constants
  ChirpFigureConfig              -> dsplot.figures.motivator.MotivatorConfig
  MOTIVATOR_VERSIONS             -> dsplot.figures.motivator.VERSIONS
  render_motivator(cfg)          -> dsplot.figures.motivator.render_one
  generate_motivator_versions()  -> dsplot.figures.motivator.render_all
  generate_alignment_diagnostic  -> dsplot.figures.alignment_diagnostic.render
  generate_foundations_figures   -> dispatched by dsplot.figures.__main__
  generate_all_dsp_figures()     -> dsplot.figures.__main__.main

``python research/dsp_figures.py`` still works — delegates to the new dispatcher.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "research.dsp_figures is deprecated — use research.dsplot.figures instead. "
    "See the migration map in this module's docstring.",
    DeprecationWarning,
    stacklevel=2,
)

# Foundation constants (canonical vector values for §2.4 figures).
from research.dsplot.figures.foundation_constants import (  # noqa: E402,F401
    A,
    A_PRIME,
    A_Z,
    B,
    FOUND_LIM,
)

# Motivator: aliases mapping legacy names to the new ones.
from research.dsplot.figures.motivator import (  # noqa: E402
    MotivatorConfig as ChirpFigureConfig,
    VERSIONS as MOTIVATOR_VERSIONS,
    render_all as generate_motivator_versions,
    render_one as render_motivator,
)

# Alignment diagnostic: legacy name -> new render() entry point.
from research.dsplot.figures.alignment_diagnostic import (  # noqa: E402
    render as generate_alignment_diagnostic,
)


def generate_all_dsp_figures() -> int:
    """Regenerate every DSP.md figure via the dsplot dispatcher.

    Delegates to ``research.dsplot.figures.__main__.main`` so the legacy
    command ``python research/dsp_figures.py`` keeps producing the same set
    of PNGs at the same paths.
    """
<<<<<<< Updated upstream
    from research.dsplot.figures.__main__ import main
    return main([])
=======
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

# ----- Canonical foundation-figure vectors — single source of truth ----------
# Every dispatched §2.4 figure (xy_reconstruction, projection_reconstruction,
# 3D extension) reads these. DSP.md §2.4.1 mirrors them in its alt-text and
# LaTeX block — if you change a value here, also bump DSP.md to match.
A         = (2.0, 3.0)    # subject vector  — orange (PALETTE_PRIMARY)
A_PRIME   = (-2.0, 3.0)   # sibling of A    — same orange identity, label-distinct
B         = (3.0, 2.0)    # reference vector — purple (PALETTE_SECONDARY)
A_Z       = 1.5           # extra z used only by the 3D extension figure
FOUND_LIM = 6             # symmetric axis extent for the 2D foundation figures

# ----- Per-dimension role colors — used by both 2D and 3D figures ------------
_DIM_NEUTRAL = style.VECTOR_NEUTRAL_COLOR
_DIM_SPINE   = style.VECTOR_AXIS_COLOR        # spines, droplines, axis labels
_DIM_X_COLOR = style.PALETTE_PRIMARY          # vector A (2D) | x dim (3D)
_DIM_Y_COLOR = style.PALETTE_SECONDARY        # vector B (2D) | y dim (3D)
_DIM_Z_COLOR = style.PALETTE_TERTIARY         # 3rd dim only, surfaces in 3D


def _save(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=style.DEFAULT_DPI,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _plot_vector_xy_projection(output_dir: str = None,
                                filename: str = "vector_xy_projection.png") -> str:
    """RETIRED — not in `generate_foundations_figures` dispatch. Hardcoded
    vector is a frozen snapshot; this helper does NOT track the canonical
    A constant. Revive by adding to the dispatch list and refactoring to
    read A / FOUND_LIM if you want it back in the pipeline.

    §2.4.1 — Simple projection onto the x and y axes.

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


def _plot_sharp_vector(ax, vec, *, origin=(0.0, 0.0), color=None,
                       label=None, linewidth=None, alpha=1.0,
                       linestyle="-", zorder=2,
                       head_length=0.3, head_width=0.1,
                       mutation_scale=22,
                       label_offset=0.30,
                       label_dx=0.0, label_dy=0.0):
    """Draw an arrow with a thin, sharply-pointed head via FancyArrowPatch.

    Same call shape as utilities.plotting.plot_vector but renders the head
    with `matplotlib.patches.FancyArrowPatch` so head shape and size are
    tunable independent of stroke linewidth.

    - ``head_length`` / ``head_width``: head dimensions in the arrowstyle
      DSL (larger length-to-width ratio = pointier).
    - ``mutation_scale``: overall size multiplier for the head in points.
    - ``label_offset``: where the label lands relative to the arrow tip.
        * scalar (default 0.30): multiplier on the vector's unit direction,
          pushing the label past the tip along the arrow's heading.
        * tuple/list (dx, dy): absolute offset in data coords; the label
          lands at (tip.x + dx, tip.y + dy), independent of arrow direction.
          Use when you want to REPLACE the default direction logic.
    - ``label_dx`` / ``label_dy``: additive per-axis nudges in data coords
      applied AFTER ``label_offset`` resolves. Positive dx → right, negative
      dx → left, positive dy → up, negative dy → down. Use these to bump an
      individual label in any direction without abandoning the default
      "past the tip" placement — e.g., ``label_dy=-0.3`` lowers just one
      label, leaving every other vector's label at the standard offset.
    """
    from matplotlib.patches import FancyArrowPatch

    if linewidth is None:
        linewidth = style.VECTOR_LINEWIDTH

    src = np.asarray(origin, dtype=float)
    vec_arr = np.asarray(vec, dtype=float)
    tip = src + vec_arr

    if linestyle != "-":
        # Dashed (or other non-solid) shaft: render as a plain dashed line
        # via ax.plot, then place a SOLID arrowhead at the tip via a tiny
        # FancyArrowPatch. Without this split, FancyArrowPatch applies the
        # dash pattern to the arrowhead outline as well, which renders as
        # a doubled / ghosted head over the solid triangle.
        ax.plot([src[0], tip[0]], [src[1], tip[1]],
                color=color, alpha=alpha,
                linewidth=linewidth, linestyle=linestyle,
                solid_capstyle="butt", zorder=zorder)
        norm = float(np.linalg.norm(vec_arr))
        if norm > 1e-9:
            unit = vec_arr / norm
            # Stub leading up to the tip — short enough that its line
            # contribution is invisible, but long enough that the head
            # renders at full size with the correct orientation.
            stub_start = tip - 1e-3 * unit
            head = FancyArrowPatch(
                posA=tuple(stub_start),
                posB=tuple(tip),
                arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
                mutation_scale=mutation_scale,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle="-",
                zorder=zorder + 0.1,
                shrinkA=0,
                shrinkB=0,
                joinstyle="miter",
                capstyle="butt",
            )
            ax.add_patch(head)
    else:
        # Solid: render the whole arrow as one FancyArrowPatch.
        arrow = FancyArrowPatch(
            posA=tuple(src),
            posB=tuple(tip),
            arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
            mutation_scale=mutation_scale,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
        )
        ax.add_patch(arrow)

    if label is not None:
        if isinstance(label_offset, (tuple, list, np.ndarray)):
            # Absolute (dx, dy) offset in data coords — pushes the label
            # in any direction regardless of the vector's orientation.
            offset_xy = np.asarray(label_offset, dtype=float)
            label_pos = tip + offset_xy
        else:
            # Scalar multiplier on the unit vector — pushes label past
            # the tip along the vector's direction (default behavior).
            norm = float(np.linalg.norm(vec_arr))
            if norm > 1e-9:
                unit = vec_arr / norm
                label_pos = tip + float(label_offset) * unit
            else:
                label_pos = tip
        # Additive per-axis nudges on top of whatever base offset was applied.
        label_pos = label_pos + np.array([label_dx, label_dy], dtype=float)
        ax.text(label_pos[0], label_pos[1], label,
                color=color, fontweight="bold",
                fontsize=style.VECTOR_LABEL_FONT_SIZE,
                ha="center", va="center",
                zorder=zorder + 1)


def _panel_titles(ax, title: str, subtitle: str):
    """Apply a two-tier panel title (title + italic subtitle) to an axis.

    Uses style.py's canonical font hierarchy (TITLE_FONT_SIZE for the
    title, TICK_LABEL_SIZE for the subtitle). Pad on the title is sized
    so the subtitle (placed at y=1.02 in axes coords) sits cleanly
    between the title and the axes top. Shared by every §2.4.1 figure
    so the panel-title styling is one-source-of-truth.
    """
    ax.set_title(title,
                 color=style.TICK_LABEL_COLOR,
                 fontsize=style.TITLE_FONT_SIZE,
                 loc="center", pad=38)
    ax.text(0.5, 1.02, subtitle,
            transform=ax.transAxes,
            fontsize=style.TICK_LABEL_SIZE,
            color=style.TICK_LABEL_COLOR, alpha=0.80,
            ha="center", va="bottom",
            style="italic")


def _draw_dashed_tip_to_tail(ax, vec, *, first_axis: str = "x",
                             color=None, zorder: int = 3):
    """Draw the two-segment tip-to-tail decomposition of `vec` using dashed
    neutral component arrows.

    `first_axis="x"` draws the x-component from origin, then the y-component
    from the x-component's tip. `first_axis="y"` does the reverse. Both
    segments share the same color and linestyle so they read as one
    scaffolding pair, not two competing arrows.
    """
    x_comp = (vec[0], 0.0)
    y_comp = (0.0, vec[1])
    if first_axis == "x":
        seg1, seg2_origin, seg2 = x_comp, x_comp, y_comp
    elif first_axis == "y":
        seg1, seg2_origin, seg2 = y_comp, y_comp, x_comp
    else:
        raise ValueError(f"first_axis must be 'x' or 'y', got {first_axis!r}")
    _plot_sharp_vector(ax, seg1, color=color,
                       alpha=0.95, zorder=zorder, linestyle="--")
    _plot_sharp_vector(ax, seg2, origin=seg2_origin, color=color,
                       alpha=0.95, zorder=zorder, linestyle="--")


def _plot_vector_xy_reconstruction(
        output_dir: str = None,
        filename: str = "components_recombine_either_order_v18.png") -> str:
    """§2.4.1 — Basic Vector Projection (3-panel figure with overall title).

    Reads canonical vectors A and A_PRIME from the module-level constants
    block. Comments here describe panel ROLES, not specific numeric values —
    bump A / A_PRIME / FOUND_LIM to change the geometry, and every panel
    follows automatically.

    Panel 1 — "Vector Projection onto Axes":
        Vector A drawn from origin with its x and y components shown as
        dashed neutral arrows along the axes. Dashed droplines from A's
        tip make the right-angle decomposition explicit. Establishes
        "component = shadow along reference axis."

    Panel 2 — "Tip-To-Tail Reconstruction":
        Same A, with BOTH reconstruction orders (x-then-y and y-then-x)
        overlaid on the same axes — together they form a bounding
        rectangle around A. Visually proves the components can be
        combined in any order.

    Panel 3 — "Perpendicular Components":
        Adds sibling vector A_PRIME. The pedagogy is "the two vectors
        share one component but differ in the other — measuring along
        that one axis is unaffected by what happens along the other."
        A and A_PRIME share the PRIMARY orange identity — the prime is
        the only label distinction.
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=3,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.35,
    )

    fig.suptitle("Basic Vector Projection",
                 color=style.TICK_LABEL_COLOR,
                 fontsize=style.SUPTITLE_FONT_SIZE,
                 fontweight="bold",
                 y=style.SUPTITLE_Y)

    ax_comp    = (A[0], 0.0)
    ay_comp    = (0.0, A[1])

    a_color    = _DIM_X_COLOR             # vectors A and A_PRIME — orange
    comp_color = _DIM_NEUTRAL             # dashed component shadows
    label_size = style.VECTOR_LABEL_FONT_SIZE
    axis_kwargs = dict(lim=FOUND_LIM, show_border=False,
                       axis_style="arrow", axis_labels=True)

    def _label(ax, x, y, text, *, ha, va):
        ax.text(x, y, text, color=a_color, fontweight="bold",
                fontsize=label_size, ha=ha, va=va)

    # ----- Panel 1: classic projection onto x and y axes -----
    setup_vector_axes(axes[0], **axis_kwargs)
    _panel_titles(axes[0],
                  "Vector Projection onto Axes",
                  r"Projecting $\vec{a}$ onto $\hat{x}$ and $\hat{y}$")
    # Droplines — thinner than the vector and component arrows.
    axes[0].plot([A[0], A[0]], [A[1], 0],
                 color=style.VECTOR_DROPLINE_COLOR,
                 alpha=style.VECTOR_DROPLINE_ALPHA,
                 linewidth=2, linestyle="--", zorder=1)
    axes[0].plot([A[0], 0], [A[1], A[1]],
                 color=style.VECTOR_DROPLINE_COLOR,
                 alpha=style.VECTOR_DROPLINE_ALPHA,
                 linewidth=2, linestyle="--", zorder=1)
    _plot_sharp_vector(axes[0], ax_comp, color=comp_color,
                       alpha=0.95, zorder=2, linestyle="--")
    _plot_sharp_vector(axes[0], ay_comp, color=comp_color,
                       alpha=0.95, zorder=2, linestyle="--")
    _label(axes[0], A[0] / 2, -0.30,    "aₓ", ha="center", va="top")
    _label(axes[0], -0.30,    A[1] / 2, "aᵧ", ha="right",  va="center")
    _plot_sharp_vector(axes[0], A, color=a_color, label="a",
                       alpha=1.0, zorder=3,
                       linewidth=style.VECTOR_BOLD_LINEWIDTH)

    # ----- Panel 2: both reconstruction orders on the same axes -----
    setup_vector_axes(axes[1], **axis_kwargs)
    _panel_titles(axes[1],
                  "Tip-To-Tail Reconstruction",
                  r"$\hat{x}$ then $\hat{y}$  |  $\hat{y}$ then $\hat{x}$")
    _draw_dashed_tip_to_tail(axes[1], A, first_axis="x", color=comp_color)
    _draw_dashed_tip_to_tail(axes[1], A, first_axis="y", color=comp_color)
    # Label each edge of the bounding rectangle so the reader sees the same
    # component twice in each direction (one per reconstruction order).
    _label(axes[1], A[0] / 2, -0.30,       "aₓ", ha="center", va="top")
    _label(axes[1], A[0] / 2, A[1] + 0.24, "aₓ", ha="center", va="bottom")
    _label(axes[1], -0.30,       A[1] / 2, "aᵧ", ha="right",  va="center")
    _label(axes[1], A[0] + 0.24, A[1] / 2, "aᵧ", ha="left",   va="center")
    _plot_sharp_vector(axes[1], A, color=a_color, label="a",
                       alpha=1.0, zorder=4,
                       linewidth=style.VECTOR_BOLD_LINEWIDTH)

    # ----- Panel 3: perpendicular components (A and A_PRIME) -----
    setup_vector_axes(axes[2], **axis_kwargs)
    _panel_titles(axes[2],
                  "Perpendicular Components",
                  "PLACEHOLDER")
    _draw_dashed_tip_to_tail(axes[2], A,       first_axis="x", color=comp_color)
    _draw_dashed_tip_to_tail(axes[2], A_PRIME, first_axis="x", color=comp_color)
    # Component labels — each placed on the outer side of its component so
    # neither label sits inside the V formed by the two vectors. With both
    # vectors reconstructed x-first, the x-components sit on the bottom edge.
    _label(axes[2], A[0] / 2,       -0.30,             "aₓ",  ha="center", va="top")
    _label(axes[2], A[0] + 0.24,    A[1] / 2,          "aᵧ",  ha="left",   va="center")
    _label(axes[2], A_PRIME[0] / 2, -0.30,             "a′ₓ", ha="center", va="top")
    _label(axes[2], A_PRIME[0] - 0.24, A_PRIME[1] / 2, "a′ᵧ", ha="right",  va="center")
    _plot_sharp_vector(axes[2], A, color=a_color, label="a",
                       alpha=1.0, zorder=4,
                       linewidth=style.VECTOR_BOLD_LINEWIDTH)
    _plot_sharp_vector(axes[2], A_PRIME, color=a_color, label="a′",
                       alpha=1.0, zorder=4,
                       linewidth=style.VECTOR_BOLD_LINEWIDTH)

    # Leave a generous gap between the suptitle and the subplots so the
    # figure title, panel titles, and panel subtitles each have breathing room.
    fig.subplots_adjust(top=0.74)

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
    """RETIRED — not in `generate_foundations_figures` dispatch. Hardcoded
    vectors are a frozen snapshot; does NOT track the canonical A / B.

    §2.4.1 — Two oblique pairs: one acute (kind of similar), one obtuse
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
    """RETIRED — not in `generate_foundations_figures` dispatch. Hardcoded
    vectors are a frozen snapshot; does NOT track the canonical A / B.

    §2.4.2 — Vector projection ("shadow") in two regimes: a long shadow
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
    """RETIRED — not in `generate_foundations_figures` dispatch. Hardcoded
    vectors are a frozen snapshot; does NOT track the canonical A / B.

    §2.4.1 — Projection onto a reference direction (3 panels juxtaposed).

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


def _plot_dot_product_symmetry(
        output_dir: str = None,
        filename: str = "dot_product_symmetry.png") -> str:
    """RETIRED — not in `generate_foundations_figures` dispatch. Hardcoded
    vectors are a frozen snapshot; does NOT track the canonical A / B.
    Superseded by `_plot_projection_reconstruction_either_order`.

    §2.4.1 — Dot-product symmetry (a · b = b · a) made visible.

    Two panels juxtaposed:
      Panel 1 ("a onto b"): a projected onto b — shadow lands along b's direction.
      Panel 2 ("b onto a"): roles reversed — shadow lands along a's direction.

    The shadows look visually different (different lengths along different
    reference directions), yet the dot product comes out identical:
    a · b = b · a = 1.00. The reference direction is a choice, not a
    constraint — that's the symmetry.

    Color palette (sourced from style.PALETTE_*):
      - Vector a   → PALETTE_PRIMARY   (orange) — stable identity in both panels
      - Vector b   → PALETTE_SECONDARY (purple) — stable identity in both panels
      - Projection shadow + droplines  → neutral (shadows are neutral)
      - Spines / axis crosshairs       → neutral grey (scaffolding)

    Math sits in the markdown beneath the figure as LaTeX blocks. Numbers:
      a = (1.0, 0.5),  b = (0.6, 0.8)
      a · b = (1.0)(0.6) + (0.5)(0.8) = 1.00
      b · a = (0.6)(1.0) + (0.8)(0.5) = 1.00
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=2,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.25,
        height=style.VECTOR_FIGSIZE_HEIGHT,
    )

    axis_kwargs = dict(
        x_color=_DIM_SPINE,
        y_color=_DIM_SPINE,
        axis_alpha=0.55,
        axis_linewidth=1.2,
    )

    a = (1.0, 0.5)
    b = (0.6, 0.8)
    a_color = _DIM_X_COLOR    # primary  — vector a, in both panels
    b_color = _DIM_Y_COLOR    # secondary — vector b, in both panels

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

    setup_vector_axes(axes[0], lim=1.25,
                      panel_title="a onto b",
                      **axis_kwargs)
    _draw_proj(axes[0], source=a, target=b,
               source_color=a_color, target_color=b_color,
               source_label="a", target_label="b")

    setup_vector_axes(axes[1], lim=1.25,
                      panel_title="b onto a",
                      **axis_kwargs)
    _draw_proj(axes[1], source=b, target=a,
               source_color=b_color, target_color=a_color,
               source_label="b", target_label="a")

    return _save(fig, output_dir, filename)


def _plot_projection_reconstruction_either_order(
        output_dir: str = None,
        filename: str = "projection_reconstruction_either_order_v9.png") -> str:
    """§2.4.1 — Vector Projection Symmetry (2-panel figure matching figure 1's style).

    Reads canonical vectors A and B from the module-level constants block.
    Bump A / B / FOUND_LIM to change the geometry — both panels follow
    automatically. DSP.md §2.4.1 LaTeX block mirrors the dot product
    (a·b = A[0]*B[0] + A[1]*B[1]); bump there too on any change.

    Demonstrates that a·b = b·a by showing two parallelogram decompositions
    side by side. Each panel projects one vector onto the other and overlays
    both reconstruction paths (parallel-first vs perpendicular-first), so
    "components combine in any order" reads at-a-glance.

    Panel 1 — "Project a onto b":
        Reference = b. Decompose a into:
          - a_parallel = (a·b / b·b) * b   (along b — the projection shadow)
          - a_perp     = a - a_parallel    (orthogonal to b)
        Both component arrows drawn from origin as neutral shadows. Two
        dashed reconstruction paths (parallel→perp and perp→parallel)
        close the parallelogram at a's tip.

    Panel 2 — "Project b onto a":
        Roles flipped — reference = a, subject = b. Same parallelogram
        structure with the components of b along/orthogonal to a.

    Visual style is shared with the components_recombine figure
    (`_plot_vector_xy_reconstruction`): canonical SUPTITLE/TITLE/TICK
    font hierarchy from style.py, sharp arrowheads via `_plot_sharp_vector`,
    `FOUND_LIM` axes, `_panel_titles` for per-axis title + subtitle.

    Color palette:
      - Vector a   → PALETTE_PRIMARY   (orange) — stable identity in both panels
      - Vector b   → PALETTE_SECONDARY (purple) — stable identity in both panels
      - Component arrows (parallel / perp)       → neutral (shadows)
      - Reconstruction paths (dashed)            → neutral (scaffolding)
      - Spines / axis crosshairs                 → neutral grey
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=2,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.35,
    )

    fig.suptitle("Vector Projection Symmetry",
                 color=style.TICK_LABEL_COLOR,
                 fontsize=style.SUPTITLE_FONT_SIZE,
                 fontweight="bold",
                 y=style.SUPTITLE_Y)

    axis_kwargs = dict(
        lim=FOUND_LIM,
        show_border=False,
        axis_style="arrow",
        axis_labels=True,
        x_color=_DIM_SPINE,
        y_color=_DIM_SPINE,
        axis_alpha=0.55,
        axis_linewidth=1.2,
    )

    a_color = _DIM_X_COLOR    # primary  — vector A (orange)
    b_color = _DIM_Y_COLOR    # secondary — vector B (purple)

    def _draw_reconstruction(ax, *, source, target,
                             source_color, target_color,
                             source_label, target_label):
        """Decompose `source` onto `target` and draw both reconstruction paths.

        - source = the vector being decomposed (subject of the panel)
        - target = the reference direction onto which `source` is projected
        """
        src = np.asarray(source, dtype=float)
        tgt = np.asarray(target, dtype=float)
        scale = float(np.dot(tgt, src)) / float(np.dot(tgt, tgt))
        par = scale * tgt          # parallel component (along target)
        per = src - par            # perpendicular component (orthogonal)

        # Reference vector (target) — drawn first, sits beneath component
        # arrows so the parallel shadow visually rides along it.
        _plot_sharp_vector(ax, target, color=target_color, label=target_label,
                    alpha=1.0, zorder=2,
                    linewidth=style.VECTOR_BOLD_LINEWIDTH)

        # Two component arrows from origin — both neutral shadows.
        _plot_sharp_vector(ax, tuple(par),
                    color=_DIM_NEUTRAL,
                    linewidth=style.VECTOR_LINEWIDTH + 0.4,
                    alpha=0.9, zorder=3)
        _plot_sharp_vector(ax, tuple(per),
                    color=_DIM_NEUTRAL,
                    linewidth=style.VECTOR_LINEWIDTH + 0.4,
                    alpha=0.9, zorder=3)

        # Two dashed reconstruction paths — completing the parallelogram.
        # Path 1: par-tip → source-tip   (parallel first, then perp)
        ax.plot([par[0], src[0]], [par[1], src[1]],
                color=style.VECTOR_DROPLINE_COLOR,
                alpha=style.VECTOR_DROPLINE_ALPHA,
                linewidth=1.2, linestyle="--", zorder=1)
        # Path 2: per-tip → source-tip   (perp first, then parallel)
        ax.plot([per[0], src[0]], [per[1], src[1]],
                color=style.VECTOR_DROPLINE_COLOR,
                alpha=style.VECTOR_DROPLINE_ALPHA,
                linewidth=1.2, linestyle="--", zorder=1)

        # Subject vector on top — primary stroke, the destination of both paths.
        _plot_sharp_vector(ax, source, color=source_color, label=source_label,
                    alpha=1.0, zorder=4,
                    linewidth=style.VECTOR_BOLD_LINEWIDTH)

    # ----- Panel 1: a onto b -----
    setup_vector_axes(axes[0], **axis_kwargs)
    _panel_titles(axes[0],
                  "Project a onto b",
                  r"shadow of $\vec{a}$ along $\hat{b}$")
    _draw_reconstruction(axes[0], source=A, target=B,
                         source_color=a_color, target_color=b_color,
                         source_label="a", target_label="b")

    # ----- Panel 2: b onto a -----
    setup_vector_axes(axes[1], **axis_kwargs)
    _panel_titles(axes[1],
                  "Project b onto a",
                  r"shadow of $\vec{b}$ along $\hat{a}$")
    _draw_reconstruction(axes[1], source=B, target=A,
                         source_color=b_color, target_color=a_color,
                         source_label="b", target_label="a")

    # Leave a generous gap between the suptitle and the panel titles, same
    # as figure 1 (`_plot_vector_xy_reconstruction`).
    fig.subplots_adjust(top=0.74)

    return _save(fig, output_dir, filename)


def _plot_dot_product_twin_rectangles(
        output_dir: str = None,
        filename: str = "dot_product_twin_rectangles_v2.png") -> str:
    """§2.4.2 PREVIEW — Dot product as rectangle area, twin-panel symmetry.

    Each panel projects one vector onto the other and adds a translucent
    rectangle whose **area** equals the dot product. The two rectangles have
    visibly different shapes (different side lengths) but the **same area**,
    making `a · b = b · a` visible at a glance.

    Uses local vectors `_TR_A = (1, 4)` and `_TR_B = (3, 1)` (not the canonical
    `A`, `B`) — chosen so `|a| ≠ |b|`, which makes the two rectangles end up
    with visibly different proportions (one squat, one tall) instead of
    congruent-but-rotated. Dot product = 7.

    Rectangle geometry per panel:
      - Subject S, Reference R, projection length `p = S·R / |R|`
      - One side along R-direction, length = p (the shadow length)
      - Adjacent side perpendicular to R, length = |R|
      - Area = p × |R| = S · R
      - Perpendicular extends on the side where S sits, so the rectangle
        wraps around the subject vector visually.

    The shadow (rectangle's bottom edge, riding on R) is drawn as a thick
    bright neutral line on top of the rectangle outline so the projection
    length is unmissable. The dot product scalar floats large in the upper
    corner — no math text inside the panel.

    Color palette:
      - Vector a (orange) and b (purple) — stable identity across panels.
      - Shadow line — bright neutral, on the reference vector's path.
      - Rectangle fill — single neutral translucent across both panels so
        "same area" reads as one quantity, twice.
      - Spines / axis crosshairs — neutral grey.
    """
    _TR_A = (1.0, 4.0)   # subject in panel 1 (tall) — orange
    _TR_B = (3.0, 1.0)   # subject in panel 2 (wide) — purple


    from matplotlib.patches import Polygon

    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    fig, axes = create_panel_row(
        n_panels=2,
        panel_size=style.VECTOR_FIGSIZE_PER_PANEL * 1.3,
        height=style.VECTOR_FIGSIZE_HEIGHT * 1.35,
    )

    fig.suptitle("Dot Product Symmetry — Equal Areas",
                 color=style.TICK_LABEL_COLOR,
                 fontsize=style.SUPTITLE_FONT_SIZE,
                 fontweight="bold",
                 y=style.SUPTITLE_Y)

    axis_kwargs = dict(
        lim=FOUND_LIM,
        show_border=False,
        axis_style="arrow",
        axis_labels=True,
        x_color=_DIM_SPINE,
        y_color=_DIM_SPINE,
        axis_alpha=0.55,
        axis_linewidth=1.2,
    )

    a_color = _DIM_X_COLOR    # orange — vector A
    b_color = _DIM_Y_COLOR    # purple — vector B

    def _draw_rectangle_panel(ax, *, subject, reference,
                              subject_color, reference_color,
                              subject_label, reference_label):
        S = np.asarray(subject, dtype=float)
        R = np.asarray(reference, dtype=float)
        R_norm = float(np.linalg.norm(R))
        R_hat = R / R_norm
        R_perp_hat = np.array([-R_hat[1], R_hat[0]])

        # Signed projection of S onto R's direction.
        p = float(np.dot(S, R)) / R_norm

        # Choose perpendicular direction on the S-side of R so the rectangle
        # wraps around the subject vector visually.
        S_perp = S - (float(np.dot(S, R)) / float(np.dot(R, R))) * R
        if float(np.dot(S_perp, R_perp_hat)) < 0:
            R_perp_hat = -R_perp_hat

        # Rectangle corners.
        C0 = np.array([0.0, 0.0])
        C1 = C0 + p * R_hat
        C2 = C1 + R_norm * R_perp_hat
        C3 = C0 + R_norm * R_perp_hat

        # Translucent fill — single neutral, thin outline.
        poly = Polygon([C0, C1, C2, C3],
                       closed=True,
                       facecolor=_DIM_NEUTRAL,
                       edgecolor=_DIM_NEUTRAL,
                       linewidth=0.8,
                       alpha=0.16,
                       zorder=1)
        ax.add_patch(poly)

        # Emphasize the rectangle's bottom edge — it IS the projection.
        # Bright white at full alpha so it reads cleanly even where the
        # reference vector (drawn on top) overlaps the same path.
        ax.plot([C0[0], C1[0]], [C0[1], C1[1]],
                color="#FFFFFF",
                alpha=1.0,
                linewidth=style.VECTOR_LINEWIDTH + 3.5,
                solid_capstyle="butt",
                zorder=2)

        # Reference vector — bold, the "direction we're projecting onto".
        _plot_sharp_vector(ax, reference, color=reference_color,
                           label=reference_label,
                           linewidth=style.VECTOR_BOLD_LINEWIDTH,
                           zorder=3)

        # Subject vector — bold, the thing being projected.
        _plot_sharp_vector(ax, subject, color=subject_color,
                           label=subject_label,
                           linewidth=style.VECTOR_BOLD_LINEWIDTH,
                           zorder=4)

        # Big bare scalar in the upper-right corner of axes coords.
        dot_value = float(np.dot(S, R))
        ax.text(0.94, 0.94, f"{dot_value:g}",
                transform=ax.transAxes,
                fontsize=64,
                fontweight="bold",
                color=style.TICK_LABEL_COLOR,
                ha="right", va="top", zorder=5)

    # ----- Panel 1: project a onto b -----
    setup_vector_axes(axes[0], **axis_kwargs)
    axes[0].set_title("project a onto b",
                      color=style.TICK_LABEL_COLOR,
                      fontsize=style.TITLE_FONT_SIZE,
                      loc="center", pad=20)
    _draw_rectangle_panel(axes[0],
                          subject=_TR_A, reference=_TR_B,
                          subject_color=a_color, reference_color=b_color,
                          subject_label="a", reference_label="b")

    # ----- Panel 2: project b onto a -----
    setup_vector_axes(axes[1], **axis_kwargs)
    axes[1].set_title("project b onto a",
                      color=style.TICK_LABEL_COLOR,
                      fontsize=style.TITLE_FONT_SIZE,
                      loc="center", pad=20)
    _draw_rectangle_panel(axes[1],
                          subject=_TR_B, reference=_TR_A,
                          subject_color=b_color, reference_color=a_color,
                          subject_label="b", reference_label="a")

    fig.subplots_adjust(top=0.82)

    return _save(fig, output_dir, filename)


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

    # Reuse the canonical 2D vector A, extended with A_Z as the third
    # dimension — "the same vector, plus depth out of the page." Keeps
    # the §2.4.1 → §2.4.2 transition visually continuous: the reader
    # sees a familiar arrow gain a third component rather than a brand
    # new vector with new numbers to track.
    a = np.array([A[0], A[1], A_Z])

    # Symmetric lims matched to the 2D foundation figures' FOUND_LIM so
    # the 3D plot reads at the same scale as the 2D panels it extends.
    # Origin sits centered; spines extend equally in both directions
    # through (0, 0, 0).
    lim = FOUND_LIM
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
    # Offsets scale with lim so the labels sit a constant fraction past
    # the spine end regardless of FOUND_LIM.
    label_size  = style.VECTOR_LABEL_FONT_SIZE + 2
    label_pad   = lim * 0.05
    ax.text(lim + label_pad, 0, 0, "x",
            color=_DIM_SPINE, fontsize=label_size,
            fontweight="bold", fontstyle="italic",
            ha="left", va="center")
    ax.text(0, lim + label_pad, 0, "y",
            color=_DIM_SPINE, fontsize=label_size,
            fontweight="bold", fontstyle="italic",
            ha="left", va="center")
    ax.text(0, 0, lim + label_pad, "z",
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
    vec_label_pad = lim * 0.04
    ax.text(a[0] + vec_label_pad, a[1] + vec_label_pad, a[2] + vec_label_pad, "a",
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
      - components_recombine_either_order  §2.4.1 beat 2 (xy / yx ordering)
      - projection_reconstruction_either_order  §2.4.1 beat 3 (par / perp ordering)
      - dot_product_geometry.png (4 pan) . §2.4.1 beat 5 (angle → sign)
      - vector_projection_3d.png ........ §2.4.2 (3D + superposition)
    Helpers retained but not dispatched here (available for future re-use):
      _plot_vector_xy_projection (replaced by components_recombine_either_order),
      _plot_projection_reference_directions (3-panel juxtaposition; split into
      the two reconstruction figures), _plot_dot_product_symmetry (2-panel
      projection-only — superseded by _plot_projection_reconstruction_either_order
      which adds the perpendicular complement and both reconstruction paths),
      _plot_vector_similarity (absorbed by the oblique panel of the 4-panel
      sign-cases figure), _plot_vector_projection (long/short shadow — its
      message is carried by the 4-panel figure's parallel + oblique cases).
    """
    if output_dir is None:
        output_dir = IMAGES_DSP_DIR

    print_section_start(f"Generating §2 foundation figures -> {output_dir}/")
    paths = [
        _plot_vector_basics(output_dir),
        _plot_vector_xy_reconstruction(output_dir),
        _plot_projection_reconstruction_either_order(output_dir),
        _plot_dot_product_geometry(output_dir),
        _plot_vector_projection_3d(output_dir),
    ]
    for p in paths:
        print(f"  Saved -> {p}")
    print_section_end()
    return paths
>>>>>>> Stashed changes


if __name__ == "__main__":
    raise SystemExit(generate_all_dsp_figures())
