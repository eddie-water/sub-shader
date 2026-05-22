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

from .. import style

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
# Wider than the global default so the row-title column has visible
# breathing room before the y-axis label / ticks (D-05 local override).
LAYOUT_WSPACE       = style.DEFAULT_WSPACE * 2.5   # label-column ↔ data-column gap
LAYOUT_LABEL_RATIO  = style.DEFAULT_LABEL_RATIO    # left "label column" width
LAYOUT_MARGIN       = style.DEFAULT_PANEL_MARGIN   # outer figure margin
ROW_LABEL_FONT_SIZE = style.DEFAULT_ROW_LABEL_SIZE
AXIS_LABEL_FONT_SIZE = style.DEFAULT_AXIS_LABEL_SIZE
TICK_LABEL_FONT_SIZE = style.DEFAULT_TICK_LABEL_SIZE

# Extra space between "f (Hz)" / "t (s)" labels and their tick labels.
LAYOUT_AXIS_LABEL_PAD = 12

# Trailing whitespace past the data on the time axis. 0.0 keeps the final
# tick (= duration_s) flush with the panel edge.
LAYOUT_X_TAIL_PAD = 0.0  # fraction of duration_s

DEFAULT_FIGSIZE = (18.0, 13.0)


def _friendly_xticks(duration_s: float):
    """Pick xticks at clean step sizes covering [0, duration_s].

    First pass tries for an exact landing on duration_s (clean durations
    like 1.0, 0.5, 2.0). If duration_s is "messy" (e.g. 0.9429 from a
    CWT-trim re-zero), fall back to a clean step that gives 4–6 ticks —
    the final tick may fall short of duration_s, which is fine; the
    right-edge padding just shows a sliver of empty time.
    """
    import numpy as np
    # Pass 1 — exact match (final tick == duration_s)
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        n = round(duration_s / step)
        if 3 <= n <= 6 and abs(n * step - duration_s) < step * 0.01:
            return np.round(step * np.arange(n + 1), 6)
    # Pass 2 — clean step, final tick may fall short of duration_s
    for step in (0.2, 0.25, 0.1, 0.5, 0.05, 1.0):
        n = int(duration_s / step)
        if 3 <= n <= 6:
            return np.round(step * np.arange(n + 1), 6)
    # Last resort
    n = 4
    step = duration_s / n
    return np.round(step * np.arange(n + 1), 6)


@dataclass
class MotivatorConfig:
    """Per-version config for a motivator figure."""
    name: str
    chirp: "BouncingChirpConfig | WaypointChirpConfig"
    output_filename: str
    figsize: tuple[float, float] = DEFAULT_FIGSIZE
    # Optional override for the displayed frequency range (Hz). Defaults to
    # the chirp's own [f_lo, f_hi]. Set wider to expose unused frequency
    # bands (e.g. (20, 21500) for a 200 Hz – 20 kHz chirp shown on a full
    # audible-spectrum axis with breathing room above 20 kHz).
    display_freq_lim_hz: tuple[float, float] | None = None
    # Optional explicit y-axis tick values in Hz. None = auto (canonical
    # 20/200/2k/20k filtered to whatever falls inside display_freq_lim_hz).
    # Override to drop or add ticks (e.g. drop 20 Hz when bottom is at 20 Hz
    # but you don't want a label crammed against the panel edge).
    display_freq_ticks: tuple[float, ...] | None = None
    # Optional figure suptitle. None = no suptitle.
    title: str | None = None


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
        output_filename="v4_100-2000hz_0.5s.png",
    ),
    MotivatorConfig(
        name="v5",
        chirp=BouncingChirpConfig(
            duration_s=1.0,
            f_decades=(50.0, 5000.0),
            bounces_per_decade=3,
        ),
        output_filename="v5_50-5000hz_1.0s.png",
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
        output_filename="vw1_gentle_100-2000hz_0.5s.png",
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
        output_filename="vw2_moderate_100-2000hz_0.5s.png",
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
        output_filename="vw3_aggressive_100-2000hz_0.5s.png",
    ),
    # §1 motivator: waypoint chirp shaped like vw4 — three audibly distinct
    # bounces between 200 Hz and ~4.5 kHz, each peak higher than the last,
    # then a final ascending leg to 20 kHz that visually suggests the
    # pattern would continue forever. Keeping most of the energy below
    # ~5 kHz also keeps the time-series waveform visible across the panel
    # (cycles at 20 kHz are too narrow for fill_between to render).
    MotivatorConfig(
        name="section1",
        chirp=WaypointChirpConfig(
            # 2.36 s spline → CWT trim ~175 ms each side (22 Hz wavelet's
            # 4-cycle support) → snap to nearest zero-crossing on both ends
            # (~7 ms more from the head) → ~2.0 s visible (clean final tick).
            duration_s=2.36,
            # Sub-trim 22 Hz waypoint at frac 0.05 (~0.118 s, well inside
            # head trim of ~0.175 s) — invisible, but bleeds 22 Hz CWT
            # energy forward into the visible window via the wavelet's
            # ~180 ms time support. Same "old vw4 hack" — gives the CWT
            # panel its characteristic glow entering from the left edge,
            # and by the time the visible window starts the chirp has
            # already risen past ~200 Hz on its climb to peak 1.
            #
            # Two visible peaks with aggressive amplitude. Only the deep 60 Hz
            # dip is intentionally below the STFT smear threshold (~200 Hz);
            # the secondary trough stays at 2 kHz so STFT doesn't smear there.
            # Final off-screen waypoint lets the spline rise cleanly off the
            # right edge — no plateau.
            waypoints=(
                (0.00, 50.0),
                (0.05, 22.0),      # sub-trim — left-edge CWT glow
                (0.18, 800.0),     # peak 1 — displayed ≈ 0.25 s
                (0.32, 60.0),      # deep dip — displayed ≈ 0.58 s — only STFT-smearing trough
                (0.55, 12000.0),   # peak 2 — big bounce, displayed ≈ 1.12 s
                (0.75, 2000.0),    # modest trough — displayed ≈ 1.59 s — above smear range
                (1.00, 25000.0),   # tail-trim — clean off-screen ascent
            ),
            clip_to_waypoints=False,
        ),
        output_filename="section1_20-20000hz_2.0s.png",
        display_freq_lim_hz=(20.0, 21500.0),
        display_freq_ticks=(200, 2000, 20000),
        title="Fourier vs Wavelet Analysis",
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
        output_filename="vw4_aggressive_20-20000hz_2.0s.png",
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
    import numpy as np

    signal, inst_freq, t, f_lo, f_hi = _build_signal(cfg)
    sr = cfg.chirp.sr

    # Displayed frequency range — defaults to the chirp's own range, but can
    # be widened (e.g. 20–21500 Hz full audible band) via cfg.display_freq_lim_hz.
    # The CWT wavelet bank extends down to min(f_lo, disp_lo) so the panel's
    # low-freq region (below the chirp's f_lo but inside the display range)
    # is filled with REAL wavelet response — wavelet-bandwidth leakage from
    # nearby chirp content produces a natural fade, not a hard zero-pad floor.
    # Trade-off: lower CWT root → wider slowest-wavelet time support → bigger
    # chunk-boundary trim. Pick duration_s wide enough to absorb the trim.
    disp_lo, disp_hi = cfg.display_freq_lim_hz or (f_lo, f_hi)

    cwt_root_hz = min(f_lo, disp_lo)
    # +1 octave so cwt_freqs[-1] lands past disp_hi (so disp_hi tick has a
    # valid bin position).
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, sr,
        root_note_hz=cwt_root_hz,
        num_octaves=num_octaves,
    )

    # ── OLD-VW4 HACK ─────────────────────────────────────────────────────────
    # Trim companion arrays to the CWT's reliable range and re-zero time. All
    # three panels then share x ∈ [0, trimmed_duration] with no edge gaps.
    # Trade-off: displayed duration is slightly shorter than cfg.chirp.duration_s
    # (head + tail of the chirp's waypoints get clipped). At root_note_hz=f_lo
    # the trim is small (~30 ms total for a chirp starting at 200 Hz).
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    # Snap both ends to the nearest zero-crossing. Without this, the
    # trim+re-zero leaves signal[0] / signal[-1] at arbitrary phase, and
    # plot_time_series's fill_between(signal, 0) draws a tall slab from
    # that phase value down to y=0 at the panel edges. Shift is at most
    # half a cycle (~2.5 ms at 200 Hz) — imperceptible against a 2.0 s
    # panel. cwt_data is sliced to match so the three panels stay aligned.
    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / sr

    # Two-column grid: hidden left "label column" + data column for the
    # three panels. Row titles get overlaid via fig.text after layout
    # finalizes (so we can read the data-column position precisely).
    fig, axes = create_grid_scaffold(
        n_rows=3, n_cols=2,
        figsize=cfg.figsize,
        hspace=LAYOUT_HSPACE,
        wspace=LAYOUT_WSPACE,
        width_ratios=[LAYOUT_LABEL_RATIO, 1.0],
    )
    for r in range(3):
        axes[r][0].axis("off")

    ax_top, ax_stft, ax_cwt = axes[0][1], axes[1][1], axes[2][1]

    # Lock the x-axis across all three data panels — the plot_* helpers
    # each call ax.set_xlim with slightly different right-edge values
    # (t[-1] = (N-1)/sr for time-series/inst-freq vs duration_s for
    # STFT/CWT; scipy's pcolormesh also extends past signal end by a
    # half-cell). sharex forces every xlim change to propagate, so any
    # late override below applies uniformly.
    ax_stft.sharex(ax_top)
    ax_cwt.sharex(ax_top)

    # Row 0 — TWIN-AXIS ESCAPE HATCH: time-series + inst-freq on
    # independent y-scales sharing one x. Time-series y-axis hidden so
    # the inst-freq twin tells the whole "y = frequency" story.
    plot_time_series(ax_top, signal, cfg.chirp.sr)
    ax_top.set_yticks([])
    ax_top.spines['left'].set_visible(False)

    ax_top_twin = ax_top.twinx()
    plot_inst_freq(ax_top_twin, inst_freq, t, cwt_freqs)
    # Freq axis lives on the LEFT (visual rhyme with the row labels):
    # tick + label positions explicit since twinx() defaults to the right.
    ax_top_twin.yaxis.tick_left()
    ax_top_twin.yaxis.set_label_position("left")
    ax_top_twin.tick_params(axis='y', which='both', right=False, labelright=False)
    ax_top_twin.minorticks_off()
    ax_top_twin.patch.set_visible(False)
    for spine in ax_top_twin.spines.values():
        spine.set_edgecolor(style.SPINE_COLOR)
        spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)

    plot_stft_spectrogram(ax_stft, signal, cfg.chirp.sr,
                          freq_lim_hz=(disp_lo, disp_hi),
                          duration_s=duration_s)
    # CWT spectrogram fills the trimmed [0, duration_s] window (helper default).
    plot_cwt_spectrogram(ax_cwt, cwt_data, duration_s, cwt_freqs)

    # Clip CWT + inst-freq panels (both bin-indexed) to the displayed
    # frequency range. cwt_freqs is geometric (log-spaced), so the bin
    # position of any target freq — even outside the cwt_freqs range — is
    # a linear function of log2(freq). This lets us position 20 Hz below
    # bin 0 to expose the empty band below the chirp's f_lo.
    log_f0 = np.log2(cwt_freqs[0])
    log_step = np.log2(cwt_freqs[1] / cwt_freqs[0])
    freq_to_bin = lambda f: (np.log2(f) - log_f0) / log_step
    disp_hi_bin = float(freq_to_bin(disp_hi))
    disp_lo_bin = float(freq_to_bin(disp_lo))
    for ax in (ax_top_twin, ax_cwt):
        ax.set_ylim(disp_lo_bin, disp_hi_bin)

    # Override y-tick set when an explicit list is provided (e.g. drop the
    # bottom-edge 20 Hz tick when disp_lo sits at 20 Hz). Bin-indexed panels
    # need bin positions; STFT (log-Hz) takes Hz directly.
    if cfg.display_freq_ticks is not None:
        hz_ticks = cfg.display_freq_ticks
        labels = [f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}' for f in hz_ticks]
        for ax in (ax_top_twin, ax_cwt):
            bin_pos = [float(freq_to_bin(f)) for f in hz_ticks]
            ax.set_yticks(bin_pos)
            ax.set_yticklabels(labels)
        ax_stft.set_yticks(list(hz_ticks))
        ax_stft.set_yticklabels(labels)

    # STFT + CWT freq ticks stay on the LEFT (matplotlib default — explicit
    # so future style edits don't drift). Suppress any right-side mirror ticks.
    for ax in (ax_stft, ax_cwt):
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis='y', which='both', right=False, labelright=False)

    # Axis units: "f (Hz)" vertical on the LEFT of every row, "t (s)"
    # under the bottom panel only. labelpad pushes the unit text out so the
    # tick labels and unit text don't crowd each other.
    for ax in (ax_top_twin, ax_stft, ax_cwt):
        ax.set_ylabel("f (Hz)", fontsize=AXIS_LABEL_FONT_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=LAYOUT_AXIS_LABEL_PAD,
                      rotation=90, va='center')

    # Time axis spans exactly [0, duration_s] (LAYOUT_X_TAIL_PAD = 0 means
    # the final tick sits flush with the right edge).
    xlim_right = duration_s * (1.0 + LAYOUT_X_TAIL_PAD)
    xticks = _friendly_xticks(duration_s)
    for ax in (ax_top_twin, ax_stft, ax_cwt):
        ax.set_xlim(0, xlim_right)
        ax.set_xticks(xticks)
    ax_top.set_xlim(0, xlim_right)  # base axes shares x with the twin
    ax_top.set_xticks(xticks)
    ax_cwt.set_xlabel("t (s)", fontsize=AXIS_LABEL_FONT_SIZE,
                      color=style.TICK_LABEL_COLOR, labelpad=LAYOUT_AXIS_LABEL_PAD)

    for ax in (ax_top, ax_top_twin, ax_stft, ax_cwt):
        ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)

    # X-tick labels on bottom panel only.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top_twin.get_xticklabels(), visible=False)
    plt.setp(ax_stft.get_xticklabels(), visible=False)

    # Symmetric margins (in absolute inches): horizontal margin is scaled
    # by H/W so left/right padding equals top/bottom padding in inches.
    # When a suptitle is present, reserve extra space at the top.
    fig_w, fig_h = cfg.figsize
    v_margin = LAYOUT_MARGIN
    h_margin = v_margin * fig_h / fig_w
    top_pad = 0.08 if cfg.title else v_margin
    fig.subplots_adjust(
        left=h_margin, right=1 - h_margin,
        top=1 - top_pad, bottom=v_margin,
        hspace=LAYOUT_HSPACE,
        wspace=LAYOUT_WSPACE,
    )

    if cfg.title:
        fig.suptitle(cfg.title,
                     fontsize=style.DEFAULT_SUPTITLE_FONT_SIZE,
                     color=style.TICK_LABEL_COLOR,
                     fontweight="bold",
                     y=1 - top_pad / 2)

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
