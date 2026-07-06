"""Figure 1 — Fourier vs Wavelet Analysis motivator (dsplot rebuild).

Same content as the old `motivator.py::section1`, rebuilt with the dsplot
Panel/Plottable framework. Three vertically stacked panels share a 0–2 s
time axis:

  Row 1 — chirp time-series (TimeSeriesPanel + TimeSeries)
  Row 2 — STFT magnitude (HeatmapPanel + Heatmap, log-spaced bins)
  Row 3 — CWT magnitude  (HeatmapPanel + Heatmap, log-spaced bins)

Data prep mirrors the old motivator: 2.36 s chirp spline → compute_full_cwt
trims the chunk-boundary edge-effect regions (~175 ms each side) →
zero-crossing snap removes the boundary slab in the time-series panel →
~2.0 s visible window with a clean final x-tick at 2.0 s.

STFT is computed on linear-spaced scipy bins and then resampled onto the
CWT's log-spaced frequency grid so both spectrogram panels share the same
y-axis scale (and `Heatmap(log_freq=True, tick_freqs=...)` places ticks
identically on both).
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager

import numpy as np

from .. import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    Line,
    SuptitlePanel,
    TextPanel,
    TimeSeries,
    TimeSeriesPanel,
    nb_compact_style,
    style,
)
from utilities import compute_full_cwt
from utilities.dsp_helpers import (
    build_waypoint_chirp,
    build_log_sweep_oscillating,
    build_click_plus_tone,
    build_low_vibrato,
)


# ============================================================
# Chirp design (mirrors motivator.py::section1 — DO NOT CHANGE without
# re-tuning duration_s / trim accounting; see comments below).
# ============================================================
SR = 44100
DURATION_S = 2.36
WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.00, 50.0),
    (0.05, 22.0),      # sub-trim — bleeds 22 Hz CWT energy into the left edge
    (0.18, 800.0),     # peak 1
    (0.32, 60.0),      # deep dip — the only intentional STFT-smearing trough
    (0.55, 12000.0),   # peak 2
    (0.75, 2000.0),    # modest trough (above STFT smear threshold)
    (1.00, 25000.0),   # tail-trim — clean off-screen ascent
)
DISPLAY_FREQ_LIM_HZ = (20.0, 21500.0)
DISPLAY_FREQ_TICKS = (200, 2000, 20000)
XTICKS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Row labels: title sits in the panel's chrome zone (above the cell) via
# CompositePanel.title; caption (parenthetical) is the body TextPanel inside
# the one and only inner cell. Mirrors the JARGON panel structure in
# `sample_template._r3_jargon_panel` so chrome reads consistently across the
# whole library.
ROW1_TITLE   = "Audio"
ROW1_CAPTION = (
    "A signal whose frequency sweeps continuously from mid to low to high. "
    "A burst of high-frequency clicks (abrupt broadband transients) are "
    "introduced into the signal at its halfway point."
)
ROW2_TITLE   = "Fourier"
ROW2_CAPTION = (
    "The Short Time Fourier Transform (STFT) resolves the signal reasonably "
    "well, but notice how low frequency measurements smear and bleed into "
    "their neighbors. High frequencies appear weak and jagged, struggling "
    "especially when trying to capture the rapid succession of clicks."
)
ROW3_TITLE   = "Wavelet"
ROW3_CAPTION = (
    "The Continuous Wavelet Transform (CWT) traces the contour of the "
    "frequency sweep with smoother and tighter definition. The onsets of "
    "each click are captured with sharper resolution as cleaner, more "
    "distinct events."
)
# Square 1-unit label cells. Caption font size auto-shrinks to fit the cell.
LABEL_PANEL_UNITS = (1, 1)
# Local row-gutter override (template default = 2.0"). Plots get more
# vertical breathing room so x/y tick labels and axis-labels don't crowd
# adjacent panels.
ROW_GUTTER_INCHES = 1.0
# Suptitle gets a full 1.4" top reserve (matches earlier passes).
TOP_RESERVE_INCHES = 1.4
# Bottom reserve hosts the "Figure 1" footer SuptitlePanel. Sized like the
# suptitle row — short single-line band at the bottom of the figure.
BOTTOM_RESERVE_INCHES = 0.6
# Default footer text — shown as a bottom-positioned SuptitlePanel.
FOOTER_TEXT = "Figure 1"

# ============================================================
# Hero — bouncy chirp (200 → 60 → 1.8k → 70 → 20k Hz) + broadband click
# at midpoint. Same STFT/CWT analysis as v8 — only the signal varies.
#
# Story: starts at 200 Hz where STFT resolves fine, dips to 60/70 Hz where
# STFT smears but CWT tracks the contour, peaks at 1.8 kHz, ends smooth at
# 20 kHz. A 5 ms broadband click at t=0.5·duration adds an abrupt transient
# that STFT averages away (window ≫ click) and CWT shows as a vertical
# streak — showcases contour tracking AND transient localization in one
# figure.
# ============================================================
HERO_DURATION_S = 2.36  # build duration; ~2.0 visible after v8-style CWT trim + ZC snap
HERO_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.00, 5500.0),     # boundary: trim brings visible-start down to ~1.9 kHz
    (0.55, 180.0),      # mid-signal dip — spline undershoots to ~98 Hz visible (32 ms CWT lag)
    (0.85, 13000.0),    # intermediate ascent anchor — controls climb rate so the visible right edge actually reaches ~20 kHz
    (1.00, 27000.0),    # tail-trim: overshoot display ceiling, the steepest spline portion gets eaten by CWT trim, leaving visible right edge at ~21 kHz
)
HERO_CLICK_T_FRAC = 0.525   # places cluster centre near the 1.0 s tick (trim offset ≈ 0.233 s)
HERO_CLICK_DURATION_S = 0.008      # 8 ms FWHM per click — narrow envelope keeps each burst's spectrum flat
HERO_CLICK_AMP = 7.0               # well above chirp peak so each click reads bright in CWT
HERO_CHIRP_AMP = 2.0               # multiplier applied to chirp post-build; click normalization uses the pre-amp peak so the click absolute level is unchanged when this knob is raised
HERO_CLICK_COUNT = 5               # N broadband bursts in the cluster — STFT smears into one blob, CWT resolves each
HERO_CLICK_SPACING_S = 0.020       # 20 ms between cluster members
HERO_MIRROR_PAD_S = 0.20           # symmetric mirror pad applied to BOTH STFT & CWT input
HERO_PANEL_UNITS = (3, 1)          # data panel is 3 units wide; label sits beside it (1 unit) — total row width = 4
HERO_DISPLAY_FREQ_LIM_HZ: tuple[float, float] = (20.0, 21500.0)
HERO_DISPLAY_FREQ_TICKS: tuple[int, ...] = (20, 200, 2000, 20000)

# ============================================================
# Anti-hero — on hold; see project_figure_1.md memory for design state.
# ============================================================
ANTIHERO_DURATION_S = 1.5
ANTIHERO_CARRIER_HZ = 60.0
ANTIHERO_DEPTH_HZ = 20.0
ANTIHERO_MOD_HZ = 8.0

# ============================================================
# Contender — low-frequency visible-cycles chirp, waypoint-spline contour
# (mid → dip → climb, kept within ~200 Hz). Hybrid of the v8/hero figure
# (same `build_waypoint_chirp` spline) and the visible-cycles aesthetic: the
# frequency stays low enough that individual oscillations are visible in the
# time-series, with the bright orange PRIMARY_COLOR inst-freq overlay (not
# the gold INST_FREQ_COLOR the v8/hero rows use).
#
# Story: an ASYMMETRIC low-valley chirp on log-frequency — dip low EARLY, hold long,
# rise high LATE. It starts high (~1 kHz) but dips fast right away to a flat floor
# (~38 Hz) it reaches by ~20% across, HOLDS that floor for ~1.4 s so the low
# frequencies stay visible a long while (slow, readable time-series cycles + a clean
# low CWT line), then ramps back UP to high only near the end. The flat valley is a
# CONSTANT low tone, which the CWT draws as a crisp horizontal line (the smudge only
# comes from MOVING through low f — the early descent is clean, and the late ascent is
# eased just enough not to fan). The STFT's fixed window blurs the
# fast-moving high ends of the U, while the CWT keeps the whole curve crisp. The
# endpoints overshoot above the display top (1.6 kHz) because the CWT trim eats
# time off both edges, landing the visible ends near 1 kHz.
# STFT/CWT analysis settings are unchanged from v8 — only the signal varies.
# ============================================================
CONTENDER_DURATION_S = 2.62   # build duration; ~2.0 s visible after CWT trim (~0.44 s front, ~0.17 s back) + ZC snap
# --- Oscillating log sweep (current contender shape) ----------------------------
# Hybrid: a monotone log-linear sweep upward (low -> high, asymmetric -> no
# dip-and-return to mirror-smudge) with a gentle sine wobble in log-frequency
# riding on top. The wobble turns the low-frequency CWT smear into a textured,
# undulating ribbon instead of one obvious symmetric fan. Built via
# build_log_sweep_oscillating(); replaces the bull-horn waypoint shape below
# (kept for reference / quick A-B).
CONTENDER_SWEEP_F_START = 24.0    # build-time start (Hz); really low — slow, readable cycles
CONTENDER_SWEEP_F_END = 650.0     # build-time end (Hz); high but not so high the right half packs solid
CONTENDER_SWEEP_OSC_OCTAVES = 0.0   # clean log sweep — no wobble (decramps the time series)
CONTENDER_SWEEP_N_OSC = 3.0       # (unused while OSC_OCTAVES = 0)
CONTENDER_SWEEP_OSC_PHASE = 0.0   # wobble phase offset (radians)
CONTENDER_SWEEP_OSC_DECAY = 0.0   # (unused while OSC_OCTAVES = 0)
CONTENDER_SWEEP_RAMP_POWER = 5.2  # >1 dwells low; hugs the ~24 Hz floor well past the midpoint clicks, then rises steeply late

CONTENDER_WAYPOINTS: tuple[tuple[float, float], ...] = (
    # "Bull horns": high on both ends, a long FLAT low valley, AGGRESSIVE steep walls.
    # Built with interp="pchip" (shape-preserving) — CubicSpline overshoots violently
    # on walls this steep (horns rocket off-screen, valley sags). PCHIP holds the horn
    # tips at their waypoint value (no rocket) and the equal-valued valley points flat
    # (no sag), so the walls can be near-vertical cleanly.
    # ASYMMETRIC: dip low EARLY, hold long, rise high LATE.
    (0.000, 1120.0), # brief high start
    (0.180, 1080.0), # hold high only briefly, then DIP early
    (0.300,   38.0), # bottom of the early descent wall (reaches the floor by ~20% of view)
    (0.550,   38.0), # flat valley floor (~38 Hz held — low freqs visible ~1.4 s, slow cycles)
    (0.780,   38.0), # valley held long, well past center
    (0.940, 1100.0), # LATE ascent — rises high only near the visible end (eased so no fan)
    (1.000, 1120.0), # high end
)
# Ceiling raised to ~1.5 kHz so the high-band click cluster floats in clear space
# ABOVE the chirp ribbon: the chirp owns the bottom decade (~14-160 Hz), the
# ticks sit up near 750 Hz with a gap below them. A broadband click would have to
# cone out toward the bottom (the CWT's 1/f time resolution warps it into a fan);
# band-limiting the clicks to a HIGH band keeps the CWT response a clean, near-
# vertical tick with no low-frequency fan. The CWT's transient advantage still
# reads — its short support at 750 Hz resolves each tick, while the STFT's fixed
# ~100 ms window smears the whole cluster into one time-blurred blob.
CONTENDER_DISPLAY_FREQ_LIM_HZ: tuple[float, float] = (9.0, 1500.0)
CONTENDER_DISPLAY_FREQ_TICKS: tuple[int, ...] = (10, 100, 1000)
CONTENDER_PANEL_UNITS = (3, 1)
# 80 ms STFT window (3528 = 0.080 × 44100). Tuned to lose on BOTH features while
# still reading as a "fat" ridge AND reaching low enough to avoid a visible cutoff:
#   - FREQUENCY loss: coarse frequency resolution (Δf = sr/nperseg = 12.5 Hz) keeps
#     the chirp ridge a FAT, soft, smeared band vs the CWT's razor-thin contour.
#   - TIME loss: 80 ms ≫ the 28 ms click spacing, so Fourier FUSES the click cluster
#     into one blob (the CWT resolves all five).
# The window length ALSO sets the STFT low-frequency FLOOR = sr/nperseg (an STFT
# can't measure a period longer than its window, and `_stft_on_log_bins` zeros
# everything below the lowest bin via np.interp left=0.0). 80 ms → floor 12.5 Hz,
# which sits BELOW the chirp's ~45 Hz dip with margin, so the fat ridge's lower
# skirt tapers naturally instead of hard-clipping flat. Shorter windows fatten the
# ridge more but raise the floor (50 ms → 20 Hz clipped the dip's skirt flat; 23 ms
# → ~43 Hz blinds the STFT to the dip AND starts resolving the clicks). Longer
# (≥110 ms) would clear the 9 Hz display bottom entirely but thins the ridge. CWT is
# window-independent (constant-Q) — smooth low contour, resolved ticks, full reach.
CONTENDER_STFT_NPERSEG = 3528
# High-band click CLUSTER: N short Gabor atoms (Gaussian-windowed tone bursts)
# centered well ABOVE the chirp ribbon, so they read as crisp vertical ticks
# floating in clear space — NOT broadband cones rooted in the ribbon. A broadband
# click must cone out at low frequency (the CWT's 1/f resolution fans it into a
# warp); band-limiting it to a high band keeps the CWT response a clean tick with
# no low-f fan. The 2 ms FWHM envelope is sharp in time (CWT crisp, STFT smears)
# and gives a frequency footprint around the carrier. The ticks are spaced 28 ms
# apart — far below the STFT's ~100 ms time resolution, so Fourier FUSES them into
# one blurred blob, while the CWT (short support at 750 Hz) RESOLVES each. Cluster
# centre is in DISPLAY time (x-axis seconds), converted to build time in
# _prepare_contender via the CWT-trim + ZC-snap offset.
CONTENDER_CLICK_CENTER_S = 1.0       # cluster centre on the rendered x-axis
CONTENDER_CLICK_COUNT = 5            # ticks in the cluster
CONTENDER_CLICK_SPACING_S = 0.028    # 28 ms apart — CWT splits; still << 100 ms STFT window, so STFT fuses
CONTENDER_CLICK_CARRIER_HZ = 750.0   # tone-burst carrier — floats clear above the ~160 Hz ribbon
CONTENDER_CLICK_DURATION_S = 0.003   # 3 ms FWHM — a touch wider in time (fatter tick), still CWT-crisp
# × chirp peak. Band-concentrated energy reads bright per-bin, so this is
# moderate — enough that the ticks sit clearly above the dB floor without blowing
# out. vmax stays pinned to the chirp ridge (see _prepare_contender) so the chirp
# keeps its brightness and the ticks ride the scale as accents.
CONTENDER_CLICK_AMP = 12.0
# Spectrograms are displayed in dB (log magnitude) referenced to the chirp
# ridge (0 dB). dB compresses the wide dynamic range so the strong tonal chirp
# AND the weak broadband clicks are both visible — the standard reason audio
# spectrograms are shown in dB. This floor is the darkest level shown (vmin);
# anything quieter clamps to black. Raise toward 0 to hide more of the floor.
CONTENDER_DB_FLOOR = -18.0
# Audio-panel y-axis is pinned to ±(chirp peak × this) so the ramping sine
# FILLS the time-series plot like the clickless version. The clicks (many×
# louder) shoot past and clip cleanly at the frame edges — reading as abrupt
# transient markers without crushing the chirp's visible cycles. Decouples the
# click loudness (which drives spectrogram brightness) from the waveform's look.
CONTENDER_TS_YLIM_PAD = 1.04
CONTENDER_ROW1_TITLE = "Audio"
CONTENDER_ROW1_CAPTION = (
    "A chirp signal whose frequency sweeps from mid to low to high. A burst of "
    "high-frequency clicks (abrupt broadband transients) are introduced at the "
    "halfway point."
)
CONTENDER_ROW2_CAPTION = (
    "The Short Time Fourier Transform resolves the signal reasonably well, but "
    "notice how the low-frequency dip smears and bleeds into its neighbors. "
    "High frequencies appear weak and jagged, struggling especially to capture "
    "the rapid succession of clicks."
)
CONTENDER_ROW3_CAPTION = (
    "The Continuous Wavelet Transform traces the contour of the frequency "
    "sweep with smoother and better definition. The onsets of each click are " 
    "captured with sharper resolution as cleaner, more disctinct events."
)
# Caption body font for the (now larger) 2×2 caption square. Above the shared
# style default so the text scales with the bigger box; the tick + axis label
# sizes below are derived from it so the whole figure-1 type system grows
# together. Defined here (before CONTENDER_TIGHT_STYLE) because the style dict
# references it.
CONTENDER_CAPTION_FONT_SIZE = 44
# Tight-padding profile scoped to the contender (the user signed off on this
# look for figure 1 only — NOT yet promoted to style.py defaults, which would
# retighten every figure). PAD 1.5 → 0.7 grows the plot's cell-fill from ~57%
# to ~74%; axis-label insets sit close to the cell border so the "Hz"/"s"
# labels read as "just enough room". Applied via a temporary style override
# during build + render (same mechanism as nb_compact_style), then restored.
CONTENDER_TIGHT_STYLE = {
    "DEFAULT_PAD_INCHES": 0.4,
    # Cell-border weight scaled to THIS figure's width so it reads at 2.4.1's
    # border thickness when each figure is viewed at a common display width
    # (the cross-figure border rule — wider canvas needs a heavier pt line to
    # look equally thick). Fig 1 is ~46.8" wide vs 241's 31.1" reference, so
    # 3.0 × 46.8/31.1 ≈ 4.5pt. (Base 3.0 lives in style.DEFAULT_FRAME_LINEWIDTH.)
    "DEFAULT_FRAME_LINEWIDTH": 4.5,
    # Perimeter margin (panel-border → figure edge) cut hard: at 1.05" it left
    # ~1.3" of dead space between the page edge and the y-axis labels. The
    # labels render INTO this margin, so shrinking it pulls "Hz"/numbers close
    # to the page edge. Safe here because the suptitle/footer are gridspec rows
    # (their band height comes from row_heights, not the margin) and the caption
    # fills to the figure edge independently of the margin.
    "DEFAULT_MARGIN_INCHES": 0.25,
    # Soft white (#EEEEEE = NEUTRAL_COLOR, the template-showcase white) for
    # EVERYTHING — one consistent value. TICK_LABEL_COLOR drives tick labels,
    # axis labels, panel titles, captions, heatmap labels AND the row-1
    # time-series waveform (passed the same constant). SUPTITLE_COLOR (bound
    # separately at import) covers the suptitle + footer. SPINE_COLOR whitens
    # the plot box around each panel to match the cell-border rectangles
    # (already drawn in NEUTRAL_COLOR).
    "TICK_LABEL_COLOR": "#EEEEEE",
    "SUPTITLE_COLOR": "#EEEEEE",
    # No plot box: spines invisible so each plot reads as bare content (heatmap
    # tick marks use TICK_LABEL_COLOR, so they survive). Drives host AND twin
    # spines across every panel in one knob.
    "SPINE_COLOR": "none",
    # Tick numbers + axis labels grow WITH the (now larger) caption so the axis
    # chrome and body text stay one type system. The caption is a figure-1-local
    # override (CONTENDER_CAPTION_FONT_SIZE), so the label sizes are local here
    # too — NOT bumped in shared style.py, which would grow the other figures'
    # labels without their boxes/captions growing. Relationship preserved from the
    # shared defaults: tick numbers == caption size; axis label ≈ 1.2× tick (the
    # dominant label voice). The taller 2-unit plots have ample room for the 3
    # y-ticks (1k/100/10) at this size.
    "DEFAULT_TICK_LABEL_SIZE": CONTENDER_CAPTION_FONT_SIZE,
    "DEFAULT_AXIS_LABEL_SIZE": round(CONTENDER_CAPTION_FONT_SIZE * 1.2),
    # Row gutter seats the x-axis "s" + numbers below row 3; column gutter is the
    # plot↔caption gap. The y-axis labels live in the in-plot label strip
    # (content_left_pad_inches), not the column gutter.
    "DEFAULT_GUTTER_INCHES": 0.7,
    "DEFAULT_COLUMN_GUTTER_INCHES": 0.55,
    # Each axis label sits PAST its tick numbers with a matched gap. The y-inset
    # places "Hz" inside the in-plot label strip (STACK_LABEL_PAD_INCHES = 1.9"),
    # left of the numbers; the x-inset places "s" below the x numbers in the row
    # gutter beneath row 3.
    "DEFAULT_X_AXIS_LABEL_INSET_INCHES": 1.2,
    "DEFAULT_Y_AXIS_LABEL_INSET_INCHES": 2.1,
    "DEFAULT_AXIS_LABEL_INSET_INCHES": 0.5,
}


@contextmanager
def _contender_tight_style():
    """Temporarily apply CONTENDER_TIGHT_STYLE to the style module.

    Layout math (margins, gutters, axis-label insets) is read from
    ``dsplot.style.*`` at compose()/render() time, so overriding here and
    restoring on exit scopes the tighter padding to the contender without
    touching style.py or any other figure.
    """
    orig = {k: getattr(style, k) for k in CONTENDER_TIGHT_STYLE}
    try:
        for k, v in CONTENDER_TIGHT_STYLE.items():
            setattr(style, k, v)
        yield
    finally:
        for k, v in orig.items():
            setattr(style, k, v)


def _prepare() -> dict:
    """Build chirp, run CWT, trim + zero-crossing snap. Returns a bundle."""
    signal, inst_freq, t = build_waypoint_chirp(
        SR, DURATION_S, WAYPOINTS, clip_to_waypoints=False
    )
    f_lo = min(f for _, f in WAYPOINTS)
    disp_lo, disp_hi = DISPLAY_FREQ_LIM_HZ

    cwt_root_hz = min(f_lo, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    # Trim + re-zero — discards CWT edge-effect regions; all three panels
    # share x ∈ [0, trimmed_duration] with no edge gaps.
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    # Zero-crossing snap — removes the boundary slab that fill_between
    # otherwise draws at t=0 (because trim leaves signal[0] at arbitrary
    # phase). Shift is ≤ half a cycle, imperceptible against the 2.0 s panel.
    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _stft_on_log_bins(signal: np.ndarray, sr: int, log_freqs: np.ndarray,
                      nperseg: int | None = None) -> np.ndarray:
    """Compute STFT magnitude and resample onto the supplied log-spaced bin grid.

    Lets the STFT spectrogram render through `Heatmap(log_freq=True)` with
    the SAME y-axis as the CWT panel (otherwise scipy's linear-spaced bins
    would force a log-yscale dance that imshow can't do cleanly).

    ``nperseg`` (when given) sets the STFT window length in samples directly.
    A LONGER window lowers the STFT frequency floor (= sr/nperseg) so the
    transform reaches lower frequencies — at the cost of time resolution
    (the window spans nperseg/sr seconds). When None, falls back to the v8
    auto value (≤1024) for apples-to-apples with the legacy chirp path.
    """
    from scipy.signal import stft as scipy_stft
    n = len(signal)
    if nperseg is None:
        nperseg = min(1024, max(64, 1 << int(math.log2(max(64, n // 4)))))
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2
    f_lin, _t, Zxx = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag_lin = np.abs(Zxx)
    # Drop DC bin so log interp is well-defined.
    f_lin = f_lin[1:]
    mag_lin = mag_lin[1:]

    mag_log = np.empty((len(log_freqs), mag_lin.shape[1]), dtype=mag_lin.dtype)
    for j in range(mag_lin.shape[1]):
        mag_log[:, j] = np.interp(log_freqs, f_lin, mag_lin[:, j],
                                  left=0.0, right=0.0)
    return mag_log


def _auto_xticks(duration_s: float) -> list[float]:
    """Generate 0.0, 0.5, 1.0, … xticks up to duration_s."""
    return [round(x, 1) for x in np.arange(0.0, duration_s + 0.01, 0.5).tolist()
            if x <= duration_s + 0.01]


def _mirror_pad_for_analysis(signal: np.ndarray, pad_samples: int) -> np.ndarray:
    """Symmetrically reflect both ends of `signal` to provide out-of-window context.

    Applied identically to both STFT and CWT inputs so edge artifacts and the
    CWT's intrinsic low-f time-support delay are pushed outside the visible
    window without introducing asymmetric treatment between the two transforms.
    Uses numpy reflect mode (value at boundary repeats so reflection is smooth).
    """
    if pad_samples <= 0 or len(signal) == 0:
        return signal
    pad_samples = min(pad_samples, len(signal) - 1)
    return np.pad(signal, pad_samples, mode="reflect")


def _stft_visible_window(signal: np.ndarray, sr: int, log_freqs: np.ndarray,
                         t_start_s: float, t_end_s: float) -> np.ndarray:
    """Compute STFT on `signal` (the full padded buffer) and slice time bins to
    the visible window [t_start_s, t_end_s] before resampling onto `log_freqs`.

    Same STFT settings as `_stft_on_log_bins` — apples-to-apples with the
    visible-signal STFT path; only the analysed buffer is wider.
    """
    from scipy.signal import stft as scipy_stft
    n = len(signal)
    nperseg = min(1024, max(64, 1 << int(math.log2(max(64, n // 4)))))
    noverlap = nperseg // 2
    f_lin, t_stft, Zxx = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag_lin = np.abs(Zxx)
    f_lin = f_lin[1:]
    mag_lin = mag_lin[1:]

    bin_lo = int(np.searchsorted(t_stft, t_start_s, side="left"))
    bin_hi = int(np.searchsorted(t_stft, t_end_s, side="right"))
    bin_lo = max(0, min(bin_lo, mag_lin.shape[1] - 1))
    bin_hi = max(bin_lo + 1, min(bin_hi, mag_lin.shape[1]))
    mag_lin = mag_lin[:, bin_lo:bin_hi]

    mag_log = np.empty((len(log_freqs), mag_lin.shape[1]), dtype=mag_lin.dtype)
    for j in range(mag_lin.shape[1]):
        mag_log[:, j] = np.interp(log_freqs, f_lin, mag_lin[:, j],
                                  left=0.0, right=0.0)
    return mag_log


def _row_label_panel(title: str, caption: str,
                     caption_font_size: float | None = None) -> CompositePanel:
    """Row label cell modeled after `sample_template._r3_jargon_panel`.

    Title renders in the chrome zone above the cell via
    ``CompositePanel.title``; caption fills the cell body as the one and
    only inner TextPanel. Both inherit current style.* values lazily so
    the notebook compact profile reshapes them automatically.

    ``caption_font_size`` (when set) PINS the caption to a fixed size — pass
    the same value for every row so all captions render at one uniform size
    instead of auto-shrinking to a different size per text length.
    """
    # EXACT match of sample_template._r3_jargon_panel — that pattern works.
    # show_ghost_border=True is the key: the library already extends the
    # content rect OUT to the cell border (by style.DEFAULT_PAD_INCHES) and
    # insets it by 1/8 of cell width.
    pinned = caption_font_size is not None
    body = TextPanel(
        caption,
        units=LABEL_PANEL_UNITS,
        font_size=caption_font_size if pinned else style.DEFAULT_TITLE_FONT_SIZE - 4,
        # When pinned, min == font_size so auto_shrink can't vary it (uniform
        # across rows). Otherwise keep shrink headroom so a caption that
        # outgrows its cell shrinks instead of overflowing.
        min_font_size=caption_font_size if pinned else 9,
        color=style.TICK_LABEL_COLOR,
        fontweight="bold",
        auto_shrink=not pinned,
        cell_padding_frac=0.0,
        justify=False,
        show_ghost_border=True,
        top_anchor=True,
    )
    return CompositePanel(
        units=LABEL_PANEL_UNITS,
        title=title,
        rows=[[body]],
    )


def _build_3row_figure(
    data: dict,
    *,
    xticks: list[float],
    suptitle: str | None = None,
    footer: str | None = None,
    display_freq_lim_hz: tuple[float, float] | None = None,
    display_freq_ticks: tuple[int, ...] | None = None,
    panel_units: tuple[int, int] = (3, 1),
    inst_freq_color: str | None = None,
    row_captions: tuple[str, str, str] | None = None,
    caption_font_size: float | None = None,
    stft_nperseg: int | None = None,
    show_xticklabels_all_rows: bool = False,
    dpi: int | None = None,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
    row_gutter_inches: float | None = None,
    bottom_reserve_inches: float | None = None,
    footer_row_height: float = 0.25,
    debug: bool = False,
) -> Figure:
    """Shared 3-row Fourier-vs-Wavelet layout used by v8 chirp + hero + anti-hero.

    Renders the time-series + STFT + CWT story with consistent visual chrome:
    orange inst-freq overlay on all 3 rows, log-spaced frequency grid via
    `_stft_on_log_bins`, twin-axis Hz labels on row 1 that rhyme with the
    spectrogram bin positions on rows 2 and 3.

    STFT/CWT analysis settings are held identical across callers (see
    `_stft_on_log_bins` and the v8 `compute_full_cwt` defaults) so figures
    differ only by their input signal — apples-to-apples comparison.

    If ``data["stft_mag_log"]`` is present (caller pre-computed STFT, e.g.
    for mirror-padded analysis), it is used directly. Otherwise STFT is
    computed here from the visible signal.

    ``display_freq_lim_hz`` (when provided) slices the CWT/STFT data and
    freq grid to the visible band so log-spaced panels fill instead of
    leaving empty rows. ``panel_units`` controls the panel aspect ratio.
    """
    duration_s = data["duration_s"]
    cwt_freqs = data["cwt_freqs"]
    cwt_data = data["cwt_data"]
    if "stft_mag_log" in data:
        stft_mag_log = data["stft_mag_log"]
    else:
        stft_mag_log = _stft_on_log_bins(data["signal"], SR, cwt_freqs,
                                         nperseg=stft_nperseg)

    if display_freq_ticks is None:
        display_freq_ticks = DISPLAY_FREQ_TICKS

    if display_freq_lim_hz is not None:
        disp_lo, disp_hi = display_freq_lim_hz
        bin_lo = int(np.searchsorted(cwt_freqs, disp_lo, side="left"))
        bin_hi = int(np.searchsorted(cwt_freqs, disp_hi, side="right"))
        bin_lo = max(0, min(bin_lo, len(cwt_freqs) - 1))
        bin_hi = max(bin_lo + 1, min(bin_hi, len(cwt_freqs)))
        cwt_freqs = cwt_freqs[bin_lo:bin_hi]
        cwt_data = cwt_data[bin_lo:bin_hi, :]
        stft_mag_log = stft_mag_log[bin_lo:bin_hi, :]

    log_f0 = float(np.log2(cwt_freqs[0]))
    log_step = float(np.log2(cwt_freqs[1] / cwt_freqs[0]))
    freq_to_bin = lambda f: (np.log2(f) - log_f0) / log_step

    t_axis = np.arange(len(data["signal"])) / SR
    inst_freq_bins = np.interp(
        data["inst_freq"], cwt_freqs, np.arange(len(cwt_freqs))
    )

    twin_ytick_positions = [float(freq_to_bin(f)) for f in display_freq_ticks]
    twin_ytick_labels = [
        f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}'
        for f in display_freq_ticks
    ]
    twin_ylim = (0.0, float(len(cwt_freqs)))

    # HeatmapPanel.render() only overrides StaticPanel's (-1.25, 1.25)
    # default xlim when extent is set explicitly.
    spec_extent = (0.0, duration_s, 0.0, float(len(cwt_freqs)))

    # When every row shows its own time ticks, give every row the "s" axis
    # label too so each panel reads as fully self-contained.
    row_x_label = "s" if show_xticklabels_all_rows else None

    row1 = TimeSeriesPanel(
        units=panel_units,
        x_label=row_x_label,
        xticks=xticks,
        # Time axis is shared across all 3 rows — by default only row 3 owns
        # the x-tick labels at the figure bottom (showing them on row 1 too
        # eats the chrome zone between rows 1 and 2). Set
        # ``show_xticklabels_all_rows=True`` to give every plot its own time
        # ticks (each panel reads as self-contained).
        show_xticklabels=show_xticklabels_all_rows,
        # Pin the waveform y-axis (when the caller supplies it) so the chirp
        # fills the panel and the louder click spikes clip at the frame edges
        # instead of autoscaling the chirp down to a thin band.
        ylim=data.get("ts_ylim"),
        twin_y=True,
        twin_y_label="Hz",
        twin_y_side="left",
        twin_yticks=twin_ytick_positions,
        twin_ytick_labels=twin_ytick_labels,
        twin_ylim=twin_ylim,
    )
    line_color = inst_freq_color if inst_freq_color is not None else style.INST_FREQ_COLOR
    row1.add(TimeSeries(data["signal"], SR, color=style.TICK_LABEL_COLOR))
    row1.add_twin(Line(
        t_axis,
        inst_freq_bins,
        color=style.BG_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 5.5,
        alpha=1.0,
    ))
    row1.add_twin(Line(
        t_axis,
        inst_freq_bins,
        color=line_color,
        linewidth=style.INST_FREQ_LINEWIDTH + 3.0,
        alpha=style.INST_FREQ_ALPHA,
    ))

    row2 = HeatmapPanel(
        units=panel_units,
        x_label=row_x_label,
        y_label="Hz",
        xticks=xticks,
        # Shared time axis — row 3 owns x-tick labels by default (see row1).
        show_xticklabels=show_xticklabels_all_rows,
    )
    row2.add(Heatmap(
        stft_mag_log,
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=display_freq_ticks,
        extent=spec_extent,
        vmin=data.get("stft_vmin", 0.0),
        vmax=data.get("stft_vmax"),
    ))

    row3 = HeatmapPanel(
        units=panel_units,
        x_label="s",
        y_label="Hz",
        xticks=xticks,
    )
    row3.add(Heatmap(
        cwt_data,
        duration_s=duration_s,
        freqs=cwt_freqs,
        log_freq=True,
        tick_freqs=display_freq_ticks,
        extent=spec_extent,
        vmin=data.get("cwt_vmin", 0.0),
        vmax=data.get("cwt_vmax"),
    ))

    # Row labels: CompositePanel(title, [[TextPanel(caption)]]) — title sits
    # in the panel's chrome zone above the cell, caption fills the cell body.
    # Same structure as `sample_template._r3_jargon_panel` so the row labels
    # read with consistent typography across the library.
    cap1, cap2, cap3 = row_captions if row_captions is not None else (
        ROW1_CAPTION, ROW2_CAPTION, ROW3_CAPTION
    )
    label1 = _row_label_panel(ROW1_TITLE, cap1, caption_font_size)
    label2 = _row_label_panel(ROW2_TITLE, cap2, caption_font_size)
    label3 = _row_label_panel(ROW3_TITLE, cap3, caption_font_size)

    # Row 0 — top SuptitlePanel spanning the full row width.
    total_row_width = LABEL_PANEL_UNITS[0] + panel_units[0]
    suptitle_row: list = [
        SuptitlePanel(suptitle or "", units=(total_row_width, 1))
    ]

    # Row 4 — bottom footer SuptitlePanel spanning the full row width. Same
    # styling as the top suptitle (SUPTITLE_* style family); positioned at
    # the bottom by virtue of being the last row in the gridspec.
    footer_row: list = [
        SuptitlePanel(footer or "", units=(total_row_width, 1))
    ]

    # Pass hspace ONLY when caller explicitly overrides row_gutter_inches.
    # Otherwise let compose derive hspace from style.DEFAULT_GUTTER_INCHES —
    # this is critical because Panel._render_chrome_titles uses the STYLE
    # gutter for chrome title placement; if the actual gridspec gutter
    # diverges from the style gutter (via an hspace_override), title text
    # overflows the cell border. Keeping them equal makes chrome math
    # consistent with cell-border math.
    hspace_override = None
    if row_gutter_inches is not None:
        effective_unit_height_inches = (
            unit_height_inches if unit_height_inches is not None
            else (unit_inches if unit_inches is not None
                  else style.DEFAULT_PANEL_UNIT_INCHES)
        )
        hspace_override = row_gutter_inches / effective_unit_height_inches
    return Figure.compose(
        rows=[
            suptitle_row,
            [label1, row1],
            [label2, row2],
            [label3, row3],
            footer_row,
        ],
        # Header (suptitle) band height matches the footer band so the figure
        # is balanced top-to-bottom.
        row_heights=[footer_row_height, 1.0, 1.0, 1.0, footer_row_height],
        hspace=hspace_override,
        bottom_reserve_inches=bottom_reserve_inches,
        dpi=dpi,
        unit_inches=unit_inches,
        unit_height_inches=unit_height_inches,
        # Same lego-kitchen-sink chrome as figure 2.4.1 — subtle gray rect
        # around each cell makes the layout structure visually explicit.
        show_cell_borders=True,
        # debug=True overlays layout guide lines + red brackets above/below
        # every rendered Text artist (titles, captions, suptitle, footer,
        # axis labels) so overflows or misalignments are visible at a glance.
        debug_guides=debug,
    )


def build_figure() -> Figure:
    return _build_3row_figure(
        _prepare(),
        xticks=XTICKS,
        suptitle="Fourier vs Wavelet Analysis",
        footer=FOOTER_TEXT,
    )


def render(output_dir: str = "assets/images/dsp/figures/figure_1",
           output_filename: str = "fourier_vs_wavelet.png") -> str:
    """Build, render, save. Returns absolute output path."""
    fig = build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show() -> Figure:
    """Build, render, and display in a notebook cell."""
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig


# ============================================================
# Hero + Anti-hero data prep and render functions.
# These extend gen_figure_1_stft_vs_cwt.py without touching the locked v8 pipeline above.
# ============================================================

def _prepare_hero() -> dict:
    """Hero pipeline: bouncy chirp + broadband click. Mirrors v8 `_prepare()`
    pattern — only the signal differs.

    1. Build chirp from HERO_WAYPOINTS (200 → 60 → 1.8k → 70 → 20k Hz) over
       HERO_DURATION_S (2.36 s build → ~2.0 s visible after trim + ZC snap).
    2. Add a Gaussian-windowed broadband white-noise click at midpoint.
    3. CWT via compute_full_cwt; trim + zero-crossing snap (same as v8).
       STFT is computed downstream in `_build_3row_figure` on the same
       visible signal — apples-to-apples with v8 and with whatever future
       anti-hero we land on.
    """
    chirp_signal, inst_freq, t_chirp = build_waypoint_chirp(
        SR, HERO_DURATION_S, HERO_WAYPOINTS, clip_to_waypoints=False
    )

    n_build = len(chirp_signal)
    chirp_peak = float(np.max(np.abs(chirp_signal)))
    sigma = HERO_CLICK_DURATION_S / 2.3548200450309493  # FWHM → sigma
    cluster_center_t = HERO_CLICK_T_FRAC * HERO_DURATION_S
    cluster_half_span = (HERO_CLICK_COUNT - 1) * HERO_CLICK_SPACING_S / 2.0

    click_burst = np.zeros(n_build, dtype=np.float64)
    for i in range(HERO_CLICK_COUNT):
        click_t = cluster_center_t - cluster_half_span + i * HERO_CLICK_SPACING_S
        rng = np.random.default_rng(seed=i)
        noise = rng.standard_normal(n_build).astype(np.float64)
        envelope = np.exp(-((t_chirp - click_t) ** 2) / (2.0 * sigma ** 2))
        single = noise * envelope
        peak = float(np.max(np.abs(single)))
        if peak > 0.0:
            single = single * (HERO_CLICK_AMP * chirp_peak / peak)
        click_burst = click_burst + single
    signal = chirp_signal * HERO_CHIRP_AMP + click_burst

    disp_lo, disp_hi = HERO_DISPLAY_FREQ_LIM_HZ
    f_lo = min(f for _, f in HERO_WAYPOINTS)
    cwt_root_hz = min(f_lo, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    # v8 pattern: slice signal/inst_freq by [start:end] (sample-rate indices);
    # cwt_data is already trimmed to its reliable region by compute_full_cwt.
    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t_chirp) > start_sample:
        t = t_chirp[start_sample:end_sample] - t_chirp[start_sample]
    else:
        t = t_chirp[:0]

    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            # v8 slices cwt_data by the same ZC indices; upper bound clips at
            # the hop-rate column count, which is the intended behaviour.
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _prepare_antihero() -> dict:
    """Build anti-hero signal (60 Hz ±20 Hz vibrato at 8 Hz), run CWT, trim + snap.

    Mirrors the structure of `_prepare()`. Returns the same bundle dict.
    CWT bank covers the full 20-21500 Hz range (same as v8/hero).
    Tighter display narrowing is a v2 task — see TODO below.
    """
    signal, inst_freq, t = build_low_vibrato(
        SR, ANTIHERO_DURATION_S,
        carrier_hz=ANTIHERO_CARRIER_HZ,
        depth_hz=ANTIHERO_DEPTH_HZ,
        mod_hz=ANTIHERO_MOD_HZ,
    )

    disp_lo, disp_hi = DISPLAY_FREQ_LIM_HZ
    cwt_root_hz = min(ANTIHERO_CARRIER_HZ - ANTIHERO_DEPTH_HZ, disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)

    from research.utilities import compute_full_cwt
    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, first_zc:last_zc + 1]

    duration_s = len(signal) / SR
    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_data,
        "cwt_freqs": cwt_freqs,
    }


def _build_hero_figure(
    *,
    dpi: int = 250,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
    row_gutter_inches: float | None = None,
    bottom_reserve_inches: float | None = None,
    debug: bool = False,
) -> Figure:
    data = _prepare_hero()
    return _build_3row_figure(
        data,
        xticks=_auto_xticks(data["duration_s"]),
        suptitle="Fourier vs Wavelet",
        footer=FOOTER_TEXT,
        display_freq_lim_hz=HERO_DISPLAY_FREQ_LIM_HZ,
        display_freq_ticks=HERO_DISPLAY_FREQ_TICKS,
        panel_units=HERO_PANEL_UNITS,
        dpi=dpi,
        unit_inches=unit_inches,
        unit_height_inches=unit_height_inches,
        debug=debug,
        row_gutter_inches=row_gutter_inches,
        bottom_reserve_inches=bottom_reserve_inches,
    )


def _build_antihero_figure() -> Figure:
    # On hold per user direction — defaults match v8 (apples-to-apples).
    data = _prepare_antihero()
    return _build_3row_figure(
        data,
        xticks=_auto_xticks(data["duration_s"]),
        suptitle="Anti-hero — TBD",
        footer=FOOTER_TEXT,
    )


def render_hero(
    output_dir: str = "assets/images/dsp/figures/figure_1",
    output_filename: str = "hero_click_plus_tone_v1.png",
) -> str:
    """Build, render, and save the hero figure. Returns absolute output path."""
    fig = _build_hero_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def render_antihero(
    output_dir: str = "assets/images/dsp/figures/figure_1",
    output_filename: str = "antihero_low_vibrato_v1.png",
) -> str:
    """Build, render, and save the anti-hero figure. Returns absolute output path."""
    fig = _build_antihero_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def _to_db(mag: np.ndarray, ref: float, floor_db: float) -> np.ndarray:
    """Magnitude → dB referenced to `ref` (0 dB == ref), clamped at `floor_db`.

    20·log10(mag/ref): the chirp ridge (mag ≈ ref) lands near 0 dB and stays
    bright, while broadband clicks — far below ref per-bin — lift off the floor
    instead of vanishing the way they do on a linear scale.
    """
    mag = np.abs(np.asarray(mag, dtype=np.float64))
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    return np.maximum(db, floor_db)


def _prepare_contender() -> dict:
    """Contender pipeline: low-freq waypoint-spline chirp (mid → dip → climb,
    kept within ~10-100 Hz) PLUS short broadband clicks. The chirp is the
    time-frequency contour; the clicks are sharp transients that expose the
    time-resolution half of the STFT-vs-CWT tradeoff. Mirrors v8 `_prepare()`
    — only the signal differs.
    """
    chirp_signal, inst_freq, t = build_log_sweep_oscillating(
        SR, CONTENDER_DURATION_S,
        CONTENDER_SWEEP_F_START, CONTENDER_SWEEP_F_END,
        osc_octaves=CONTENDER_SWEEP_OSC_OCTAVES,
        n_osc=CONTENDER_SWEEP_N_OSC,
        osc_phase=CONTENDER_SWEEP_OSC_PHASE,
        osc_decay=CONTENDER_SWEEP_OSC_DECAY,
        ramp_power=CONTENDER_SWEEP_RAMP_POWER,
    )

    n_build = len(chirp_signal)
    chirp_peak = float(np.max(np.abs(chirp_signal))) or 1.0

    disp_lo, disp_hi = CONTENDER_DISPLAY_FREQ_LIM_HZ
    cwt_root_hz = min(float(np.min(inst_freq)), disp_lo)
    num_octaves = max(1, math.ceil(math.log2(disp_hi / cwt_root_hz)) + 1)

    # Chirp-referenced 0 dB level: take the chirp-ALONE ridge magnitude as the
    # 0 dB reference for each panel so the bright end of the dB colormap tracks
    # the chirp — NOT the broadband clicks. The chirp ridge then stays pinned
    # bright regardless of click loudness, while the clicks lift off the dB
    # floor (see _to_db / CONTENDER_DB_FLOOR). 99.5th percentile (not max) so a
    # lone hot pixel doesn't set the reference.
    chirp_cwt, chirp_freqs, chirp_start, chirp_end = compute_full_cwt(
        chirp_signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )
    cwt_ref = float(np.percentile(np.abs(chirp_cwt), 99.5)) or 1.0
    chirp_stft = _stft_on_log_bins(
        chirp_signal, SR, chirp_freqs, nperseg=CONTENDER_STFT_NPERSEG
    )
    stft_ref = float(np.percentile(chirp_stft, 99.5)) or 1.0

    # Display→build time offset: the visible window starts at build sample
    # (chirp_start + first zero-crossing), so a click meant to land at DISPLAY
    # time D must be placed at build time D + offset. The trim is set by the
    # wavelet support (independent of the clicks, which sit mid-signal far from
    # both edges), so the chirp-only bounds give the correct offset.
    chirp_trimmed = chirp_signal[chirp_start:chirp_end]
    chirp_zc = np.where(np.diff(np.sign(chirp_trimmed)) != 0)[0]
    first_zc_offset = int(chirp_zc[0]) + 1 if len(chirp_zc) else 0
    display_to_build_s = (chirp_start + first_zc_offset) / SR

    # Layer in a CLUSTER of high-band Gabor atoms (Gaussian-windowed tone bursts),
    # NOT broadband impulses. A broadband click must cone out toward low frequency
    # in the CWT (1/f time resolution → a warped fan); band-limiting the burst to
    # a high carrier keeps the CWT response a clean, near-vertical tick floating
    # clear above the chirp ribbon. The short envelope is sharp in time, so the
    # CWT (short support at 750 Hz) resolves each tick while the STFT's fixed
    # ~100 ms window smears the cluster into one blurred blob. Normalized to
    # CLICK_AMP × chirp peak so the ticks sit clear of the dB floor.
    sigma = CONTENDER_CLICK_DURATION_S / 2.3548200450309493  # FWHM → sigma
    cluster_center_t = CONTENDER_CLICK_CENTER_S + display_to_build_s
    cluster_half_span = (CONTENDER_CLICK_COUNT - 1) * CONTENDER_CLICK_SPACING_S / 2.0
    click_burst = np.zeros(n_build, dtype=np.float64)
    for i in range(CONTENDER_CLICK_COUNT):
        click_t = cluster_center_t - cluster_half_span + i * CONTENDER_CLICK_SPACING_S
        envelope = np.exp(-((t - click_t) ** 2) / (2.0 * sigma ** 2))
        single = np.cos(2.0 * np.pi * CONTENDER_CLICK_CARRIER_HZ * (t - click_t)) * envelope
        peak = float(np.max(np.abs(single)))
        if peak > 0.0:
            single *= CONTENDER_CLICK_AMP * chirp_peak / peak
        click_burst += single
    signal = chirp_signal + click_burst

    cwt_data, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        signal, SR, root_note_hz=cwt_root_hz, num_octaves=num_octaves
    )

    signal = signal[start_sample:end_sample]
    inst_freq = inst_freq[start_sample:end_sample]
    if len(t) > start_sample:
        t = t[start_sample:end_sample] - t[start_sample]
    else:
        t = t[:0]

    if len(signal) > 1:
        sign_changes = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(sign_changes) >= 2:
            first_zc = int(sign_changes[0]) + 1
            last_zc = int(sign_changes[-1]) + 1
            # cwt_data is at a HOP rate (n_cols << n_samples), so the sample-index
            # zero-crossing bounds must be mapped to COLUMN indices before
            # slicing. Indexing the hop-rate array with raw sample indices chops
            # hundreds of columns off the left and slides the whole scalogram out
            # of time-sync with the waveform (the STFT, recomputed on the trimmed
            # signal, stays aligned — so only the CWT drifts).
            n_samples = len(signal)
            n_cols = cwt_data.shape[1]
            col_lo = int(round(first_zc / n_samples * n_cols))
            col_hi = int(round((last_zc + 1) / n_samples * n_cols))
            signal = signal[first_zc:last_zc + 1]
            inst_freq = inst_freq[first_zc:last_zc + 1]
            t = t[first_zc:last_zc + 1] - t[first_zc] if len(t) > first_zc else t[:0]
            cwt_data = cwt_data[:, col_lo:col_hi]

    duration_s = len(signal) / SR

    # Convert both magnitudes to dB referenced to the chirp ridge (0 dB =
    # chirp). STFT is computed here on the trimmed visible signal (same nperseg)
    # so its dB conversion matches the CWT's. The builder uses data["stft_mag_log"]
    # when present, so it consumes these dB arrays directly.
    stft_mag = _stft_on_log_bins(signal, SR, cwt_freqs,
                                 nperseg=CONTENDER_STFT_NPERSEG)
    cwt_db = _to_db(cwt_data, cwt_ref, CONTENDER_DB_FLOOR)
    stft_db = _to_db(stft_mag, stft_ref, CONTENDER_DB_FLOOR)

    return {
        "signal": signal,
        "inst_freq": inst_freq,
        "t": t,
        "duration_s": duration_s,
        "cwt_data": cwt_db,
        "cwt_freqs": cwt_freqs,
        "stft_mag_log": stft_db,
        "cwt_vmin": CONTENDER_DB_FLOOR,
        "cwt_vmax": 0.0,
        "stft_vmin": CONTENDER_DB_FLOOR,
        "stft_vmax": 0.0,
        "ts_ylim": (-chirp_peak * CONTENDER_TS_YLIM_PAD,
                    chirp_peak * CONTENDER_TS_YLIM_PAD),
    }


def _build_contender_figure(
    *,
    dpi: int = 250,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
    row_gutter_inches: float | None = None,
    bottom_reserve_inches: float | None = None,
    debug: bool = False,
) -> Figure:
    data = _prepare_contender()
    return _build_3row_figure(
        data,
        xticks=_auto_xticks(data["duration_s"]),
        suptitle="Fourier vs Wavelet",
        footer=FOOTER_TEXT,
        display_freq_lim_hz=CONTENDER_DISPLAY_FREQ_LIM_HZ,
        display_freq_ticks=CONTENDER_DISPLAY_FREQ_TICKS,
        panel_units=CONTENDER_PANEL_UNITS,
        inst_freq_color=style.PRIMARY_COLOR,
        row_captions=(
            CONTENDER_ROW1_CAPTION, CONTENDER_ROW2_CAPTION, CONTENDER_ROW3_CAPTION
        ),
        # Pin all three captions to one uniform size (21pt = what the longest
        # caption fits at) instead of auto-shrinking each to a different size.
        caption_font_size=21,
        stft_nperseg=CONTENDER_STFT_NPERSEG,
        show_xticklabels_all_rows=True,
        # Taller footer band so "Figure 1" sits centered with balanced,
        # roomy space above (to the row-3 s-labels) and below (to the figure
        # edge) instead of being cramped in a 1.0" band at the bottom.
        footer_row_height=0.42,
        dpi=dpi,
        unit_inches=unit_inches,
        unit_height_inches=unit_height_inches,
        debug=debug,
        row_gutter_inches=row_gutter_inches,
        bottom_reserve_inches=bottom_reserve_inches,
    )


# ============================================================
# Stacked layout (v32+): the three plots tile flush on top of each other
# (hspace=0, shared time axis — only the bottom plot carries x ticks/label),
# Hz ticks form a dedicated column in the left margin, and the row labels move
# to a text column on the RIGHT. Distinct from the shared `_build_3row_figure`
# (still used by hero/anti-hero) so this restyle can't perturb those figures.
# ============================================================
# Grid layout — a SQUARE-UNIT system. One panel unit is a square (width unit ==
# height unit == STACK_SQUARE_INCHES). The caption is exactly ONE square; each
# plot spans STACK_PLOT_COLS squares wide × one square tall, in the same row as
# its caption square. The y-axis labels (Hz/amp + numbers) render INSIDE the
# plot's own panel box — a strip STACK_LABEL_PAD_INCHES wide is reserved on the
# plot's left (via the panel's content_left_pad_inches hook), so there is no
# separate label column. Three body rows + a header band + a footer band.
STACK_SQUARE_INCHES = 5.0     # the panel unit: one square (width unit == height unit)
STACK_PLOT_COLS = 6           # plot spans this many squares wide
# Caption stays SQUARE while the plot rows grew taller (2 units): a 1-col square
# would be 1 unit tall, but the body rows are STACK_PLOT_ROW_HEIGHT units tall,
# so the caption must span that many columns to stay square (side == row height).
STACK_TEXT_COLS = 2           # caption square is 2 squares wide × 2 tall
STACK_W_UNIT_INCHES = STACK_SQUARE_INCHES   # width unit == square
STACK_UNIT_INCHES = STACK_SQUARE_INCHES     # height unit == square
STACK_PLOT_UNITS = (STACK_PLOT_COLS, 1)     # K squares wide × 1 row tall
STACK_TEXT_UNITS = (STACK_TEXT_COLS, 1)     # 2×2 square (2 cols, 2-unit-tall row)
STACK_PLOT_ROW_HEIGHT = 2     # each body row is TWO squares tall (taller plots)
STACK_BAND_HEIGHT = 0.4       # header / footer bands — short relative to a square
# Strip reserved on each plot's LEFT (inside its cell box) for the y-axis label
# + tick numbers. The plot's data axes is inset rightward by this much, so the
# labels live inside the panel box instead of a separate column.
STACK_LABEL_PAD_INCHES = 2.6
# A little air between panels (inches) — uniform gutter between stacked plots
# (rows) and between each plot and its caption square. Matched to 2.4.1's
# column gutter (0.30) so Figure 1's cells are the SAME size as the other
# figures' tiles (the shared-square model) instead of ~12% larger.
STACK_GUTTER_INCHES = 0.3
# Vertical breathing room INSIDE each plot cell: the filled data axes is inset
# from the cell top + bottom by this much so the extreme y-tick labels (1k at
# top, 10 at bottom) lift off the cell-border line instead of sitting on it.
STACK_CELL_VPAD_INCHES = 0.6
# CONTENDER_CAPTION_FONT_SIZE is defined earlier (above CONTENDER_TIGHT_STYLE,
# which derives the tick + axis label sizes from it).


def _side_text_panel(
    title: str,
    caption: str,
    *,
    font_size: float | None = None,
    min_font_size: float = 24,
) -> TextPanel:
    """Right-hand row label: title lead-in + justified caption, rendered
    ENTIRELY inside the cell (no chrome-zone title) so it survives hspace=0
    stacking — a chrome title would render in the now-zero gutter and collide
    with the panel above.

    ``font_size`` is the CEILING of the auto-fit search and is set to the
    size at which the LONGEST caption (the STFT/FOURIER block) just fits its
    cell. Because every caption shares that ceiling and the shorter ones also
    fit at it, all three render at the same size — uniform across the column,
    with shorter captions leaving a void below rather than scaling up.
    """
    # Default the auto-fit CEILING to the shared caption size so axis labels and
    # caption text stay one type system across figures (style.py is the source).
    if font_size is None:
        font_size = style.DEFAULT_CAPTION_FONT_SIZE
    text = f"{title.upper()}\n\n{caption}"
    return TextPanel(
        text,
        units=STACK_TEXT_UNITS,
        font_size=font_size,
        min_font_size=min_font_size,
        color=style.TICK_LABEL_COLOR,
        fontweight="normal",
        auto_shrink=False,
        # Left-justified (ragged right) — top_anchor still drives the wrap
        # pipeline so the block starts at the cell's top-left; justify=False
        # just drops the full-width word spreading.
        justify=False,
        # Top-anchored: every caption starts at the same top edge of its cell, so
        # the three blocks share a common top-left origin (uniform size pins the
        # rest). Shorter captions leave a void below rather than floating centred.
        top_anchor=True,
        # Text starts in the cell's top-left corner with a small uniform margin
        # in from the cell border (content_margin_frac is measured from the cell
        # border, not the inset axes box). The margin is sized to clear row 1's
        # right-side amplitude tick labels (-1/0/1), which render across the
        # gutter to the cell border — a tighter inset would let "0" touch the
        # AUDIO body text. Uniform across all three captions for alignment.
        content_margin_frac=0.018,
        # No inner ghost outline — the cell border (show_cell_borders) frames
        # each caption as a single clean box matching the plot cells. The
        # inner content-margin rect drew a redundant second line (double border).
        show_ghost_border=False,
        # Transparent cell so the time-series row's right-side amplitude tick
        # labels (which render leftward into this caption cell's empty top-left
        # margin) aren't painted over. Figure bg already equals BG_COLOR, so this
        # is visually identical for the captions themselves.
        facecolor="none",
    )


def _build_contender_stacked_figure(
    *,
    dpi: int = 150,
    unit_inches: float | None = None,
    unit_height_inches: float | None = None,
    debug: bool = False,
) -> Figure:
    data = _prepare_contender()
    duration_s = data["duration_s"]
    cwt_freqs = data["cwt_freqs"]
    cwt_data = data["cwt_data"]
    stft_mag_log = data["stft_mag_log"]

    display_freq_ticks = CONTENDER_DISPLAY_FREQ_TICKS
    disp_lo, disp_hi = CONTENDER_DISPLAY_FREQ_LIM_HZ
    bin_lo = int(np.searchsorted(cwt_freqs, disp_lo, side="left"))
    bin_hi = int(np.searchsorted(cwt_freqs, disp_hi, side="right"))
    bin_lo = max(0, min(bin_lo, len(cwt_freqs) - 1))
    bin_hi = max(bin_lo + 1, min(bin_hi, len(cwt_freqs)))
    cwt_freqs = cwt_freqs[bin_lo:bin_hi]
    cwt_data = cwt_data[bin_lo:bin_hi, :]
    stft_mag_log = stft_mag_log[bin_lo:bin_hi, :]

    log_f0 = float(np.log2(cwt_freqs[0]))
    log_step = float(np.log2(cwt_freqs[1] / cwt_freqs[0]))
    freq_to_bin = lambda f: (np.log2(f) - log_f0) / log_step

    # Display the clip on a clean 0–2.0 s axis: the trimmed signal is ~1.98 s,
    # so map every time axis (waveform, inst-freq curve, both spectrograms) onto
    # an even 2.0 s span. The ~0.9% stretch is uniform across all four traces and
    # imperceptible, but it lets the x-axis carry a 2.0 s tick at the right edge.
    disp_dur = 2.0
    disp_sr = len(data["signal"]) / disp_dur
    t_axis = np.linspace(0.0, disp_dur, len(data["signal"]))
    inst_freq_bins = np.interp(
        data["inst_freq"], cwt_freqs, np.arange(len(cwt_freqs))
    )
    twin_ytick_positions = [float(freq_to_bin(f)) for f in display_freq_ticks]
    twin_ytick_labels = [
        f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}'
        for f in display_freq_ticks
    ]
    twin_ylim = (0.0, float(len(cwt_freqs)))
    spec_extent = (0.0, disp_dur, 0.0, float(len(cwt_freqs)))
    xticks = _auto_xticks(disp_dur)

    # Row 1 — chirp waveform. Its OWN amplitude scale (-1..1) labels the LEFT
    # side, dropping into the same y-tick column rows 2/3 use for Hz (so the
    # numbers column-align down the figure) and keeping the right gutter clear
    # for the caption panel. A hidden twin axis (twin_ylim maps Hz→bin) only
    # positions the orange inst-freq overlay so it tracks the spectrogram bins
    # below; it carries no ticks/label.
    row1 = TimeSeriesPanel(
        units=STACK_PLOT_UNITS,
        xticks=xticks,
        xlim=(0.0, disp_dur),
        show_xticklabels=False,
        # Host axis carries the waveform on a HIDDEN amplitude scale (ts_ylim);
        # no ticks/label so the left side reads as a frequency axis. The ylim is
        # still pinned so the waveform's vertical extent (top aligned to the
        # caption text) is preserved.
        ylim=data.get("ts_ylim"),
        yticks=[],
        show_yticklabels=False,
        y_label_side="left",
        # Twin carries the Hz/bin scale on the LEFT — same ticks/label as the
        # spectrograms below, so the top plot's frequency axis column-aligns with
        # rows 2/3. The inst-freq overlay (bin space) draws onto this twin.
        twin_y=True,
        twin_y_label="Hz",
        twin_y_side="left",
        twin_yticks=twin_ytick_positions,
        twin_ytick_labels=twin_ytick_labels,
        twin_ylim=twin_ylim,
    )
    row1.add(TimeSeries(data["signal"], disp_sr, color=style.TICK_LABEL_COLOR))
    row1.add_twin(Line(
        t_axis, inst_freq_bins, color=style.BG_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 9.5, alpha=1.0,
    ))
    row1.add_twin(Line(
        t_axis, inst_freq_bins, color=style.PRIMARY_COLOR,
        linewidth=style.INST_FREQ_LINEWIDTH + 7.0, alpha=1.0,
    ))

    # Rows 2/3 — STFT then CWT. Only row 3 owns the shared x axis.
    row2 = HeatmapPanel(
        units=STACK_PLOT_UNITS, y_label="Hz", xticks=xticks,
        show_xticklabels=False,
    )
    row2.add(Heatmap(
        stft_mag_log, duration_s=duration_s, freqs=cwt_freqs, log_freq=True,
        tick_freqs=display_freq_ticks, extent=spec_extent,
        vmin=data.get("stft_vmin", 0.0), vmax=data.get("stft_vmax"),
    ))
    row3 = HeatmapPanel(
        units=STACK_PLOT_UNITS, x_label="s", y_label="Hz", xticks=xticks,
        show_xticklabels=True,
    )
    row3.add(Heatmap(
        cwt_data, duration_s=duration_s, freqs=cwt_freqs, log_freq=True,
        tick_freqs=display_freq_ticks, extent=spec_extent,
        vmin=data.get("cwt_vmin", 0.0), vmax=data.get("cwt_vmax"),
    ))

    # Reserve the in-panel label strip on every plot — the y-axis label + tick
    # numbers render inside this strip, inside each plot's own cell box, so no
    # separate y-tick column is needed.
    row1.content_left_pad_inches = STACK_LABEL_PAD_INCHES
    row2.content_left_pad_inches = STACK_LABEL_PAD_INCHES
    row3.content_left_pad_inches = STACK_LABEL_PAD_INCHES
    # Maximize each plot vertically — expand the data axes to fill its panel cell
    # height (the left label strip is preserved by fill_cell_vertical), but inset
    # top + bottom by STACK_CELL_VPAD_INCHES so the extreme y-ticks (1k/10) don't
    # sit on the cell-border line.
    row1.fill_cell_vertical = True
    row2.fill_cell_vertical = True
    row3.fill_cell_vertical = True
    row1.fill_cell_pad_inches = STACK_CELL_VPAD_INCHES
    row2.fill_cell_pad_inches = STACK_CELL_VPAD_INCHES
    row3.fill_cell_pad_inches = STACK_CELL_VPAD_INCHES

    # Caption font bumped above the shared default: the caption square grew from
    # 5"→10" per side (taller plots → bigger square), so the body text scales up
    # with it to stay proportionate (passed locally so other figures keep the
    # shared style.DEFAULT_CAPTION_FONT_SIZE).
    text1 = _side_text_panel(ROW1_TITLE, CONTENDER_ROW1_CAPTION, font_size=CONTENDER_CAPTION_FONT_SIZE)
    text2 = _side_text_panel(ROW2_TITLE, CONTENDER_ROW2_CAPTION, font_size=CONTENDER_CAPTION_FONT_SIZE)
    text3 = _side_text_panel(ROW3_TITLE, CONTENDER_ROW3_CAPTION, font_size=CONTENDER_CAPTION_FONT_SIZE)

    total_w = STACK_PLOT_UNITS[0] + STACK_TEXT_UNITS[0]
    suptitle_row = [SuptitlePanel(
        "Figure 1 - Fourier vs Wavelet Analysis", units=(total_w, 1), font_size=44
    )]
    # Empty bottom band: the footer text is removed, but the row is kept as a
    # spacer so row 3's x-axis tick numbers + "s" label have room below the plot
    # (without it they'd render into the thin figure margin and clip).
    footer_row = [SuptitlePanel("", units=(total_w, 1), font_size=34)]

    # Square units: width unit == height unit. Columns are integer multiples of
    # u_w, rows of u_h.
    u_w = unit_inches if unit_inches is not None else STACK_W_UNIT_INCHES
    u_h = unit_height_inches if unit_height_inches is not None else STACK_UNIT_INCHES
    # mpl hspace/wspace are fractions of the AVERAGE cell dimension. A body cell
    # is (u_w wide) × (u_h*STACK_PLOT_ROW_HEIGHT tall), so these fractions land
    # an exact STACK_GUTTER_INCHES gap in both directions.
    hspace = STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT)
    wspace = STACK_GUTTER_INCHES / u_w
    with _contender_tight_style():
        return Figure.compose(
            rows=[
                suptitle_row,
                [row1, text1],
                [row2, text2],
                [row3, text3],
                footer_row,
            ],
            row_heights=[
                STACK_BAND_HEIGHT,
                STACK_PLOT_ROW_HEIGHT, STACK_PLOT_ROW_HEIGHT, STACK_PLOT_ROW_HEIGHT,
                STACK_BAND_HEIGHT,
            ],
            # A small uniform gutter between plots (rows) and between the
            # y-tick / plot / caption columns — breathing room, not flush.
            hspace=hspace,
            wspace=wspace,
            dpi=dpi,
            unit_inches=u_w,
            unit_height_inches=u_h,
            # Template/inspection cell-border grid ON.
            show_cell_borders=True,
            debug_guides=debug,
        )


def render_contender(
    output_dir: str = "assets/images/dsp/figures/figure_1",
    output_filename: str = "contender_low_chirp_v1.png",
) -> str:
    """Build, render, and save the contender figure. Returns absolute path."""
    with _contender_tight_style():
        fig = _build_contender_stacked_figure()
        fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show_contender(debug: bool = False) -> Figure:
    """Build, render, and display the contender figure in a notebook cell.

    Notebook-tuned exactly like `show_hero`: compact style profile + 75%
    canvas width + ipympl chrome suppression.
    """
    import matplotlib.pyplot as plt
    with nb_compact_style():
        fig = _build_contender_figure(
            dpi=80,
            unit_inches=2.5,
            unit_height_inches=2.5,
            debug=debug,
        )
        fig._display_width = "75%"
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


def render_all() -> tuple[str, str, str]:
    """Render the v8 chirp + hero + anti-hero into assets/images/dsp/figures/figure_1/."""
    chirp_path = render()
    hero_path = render_hero()
    antihero_path = render_antihero()
    return chirp_path, hero_path, antihero_path


def show_hero(debug: bool = False) -> Figure:
    """Build, render, and display the hero figure in a notebook cell.

    Notebook-tuned: a fully compact style profile (smaller gutters, margins,
    font sizes) is active during construction + render so the chrome:cell
    ratio matches print. The canvas widget is capped at 75% of the cell
    width so the figure doesn't dominate the notebook. The static PNG path
    via ``render_hero()`` keeps the print-scale chrome.

    ``debug=True`` overlays figure-level guide lines (cyan/yellow/orange/lime
    for chrome bands and cell gutters) plus a red bracket above and below
    every rendered Text artist — titles, captions, suptitle, footer, axis
    labels, tick labels. Anything that overflows its band shows as text
    crossing a guide line.
    """
    import matplotlib.pyplot as plt
    with nb_compact_style():
        # unit_inches == unit_height_inches so that 1×1 cells (label panels,
        # suptitle, footer) render as squares. Data cells at (3, 1) become
        # 3× wider than tall — wide spectrogram aspect ratio falls out
        # naturally from the unit ratio, not from a separate unit_height.
        # NO bottom_reserve_inches override: compose lifts the footer
        # SuptitlePanel into a bottom band sized from its row height. Passing
        # bottom_reserve_inches=0 here would clobber that and the footer
        # would have no room to render.
        fig = _build_hero_figure(
            dpi=80,
            unit_inches=2.5,
            unit_height_inches=2.5,
            debug=debug,
        )
        fig._display_width = "75%"
        fig.render()
    # Suppress the ipympl widget chrome ("Figure 1" / toolbar) — that header
    # is matplotlib canvas chrome, distinct from the figure-bottom footer
    # SuptitlePanel that we render in the gridspec.
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
