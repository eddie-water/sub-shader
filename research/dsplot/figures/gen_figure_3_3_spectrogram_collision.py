"""Figure 3.3 - Spectrogram Collision (two signals the STFT cannot tell apart).

The Short Time Fourier Transform analyses every frequency through one window of
one fixed length. Whatever that length is, structure faster than the window is
averaged away: the window reports how much energy sat at a frequency, not how it
was distributed inside the window.

Two signals are built to exploit exactly that. Both carry the same 200 Hz tone
(the control - long time support, identical in both). They differ only in the
2 kHz band, a decade above it:

    Signal A   2 kHz tone sounding continuously
    Signal B   the same 2 kHz tone, amplitude-modulated to full depth at 40 Hz
               so it dies away and swells back once every 25 ms, turned up so
               each STFT window receives the same energy as A's continuous tone

At `STFT_NPERSEG` = 4096 (92.9 ms) each analysis window spans ~3.7 of B's
modulation cycles, so both signals deposit the same magnitude in the same bin
and the two spectrograms read as the same picture. The CWT's window at 2 kHz is
`num_cycles / f` ~= 3 ms - an eighth of one modulation cycle - so it resolves B
into a train of discrete events while A stays a solid line.

The 200 Hz row is the honest control: it is identical between the two signals,
so any difference visible at 2 kHz is a property of the transform, not of the
rendering. In signal B's wavelet panel the 200 Hz line also stays solid, because
the wavelet down there is ~30 ms long and cannot resolve the modulation either -
the resolution scales with frequency, which is the constant-Q argument making
itself.

Why AM and not a loud-short vs quiet-long burst pair
----------------------------------------------------
The intuitive construction - one 1 ms burst at amplitude A against one 10 ms
burst at amplitude A/10 - does NOT collide. Both transforms obey the same
uncertainty bound: an event short enough for the CWT to call it an impulse is
necessarily *wideband*, and a long-window STFT has fine enough frequency
resolution to see that width. Rendered, the 1 ms burst is a tall vertical smear
and the 10 ms burst a thin horizontal dash - same brightness, obviously
different shapes. The bandwidths have to match for the STFTs to match, which
means the durations have to match, which means the difference has to live in how
the energy is *arranged in time*.

Analysis settings are held identical between the two signals (same nperseg, same
CWT bank, same dB reference) so the panels differ only by their input - the
apples-to-apples rule that governs Figure 1.
"""
from __future__ import annotations

import os

import numpy as np

from .. import (
    Figure,
    Heatmap,
    HeatmapPanel,
    SuptitlePanel,
    TimeSeries,
    TimeSeriesPanel,
    style,
)
from .gen_figure_1_stft_vs_cwt import (
    STACK_BAND_HEIGHT,
    STACK_CELL_VPAD_INCHES,
    STACK_GUTTER_INCHES,
    STACK_LABEL_PAD_INCHES,
    STACK_MID_BAND_TRIM_INCHES,
    STACK_PLOT_COLS,
    STACK_PLOT_ROW_HEIGHT,
    STACK_PLOT_UNITS,
    STACK_TEXT_COLS,
    _contender_tight_style,
    _side_text_panel,
    _stft_on_log_bins,
    _to_db,
)
from utilities import compute_full_cwt


# ============================================================
# Signal design - locked. Changing any of these re-opens the collision question;
# re-run `report_collision()` afterwards and check the residual.
# ============================================================
SR = 44100

# Built long, displayed short. The CWT discards its own edge-effect regions and
# the visible window is cropped further inside that, so neither spectrogram
# carries a boundary artifact. No mirror padding is used: a reflected sine has a
# derivative corner at the seam, which splashes broadband energy into the edge
# STFT columns (visible as a bright block).
BUILD_DURATION_S = 0.90
VISIBLE_DURATION_S = 0.40

LOW_HZ = 200.0            # the control - identical in both signals
LOW_AMP = 1.0
HIGH_HZ = 2000.0          # a decade above the control (the 1:10 range)
HIGH_AMP_A = 1.0          # signal A: continuous

# Signal B's amplitude modulation: 100% depth at 40 Hz, so the 2 kHz tone swells
# and dies away once every 25 ms.
#
# The rate is boxed in from both sides and 40 Hz sits in the middle of the box:
#   lower bound  the STFT window must average MANY cycles rather than resolve
#                them - the window is 92.9 ms, so the period must sit well under
#                it.
#   upper bound  100% AM puts sidebands at exactly HIGH_HZ +/- MOD_RATE_HZ and
#                nowhere else. Both must stay inside one chromatic bin (~119 Hz
#                wide at 2 kHz) or they render as extra rows in the STFT panel
#                and the collision breaks.
#
# A hard on/off gate was tried first and rejected: even with raised-cosine edges,
# its harmonics land on the neighbouring chromatic bins ~113 Hz out and put a
# visible stripe under B's 2 kHz line (measured at -16.6 dB against a -40 dB
# floor, where signal A is silent). Pure AM is the minimum-bandwidth way to
# interrupt a tone.
MOD_RATE_HZ = 40.0

# 4096 samples = 92.9 ms. Long enough to resolve 200 Hz cleanly in frequency,
# which is exactly what forces it to be blind to 25 ms structure at 2 kHz.
STFT_NPERSEG = 4096

CWT_ROOT_HZ = 150.0
CWT_NUM_OCTAVES = 6

DISPLAY_FREQ_LIM_HZ: tuple[float, float] = (150.0, 4000.0)
DISPLAY_FREQ_TICKS: tuple[int, ...] = (200, 2000)
DB_FLOOR = -40.0

# The detail row is a decomposition legend, not an A/B comparison (the Audio band
# already does A/B). Left panel: the whole signal. Right panel: exactly one cycle
# of each constituent on one shared time axis - the low tone's cycle spans the
# panel, the high tone's single cycle is zero-padded out to the same span and
# centred in the middle of the low tone's support. Drawn to the same scale, so
# the 10:1 ratio of time supports is not asserted in a caption, it is visible.
ONE_CYCLE_DURATION_S = 1.0 / LOW_HZ                  # 5 ms
ONE_CYCLE_XTICKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]    # milliseconds

# The shared time axis for the full-clip rows. `_auto_xticks` steps in
# half-seconds, which yields a single 0.0 tick on a 0.4 s clip - this figure
# needs its own scale.
XTICKS = [0.0, 0.1, 0.2, 0.3, 0.4]

FIGURE_TITLE = "Figure 3.3 - Spectrogram Collision"

# Row headers name the transform; captions name the signal. Scaffolding prose -
# final copy is authored by hand.
BAND_TITLES = ("Audio Signal", "Time Support", "Fourier Analysis",
               "Wavelet Analysis")
CAPTION_A_AUDIO = (
    "A 200 Hz tone and a 2 kHz tone, a decade apart, both sounding "
    "continuously for the whole clip. The mix never changes - every slice of "
    "this signal looks like every other slice."
)
CAPTION_B_AUDIO = (
    "The same 200 Hz tone, but the 2 kHz tone dies away and swells back every "
    "25 ms, turned up to compensate. Each 92.9 ms Fourier window still "
    "receives the same 2 kHz energy as the signal above."
)
CAPTION_ONE_CYCLE = (
    "Left: the whole signal. Right: one cycle of each tone, on the same time "
    "axis. The 200 Hz cycle fills all 5 ms; the 2 kHz cycle takes 0.5 ms and "
    "is centred inside it, zero either side. One event ten times shorter than "
    "the other - that span is what a single fixed window has to cover, and "
    "cannot."
)
CAPTION_A_STFT = (
    "Two solid lines, one per tone. A 92.9 ms window places both frequencies "
    "precisely - this is the STFT doing its job well, on a signal that never "
    "changes."
)
CAPTION_B_STFT = (
    "The same two solid lines. The window spans several modulation cycles and "
    "reports only their average, so the interruptions leave no trace. The "
    "STFT cannot tell these two signals apart."
)
CAPTION_A_CWT = (
    "Two solid lines again. Every wavelet in the bank agrees with the Fourier "
    "reading, because there is genuinely nothing here that changes over "
    "time."
)
CAPTION_B_CWT = (
    "The 200 Hz line is unchanged - but the 2 kHz line breaks into a train of "
    "distinct events. The wavelet at 2 kHz spans about 3 ms, short enough to "
    "see between the swells."
)

# Ceiling for the caption column. Set to the size at which the LONGEST caption
# above still fits its square, so all of them render at one size instead of each
# shrinking to its own.
CAPTION_FONT_SIZE = 34


def _modulation_envelope(t: np.ndarray) -> np.ndarray:
    """100%-depth amplitude modulation at `MOD_RATE_HZ`: 0.5 * (1 - cos).

    Touches zero once per cycle - a genuine interruption, not a wobble - while
    spending its entire spectrum on two sidebands at +/- MOD_RATE_HZ. Anything
    with sharper edges is wider in frequency, and width is the one thing the
    STFT can still see.
    """
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * MOD_RATE_HZ * t))


def _stft_high_band_peak(signal: np.ndarray) -> float:
    """Peak linear STFT magnitude in a band around `HIGH_HZ`.

    Used to calibrate signal B's amplitude empirically rather than trusting the
    theoretical 2x - the analysis window's own taper bends the exact value.
    """
    from scipy.signal import stft as scipy_stft
    freqs, _t, Zxx = scipy_stft(signal, fs=SR, nperseg=STFT_NPERSEG,
                                noverlap=STFT_NPERSEG // 2)
    mag = np.abs(Zxx)[1:]
    freqs = freqs[1:]
    band = (freqs > HIGH_HZ * 0.85) & (freqs < HIGH_HZ * 1.15)
    return float(mag[band].max())


def _build_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (t, signal_a, signal_b, modulated_amp) over the full build."""
    t = np.arange(int(SR * BUILD_DURATION_S)) / SR
    low = LOW_AMP * np.sin(2.0 * np.pi * LOW_HZ * t)
    high_continuous = HIGH_AMP_A * np.sin(2.0 * np.pi * HIGH_HZ * t)
    high_modulated_unit = np.sin(2.0 * np.pi * HIGH_HZ * t) * _modulation_envelope(t)

    modulated_amp = (_stft_high_band_peak(high_continuous)
                     / _stft_high_band_peak(high_modulated_unit))
    return t, low + high_continuous, low + modulated_amp * high_modulated_unit, modulated_amp


def _prepare() -> dict:
    """Run both signals through the same CWT bank and STFT, crop to the visible
    window, and convert both to dB against ONE shared reference.

    A shared reference is what makes the two spectrograms comparable at all: a
    per-signal reference would renormalize B's brighter modulated peaks and hide
    the very thing being measured.
    """
    t, sig_a, sig_b, modulated_amp = _build_signals()

    cwt_a, cwt_freqs, start_sample, end_sample = compute_full_cwt(
        sig_a, SR, root_note_hz=CWT_ROOT_HZ, num_octaves=CWT_NUM_OCTAVES
    )
    cwt_b, _f, start_b, end_b = compute_full_cwt(
        sig_b, SR, root_note_hz=CWT_ROOT_HZ, num_octaves=CWT_NUM_OCTAVES
    )
    if (start_b, end_b) != (start_sample, end_sample):
        raise RuntimeError(
            "CWT trim windows diverged between the two signals "
            f"({start_sample},{end_sample}) vs ({start_b},{end_b}) - the panels "
            "would no longer share a time axis."
        )

    # Crop to a centred VISIBLE_DURATION_S window strictly inside the CWT's own
    # trim, in BOTH sample space (waveform, STFT) and column space (CWT).
    n_visible = int(SR * VISIBLE_DURATION_S)
    span = end_sample - start_sample
    if n_visible > span:
        raise ValueError(
            f"VISIBLE_DURATION_S={VISIBLE_DURATION_S}s needs {n_visible} samples "
            f"but the CWT keeps only {span}. Lengthen BUILD_DURATION_S."
        )
    lo = start_sample + (span - n_visible) // 2
    hi = lo + n_visible
    n_cols = cwt_a.shape[1]
    col_lo = int(round((lo - start_sample) / span * n_cols))
    col_hi = int(round((hi - start_sample) / span * n_cols))

    wave_a = sig_a[lo:hi]
    wave_b = sig_b[lo:hi]
    cwt_a = cwt_a[:, col_lo:col_hi]
    cwt_b = cwt_b[:, col_lo:col_hi]

    # STFT runs on the FULL build and is then sliced to the visible window, so no
    # analysis window ever straddles the ends of the buffer.
    stft_a_full = _stft_on_log_bins(sig_a, SR, cwt_freqs, nperseg=STFT_NPERSEG)
    stft_b_full = _stft_on_log_bins(sig_b, SR, cwt_freqs, nperseg=STFT_NPERSEG)
    n_stft = stft_a_full.shape[1]
    s_lo = int(round(lo / len(sig_a) * n_stft))
    s_hi = max(s_lo + 1, int(round(hi / len(sig_a) * n_stft)))
    stft_a = stft_a_full[:, s_lo:s_hi]
    stft_b = stft_b_full[:, s_lo:s_hi]

    stft_ref = float(np.percentile(stft_a, 99.9)) or 1.0
    cwt_ref = float(np.percentile(np.abs(cwt_a), 99.9)) or 1.0

    # Waveforms share one normalization so B's taller modulated peaks read as
    # genuinely louder rather than being scaled back to the same height.
    wave_peak = max(float(np.max(np.abs(wave_a))),
                    float(np.max(np.abs(wave_b)))) or 1.0

    return {
        "duration_s": VISIBLE_DURATION_S,
        "cwt_freqs": cwt_freqs,
        "modulated_amp": modulated_amp,
        "wave_peak": wave_peak,
        "wave_a": wave_a / wave_peak,
        "wave_b": wave_b / wave_peak,
        "stft_a": _to_db(stft_a, stft_ref, DB_FLOOR),
        "stft_b": _to_db(stft_b, stft_ref, DB_FLOOR),
        "cwt_a": _to_db(np.abs(cwt_a), cwt_ref, DB_FLOOR),
        "cwt_b": _to_db(np.abs(cwt_b), cwt_ref, DB_FLOOR),
        "stft_ref": stft_ref,
    }


def _slice_display_band(data: dict) -> dict:
    """Restrict every spectrogram to DISPLAY_FREQ_LIM_HZ so the log-spaced panels
    fill their cells instead of leaving empty rows top and bottom."""
    freqs = data["cwt_freqs"]
    disp_lo, disp_hi = DISPLAY_FREQ_LIM_HZ
    bin_lo = max(0, min(int(np.searchsorted(freqs, disp_lo, "left")), len(freqs) - 1))
    bin_hi = max(bin_lo + 1, min(int(np.searchsorted(freqs, disp_hi, "right")), len(freqs)))
    out = dict(data)
    out["cwt_freqs"] = freqs[bin_lo:bin_hi]
    for key in ("stft_a", "stft_b", "cwt_a", "cwt_b"):
        out[key] = data[key][bin_lo:bin_hi, :]
    return out


def report_collision(data: dict | None = None) -> dict:
    """Measure how close the two STFTs actually are, and how far apart the CWTs.

    Run this after touching any signal constant. The figure's claim is only as
    good as these numbers.
    """
    if data is None:
        data = _prepare()
    band = _slice_display_band(data)
    stft_diff = np.abs(band["stft_a"] - band["stft_b"])
    cwt_diff = np.abs(band["cwt_a"] - band["cwt_b"])
    metrics = {
        "modulated_amp": data["modulated_amp"],
        "stft_max_db_diff": float(stft_diff.max()),
        "stft_p99_db_diff": float(np.percentile(stft_diff, 99)),
        "stft_mean_db_diff": float(stft_diff.mean()),
        "cwt_max_db_diff": float(cwt_diff.max()),
        "cwt_mean_db_diff": float(cwt_diff.mean()),
    }
    metrics["separation_ratio"] = (
        metrics["cwt_mean_db_diff"] / max(metrics["stft_mean_db_diff"], 1e-9)
    )
    return metrics


# ============================================================
# Layout - body rows grouped BY TRANSFORM, not by signal, so the two spectrograms
# being compared are always vertically adjacent. Chrome, spacing and caption
# treatment are Figure 1's stacked (v47) system, reused verbatim.
# ============================================================
HALF_PLOT_UNITS = (STACK_PLOT_COLS // 2, 1)


def _spectrogram_panel(matrix: np.ndarray, freqs: np.ndarray, xticks: list[float],
                       *, show_xaxis: bool) -> HeatmapPanel:
    panel = HeatmapPanel(
        units=STACK_PLOT_UNITS,
        x_label="s" if show_xaxis else None,
        y_label="Hz",
        xticks=xticks,
        show_xticklabels=show_xaxis,
    )
    panel.add(Heatmap(
        matrix, duration_s=VISIBLE_DURATION_S, freqs=freqs, log_freq=True,
        tick_freqs=DISPLAY_FREQ_TICKS,
        extent=(0.0, VISIBLE_DURATION_S, 0.0, float(len(freqs))),
        vmin=DB_FLOOR, vmax=0.0,
    ))
    return panel


def _waveform_panel(wave: np.ndarray, xticks: list[float]) -> TimeSeriesPanel:
    panel = TimeSeriesPanel(
        units=STACK_PLOT_UNITS,
        x_label=None,
        xticks=xticks,
        xlim=(0.0, VISIBLE_DURATION_S),
        show_xticklabels=False,
        ylim=(-1.0, 1.0),
        yticks=[-1.0, 0.0, 1.0],
        show_yticklabels=True,
        y_label_side="left",
    )
    panel.add(TimeSeries(wave, len(wave) / VISIBLE_DURATION_S,
                         color=style.NEUTRAL_COLOR))
    return panel


def _one_cycle_traces() -> tuple[np.ndarray, np.ndarray]:
    """One cycle of the low tone, and one cycle of the high tone zero-padded to
    the same span and centred in it.

    Both come back on the same sample grid so a single panel can carry them on
    one axis with no rescaling - the point of the panel is that their widths are
    directly comparable.
    """
    n = int(round(ONE_CYCLE_DURATION_S * SR))
    t = np.arange(n) / SR
    low = LOW_AMP * np.sin(2.0 * np.pi * LOW_HZ * t)

    high = np.zeros(n, dtype=np.float64)
    n_high = int(round(SR / HIGH_HZ))          # one cycle of the high tone
    start = (n - n_high) // 2                  # centred in the low tone's support
    t_high = np.arange(n_high) / SR
    high[start:start + n_high] = HIGH_AMP_A * np.sin(2.0 * np.pi * HIGH_HZ * t_high)
    return low, high


def _one_cycle_panel() -> TimeSeriesPanel:
    """The two constituent cycles overlaid on one millisecond axis.

    The high tone takes PRIMARY_COLOR so the eye separates the two without a
    legend; the low tone keeps the bone-white every other trace uses.
    """
    low, high = _one_cycle_traces()
    panel = TimeSeriesPanel(
        units=HALF_PLOT_UNITS,
        x_label=None,
        xticks=list(ONE_CYCLE_XTICKS),
        xlim=(0.0, ONE_CYCLE_DURATION_S * 1000.0),
        show_xticklabels=True,
        ylim=(-1.15, 1.15),
        yticks=[],
        show_yticklabels=False,
    )
    sr_ms = len(low) / (ONE_CYCLE_DURATION_S * 1000.0)
    panel.add(TimeSeries(low, sr_ms, color=style.NEUTRAL_COLOR))
    panel.add(TimeSeries(high, sr_ms, color=style.PRIMARY_COLOR))
    panel.no_label_strip = True
    return panel


def _whole_signal_panel(wave: np.ndarray, xticks: list[float]) -> TimeSeriesPanel:
    """Half-width view of the entire clip - the macro half of the detail row."""
    panel = TimeSeriesPanel(
        units=HALF_PLOT_UNITS,
        x_label=None,
        xticks=xticks,
        xlim=(0.0, VISIBLE_DURATION_S),
        # No numbers: this panel's 0-0.4 s axis is already carried by the
        # full-clip rows above and the figure's bottom axis, and labelling it
        # put its "0.4" flush against the ms panel's "0" across the cell border.
        # The millisecond axis to the right is the informative one.
        show_xticklabels=False,
        ylim=(-1.0, 1.0),
        yticks=[],
        show_yticklabels=False,
    )
    panel.add(TimeSeries(wave, len(wave) / VISIBLE_DURATION_S,
                         color=style.NEUTRAL_COLOR))
    panel.no_label_strip = True
    return panel


def _unit_inches() -> float:
    """One square, sized so the figure lands on the house 28in canvas width."""
    return 28.0 / (STACK_PLOT_COLS + STACK_TEXT_COLS)


def _build_figure(*, data: dict | None = None, dpi: int = 150,
                  unit_inches: float | None = None,
                  unit_height_inches: float | None = None,
                  debug: bool = False) -> Figure:
    band = _slice_display_band(data if data is not None else _prepare())
    freqs = band["cwt_freqs"]
    xticks = list(XTICKS)

    # Each band holds a list of ROWS; a row is (panels, title, caption). Most
    # rows carry one full-width panel - the detail row carries two half-width
    # ones so it costs one row instead of two.
    rows_spec = [
        (BAND_TITLES[0], [
            ([_waveform_panel(band["wave_a"], xticks)], "Signal A", CAPTION_A_AUDIO),
            ([_waveform_panel(band["wave_b"], xticks)], "Signal B", CAPTION_B_AUDIO),
        ]),
        (BAND_TITLES[1], [
            ([_whole_signal_panel(band["wave_a"], xticks), _one_cycle_panel()],
             "Time Support", CAPTION_ONE_CYCLE),
        ]),
        (BAND_TITLES[2], [
            ([_spectrogram_panel(band["stft_a"], freqs, xticks, show_xaxis=False)],
             "Signal A", CAPTION_A_STFT),
            ([_spectrogram_panel(band["stft_b"], freqs, xticks, show_xaxis=False)],
             "Signal B", CAPTION_B_STFT),
        ]),
        (BAND_TITLES[3], [
            ([_spectrogram_panel(band["cwt_a"], freqs, xticks, show_xaxis=False)],
             "Signal A", CAPTION_A_CWT),
            ([_spectrogram_panel(band["cwt_b"], freqs, xticks, show_xaxis=True)],
             "Signal B", CAPTION_B_CWT),
        ]),
    ]

    total_w = STACK_PLOT_COLS + STACK_TEXT_COLS
    rows: list[list] = [[SuptitlePanel(FIGURE_TITLE, units=(total_w, 1), font_size=44)]]
    row_heights: list[float] = [STACK_BAND_HEIGHT]
    u_h = unit_height_inches if unit_height_inches is not None else _unit_inches()
    mid_band = STACK_BAND_HEIGHT - STACK_MID_BAND_TRIM_INCHES / u_h

    for band_index, (band_title, entries) in enumerate(rows_spec):
        rows.append([SuptitlePanel(band_title, units=(total_w, 1), font_size=44)])
        row_heights.append(STACK_BAND_HEIGHT if band_index == 0 else mid_band)
        for panels, title, caption in entries:
            for panel in panels:
                # Panels that carry no y-axis labels skip the label strip; left
                # in, it shoves each trace off-centre inside its cell.
                panel.content_left_pad_inches = (
                    0.0 if getattr(panel, "no_label_strip", False)
                    else STACK_LABEL_PAD_INCHES)
                panel.fill_cell_vertical = True
                panel.fill_cell_pad_inches = STACK_CELL_VPAD_INCHES
            rows.append([*panels, _side_text_panel(
                title, caption, font_size=CAPTION_FONT_SIZE)])
            row_heights.append(STACK_PLOT_ROW_HEIGHT)

    rows.append([SuptitlePanel("", units=(total_w, 1), font_size=34)])
    row_heights.append(STACK_BAND_HEIGHT)

    u_w = unit_inches if unit_inches is not None else _unit_inches()
    return Figure.compose(
        rows=rows,
        row_heights=row_heights,
        hspace=STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT),
        wspace=STACK_GUTTER_INCHES / u_w,
        dpi=dpi,
        unit_inches=u_w,
        unit_height_inches=u_h,
        show_cell_borders=True,
        debug_guides=debug,
    )


def render(output_dir: str = "assets/images/dsp/figures/by_figure/fig_3_3_spectrogram_collision",
           output_filename: str = "fig_3_3_spectrogram_collision_v1.png") -> str:
    """Build, render, and save at production DPI. Returns the absolute path."""
    # Build AND render inside the style context: panels resolve colours and sizes
    # at construction time, so composing outside it bakes in the bare template
    # defaults and the context only reaches the render pass.
    with _contender_tight_style():
        fig = _build_figure()
        fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


def show(debug: bool = False) -> Figure:
    """Notebook-tuned render for inline display in dsp.ipynb."""
    with _contender_tight_style():
        fig = _build_figure(dpi=60, unit_inches=1.6, unit_height_inches=1.6,
                            debug=debug)
        fig.render()
    return fig
