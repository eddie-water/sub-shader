"""DSP helper functions for the benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft as scipy_stft, resample as scipy_resample, chirp as scipy_chirp

from . import constants


@dataclass
class BouncingChirpConfig:
    sr: int = 44_100
    duration_s: float = 2.0
    f_decades: tuple[float, ...] = (20.0, 200.0, 2000.0, 20000.0)
    bounces_per_decade: int = 3
    seed: int | None = None


@dataclass
class WaypointChirpConfig:
    """Inst-freq curve defined by user-specified (time_frac, freq_hz) waypoints.

    Time fractions are in [0, 1] of duration_s. Waypoints are cubic-splined in
    log-frequency space. Waypoints must be strictly increasing in time and
    must include both endpoints (0 and 1).

    `clip_to_waypoints` clamps the resulting curve to [min_wp_freq,
    max_wp_freq]. Set False when an endpoint waypoint sits AT the desired
    extreme (e.g. ending at 20 kHz) — the cubic spline will tend to overshoot
    that extreme inside the last segment, and the clamp pins the curve into
    a visible plateau. With clip off, brief spline overshoot just clips
    against the panel edge in the plot.
    """
    sr: int = 44_100
    duration_s: float = 0.5
    waypoints: tuple[tuple[float, float], ...] = ()
    bc_type: str = "natural"
    clip_to_waypoints: bool = True
    seed: int | None = None  # unused, kept for API symmetry


def build_fm_chirp(sr: int, duration_s: float, fc: float, delta_f: float,
                   f_mod_start: float, f_mod_end: float, f_floor: float = 25.0):
    """Generate a biased FM signal with chirped modulation rate.

    Instantaneous frequency: fc + delta_f * sin(chirped_mod_phase)
    The modulation rate itself is a linear chirp from f_mod_start to f_mod_end,
    so the bouncing arcs accelerate over time.

    Args:
        fc: Bias/center frequency in Hz
        delta_f: Peak frequency deviation in Hz (hill height)
        f_mod_start: Initial modulation rate in Hz (slow bounces at start)
        f_mod_end: Final modulation rate in Hz (fast bounces at end)
        f_floor: Minimum allowed frequency in Hz

    Returns:
        (signal, inst_freq, t) — all 1-D arrays of length int(sr * duration_s)
    """
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, dtype=np.float64)

    # Modulator: linear chirp phase (accelerating bounce rate)
    k = (f_mod_end - f_mod_start) / duration_s
    mod_phase = 2 * np.pi * (f_mod_start * t + 0.5 * k * t**2)

    # Instantaneous frequency: biased sinusoidal
    inst_freq = fc + delta_f * np.sin(mod_phase)
    inst_freq = np.clip(inst_freq, f_floor, None)

    # Integrate inst_freq to get signal phase
    signal_phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(signal_phase)

    return signal.astype(np.float64), inst_freq, t


def build_fm_chirp_chunks(fc, delta_f, f_mod_start, f_mod_end, sr, chunk_size,
                          overlap_factor, n_frames, f_floor=25.0):
    """Generate a chirped-modulation FM signal pre-sliced into overlapping chunks.

    Returns:
        (chunks, signal, inst_freq, t) — chunks is a list of arrays,
        the rest are the full signal/freq/time for reference plotting.
    """
    hop_size = int(chunk_size * (1 - overlap_factor))
    total_samples = hop_size * n_frames + chunk_size
    duration_s = total_samples / sr

    signal, inst_freq, t = build_fm_chirp(
        sr, duration_s, fc, delta_f, f_mod_start, f_mod_end, f_floor=f_floor
    )

    chunks = [signal[i * hop_size: i * hop_size + chunk_size] for i in range(n_frames)]
    return chunks, signal, inst_freq, t


def build_wandering_chirp(sr: int, duration_s: float, f_lo: float, f_hi: float,
                          num_waypoints: int = 8, seed: int = 42,
                          margin: float = 0.15):
    """Generate a chirp whose frequency random-walks between bands.

    Picks random waypoints in [f_lo, f_hi] (log-spaced) with an inward margin,
    then cubic-interpolates to get a smooth instantaneous frequency curve.
    The margin keeps the curve from hugging the frequency bounds.

    Args:
        margin: Fraction of the log-freq range to inset waypoints from each edge.
                0.15 means waypoints stay in the inner 70% of the log-freq range.

    Returns:
        (signal, inst_freq, t)  — all 1-D arrays of length int(sr * duration_s)
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Inset waypoint range so cubic spline doesn't overshoot into bounds
    log_lo, log_hi = np.log(f_lo), np.log(f_hi)
    log_range = log_hi - log_lo
    wp_lo = log_lo + margin * log_range
    wp_hi = log_hi - margin * log_range

    # Random waypoints in log-frequency space (within inset range)
    wp_times = np.linspace(0, t[-1], num_waypoints)
    wp_freqs_log = rng.uniform(wp_lo, wp_hi, size=num_waypoints)
    # Start and end at mid-range for a clean loop
    log_mid = np.log(np.sqrt(f_lo * f_hi))
    wp_freqs_log[0] = log_mid
    wp_freqs_log[-1] = log_mid

    # Cubic interpolation for smooth wandering
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(wp_times, wp_freqs_log, bc_type='clamped')
    inst_freq = np.exp(cs(t))

    # Safety clamp (should rarely activate with margin)
    inst_freq = np.clip(inst_freq, f_lo, f_hi)

    # Integrate instantaneous frequency to get phase
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(phase)

    return signal.astype(np.float64), inst_freq, t


def build_wandering_chirp_chunks(f_lo, f_hi, sr, chunk_size, overlap_factor,
                                 n_frames, num_waypoints=8, seed=42):
    """Generate a frequency-wandering chirp pre-sliced into overlapping chunks.

    Returns:
        (chunks, signal, inst_freq, t) — chunks is a list of arrays,
        the rest are the full signal/freq/time for reference plotting.
    """
    hop_size = int(chunk_size * (1 - overlap_factor))
    total_samples = hop_size * n_frames + chunk_size
    duration_s = total_samples / sr

    signal, inst_freq, t = build_wandering_chirp(
        sr, duration_s, f_lo, f_hi, num_waypoints=num_waypoints, seed=seed
    )

    chunks = [signal[i * hop_size: i * hop_size + chunk_size] for i in range(n_frames)]
    return chunks, signal, inst_freq, t


def build_bouncing_chirp(sr: int, duration_s: float,
                         f_decades: list = None,
                         bounces_per_decade: int = 3,
                         seed: int = None) -> tuple:
    """Generate a chirp whose frequency ascends across decades with parabolic dips.

    The frequency contour rises overall from ~20 Hz to ~20 kHz across 3 frequency
    decades, with periodic parabolic dips — like a ball bouncing upward. Each bounce
    starts near the previous decade floor, rises toward the next decade ceiling, then
    dips partway back before rising higher.

    This shape is designed to showcase SubShader's time-frequency resolution advantage:
    the bouncing contour is clearly distinct from both stationary tones and linear chirps.

    Args:
        sr: Sample rate in Hz
        duration_s: Signal duration in seconds
        f_decades: List of frequency decade boundaries in Hz.
                   Default: [20, 200, 2000, 20000] (3 decades)
        bounces_per_decade: Number of parabolic arcs per decade segment
        seed: Random seed for optional jitter (None = deterministic)

    Returns:
        (signal, inst_freq, t) — all 1-D arrays of length int(sr * duration_s)
    """
    if f_decades is None:
        f_decades = [20, 200, 2000, 20000]

    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Build instantaneous frequency in log space, then exponentiate
    log_decades = np.log(np.array(f_decades, dtype=np.float64))
    n_decades = len(f_decades) - 1
    total_bounces = n_decades * bounces_per_decade

    # Build waypoints: one (peak, dip) pair per bounce, plus start and end anchors.
    # Time is split evenly across all bounces; each bounce occupies one time slot.
    # Within each slot: peak at 70% through, dip at end (=start of next slot).
    # To avoid duplicate times at slot boundaries, the dip lands at 97% of the slot
    # and the next peak starts at 3% into the next slot. This ensures strictly
    # increasing times while preserving the shape.
    time_per_bounce = duration_s / total_bounces

    waypoint_times = [0.0]
    waypoint_log_freqs = [log_decades[0]]

    for bounce_idx in range(total_bounces):
        d = bounce_idx // bounces_per_decade       # which decade
        b = bounce_idx % bounces_per_decade        # which bounce within decade

        log_floor = log_decades[d]
        log_ceil = log_decades[d + 1]
        log_range = log_ceil - log_floor

        # Overall ascent progress for this bounce (0 → 1 across the whole signal)
        overall_progress_at_peak = (bounce_idx + 0.70) / total_bounces
        overall_progress_at_dip = (bounce_idx + 0.97) / total_bounces

        t_peak = overall_progress_at_peak * duration_s
        t_dip = overall_progress_at_dip * duration_s

        # Peak: proportional progress along the full log-frequency range
        log_peak = log_decades[0] + (log_decades[-1] - log_decades[0]) * overall_progress_at_peak

        # Dip: pull back ~35% from peak toward the overall floor at this time point
        log_at_dip_baseline = log_decades[0] + (log_decades[-1] - log_decades[0]) * (bounce_idx / total_bounces)
        log_dip = log_peak - 0.35 * (log_peak - log_at_dip_baseline)

        waypoint_times.extend([t_peak, t_dip])
        waypoint_log_freqs.extend([log_peak, log_dip])

    # Final anchor at the top frequency
    waypoint_times.append(duration_s)
    waypoint_log_freqs.append(log_decades[-1])

    waypoint_times = np.array(waypoint_times, dtype=np.float64)
    waypoint_log_freqs = np.array(waypoint_log_freqs, dtype=np.float64)

    # Smooth interpolation via cubic spline
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(waypoint_times, waypoint_log_freqs)
    log_inst_freq = cs(t)

    # Hard clamp to stay within decade boundaries
    log_inst_freq = np.clip(log_inst_freq, log_decades[0], log_decades[-1])
    inst_freq = np.exp(log_inst_freq)

    # Truncate once inst_freq first reaches the ceiling so the signal
    # ends naturally instead of lingering at 20 kHz.
    ceiling = f_decades[-1]
    ceiling_hits = np.where(inst_freq >= ceiling * 0.999)[0]
    if len(ceiling_hits) > 0:
        cut = ceiling_hits[0]
        inst_freq = inst_freq[:cut]
        t = t[:cut]

    # Integrate instantaneous frequency to get phase
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(phase)

    return signal.astype(np.float64), inst_freq, t


def build_waypoint_chirp(sr: int, duration_s: float,
                         waypoints: tuple[tuple[float, float], ...],
                         *, bc_type: str = "natural",
                         interp: str = "cubic",
                         clip_to_waypoints: bool = True) -> tuple:
    """Generate a chirp by splining log-frequency through user waypoints.

    Args:
        sr: Sample rate in Hz
        duration_s: Signal duration in seconds
        waypoints: Sequence of (time_frac, freq_hz). Time fractions are in
                   [0, 1] of duration_s, must be strictly increasing, and must
                   include both endpoints (0 and 1).
        bc_type: scipy.interpolate.CubicSpline boundary condition (only used when
                 interp="cubic"). "natural" (zero second derivative at endpoints)
                 gives a soft start/end; "clamped" (zero first derivative) gives
                 flat takeoff/landing.
        interp: interpolation kernel for the log-frequency curve.
                "cubic"  — CubicSpline: C^2-smooth, cleanest for gentle shapes,
                           but OVERSHOOTS badly on steep/aggressive walls.
                "pchip"  — PchipInterpolator: shape-preserving (monotone between
                           waypoints, NO overshoot) — use for aggressive dips/ramps
                           and flat plateaus that must not sag.
                "akima"  — Akima1DInterpolator: low-overshoot, slightly smoother
                           than pchip but can still wiggle a little.

    Returns:
        (signal, inst_freq, t) — all 1-D arrays of length int(sr * duration_s)
    """
    if len(waypoints) < 2:
        raise ValueError(f"Need at least 2 waypoints, got {len(waypoints)}")

    wp = np.asarray(waypoints, dtype=np.float64)
    wp_times = wp[:, 0] * duration_s
    wp_log_freqs = np.log(wp[:, 1])

    if not np.isclose(wp[0, 0], 0.0) or not np.isclose(wp[-1, 0], 1.0):
        raise ValueError(
            f"Waypoints must span [0, 1] in time_frac; "
            f"got [{wp[0, 0]}, {wp[-1, 0]}]"
        )
    if not np.all(np.diff(wp_times) > 0):
        raise ValueError("Waypoint time fractions must be strictly increasing")

    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # CubicSpline (C^2 continuous, smooth second derivative across waypoints)
    # gives the cleanest bounded humps for gently-paced chirps. Akima and
    # Pchip both produce visible "kinks" at waypoints (discontinuous second
    # derivative) — fine for protecting against ringing on aggressive slope
    # flips, but visibly angular on smoother shapes. CubicSpline's only
    # weakness is overshoot when slopes flip sharply between waypoints; for
    # aggressive walls (steep dips/ramps) that overshoot explodes, so pick
    # interp="pchip" (shape-preserving, no overshoot) there instead.
    if interp == "pchip":
        from scipy.interpolate import PchipInterpolator
        spline = PchipInterpolator(wp_times, wp_log_freqs)
    elif interp == "akima":
        from scipy.interpolate import Akima1DInterpolator
        spline = Akima1DInterpolator(wp_times, wp_log_freqs)
    elif interp == "cubic":
        from scipy.interpolate import CubicSpline
        spline = CubicSpline(wp_times, wp_log_freqs, bc_type=bc_type)
    else:
        raise ValueError(f"interp must be 'cubic', 'pchip', or 'akima', got {interp!r}")
    log_inst_freq = spline(t)

    if clip_to_waypoints:
        log_inst_freq = np.clip(log_inst_freq, wp_log_freqs.min(), wp_log_freqs.max())
    inst_freq = np.exp(log_inst_freq)

    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(phase)

    return signal.astype(np.float64), inst_freq, t


def build_log_sweep_oscillating(sr: int, duration_s: float,
                                f_start: float, f_end: float, *,
                                osc_octaves: float = 0.33,
                                n_osc: float = 2.5,
                                osc_phase: float = 0.0,
                                osc_decay: float = 0.0,
                                ramp_power: float = 1.0) -> tuple:
    """Log-frequency sweep with a gentle sine wobble in log-freq.

    The base contour rises in log-frequency from f_start to f_end, monotone
    and therefore asymmetric (no dip-and-return arc to fan symmetrically).
    With ramp_power=1 the rise is LINEAR in log-freq (a classic exponential
    chirp, equal time per octave); ramp_power>1 warps the time curve so the
    contour DWELLS near f_start for longer and then ramps up steeply toward
    the end ("stay low for longer, rise late"). A slow sine ripples the
    contour by +/- osc_octaves so the time-frequency ridge undulates instead
    of tracing one clean line. The wobble breaks the low-frequency CWT smear
    into a gently textured ribbon rather than a single obvious symmetric fan —
    the smear reads as intentional motion, not as an artifact.

    Args:
        sr: Sample rate in Hz
        duration_s: Signal duration in seconds
        f_start: Frequency at t=0 (Hz) — set low for slow, readable cycles
        f_end: Frequency at t=duration_s (Hz)
        osc_octaves: Wobble amplitude in octaves (peak deviation each way).
                     ~0.3 is a gentle undulation; >0.5 starts to look wavy.
        n_osc: Number of full wobble cycles across the build duration.
        osc_phase: Phase offset (radians) of the wobble — shifts where the
                   undulations land relative to the trim window.
        osc_decay: Exponential decay rate of the wobble amplitude over the
                   sweep (0 = constant wobble; >0 fades the wobble as the
                   sweep rises, so it textures the low end but not the high).
        ramp_power: Time-curve exponent for the base rise. 1.0 = linear in
                    log-freq (constant rate). >1 dwells low and rises late
                    (e.g. 2.0-2.5 holds near f_start for the first ~half before
                    climbing). <1 rises fast then flattens high.

    Returns:
        (signal, inst_freq, t) — all 1-D arrays of length int(sr * duration_s)
    """
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    frac = t / duration_s

    ramp = frac ** ramp_power
    base = np.log(f_start) + (np.log(f_end) - np.log(f_start)) * ramp
    amp = osc_octaves * np.log(2.0)
    if osc_decay:
        amp = amp * np.exp(-osc_decay * frac)
    wobble = amp * np.sin(2 * np.pi * n_osc * frac + osc_phase)
    inst_freq = np.exp(base + wobble)

    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(phase)

    return signal.astype(np.float64), inst_freq, t


def build_click_plus_tone(
    sr: int,
    duration_s: float,
    tone_hz: float = 60.0,
    click_t: float = 0.7,
    click_center_hz: float | None = None,
    click_duration_s: float = 0.008,
    click_amp: float = 0.7,
    tone_amp: float = 1.0,
    click_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sustained sine tone with a single Gaussian-windowed noise click.

    When ``click_center_hz`` is None (default), the click is a true broadband
    Gaussian-windowed white-noise impulse — energy at every frequency
    simultaneously, producing a vertical streak across the full spectrogram.
    When ``click_center_hz`` is a number, the noise is rectangular-bandpassed
    to a 2 kHz-wide band around that centre (the "spectrally-centred burst"
    variant — more like a tone burst than a click).

    The instantaneous frequency curve is constant at ``tone_hz`` (the click is
    broadband and has no single inst-freq). Render-time callers decide whether
    to mask the click region.

    Args:
        sr: Sample rate in Hz.
        duration_s: Total signal duration in seconds.
        tone_hz: Sustained tone frequency in Hz.
        click_t: Time of click centre in seconds.
        click_center_hz: When None, no bandpass — true broadband click. When
            numeric, centre frequency of a 2 kHz-wide rectangular bandpass.
        click_duration_s: Gaussian FWHM of the click envelope in seconds.
        click_amp: Peak absolute amplitude of the click burst.
        tone_amp: Amplitude of the sustained sine tone.
        click_seed: RNG seed for reproducibility.

    Returns:
        (signal, inst_freq, t) — all 1-D float64 arrays of length
        ``int(sr * duration_s)``.
    """
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    tone = tone_amp * np.sin(2.0 * np.pi * tone_hz * t)

    rng = np.random.default_rng(click_seed)
    noise = rng.standard_normal(n).astype(np.float64)

    if click_center_hz is not None:
        noise_fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        band_lo = click_center_hz - 1000.0
        band_hi = click_center_hz + 1000.0
        mask = (freqs >= band_lo) & (freqs <= band_hi)
        noise_fft[~mask] = 0.0
        noise = np.fft.irfft(noise_fft, n=n)

    # Gaussian window centred at click_t with FWHM = click_duration_s.
    sigma = click_duration_s / 2.3548200450309493  # FWHM → sigma
    envelope = np.exp(-((t - click_t) ** 2) / (2.0 * sigma ** 2))
    click_burst = noise * envelope

    peak = np.max(np.abs(click_burst))
    if peak > 0.0:
        click_burst = click_burst * (click_amp / peak)

    signal = tone + click_burst
    inst_freq = np.full(n, tone_hz, dtype=np.float64)

    return signal.astype(np.float64), inst_freq, t


def build_low_vibrato(
    sr: int,
    duration_s: float,
    carrier_hz: float = 60.0,
    depth_hz: float = 20.0,
    mod_hz: float = 8.0,
    amp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60 Hz carrier with slow sinusoidal vibrato modulation.

    Instantaneous frequency oscillates as::

        f(t) = carrier_hz + depth_hz * sin(2π · mod_hz · t)

    Phase is computed by integrating the inst-freq curve exactly (cumsum),
    matching the ``build_waypoint_chirp`` convention.

    Args:
        sr: Sample rate in Hz.
        duration_s: Signal duration in seconds.
        carrier_hz: Centre frequency in Hz.
        depth_hz: Peak frequency deviation in Hz.
        mod_hz: Vibrato modulation rate in Hz.
        amp: Signal amplitude.

    Returns:
        (signal, inst_freq, t) — all 1-D float64 arrays of length
        ``int(sr * duration_s)``.
    """
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    inst_freq = (carrier_hz + depth_hz * np.sin(2.0 * np.pi * mod_hz * t)).astype(np.float64)
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
    signal = amp * np.sin(phase)

    return signal.astype(np.float64), inst_freq, t


def build_bouncing_chirp_chunks(sr, chunk_size, overlap_factor, n_frames,
                                f_decades=None, bounces_per_decade=3, seed=None):
    """Generate a bouncing chirp pre-sliced into overlapping chunks.

    Returns:
        (chunks, signal, inst_freq, t) — chunks is a list of arrays,
        the rest are the full signal/freq/time for reference plotting.
    """
    hop_size = int(chunk_size * (1 - overlap_factor))
    total_samples = hop_size * n_frames + chunk_size
    # Overshoot duration to compensate for ceiling truncation in build_bouncing_chirp
    duration_s = total_samples / sr * 1.15

    signal, inst_freq, t = build_bouncing_chirp(
        sr, duration_s,
        f_decades=f_decades,
        bounces_per_decade=bounces_per_decade,
        seed=seed,
    )

    # Only yield as many full chunks as the (possibly truncated) signal supports
    max_chunks = max(0, (len(signal) - chunk_size) // hop_size + 1)
    actual_n_frames = min(n_frames, max_chunks)
    chunks = [signal[i * hop_size: i * hop_size + chunk_size] for i in range(actual_n_frames)]
    return chunks, signal, inst_freq, t


def build_chirp_chunks(f0, f1, sr, chunk_size, overlap_factor, n_frames):
    """
    Generate synthetic chirp audio pre-sliced into overlapping chunks.

    Creates a linear chirp from f0 to f1 with the given sample rate, then slices
    it into overlapping chunks matching the audio processing parameters.

    Args:
        f0: Start frequency in Hz
        f1: End frequency in Hz
        sr: Sample rate in Hz
        chunk_size: Samples per chunk
        overlap_factor: Overlap factor (0 < overlap_factor < 1)
        n_frames: Number of chunks to generate

    Returns:
        List of n_frames numpy arrays, each of length chunk_size (float64)
    """
    hop_size = int(chunk_size * (1 - overlap_factor))
    total_samples = hop_size * n_frames + chunk_size

    t = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
    signal = scipy_chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear").astype(np.float64)

    chunks = [signal[i * hop_size : i * hop_size + chunk_size] for i in range(n_frames)]
    return chunks
