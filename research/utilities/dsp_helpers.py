"""DSP helper functions for the benchmark suite."""

import numpy as np
from scipy.signal import stft as scipy_stft, resample as scipy_resample, chirp as scipy_chirp

from . import constants


def compute_stft_frame(chunk, sr, nperseg, freq_mask, cropped_freqs, cwt_freqs, target_w):
    """
    Run STFT on one audio chunk, resample to log-freq bins matching cwt_freqs.

    Performs:
      1. STFT with scipy.signal.stft
      2. Magnitude extraction and frequency masking
      3. Time-domain resampling to target_w bins
      4. Log-frequency interpolation to cwt_freqs

    Args:
        chunk: Input audio chunk (1D array)
        sr: Sample rate in Hz
        nperseg: STFT window size
        freq_mask: Boolean mask for frequency filtering
        cropped_freqs: STFT frequencies after masking
        cwt_freqs: Target frequency bins (log-spaced CWT frequencies)
        target_w: Target number of time bins

    Returns:
        (n_cwt_freqs, target_w) float32 array with log-interpolated STFT magnitudes
    """
    # Compute STFT and extract magnitude
    _, _, Zxx = scipy_stft(chunk, fs=sr, nperseg=nperseg)
    stft_mag = np.abs(Zxx)[freq_mask, :]

    # Resample to target width (number of time bins)
    stft_resampled = scipy_resample(stft_mag, target_w, axis=1)
    stft_resampled = np.clip(stft_resampled, 0, None)

    # Interpolate from STFT frequencies to log-spaced CWT frequencies
    n_cwt_freqs = len(cwt_freqs)
    stft_log = np.zeros((n_cwt_freqs, target_w))
    for col in range(target_w):
        stft_log[:, col] = np.interp(
            cwt_freqs, cropped_freqs, stft_resampled[:, col], left=0.0, right=0.0
        )

    return stft_log.astype(np.float32)


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

    # Integrate instantaneous frequency to get phase
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    signal = np.sin(phase)

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
    duration_s = total_samples / sr

    signal, inst_freq, t = build_bouncing_chirp(
        sr, duration_s,
        f_decades=f_decades,
        bounces_per_decade=bounces_per_decade,
        seed=seed,
    )

    chunks = [signal[i * hop_size: i * hop_size + chunk_size] for i in range(n_frames)]
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
