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
