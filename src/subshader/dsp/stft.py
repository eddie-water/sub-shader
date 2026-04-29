"""
STFT backend for SubShader.

STFT wraps scipy.signal.stft with log-frequency interpolation to match the
chromatic frequency bins used by CWT backends. Used for comparison and
benchmarking only, not production pipeline use. pre/post are stubs (D-14).
"""

from __future__ import annotations

import numpy as np

from subshader.config import CWTConfig
from subshader.dsp.dsp import DSP
from subshader.utils.logging import get_logger
from subshader.utils.timing import timed

log = get_logger(__name__)


class STFT(DSP):
    """Short-Time Fourier Transform backend — used for comparison only.

    Computes STFT via scipy.signal.stft, resamples the time axis to
    target_width, then log-frequency-interpolates frequency bins to match
    the chromatic CWT frequency grid. Output shape matches CWT backends.

    pre() and post() are stubs per D-14. The transform() method performs the
    full STFT + resample + interpolation pipeline.
    """

    def __init__(self, config: CWTConfig) -> None:
        super().__init__(config)

        self.sample_rate: float = float(config.sample_rate)
        self.nyquist_freq: float = self.sample_rate / 2.0
        self.target_width: int = config.target_width

        # Build chromatic frequency scale (mirrors CWT frequency grid)
        scale_factor = 2 ** (1 / config.notes_per_octave)
        i = np.arange(0, config.notes_per_octave * config.num_octaves, dtype=np.float64)
        freqs = np.float64(config.root_note_hz) * (scale_factor ** i)
        self.freqs: np.ndarray = freqs[freqs < self.nyquist_freq].astype(np.float64)

        # STFT window size — use chunk_size as nperseg for maximum frequency resolution
        self.nperseg: int = int(config.chunk_size)

        # Pre-compute the STFT frequency axis and frequency mask covering the CWT range
        from scipy.signal import stft as scipy_stft

        # Frequency axis for a full STFT at this nperseg
        stft_freqs = np.fft.rfftfreq(self.nperseg, d=1.0 / self.sample_rate)
        f_lo = self.freqs[0] * 0.9
        f_hi = self.freqs[-1] * 1.1
        self.freq_mask: np.ndarray = (stft_freqs >= f_lo) & (stft_freqs <= f_hi)
        self.cropped_freqs: np.ndarray = stft_freqs[self.freq_mask]

    def pre(self, chunk: np.ndarray) -> np.ndarray:
        """Pass-through stub (D-14 — no pre-processing defined for STFT backend)."""
        return chunk

    @timed
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Compute STFT with log-frequency resampling to match CWT output shape.

        Steps:
            1. scipy.signal.stft with nperseg=chunk_size
            2. Magnitude extraction and frequency masking
            3. Time-axis resampling to target_width bins
            4. Log-frequency interpolation from STFT bins to CWT chromatic bins

        Args:
            data: 1D audio samples (float64).

        Returns:
            float32 array of shape (num_cwt_freqs, target_width).
        """
        from scipy.signal import stft as scipy_stft, resample as scipy_resample

        _, _, Zxx = scipy_stft(data, fs=self.sample_rate, nperseg=self.nperseg)
        stft_mag = np.abs(Zxx)[self.freq_mask, :]

        # Resample time axis to target_width
        stft_resampled = scipy_resample(stft_mag, self.target_width, axis=1)
        stft_resampled = np.clip(stft_resampled, 0, None)

        # Interpolate from STFT linear frequency bins to log-spaced CWT bins
        n_cwt_freqs = len(self.freqs)
        stft_log = np.zeros((n_cwt_freqs, self.target_width))
        for col in range(self.target_width):
            stft_log[:, col] = np.interp(
                self.freqs, self.cropped_freqs, stft_resampled[:, col],
                left=0.0, right=0.0,
            )

        return stft_log.astype(np.float32)

    def post(self, raw: np.ndarray) -> np.ndarray:
        """Pass-through stub (D-14 — no post-processing defined for STFT backend)."""
        return raw
