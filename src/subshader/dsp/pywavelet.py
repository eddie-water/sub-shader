"""
PyWavelets CWT backend for SubShader.

PywaveletCWT wraps the pywt.cwt() library function, used for comparison and
validation against SubShader's hand-rolled ANTS implementation. Not intended
for production pipeline use — pre/post are stubs (D-14).
"""

from __future__ import annotations

import numpy as np
import pywt

from subshader.config import CWTConfig
from subshader.dsp.dsp import DSP
from subshader.utils.logging import get_logger
from subshader.utils.timing import timed

log = get_logger(__name__)


class PywaveletCWT(DSP):
    """PyWavelets library CWT backend — used for comparison and testing only.

    Wraps pywt.cwt() with a complex Morlet wavelet, using scale arrays derived
    from the same chromatic frequency list as the ANTS CWT backends. Scale-
    dependent normalization divides by sqrt(scale) to match the energy response
    of L1-normalized ANTS kernels.

    pre() and post() are stubs per D-14. Callers use process() or transform()
    directly for comparison harness usage.
    """

    def __init__(self, config: CWTConfig) -> None:
        super().__init__(config)

        self.sample_rate: np.float64 = np.float64(config.sample_rate)
        self.nyquist_freq: np.float64 = self.sample_rate / 2.0
        self.sampling_period: np.float64 = 1.0 / self.sample_rate

        # Complex Morlet wavelet (cmor bandwidth-center)
        self.wavelet_name: str = "cmor1.5-1.0"

        # Build chromatic frequency scale
        scale_factor = 2 ** (1 / config.notes_per_octave)
        i = np.arange(0, config.notes_per_octave * config.num_octaves, dtype=np.float64)
        freqs = np.float64(config.root_note_hz) * (scale_factor ** i)
        self.freqs: np.ndarray = freqs[freqs < self.nyquist_freq].astype(np.float64)

        # Convert frequencies to pywt scales
        f_norm: np.ndarray = self.freqs / self.sample_rate
        self.scales: np.ndarray = pywt.frequency2scale(self.wavelet_name, f_norm)

    def pre(self, chunk: np.ndarray) -> np.ndarray:
        """Pass-through stub (D-14 — no pre-processing defined for PyWavelets backend)."""
        return chunk

    @timed
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Compute CWT using pywt.cwt() with scale-dependent normalization.

        Args:
            data: 1D float64 audio samples.

        Returns:
            Scale-normalized complex CWT coefficients (num_freqs, input_n).
        """
        coefs_raw, _ = pywt.cwt(
            data=data,
            scales=self.scales,
            wavelet=self.wavelet_name,
            sampling_period=self.sampling_period,
        )

        # Divide by sqrt(scale) to compensate for scale-dependent energy accumulation.
        # pywt.cwt() returns raw convolution output; longer wavelets at low frequencies
        # accumulate proportionally more energy. sqrt(scale) normalization makes
        # equal-amplitude tones appear equally bright across all frequencies.
        return coefs_raw / np.sqrt(self.scales)[:, np.newaxis]

    def post(self, raw: np.ndarray) -> np.ndarray:
        """Pass-through stub (D-14 — no post-processing defined for PyWavelets backend)."""
        return raw
