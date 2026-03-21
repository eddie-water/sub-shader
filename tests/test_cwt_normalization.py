"""
Tests for CWT kernel normalization and frequency-band brightness bias.

These tests drive the fix for the root cause of disproportionately bright
low-frequency bands: wavelet kernel L1 norm scales as 1/f without normalization.
"""

import numpy as np
import pytest

from subshader.config import WaveletConfig
from subshader.dsp.wavelet_kernel import WaveletKernel
from subshader.dsp.wavelet import NpWavelet


# =============================================================================
# TESTS: Wavelet Kernel L1 Normalization
# =============================================================================

class TestWaveletKernelNormalization:
    """WaveletKernel must produce unit L1 norm kernels at any frequency."""

    def test_kernel_has_unit_l1_norm(self):
        """Kernel at 1000 Hz must have L1 norm within 1e-5 of 1.0."""
        kernel = WaveletKernel(
            f=1000.0,
            sample_rate=44100,
            num_cycles=6,
            num_fwhm_cycles=3,
            input_n=16384,
        )
        l1_norm = np.sum(np.abs(kernel.kernel_t))
        assert abs(l1_norm - 1.0) < 1e-5, (
            f"Expected L1 norm ≈ 1.0, got {l1_norm:.6f}. "
            "WaveletKernel.kernel_t must be normalized to unit L1 area."
        )

    @pytest.mark.parametrize("freq", [100.0, 1000.0, 10000.0])
    def test_kernel_l1_norm_consistent_across_frequencies(self, freq):
        """Kernel L1 norm must be within 1e-5 of 1.0 at any frequency."""
        kernel = WaveletKernel(
            f=freq,
            sample_rate=44100,
            num_cycles=6,
            num_fwhm_cycles=3,
            input_n=16384,
        )
        l1_norm = np.sum(np.abs(kernel.kernel_t))
        assert abs(l1_norm - 1.0) < 1e-5, (
            f"Kernel at {freq} Hz: L1 norm = {l1_norm:.6f}, expected ≈ 1.0. "
            "All kernels must be L1-normalized regardless of center frequency."
        )


# =============================================================================
# TESTS: CWT Brightness Bias
# =============================================================================

class TestCwtBrightnessBias:
    """Equal-amplitude tones across the frequency range must produce comparable CWT magnitudes."""

    def test_equal_amplitude_tones_produce_comparable_magnitudes(self, numpy_wavelet):
        """
        Synthetic signal with equal-amplitude sinusoids at 100, 500, 1000, 5000,
        and 10000 Hz must produce CWT peak magnitudes within 2x of each other.

        Without kernel normalization, the ratio is ~100x because low-frequency
        kernels integrate far more energy due to their wider time support.
        """
        sample_rate = 44100
        input_n = 16384
        test_freqs = [100, 500, 1000, 5000, 10000]

        t = np.arange(input_n) / sample_rate
        signal = sum(np.sin(2 * np.pi * f * t) for f in test_freqs) / len(test_freqs)
        signal = signal.astype(np.float64)

        cwt_output = numpy_wavelet.cwt(signal)

        # For each test frequency, find the closest row in the wavelet freq array
        peak_magnitudes = []
        for test_f in test_freqs:
            closest_idx = np.argmin(np.abs(numpy_wavelet.freqs - test_f))
            row_max = np.max(cwt_output[closest_idx])
            peak_magnitudes.append(row_max)

        max_mag = max(peak_magnitudes)
        min_mag = min(peak_magnitudes)

        assert min_mag > 0, "All frequency bands should have non-zero response."

        ratio = max_mag / min_mag
        assert ratio < 2.0, (
            f"Max/min magnitude ratio across frequency bands = {ratio:.2f}, expected < 2.0. "
            f"Peak magnitudes: {dict(zip(test_freqs, [f'{m:.4f}' for m in peak_magnitudes]))}. "
            "Kernel L1 normalization must equalize CWT response across frequencies."
        )


# =============================================================================
# TESTS: normalize_by_scale No-Op
# =============================================================================

class TestNormalizeByScale:
    """AntsWavelet.normalize_by_scale must be a no-op after kernel normalization handles bias."""

    def test_normalize_by_scale_is_noop(self, numpy_wavelet):
        """normalize_by_scale must return its input array unchanged."""
        rng = np.random.default_rng(42)
        num_freqs = numpy_wavelet.num_freqs
        input_n = numpy_wavelet.input_n
        coefs = rng.standard_normal((num_freqs, input_n)) + 1j * rng.standard_normal((num_freqs, input_n))
        coefs = coefs.astype(np.complex128)

        result = numpy_wavelet.normalize_by_scale(coefs)

        assert np.array_equal(result, coefs), (
            "normalize_by_scale must return its input unchanged. "
            "Energy bias is corrected at the kernel construction level."
        )
