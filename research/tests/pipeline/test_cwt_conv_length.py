"""
Tests for the CWT FFT-convolution length.

These tests drive the Fix 1 performance change (see assets/timing/LEDGER.md):
padding max_conv_n to a fast FFT size. The exact convolution length
(input_n + widest_time_support - 1) can land on a length with a large prime
factor - 26005 = 5 * 7 * 743 at default config - which forces FFT libraries
into the slow Bluestein fallback. Padding with zeros to the next fast length
is mathematically transparent: the kept [:input_n] region of the linear
convolution is unchanged.
"""

import numpy as np
import pytest

from subshader.config import CWTConfig
from subshader.dsp.cwt import CpuCWT


def small_config() -> CWTConfig:
    """Small chromatic range so kernel construction stays fast in tests."""
    return CWTConfig(
        chunk_size=2048,
        num_octaves=4,
        root_note_hz=220.0,
        target_width=32,
    )


def largest_prime_factor(n: int) -> int:
    largest, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            largest, n = d, n // d
        d += 1
    return max(largest, n) if n > 1 else largest


class TestConvolutionLength:
    """max_conv_n must be a fast FFT size without changing the math."""

    def test_conv_length_is_fast_fft_size(self):
        """max_conv_n must have no prime factor above 11 (fast FFT radices).

        scipy.fft.next_fast_len returns 11-smooth lengths; pocketfft and cuFFT
        both have fast kernels through radix 11. Large prime factors (e.g. 743
        in the unpadded default length 26005) trigger the slow Bluestein
        fallback instead.
        """
        cwt = CpuCWT(small_config())
        factor = largest_prime_factor(cwt.max_conv_n)
        assert factor <= 11, (
            f"max_conv_n = {cwt.max_conv_n} has prime factor {factor}; "
            "lengths with large prime factors force the Bluestein FFT fallback "
            "(~3x slower on GPU). Pad to scipy.fft.next_fast_len."
        )

    def test_conv_length_covers_linear_convolution(self):
        """Padding may only grow max_conv_n, never below the exact conv length."""
        cwt = CpuCWT(small_config())
        exact_n = max(w.get_conv_n() for w in cwt.wavelets)
        assert cwt.max_conv_n >= exact_n, (
            f"max_conv_n = {cwt.max_conv_n} is below the exact convolution "
            f"length {exact_n}; the FFT convolution would wrap circularly."
        )

    def test_transform_matches_direct_convolution(self):
        """FFT convolution at the padded length must equal direct linear convolution.

        This is the parity guarantee for Fix 1: zero-padding the FFT length
        changes performance, never results.
        """
        config = small_config()
        cwt = CpuCWT(config)

        rng = np.random.default_rng(seed=42)
        data = rng.standard_normal(config.chunk_size)

        result = cwt.transform(data)
        assert result.shape == (cwt.num_freqs, config.chunk_size)

        for row in (0, cwt.num_freqs // 2, cwt.num_freqs - 1):
            kernel_t = cwt.wavelets[row].kernel_t
            direct = np.convolve(data, kernel_t)[: config.chunk_size]
            np.testing.assert_allclose(
                result[row], direct, rtol=1e-3, atol=1e-6,
                err_msg=(
                    f"CWT row {row} ({cwt.freqs[row]:.1f} Hz) diverges from "
                    "direct linear convolution - the FFT length change altered "
                    "the transform output."
                ),
            )

    def test_output_shape_unchanged_by_padding(self):
        """Pipeline-facing output shape stays (num_freqs, target_width)."""
        config = small_config()
        cwt = CpuCWT(config)
        rng = np.random.default_rng(seed=7)
        processed = cwt.post(cwt.transform(rng.standard_normal(config.chunk_size)))
        assert processed.shape == (cwt.num_freqs, config.target_width)
