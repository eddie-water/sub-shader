"""
Pytest tests for SubShader wavelet kernel internals.

Covers test categories 4, 5, 9:
  4. Kernel FFT bank integrity
  5. Impulse response center frequency
  9. Wavelet kernel energy per scale
"""

import numpy as np
from numpy.fft import fft

from subshader.dsp.wavelet import NumPyWavelet
from conftest import _make_wavelet


# =============================================================================
# TEST 4: KERNEL FFT BANK INTEGRITY
# =============================================================================

def test_kernel_fft_bank_integrity():
    """Each kernel in the bank should be FFT'd at max_conv_n, not its own conv_n."""
    npwt = _make_wavelet(NumPyWavelet)

    mismatches = []

    for i, w in enumerate(npwt.wavelets):
        expected = fft(w.kernel_t, npwt.max_conv_n)
        stored = npwt.kernel_f_bank[i]

        if not np.allclose(expected, stored, atol=1e-5):
            mismatches.append(i)

    print(f"  Kernel FFT bank: {len(mismatches)} mismatches out of {npwt.num_wavelets} kernels"
          + (f" (bins: {mismatches[:5]})" if mismatches else ""))

    assert len(mismatches) == 0, (
        f"Kernel FFT bank: {len(mismatches)} mismatches out of {npwt.num_wavelets} kernels"
        + (f" (bins: {mismatches[:5]})" if mismatches else "")
    )


# =============================================================================
# TEST 5: IMPULSE RESPONSE CENTER FREQUENCY
# =============================================================================

def test_impulse_response_center_freq():
    """Each kernel's FFT magnitude should peak at its declared center frequency."""
    npwt = _make_wavelet(NumPyWavelet)
    sr = int(npwt.sample_rate)

    worst_error_hz = 0.0
    failures = []

    for i, w in enumerate(npwt.wavelets):
        kernel_f = fft(w.kernel_t, npwt.max_conv_n)
        mag = np.abs(kernel_f[:npwt.max_conv_n // 2])
        fft_freqs = np.fft.rfftfreq(npwt.max_conv_n, d=1.0 / sr)[:len(mag)]

        peak_idx = np.argmax(mag)
        peak_freq = fft_freqs[peak_idx]
        error_hz = abs(peak_freq - w.freq)
        worst_error_hz = max(worst_error_hz, error_hz)

        bin_width = sr / npwt.max_conv_n
        ok = error_hz <= bin_width * 1.5

        if not ok:
            failures.append(
                f"Kernel {i} ({w.freq:.1f} Hz): FFT peak at {peak_freq:.1f} Hz, "
                f"error={error_hz:.2f} Hz (bin width={bin_width:.2f} Hz)"
            )

    print(f"  All {npwt.num_wavelets} kernels checked: worst error={worst_error_hz:.2f} Hz")

    assert len(failures) == 0, (
        f"{len(failures)} kernels have FFT peak outside 1.5 bin widths of center freq:\n"
        + "\n".join(failures[:5])
    )


# =============================================================================
# TEST 9: WAVELET KERNEL ENERGY PER SCALE
# =============================================================================

def test_kernel_energy_per_scale():
    """
    Verify that Phase 2 L1 normalization is applied to all wavelet kernels.

    After Phase 2 (L1 kernel normalization), each kernel_t should have L1 norm ≈ 1.0
    across all frequencies. This ensures equal-amplitude tones produce comparable CWT
    output magnitudes regardless of frequency.

    Note: L2 norms scale as sqrt(f) after L1 normalization (shorter high-freq kernels
    have proportionally higher L2 per-sample energy), confirming the normalization
    geometry is correct.
    """
    npwt = _make_wavelet(NumPyWavelet)

    freqs = []
    l1_norms = []
    l2_norms = []

    for w in npwt.wavelets:
        l1 = np.sum(np.abs(w.kernel_t))
        l2 = np.sqrt(np.sum(np.abs(w.kernel_t) ** 2))
        freqs.append(w.freq)
        l1_norms.append(l1)
        l2_norms.append(l2)

    freqs = np.array(freqs)
    l1_norms = np.array(l1_norms)
    l2_norms = np.array(l2_norms)

    log_f = np.log(freqs)
    log_l2 = np.log(l2_norms)
    slope, _ = np.polyfit(log_f, log_l2, 1)

    l1_cv = np.std(l1_norms) / np.mean(l1_norms)

    print(f"    L1 norms: mean={np.mean(l1_norms):.4f}, std={np.std(l1_norms):.6f}, CV={l1_cv:.4f}")
    print(f"    L2 log-log slope vs freq: {slope:.4f} (expect ~+0.5 with L1 normalization)")

    # L1 norm should be ≈ 1.0 for all kernels after Phase 2 normalization
    assert l1_cv < 0.01, (
        f"L1 norm coefficient of variation: {l1_cv:.4f} (expect < 0.01 — all kernels should be L1-normalized)"
    )
    assert abs(np.mean(l1_norms) - 1.0) < 0.01, (
        f"Mean L1 norm: {np.mean(l1_norms):.4f} (expect ≈ 1.0 — L1 normalization applied)"
    )
    # With L1 normalization, L2 norm scales as sqrt(f): longer low-freq kernels
    # have more samples at the same per-sample magnitude → lower L2 per sample.
    assert abs(slope - 0.5) < 0.15, (
        f"L2 norm vs freq log-log slope: {slope:.3f} "
        f"(expect ~+0.5 after L1 normalization — shorter high-freq kernels have proportionally higher L2)"
    )
