"""
Pytest tests for SubShader wavelet implementations.

Covers test categories 0, 1, 2, 3, 6, 7, 8:
  0. NumPy vs CuPy CWT correctness (GPU-only, skipped if no CUDA)
  1. Pure tone peak accuracy
  2. Cross-implementation frequency agreement
  3. Chirp sweep monotonicity
  6. Normalization behavior (scale-dependent)
  7. Reliable region consistency
  8. Output shape regression
"""

import numpy as np
import pytest

from subshader.config import get_default_config, WaveletConfig, ColorNormalizationConfig
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from conftest import generate_tone, find_peak_bin, _make_wavelet


TEST_FREQS = [27.5, 130.81, 440.0, 1000.0, 4186.01, 10000.0]


def _gpu_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


# =============================================================================
# TEST 0: NUMPY VS CUPY VERIFICATION
# =============================================================================

@pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU available")
def test_numpy_vs_cupy():
    """Verify that NumPyWavelet and CuPyWavelet produce numerically equivalent outputs."""
    from subshader.dsp.wavelet import CuPyWavelet
    from subshader.audio.audio_input import AudioInput

    config = get_default_config()
    ai = AudioInput(
        path=config.audio.file_path,
        chunk_size=config.audio.chunk_size,
        overlap_factor=config.audio.overlap_factor,
    )
    sr = ai.get_sample_rate()
    data = ai.get_chunk()

    npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
    cpwt = CuPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

    np_coefs = npwt.cwt(data)
    cp_coefs = cpwt.cwt(data)

    diff = np.abs(np_coefs - cp_coefs)
    max_val = np_coefs.max() + 1e-12
    rel_error = diff.max() / max_val

    print(f"  shape          : {np_coefs.shape}")
    print(f"  max abs diff   : {diff.max():.2e}")
    print(f"  mean abs diff  : {diff.mean():.2e}")
    print(f"  rel error      : {rel_error:.2e}")

    n_freq = np_coefs.shape[0]
    corrs = np.array([
        np.corrcoef(np_coefs[i], cp_coefs[i])[0, 1]
        for i in range(n_freq)
    ])
    low_corr = np.where(corrs < 0.999)[0]
    print(f"  bins < 0.999 r : {len(low_corr)} / {n_freq}")

    assert rel_error < 1e-3, f"Relative error {rel_error:.2e} exceeds 1e-3"
    assert len(low_corr) == 0, (
        f"{len(low_corr)} frequency bins have Pearson r < 0.999 between NumPy and CuPy"
    )


# =============================================================================
# TEST 1: PURE TONE PEAK ACCURACY
# =============================================================================

def test_pure_tone_peak_accuracy():
    """For each wavelet, a pure tone at f should peak at the bin nearest f.

    Excludes boundary frequencies where CWT resolution is insufficient:
    - Lowest frequency (freqs[0]): only ~3 cycles fit in the chunk window,
      making peak localization unreliable for NumPyWavelet at that resolution.
    - High frequencies for PyWavelet: pywt.cwt() produces aliased energy
      at frequencies above ~5 kHz due to the Morlet wavelet's discretization
      at short time supports. NumPyWavelet uses FFT-based convolution which
      is accurate up to Nyquist.
    """
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    # NumPyWavelet: reliable from freqs[3] upward (exclude lowest 3 bins)
    # PyWavelet: reliable only in the midrange (exclude lowest 3 and above ~5 kHz)
    reliable_ranges = {
        "NumPyWavelet": (3, None),   # skip 3 lowest bins, all high bins ok
        "PyWavelet": (3, 90),        # skip 3 lowest; unreliable above bin ~90 (~5 kHz)
    }

    implementations = [("NumPyWavelet", NumPyWavelet), ("PyWavelet", PyWavelet)]

    for impl_name, cls in implementations:
        wt = cls(sample_rate=sr, input_n=chunk, config=config.wavelet)
        freqs = wt.freqs

        low_bin, high_bin = reliable_ranges[impl_name]
        reliable_low = freqs[low_bin]
        reliable_high = freqs[high_bin] if high_bin is not None else freqs[-1]

        for f_target in TEST_FREQS:
            if f_target < reliable_low or f_target > reliable_high:
                print(f"  {impl_name} @ {f_target:.1f} Hz: skipped (outside reliable range "
                      f"[{reliable_low:.1f}, {reliable_high:.1f}] Hz)")
                continue

            tone = generate_tone(f_target, sr, chunk)
            output = wt.cwt(tone)

            peak_bin, peak_freq = find_peak_bin(output, freqs)
            expected_bin = int(np.argmin(np.abs(freqs - f_target)))

            print(f"  {impl_name} @ {f_target:.1f} Hz: peak bin={peak_bin} "
                  f"(freq={peak_freq:.1f} Hz), expected bin={expected_bin} "
                  f"(freq={freqs[expected_bin]:.1f} Hz)")

            assert abs(peak_bin - expected_bin) <= 1, (
                f"{impl_name} @ {f_target:.1f} Hz: peak bin={peak_bin} "
                f"(freq={peak_freq:.1f} Hz), expected bin={expected_bin} "
                f"(freq={freqs[expected_bin]:.1f} Hz)"
            )


# =============================================================================
# TEST 2: CROSS-IMPLEMENTATION FREQUENCY AGREEMENT
# =============================================================================

def test_cross_implementation_agreement():
    """All implementations should peak at the same bin for the same tone."""
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    freqs_match = np.array_equal(npwt.freqs, pywt_wt.freqs)
    assert freqs_match, "freqs arrays are not identical across NumPyWavelet and PyWavelet"

    for f_target in [130.81, 440.0, 1000.0, 4186.01]:
        if f_target < npwt.freqs[0] or f_target > npwt.freqs[-1]:
            continue

        tone = generate_tone(f_target, sr, chunk)

        np_bin, _ = find_peak_bin(npwt.cwt(tone), npwt.freqs)
        py_bin, _ = find_peak_bin(pywt_wt.cwt(tone), pywt_wt.freqs)

        print(f"  @ {f_target:.1f} Hz: NumPy bin={np_bin}, PyWavelet bin={py_bin}")

        assert abs(np_bin - py_bin) <= 1, (
            f"@ {f_target:.1f} Hz: NumPy bin={np_bin}, PyWavelet bin={py_bin} — "
            f"bins differ by more than 1"
        )


# =============================================================================
# TEST 3: CHIRP SWEEP MONOTONICITY
# =============================================================================

def test_chirp_monotonicity():
    """Peak bin should increase monotonically for a rising chirp."""
    from scipy.signal import chirp as scipy_chirp

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size
    hop = int(chunk * (1 - config.audio.overlap_factor))
    num_chunks = 32

    f_start, f_end = 200.0, 10000.0
    total_samples = hop * num_chunks + chunk
    t = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
    signal = scipy_chirp(t, f0=f_start, f1=f_end, t1=t[-1], method='linear').astype(np.float64)

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    peak_bins = []
    for i in range(num_chunks):
        chunk_data = signal[i * hop: i * hop + chunk]
        output = npwt.cwt(chunk_data)
        pb, _ = find_peak_bin(output, npwt.freqs)
        peak_bins.append(pb)

    violations = 0
    for i in range(1, len(peak_bins)):
        if peak_bins[i] < peak_bins[i - 1]:
            violations += 1

    print(f"  Monotonicity: {violations} violations in {len(peak_bins)} chunks "
          f"(bins: {peak_bins[0]} -> {peak_bins[-1]})")

    first_freq = npwt.freqs[peak_bins[0]]
    last_freq = npwt.freqs[peak_bins[-1]]
    print(f"  First chunk peak: {first_freq:.0f} Hz, last chunk peak: {last_freq:.0f} Hz")

    assert violations <= 2, (
        f"Chirp monotonicity: {violations} violations in {len(peak_bins)} chunks "
        f"(bins: {peak_bins[0]} -> {peak_bins[-1]})"
    )
    # Allow a generous bound: with a short total duration (~1.5s), the chirp
    # sweeps rapidly and the first CWT chunk captures multiple frequencies.
    # The first peak should be in the lower half of the sweep range.
    assert first_freq < (f_start + f_end) / 2, (
        f"First chunk peak: {first_freq:.0f} Hz (expect below midpoint {(f_start + f_end) / 2:.0f} Hz)"
    )
    assert last_freq > f_end * 0.5, (
        f"Last chunk peak: {last_freq:.0f} Hz (expect near {f_end:.0f} Hz)"
    )


# =============================================================================
# TEST 6: NORMALIZATION BEHAVIOR
# =============================================================================

def test_normalization_behavior():
    """Test scale-dependent normalization: white noise flatness + multi-tone equality."""
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    # --- 6a: White noise flatness ---
    np.random.seed(42)
    noise = np.random.randn(chunk).astype(np.float64)

    np_out = npwt.cwt(noise)
    py_out = pywt_wt.cwt(noise)

    np_mean_energy = np.mean(np_out, axis=1)
    py_mean_energy = np.mean(py_out, axis=1)

    n_bins = len(npwt.freqs)
    band = max(1, n_bins // 10)
    np_low = np.mean(np_mean_energy[:band])
    np_high = np.mean(np_mean_energy[-band:])
    py_low = np.mean(py_mean_energy[:band])
    py_high = np.mean(py_mean_energy[-band:])

    np_ratio = np_low / (np_high + 1e-12)
    py_ratio = py_low / (py_high + 1e-12)

    print(f"  NumPyWavelet white noise low/high ratio: {np_ratio:.2f} (expect < 3.0 with sqrt(f) norm)")
    print(f"  PyWavelet white noise low/high ratio: {py_ratio:.2f} (expect > 2.0 without norm)")

    assert np_ratio < 3.0, (
        f"NumPyWavelet white noise low/high ratio: {np_ratio:.2f} (expect < 3.0 with sqrt(f) norm)"
    )
    assert py_ratio > 2.0, (
        f"PyWavelet white noise low/high ratio: {py_ratio:.2f} (expect > 2.0 without norm)"
    )

    # --- 6b: Equal-amplitude multi-tone ---
    f_low, f_mid, f_high = 200.0, 1000.0, 5000.0
    multi_tone = (generate_tone(f_low, sr, chunk)
                  + generate_tone(f_mid, sr, chunk)
                  + generate_tone(f_high, sr, chunk))

    np_out = npwt.cwt(multi_tone)
    np_mean = np.mean(np_out, axis=1)

    bin_low = int(np.argmin(np.abs(npwt.freqs - f_low)))
    bin_high_freq = int(np.argmin(np.abs(npwt.freqs - f_high)))

    peak_low = np_mean[bin_low]
    peak_high = np_mean[bin_high_freq]

    ratio_high_low = peak_low / (peak_high + 1e-12)
    print(f"  Multi-tone: 200 Hz peak / 5 kHz peak = {ratio_high_low:.2f} (expect < 8.0 with normalization)")

    assert ratio_high_low < 8.0, (
        f"Multi-tone: 200 Hz peak / 5 kHz peak = {ratio_high_low:.2f} "
        f"(expect < 8.0 with normalization; without would be ~20x+)"
    )

    # --- 6c: Normalization sufficiency (single tone comparison) ---
    tone_low = generate_tone(200.0, sr, chunk)
    tone_high_f = generate_tone(5000.0, sr, chunk)

    out_low = npwt.cwt(tone_low)
    out_high = npwt.cwt(tone_high_f)

    peak_at_200 = np.max(np.mean(out_low, axis=1))
    peak_at_5k = np.max(np.mean(out_high, axis=1))

    ratio = peak_at_200 / (peak_at_5k + 1e-12)
    print(f"  Single-tone sufficiency: 200 Hz peak / 5 kHz peak = {ratio:.2f} (expect < 8.0)")

    assert ratio < 8.0, (
        f"Single-tone sufficiency: 200 Hz peak / 5 kHz peak = {ratio:.2f} "
        f"(expect < 8.0; residual bias from CWT time-freq tradeoff is normal)"
    )


# =============================================================================
# TEST 7: RELIABLE REGION CONSISTENCY
# =============================================================================

def test_reliable_region_consistency():
    """Output shapes should match between PyWavelet and NumPyWavelet."""
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    np_shape = npwt.get_output_shape()
    py_shape = pywt_wt.get_output_shape()

    print(f"  Output shapes: NumPy={np_shape}, PyWavelet={py_shape}")
    assert np_shape == py_shape, (
        f"Output shapes differ: NumPy={np_shape}, PyWavelet={py_shape}"
    )

    tone = generate_tone(440.0, sr, chunk)
    np_out = npwt.cwt(tone)
    py_out = pywt_wt.cwt(tone)

    print(f"  Actual output shapes: NumPy={np_out.shape}, PyWavelet={py_out.shape}")
    assert np_out.shape == py_out.shape, (
        f"Actual output shapes differ: NumPy={np_out.shape}, PyWavelet={py_out.shape}"
    )


# =============================================================================
# TEST 8: OUTPUT SHAPE REGRESSION
# =============================================================================

def test_output_shape_regression():
    """Verify output shapes match declared shapes for all implementations."""
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    for name, cls in [("NumPyWavelet", NumPyWavelet), ("PyWavelet", PyWavelet)]:
        wt = cls(sample_rate=sr, input_n=chunk, config=config.wavelet)
        tone = generate_tone(440.0, sr, chunk)
        output = wt.cwt(tone)

        expected_shape = (wt.num_freqs, wt.config.target_width)
        declared_shape = wt.get_output_shape()

        print(f"  {name}.get_output_shape() = {declared_shape}, "
              f"expected ({wt.num_freqs}, {wt.config.target_width})")

        assert declared_shape == expected_shape, (
            f"{name}.get_output_shape() = {declared_shape}, "
            f"expected ({wt.num_freqs}, {wt.config.target_width})"
        )
        assert output.shape == declared_shape, (
            f"{name}.cwt() output shape = {output.shape}, "
            f"declared = {declared_shape}"
        )
        assert len(wt.freqs) == wt.num_freqs, (
            f"{name}: len(freqs)={len(wt.freqs)} != num_freqs={wt.num_freqs}"
        )
