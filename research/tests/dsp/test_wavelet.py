"""
Pytest tests for SubShader wavelet implementations.

Covers test categories 0, 1, 2, 3, 6, 7, 8:
  0. CpuCWT vs GpuCWT correctness (GPU-only, skipped if no CUDA)
  1. Pure tone peak accuracy
  2. Cross-implementation frequency agreement
  3. Chirp sweep monotonicity
  6. Normalization behavior (scale-dependent)
  7. Reliable region consistency
  8. Output shape regression
"""

import numpy as np
import pytest

from subshader.config import CWTConfig
from subshader.dsp.cwt import CpuCWT
from subshader.dsp.pywavelet import PywaveletCWT
from conftest import generate_tone, find_peak_bin, _make_wavelet, make_test_config


TEST_FREQS = [27.5, 130.81, 440.0, 1000.0, 4186.01, 10000.0]


def _gpu_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


# =============================================================================
# TEST 0: CPUCWT VS GPUCWT VERIFICATION
# =============================================================================

@pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU available")
def test_numpy_vs_cupy():
    """Verify that CpuCWT and GpuCWT produce numerically equivalent outputs."""
    from subshader.dsp.cwt import GpuCWT
    from subshader.audio.reader import AudioReader

    import os
    # Use absolute path so this test works regardless of CWD (run from project root or research/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    audio_path = os.path.join(project_root, "assets/audio/reference/beltran_sc_rip.wav")
    config = CWTConfig(file_path=audio_path)
    reader = AudioReader(config)
    data = reader.get_chunk()

    cpu_wt = CpuCWT(config)
    gpu_wt = GpuCWT(config)

    np_coefs = cpu_wt.process(data)
    cp_coefs = gpu_wt.process(data)

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
        f"{len(low_corr)} frequency bins have Pearson r < 0.999 between CpuCWT and GpuCWT"
    )


# =============================================================================
# TEST 1: PURE TONE PEAK ACCURACY
# =============================================================================

def test_pure_tone_peak_accuracy():
    """For each wavelet, a pure tone at f should peak at the bin nearest f.

    Excludes boundary frequencies where CWT resolution is insufficient:
    - Lowest frequency (freqs[0]): only ~3 cycles fit in the chunk window,
      making peak localization unreliable for CpuCWT at that resolution.
    - High frequencies for PywaveletCWT: pywt.cwt() produces aliased energy
      at frequencies above ~5 kHz due to the Morlet wavelet's discretization
      at short time supports. CpuCWT uses FFT-based convolution which
      is accurate up to Nyquist.
    """
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size

    # CpuCWT: reliable from freqs[3] upward (exclude lowest 3 bins)
    # PywaveletCWT: reliable only in the midrange (exclude lowest 3 and above ~5 kHz)
    reliable_ranges = {
        "CpuCWT": (3, None),       # skip 3 lowest bins, all high bins ok
        "PywaveletCWT": (3, 90),   # skip 3 lowest; unreliable above bin ~90 (~5 kHz)
    }

    implementations = [("CpuCWT", CpuCWT), ("PywaveletCWT", PywaveletCWT)]

    for impl_name, cls in implementations:
        wt = cls(config)
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
            output = wt.process(tone)

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
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size

    cpu_wt = CpuCWT(config)
    pywt_wt = PywaveletCWT(config)

    freqs_match = np.array_equal(cpu_wt.freqs, pywt_wt.freqs)
    assert freqs_match, "freqs arrays are not identical across CpuCWT and PywaveletCWT"

    for f_target in [130.81, 440.0, 1000.0, 4186.01]:
        if f_target < cpu_wt.freqs[0] or f_target > cpu_wt.freqs[-1]:
            continue

        tone = generate_tone(f_target, sr, chunk)

        np_bin, _ = find_peak_bin(cpu_wt.process(tone), cpu_wt.freqs)
        py_bin, _ = find_peak_bin(pywt_wt.process(tone), pywt_wt.freqs)

        print(f"  @ {f_target:.1f} Hz: CpuCWT bin={np_bin}, PywaveletCWT bin={py_bin}")

        assert abs(np_bin - py_bin) <= 1, (
            f"@ {f_target:.1f} Hz: CpuCWT bin={np_bin}, PywaveletCWT bin={py_bin} — "
            f"bins differ by more than 1"
        )


# =============================================================================
# TEST 3: CHIRP SWEEP MONOTONICITY
# =============================================================================

def test_chirp_monotonicity():
    """Peak bin should increase monotonically for a rising chirp."""
    from scipy.signal import chirp as scipy_chirp

    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size
    hop = int(chunk * (1 - config.overlap_factor))
    num_chunks = 32

    f_start, f_end = 200.0, 10000.0
    total_samples = hop * num_chunks + chunk
    t = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
    signal = scipy_chirp(t, f0=f_start, f1=f_end, t1=t[-1], method='linear').astype(np.float64)

    cpu_wt = CpuCWT(config)

    peak_bins = []
    for i in range(num_chunks):
        chunk_data = signal[i * hop: i * hop + chunk]
        output = cpu_wt.process(chunk_data)
        pb, _ = find_peak_bin(output, cpu_wt.freqs)
        peak_bins.append(pb)

    violations = 0
    for i in range(1, len(peak_bins)):
        if peak_bins[i] < peak_bins[i - 1]:
            violations += 1

    print(f"  Monotonicity: {violations} violations in {len(peak_bins)} chunks "
          f"(bins: {peak_bins[0]} -> {peak_bins[-1]})")

    first_freq = cpu_wt.freqs[peak_bins[0]]
    last_freq = cpu_wt.freqs[peak_bins[-1]]
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
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size

    cpu_wt = CpuCWT(config)
    pywt_wt = PywaveletCWT(config)

    # --- 6a: White noise flatness ---
    np.random.seed(42)
    noise = np.random.randn(chunk).astype(np.float64)

    np_out = cpu_wt.process(noise)
    py_out = pywt_wt.process(noise)

    np_mean_energy = np.mean(np.abs(np_out), axis=1)
    py_mean_energy = np.mean(np.abs(py_out), axis=1)

    n_bins = len(cpu_wt.freqs)
    band = max(1, n_bins // 10)
    np_low = np.mean(np_mean_energy[:band])
    np_high = np.mean(np_mean_energy[-band:])
    py_low = np.mean(py_mean_energy[:band])
    py_high = np.mean(py_mean_energy[-band:])

    np_ratio = np_low / (np_high + 1e-12)
    py_ratio = py_low / (py_high + 1e-12)

    print(f"  CpuCWT white noise low/high ratio: {np_ratio:.2f} (expect < 3.0 with L1 kernel norm)")
    print(f"  PywaveletCWT white noise low/high ratio: {py_ratio:.2f} (expect < 3.0 with sqrt(scale) norm)")

    assert np_ratio < 3.0, (
        f"CpuCWT white noise low/high ratio: {np_ratio:.2f} (expect < 3.0 with L1 kernel norm)"
    )
    assert py_ratio < 3.0, (
        f"PywaveletCWT white noise low/high ratio: {py_ratio:.2f} (expect < 3.0 with sqrt(scale) norm)"
    )

    # --- 6b: Equal-amplitude multi-tone ---
    f_low, f_mid, f_high = 200.0, 1000.0, 5000.0
    multi_tone = (generate_tone(f_low, sr, chunk)
                  + generate_tone(f_mid, sr, chunk)
                  + generate_tone(f_high, sr, chunk))

    np_out = cpu_wt.process(multi_tone)
    np_mean = np.mean(np_out, axis=1)

    bin_low = int(np.argmin(np.abs(cpu_wt.freqs - f_low)))
    bin_high_freq = int(np.argmin(np.abs(cpu_wt.freqs - f_high)))

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

    out_low = cpu_wt.process(tone_low)
    out_high = cpu_wt.process(tone_high_f)

    peak_at_200 = np.max(np.mean(out_low, axis=1))
    peak_at_5k = np.max(np.mean(out_high, axis=1))

    ratio = peak_at_200 / (peak_at_5k + 1e-12)
    print(f"  Single-tone sufficiency: 200 Hz peak / 5 kHz peak = {ratio:.2f} (expect < 8.0)")

    assert ratio < 8.0, (
        f"Single-tone sufficiency: 200 Hz peak / 5 kHz peak = {ratio:.2f} "
        f"(expect < 8.0; residual bias from CWT time-freq tradeoff is normal)"
    )


# =============================================================================
# TEST 6d: CROSS-IMPLEMENTATION NORMALIZATION COMPARABILITY
# =============================================================================

def test_normalization_comparability():
    """PywaveletCWT and CpuCWT should produce comparable energy profiles.

    Both implementations normalize to compensate for scale-dependent energy
    bias: SubShader uses L1 kernel normalization, PyWavelet uses 1/sqrt(scale).
    Equal-amplitude tones should produce similar peak ratios across both.
    """
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size

    cpu_wt = CpuCWT(config)
    pywt_wt = PywaveletCWT(config)

    # Equal-amplitude tones at 200 Hz and 5 kHz
    f_low, f_high = 200.0, 5000.0
    tone_low = generate_tone(f_low, sr, chunk)
    tone_high = generate_tone(f_high, sr, chunk)

    # Get peak energy at each tone's frequency for both implementations
    # Use np.abs() for PywaveletCWT which returns raw complex coefficients (stub post(), D-14)
    np_low = np.max(np.mean(np.abs(cpu_wt.process(tone_low)), axis=1))
    np_high = np.max(np.mean(np.abs(cpu_wt.process(tone_high)), axis=1))
    py_low = np.max(np.mean(np.abs(pywt_wt.process(tone_low)), axis=1))
    py_high = np.max(np.mean(np.abs(pywt_wt.process(tone_high)), axis=1))

    np_ratio = np_low / (np_high + 1e-12)
    py_ratio = py_low / (py_high + 1e-12)

    print(f"  CpuCWT 200Hz/5kHz ratio: {np_ratio:.2f}")
    print(f"  PywaveletCWT 200Hz/5kHz ratio: {py_ratio:.2f}")

    # Both should have similar-ish ratios (within 4x of each other)
    ratio_of_ratios = max(np_ratio, py_ratio) / (min(np_ratio, py_ratio) + 1e-12)
    print(f"  Ratio of ratios: {ratio_of_ratios:.2f} (expect < 4.0)")

    assert ratio_of_ratios < 4.0, (
        f"Normalization mismatch: CpuCWT ratio={np_ratio:.2f}, PywaveletCWT ratio={py_ratio:.2f}, "
        f"ratio-of-ratios={ratio_of_ratios:.2f} (expect < 4.0 for comparable normalization)"
    )


# =============================================================================
# TEST 7: RELIABLE REGION CONSISTENCY
# =============================================================================

def test_reliable_region_consistency():
    """Each backend's process() output shape must match its declared get_output_shape().

    CpuCWT and PywaveletCWT have different output shapes by design:
    - CpuCWT applies reliable-region trimming and downsampling → (num_freqs, target_width)
    - PywaveletCWT has stub post() → (num_freqs, chunk_size) — raw complex coefficients (D-14)

    This test verifies each backend's self-consistency, not cross-backend shape equality.
    """
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size
    tone = generate_tone(440.0, sr, chunk)

    for name, cls in [("CpuCWT", CpuCWT), ("PywaveletCWT", PywaveletCWT)]:
        wt = cls(config)
        declared_shape = wt.get_output_shape()
        actual_out = wt.process(tone)

        print(f"  {name}: declared={declared_shape}, actual={actual_out.shape}")
        assert actual_out.shape == declared_shape, (
            f"{name}: process() shape={actual_out.shape} does not match "
            f"get_output_shape()={declared_shape}"
        )


# =============================================================================
# TEST 8: OUTPUT SHAPE REGRESSION
# =============================================================================

def test_output_shape_regression():
    """Verify output shapes match declared shapes for all implementations."""
    config = make_test_config()
    sr = int(config.sample_rate)
    chunk = config.chunk_size

    for name, cls in [("CpuCWT", CpuCWT), ("PywaveletCWT", PywaveletCWT)]:
        wt = cls(config)
        tone = generate_tone(440.0, sr, chunk)
        output = wt.process(tone)

        declared_shape = wt.get_output_shape()

        print(f"  {name}.get_output_shape() = {declared_shape}")

        assert output.shape == declared_shape, (
            f"{name}.process() output shape = {output.shape}, "
            f"declared = {declared_shape}"
        )
        assert len(wt.freqs) == declared_shape[0], (
            f"{name}: len(freqs)={len(wt.freqs)} != output shape[0]={declared_shape[0]}"
        )


# =============================================================================
# TEST 9: @TIMED METHOD ATTRIBUTES
# =============================================================================

def test_timed_attributes_populated_after_process():
    """After process(), all @timed sub-stage attributes are set and non-negative."""
    config = make_test_config()
    sr = int(config.sample_rate)
    wt = CpuCWT(config)
    tone = generate_tone(440.0, sr, config.chunk_size)
    wt.process(tone)

    # Each @timed method sets _timing_{name}_ms on the instance after each call
    expected_attrs = [
        "_timing_transform_ms",
        "_timing__compute_mag_ms",
        "_timing_discard_unreliable_coefs_ms",
        "_timing_downsample_ms",
        "_timing_extract_hop_center_ms",
    ]
    for attr in expected_attrs:
        assert hasattr(wt, attr), f"Expected attribute {attr} not found after process()"
        val = getattr(wt, attr)
        assert isinstance(val, float), f"{attr} is not float (got {type(val)})"
        assert val >= 0.0, f"{attr} is negative ({val})"


def test_cwt_output_consistent_across_calls():
    """Repeated process() calls on the same input produce identical output."""
    config = make_test_config()
    sr = int(config.sample_rate)
    wt = CpuCWT(config)
    tone = generate_tone(440.0, sr, config.chunk_size)
    result_a = wt.process(tone)
    result_b = wt.process(tone)
    np.testing.assert_allclose(result_a, result_b, rtol=1e-10)
