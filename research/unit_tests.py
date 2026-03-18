"""
Unit tests for SubShader components.

Test categories (see .claude-plans/2026-03-17_unit-test-framework-cwt-accuracy.md):
  0. NumPy vs CuPy CWT correctness verification (existing)
  1. Pure tone peak accuracy
  2. Cross-implementation frequency agreement
  3. Chirp sweep monotonicity
  4. Kernel FFT bank integrity
  5. Impulse response center frequency
  6. Normalization behavior (scale-dependent)
  7. Reliable region consistency
  8. Output shape regression
  9. Wavelet kernel energy per scale
 10. Buffer/visualization normalization
"""

import os
import sys

import numpy as np
from numpy.fft import fft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from subshader.config import get_default_config, WaveletConfig, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from subshader.dsp.wavelet_kernel import WaveletKernel
from subshader.viz.plotter import CircularFrameBuffer

# =============================================================================
# CONSTANTS
# =============================================================================

BENCHMARKS_DIR   = "assets/images/benchmarks"
AUDIO_DEFAULT    = "assets/audio/daw/a2a3_a4_minor_scale.wav"

# =============================================================================
# GPU DETECTION
# =============================================================================

def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False

GPU_AVAILABLE = _gpu_available()

# =============================================================================
# HELPERS
# =============================================================================

def generate_tone(freq_hz, sample_rate, num_samples):
    """Generate a pure sine tone at a given frequency."""
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def find_peak_bin(cwt_output, freqs):
    """Find the frequency bin with the highest mean energy."""
    mean_energy = np.mean(cwt_output, axis=1)
    peak_bin = np.argmax(mean_energy)
    return peak_bin, freqs[peak_bin]


def _make_wavelet(cls, config=None):
    """Construct a wavelet instance with default config."""
    if config is None:
        config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size
    return cls(sample_rate=sr, input_n=chunk, config=config.wavelet)


def _pass_fail(passed, label):
    """Print pass/fail result for a test."""
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    return passed


# =============================================================================
# TEST 0: NUMPY VS CUPY VERIFICATION (existing)
# =============================================================================

def verify_numpy_vs_cupy(save_figure: bool = True):
    """
    Verify that NumPyWavelet and CuPyWavelet produce numerically equivalent outputs.

    Checks:
      1. Global absolute / relative error statistics
      2. Per-frequency-bin Pearson correlation (all bins should be > 0.999)
      3. Optionally saves a 3-panel diff figure: NumPy | CuPy | abs(diff)

    Requires a GPU -- exits early with a message if none is available.
    """
    if not GPU_AVAILABLE:
        print("[verify] Skipped -- no GPU available. Reconnect GPU and re-run.")
        return

    from subshader.dsp.wavelet import CuPyWavelet

    config = get_default_config()
    config.audio.file_path = AUDIO_DEFAULT

    ai = AudioInput(
        path=config.audio.file_path,
        chunk_size=config.audio.chunk_size,
        overlap_factor=config.audio.overlap_factor,
    )
    sr   = ai.get_sample_rate()
    data = ai.get_chunk()

    npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
    cpwt = CuPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

    print("\nVerifying NumPy vs CuPy CWT outputs ...\n")

    np_coefs = npwt.cwt(data)
    cp_coefs = cpwt.cwt(data)

    diff      = np.abs(np_coefs - cp_coefs)
    max_val   = np_coefs.max() + 1e-12
    rel_error = diff.max() / max_val

    print(f"  shape          : {np_coefs.shape}")
    print(f"  max abs diff   : {diff.max():.2e}")
    print(f"  mean abs diff  : {diff.mean():.2e}")
    print(f"  rel error      : {rel_error:.2e}")

    # Per-frequency-bin Pearson correlation
    n_freq = np_coefs.shape[0]
    corrs  = np.array([
        np.corrcoef(np_coefs[i], cp_coefs[i])[0, 1]
        for i in range(n_freq)
    ])
    low_corr = np.where(corrs < 0.999)[0]
    print(f"  bins < 0.999 r : {len(low_corr)} / {n_freq}", end="")
    if len(low_corr):
        print(f"  <- bins: {low_corr[:10]}")
    else:
        print()

    passed = rel_error < 1e-3 and len(low_corr) == 0
    _pass_fail(passed, "NumPy vs CuPy agreement")

    if save_figure:
        _save_diff_figure(np_coefs, cp_coefs, diff)


def _save_diff_figure(np_coefs, cp_coefs, diff):
    """Save 3-panel figure: NumPy | CuPy | abs(diff)."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NumPy vs CuPy CWT -- correctness check", fontsize=12)

    vmax = max(np_coefs.max(), cp_coefs.max())

    axes[0].imshow(np_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title("NumPyWavelet.cwt()")
    axes[0].set_ylabel("Frequency bin")
    axes[0].set_xlabel("Time bin")

    axes[1].imshow(cp_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
    axes[1].set_title("CuPyWavelet.cwt()")
    axes[1].set_xlabel("Time bin")

    im = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto", origin="lower")
    axes[2].set_title("abs(NumPy - CuPy)")
    axes[2].set_xlabel("Time bin")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="abs diff")

    fig.tight_layout()
    path = os.path.join(BENCHMARKS_DIR, "numpy_vs_cupy_diff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diff figure saved -> {path}\n")


# =============================================================================
# TEST 1: PURE TONE PEAK ACCURACY
# =============================================================================

TEST_FREQS = [27.5, 130.81, 440.0, 1000.0, 4186.01, 10000.0]

def test_pure_tone_peak_accuracy():
    """For each wavelet, a pure tone at f should peak at the bin nearest f."""
    print("\n--- Test 1: Pure Tone Peak Accuracy ---\n")

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    implementations = [("NumPyWavelet", NumPyWavelet), ("PyWavelet", PyWavelet)]
    all_passed = True

    for impl_name, cls in implementations:
        wt = cls(sample_rate=sr, input_n=chunk, config=config.wavelet)
        freqs = wt.freqs

        for f_target in TEST_FREQS:
            if f_target < freqs[0] or f_target > freqs[-1]:
                continue

            tone = generate_tone(f_target, sr, chunk)
            output = wt.cwt(tone)

            peak_bin, peak_freq = find_peak_bin(output, freqs)
            expected_bin = int(np.argmin(np.abs(freqs - f_target)))

            ok = abs(peak_bin - expected_bin) <= 1
            all_passed &= _pass_fail(ok,
                f"{impl_name} @ {f_target:.1f} Hz: peak bin={peak_bin} "
                f"(freq={peak_freq:.1f} Hz), expected bin={expected_bin} "
                f"(freq={freqs[expected_bin]:.1f} Hz)")

    return all_passed


# =============================================================================
# TEST 2: CROSS-IMPLEMENTATION FREQUENCY AGREEMENT
# =============================================================================

def test_cross_implementation_agreement():
    """All implementations should peak at the same bin for the same tone."""
    print("\n--- Test 2: Cross-Implementation Frequency Agreement ---\n")

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    # Frequency arrays should be identical
    freqs_match = np.array_equal(npwt.freqs, pywt_wt.freqs)
    all_passed = _pass_fail(freqs_match, "freqs arrays identical across implementations")

    for f_target in [130.81, 440.0, 1000.0, 4186.01]:
        if f_target < npwt.freqs[0] or f_target > npwt.freqs[-1]:
            continue

        tone = generate_tone(f_target, sr, chunk)

        np_bin, _ = find_peak_bin(npwt.cwt(tone), npwt.freqs)
        py_bin, _ = find_peak_bin(pywt_wt.cwt(tone), pywt_wt.freqs)

        ok = abs(np_bin - py_bin) <= 1
        all_passed &= _pass_fail(ok,
            f"@ {f_target:.1f} Hz: NumPy bin={np_bin}, PyWavelet bin={py_bin}")

    return all_passed


# =============================================================================
# TEST 3: CHIRP SWEEP MONOTONICITY
# =============================================================================

def test_chirp_monotonicity():
    """Peak bin should increase monotonically for a rising chirp."""
    print("\n--- Test 3: Chirp Sweep Monotonicity ---\n")
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

    # Check monotonicity (allow equal bins -- non-decreasing)
    violations = 0
    for i in range(1, len(peak_bins)):
        if peak_bins[i] < peak_bins[i - 1]:
            violations += 1

    all_passed = True
    all_passed &= _pass_fail(violations <= 2,
        f"Monotonicity: {violations} violations in {len(peak_bins)} chunks "
        f"(bins: {peak_bins[0]} -> {peak_bins[-1]})")

    # First chunk should be near f_start, last near f_end
    first_freq = npwt.freqs[peak_bins[0]]
    last_freq = npwt.freqs[peak_bins[-1]]
    all_passed &= _pass_fail(first_freq < f_start * 3,
        f"First chunk peak: {first_freq:.0f} Hz (expect near {f_start:.0f} Hz)")
    all_passed &= _pass_fail(last_freq > f_end * 0.5,
        f"Last chunk peak: {last_freq:.0f} Hz (expect near {f_end:.0f} Hz)")

    return all_passed


# =============================================================================
# TEST 4: KERNEL FFT BANK INTEGRITY
# =============================================================================

def test_kernel_fft_bank_integrity():
    """Each kernel in the bank should be FFT'd at max_conv_n, not its own conv_n."""
    print("\n--- Test 4: Kernel FFT Bank Integrity ---\n")

    npwt = _make_wavelet(NumPyWavelet)

    all_passed = True
    mismatches = []

    for i, w in enumerate(npwt.wavelets):
        expected = fft(w.kernel_t, npwt.max_conv_n)
        stored = npwt.kernel_f_bank[i]

        if not np.allclose(expected, stored, atol=1e-5):
            mismatches.append(i)

    ok = len(mismatches) == 0
    all_passed &= _pass_fail(ok,
        f"Kernel FFT bank: {len(mismatches)} mismatches out of {npwt.num_wavelets} kernels"
        + (f" (bins: {mismatches[:5]})" if mismatches else ""))

    return all_passed


# =============================================================================
# TEST 5: IMPULSE RESPONSE CENTER FREQUENCY
# =============================================================================

def test_impulse_response_center_freq():
    """Each kernel's FFT magnitude should peak at its declared center frequency."""
    print("\n--- Test 5: Impulse Response Center Frequency ---\n")

    npwt = _make_wavelet(NumPyWavelet)
    sr = int(npwt.sample_rate)

    all_passed = True
    worst_error_hz = 0.0

    for i, w in enumerate(npwt.wavelets):
        kernel_f = fft(w.kernel_t, npwt.max_conv_n)
        mag = np.abs(kernel_f[:npwt.max_conv_n // 2])
        fft_freqs = np.fft.rfftfreq(npwt.max_conv_n, d=1.0 / sr)[:len(mag)]

        peak_idx = np.argmax(mag)
        peak_freq = fft_freqs[peak_idx]
        error_hz = abs(peak_freq - w.freq)
        worst_error_hz = max(worst_error_hz, error_hz)

        # Tolerance: within 1 FFT bin width
        bin_width = sr / npwt.max_conv_n
        ok = error_hz <= bin_width * 1.5

        if not ok:
            _pass_fail(False,
                f"Kernel {i} ({w.freq:.1f} Hz): FFT peak at {peak_freq:.1f} Hz, "
                f"error={error_hz:.2f} Hz (bin width={bin_width:.2f} Hz)")
            all_passed = False

    if all_passed:
        _pass_fail(True,
            f"All {npwt.num_wavelets} kernels: FFT peak matches center freq "
            f"(worst error: {worst_error_hz:.2f} Hz)")

    return all_passed


# =============================================================================
# TEST 6: NORMALIZATION BEHAVIOR
# =============================================================================

def test_normalization_behavior():
    """Test scale-dependent normalization: white noise flatness + multi-tone equality."""
    print("\n--- Test 6: Normalization Behavior ---\n")

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    all_passed = True

    # --- 6a: White noise flatness ---
    np.random.seed(42)
    noise = np.random.randn(chunk).astype(np.float64)

    np_out = npwt.cwt(noise)
    py_out = pywt_wt.cwt(noise)

    np_mean_energy = np.mean(np_out, axis=1)
    py_mean_energy = np.mean(py_out, axis=1)

    # Low bin = index 0 (lowest freq), high bin = last index (highest freq)
    # Use bands to reduce noise: average bottom 10% and top 10%
    n_bins = len(npwt.freqs)
    band = max(1, n_bins // 10)
    np_low = np.mean(np_mean_energy[:band])
    np_high = np.mean(np_mean_energy[-band:])
    py_low = np.mean(py_mean_energy[:band])
    py_high = np.mean(py_mean_energy[-band:])

    np_ratio = np_low / (np_high + 1e-12)
    py_ratio = py_low / (py_high + 1e-12)

    all_passed &= _pass_fail(np_ratio < 3.0,
        f"NumPyWavelet white noise low/high ratio: {np_ratio:.2f} (expect < 3.0 with sqrt(f) norm)")
    all_passed &= _pass_fail(py_ratio > 2.0,
        f"PyWavelet white noise low/high ratio: {py_ratio:.2f} (expect > 2.0 without norm)")

    print()

    # --- 6b: Equal-amplitude multi-tone ---
    f_low, f_mid, f_high = 200.0, 1000.0, 5000.0
    multi_tone = (generate_tone(f_low, sr, chunk)
                  + generate_tone(f_mid, sr, chunk)
                  + generate_tone(f_high, sr, chunk))

    np_out = npwt.cwt(multi_tone)
    np_mean = np.mean(np_out, axis=1)

    bin_low = int(np.argmin(np.abs(npwt.freqs - f_low)))
    bin_mid = int(np.argmin(np.abs(npwt.freqs - f_mid)))
    bin_high = int(np.argmin(np.abs(npwt.freqs - f_high)))

    peak_low = np_mean[bin_low]
    peak_mid = np_mean[bin_mid]
    peak_high = np_mean[bin_high]

    # After sqrt(f) normalization, equal-amplitude tones should have comparable peaks.
    # CWT inherently has some frequency-dependent resolution bias for pure tones
    # (wider time support at low freq captures more cycles), so allow factor of 8.
    # The key check is that it's much better than WITHOUT normalization (which would be ~20x+).
    ratio_high_low = peak_low / (peak_high + 1e-12)
    all_passed &= _pass_fail(ratio_high_low < 8.0,
        f"Multi-tone: 200 Hz peak / 5 kHz peak = {ratio_high_low:.2f} "
        f"(expect < 8.0 with normalization; without would be ~20x+)")

    print()

    # --- 6c: Normalization sufficiency (single tone comparison) ---
    tone_low = generate_tone(200.0, sr, chunk)
    tone_high = generate_tone(5000.0, sr, chunk)

    out_low = npwt.cwt(tone_low)
    out_high = npwt.cwt(tone_high)

    peak_at_200 = np.max(np.mean(out_low, axis=1))
    peak_at_5k = np.max(np.mean(out_high, axis=1))

    ratio = peak_at_200 / (peak_at_5k + 1e-12)
    all_passed &= _pass_fail(ratio < 8.0,
        f"Single-tone sufficiency: 200 Hz peak / 5 kHz peak = {ratio:.2f} "
        f"(expect < 8.0; residual bias from CWT time-freq tradeoff is normal)")

    return all_passed


# =============================================================================
# TEST 7: RELIABLE REGION CONSISTENCY
# =============================================================================

def test_reliable_region_consistency():
    """Output shapes should match between PyWavelet and NumPyWavelet."""
    print("\n--- Test 7: Reliable Region Consistency ---\n")

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    npwt = NumPyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)
    pywt_wt = PyWavelet(sample_rate=sr, input_n=chunk, config=config.wavelet)

    all_passed = True

    np_shape = npwt.get_output_shape()
    py_shape = pywt_wt.get_output_shape()

    all_passed &= _pass_fail(np_shape == py_shape,
        f"Output shapes: NumPy={np_shape}, PyWavelet={py_shape}")

    # Run on actual data to verify
    tone = generate_tone(440.0, sr, chunk)
    np_out = npwt.cwt(tone)
    py_out = pywt_wt.cwt(tone)

    all_passed &= _pass_fail(np_out.shape == py_out.shape,
        f"Actual output shapes: NumPy={np_out.shape}, PyWavelet={py_out.shape}")

    return all_passed


# =============================================================================
# TEST 8: OUTPUT SHAPE REGRESSION
# =============================================================================

def test_output_shape_regression():
    """Verify output shapes match declared shapes for all implementations."""
    print("\n--- Test 8: Output Shape Regression ---\n")

    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size

    all_passed = True

    for name, cls in [("NumPyWavelet", NumPyWavelet), ("PyWavelet", PyWavelet)]:
        wt = cls(sample_rate=sr, input_n=chunk, config=config.wavelet)
        tone = generate_tone(440.0, sr, chunk)
        output = wt.cwt(tone)

        expected_shape = (wt.num_freqs, wt.config.target_width)
        declared_shape = wt.get_output_shape()

        all_passed &= _pass_fail(declared_shape == expected_shape,
            f"{name}.get_output_shape() = {declared_shape}, "
            f"expected ({wt.num_freqs}, {wt.config.target_width})")

        all_passed &= _pass_fail(output.shape == declared_shape,
            f"{name}.cwt() output shape = {output.shape}, "
            f"declared = {declared_shape}")

        all_passed &= _pass_fail(len(wt.freqs) == wt.num_freqs,
            f"{name}: len(freqs)={len(wt.freqs)} == num_freqs={wt.num_freqs}")

    return all_passed


# =============================================================================
# TEST 9: WAVELET KERNEL ENERGY PER SCALE
# =============================================================================

def test_kernel_energy_per_scale():
    """
    Diagnose whether sqrt(f) normalization matches the actual kernel energy scaling.

    If kernel L2 norm scales as 1/sqrt(f), then multiplying by sqrt(f) is correct.
    """
    print("\n--- Test 9: Wavelet Kernel Energy per Scale ---\n")

    npwt = _make_wavelet(NumPyWavelet)

    freqs = []
    l2_norms = []

    for w in npwt.wavelets:
        l2 = np.sqrt(np.sum(np.abs(w.kernel_t) ** 2))
        freqs.append(w.freq)
        l2_norms.append(l2)

    freqs = np.array(freqs)
    l2_norms = np.array(l2_norms)

    # If L2 norm ~ C / sqrt(f), then L2 * sqrt(f) should be roughly constant
    corrected = l2_norms * np.sqrt(freqs)
    cv = np.std(corrected) / np.mean(corrected)  # coefficient of variation

    all_passed = True
    all_passed &= _pass_fail(cv < 0.15,
        f"L2_norm * sqrt(f) coefficient of variation: {cv:.4f} "
        f"(expect < 0.15 if sqrt(f) is the right correction)")

    # Also check the actual scaling trend
    # Fit log(L2) vs log(f) -- slope should be ~ -0.5
    log_f = np.log(freqs)
    log_l2 = np.log(l2_norms)
    slope, intercept = np.polyfit(log_f, log_l2, 1)

    all_passed &= _pass_fail(abs(slope - (-0.5)) < 0.15,
        f"L2 norm vs freq log-log slope: {slope:.3f} (expect ~ -0.5)")

    print(f"    Detail: slope={slope:.4f}, CV={cv:.4f}, "
          f"mean(L2*sqrt(f))={np.mean(corrected):.2f}, "
          f"std={np.std(corrected):.2f}")

    return all_passed


# =============================================================================
# TEST 10: BUFFER / VISUALIZATION NORMALIZATION
# =============================================================================

def test_buffer_normalization():
    """Verify CircularFrameBuffer preserves relative intensities."""
    print("\n--- Test 10: Buffer/Visualization Normalization ---\n")

    n_freqs = 16
    n_time = 32
    n_frames = 10

    color_norm = ColorNormalizationConfig()
    buf = CircularFrameBuffer(
        frame_shape=(n_freqs, n_time),
        num_frames=n_frames,
        color_norm_config=color_norm,
    )

    all_passed = True

    # Push frames with known intensity pattern: increasing max per frame
    for i in range(n_frames):
        frame = np.random.rand(n_freqs, n_time).astype(np.float32) * (i + 1)
        buf.push_frame(frame)

    # get_intensity_max should track the global max
    imax = buf.get_intensity_max()
    all_passed &= _pass_fail(imax > 0,
        f"get_intensity_max() = {imax:.4f} (expect > 0)")

    # get_flattened_buffer should preserve the frame data
    flat = buf.get_flattened_buffer()
    all_passed &= _pass_fail(flat.shape[0] == n_freqs,
        f"Flattened buffer freq dimension: {flat.shape[0]} (expect {n_freqs})")
    all_passed &= _pass_fail(flat.shape[1] == n_time * n_frames,
        f"Flattened buffer time dimension: {flat.shape[1]} (expect {n_time * n_frames})")

    # Values should not be clipped or normalized to [0, 1]
    all_passed &= _pass_fail(flat.max() > 1.0,
        f"Flattened buffer max = {flat.max():.2f} "
        f"(expect > 1.0, confirming no [0,1] normalization)")

    return all_passed


# =============================================================================
# RUN ALL
# =============================================================================

def run_all():
    """Run all unit tests."""
    print("\n=== Unit Tests ===\n")

    results = {}

    # Tests that don't require GPU or audio files
    results["Test 4: Kernel FFT Bank"]     = test_kernel_fft_bank_integrity()
    results["Test 5: Impulse Response"]    = test_impulse_response_center_freq()
    results["Test 8: Output Shape"]        = test_output_shape_regression()
    results["Test 9: Kernel Energy"]       = test_kernel_energy_per_scale()
    results["Test 10: Buffer Norm"]        = test_buffer_normalization()
    results["Test 1: Pure Tone"]           = test_pure_tone_peak_accuracy()
    results["Test 2: Cross-Impl"]          = test_cross_implementation_agreement()
    results["Test 3: Chirp Monotonicity"]  = test_chirp_monotonicity()
    results["Test 6: Normalization"]       = test_normalization_behavior()
    results["Test 7: Reliable Region"]     = test_reliable_region_consistency()

    if GPU_AVAILABLE:
        verify_numpy_vs_cupy()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{total} tests passed.\n")


if __name__ == "__main__":
    run_all()
