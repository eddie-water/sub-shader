# Unit Test Framework: CWT Frequency Accuracy & Normalization

**Updated:** 2026-03-17

## Context

During benchmark figure development, three issues were discovered that went undetected:

1. **FFT length mismatch bug** -- `AntsWavelet.__init__()` stores kernel FFTs computed at per-kernel `conv_n` into a bank of `max_conv_n`, then multiplies with input FFT'd at `max_conv_n`. Frequency bins don't align, corrupting high-frequency wavelets most severely.

2. **Normalization asymmetry** -- `PyWavelet.normalize_by_scale` is identity (`return cwt_coefs`), `AntsWavelet` multiplies by `sqrt(f)`. This is a design difference, but the lack of tests meant there was no documentation of expected behavior.

3. **High-frequency dimming in benchmark figures** -- Both PyWavelet and SubShader (NumPyWavelet) spectrogram rows appear weaker at higher frequencies. Root cause investigation (2026-03-17):
   - **PyWavelet**: `normalize_by_scale()` is a no-op (line 320 of `wavelet.py`). Low-frequency bins naturally accumulate more energy from wider wavelets, so without `sqrt(f)` correction, high frequencies appear dim.
   - **NumPyWavelet (SubShader)**: Does apply `sqrt(f)` normalization (line 435), but may still appear dim if the correction is insufficient OR if the Gaussian envelope in `wavelet_kernel.py` is not L2-normalized per scale (it uses FWHM-based construction, not energy-normalized).
   - **Interaction with shared vmax**: Phase 3b introduced shared vmax across all 3 spectrogram rows. If STFT has much higher raw intensity at low frequencies, the shared vmax will make all 3 methods' high-frequency content appear even dimmer.

These tests should live alongside the existing `research/unit_tests.py` (invoked via `python research/benchmark.py --unit-tests`).

---

## Test Categories

### 1. Pure Tone Peak Accuracy
**What it catches:** FFT length mismatch, kernel construction errors, frequency-to-scale mapping bugs.

For each wavelet implementation (PyWavelet, NumPyWavelet, CuPyWavelet if GPU available):
- Generate a pure sine tone at a known frequency `f_target`
- Run `wavelet.cwt(tone)`
- Assert `argmax(mean(output, axis=1))` corresponds to the bin nearest `f_target` in `wavelet.freqs`
- Tolerance: peak must be within +/-1 bin of the expected bin

**Test frequencies** (span the full range, targeting known trouble spots):
- `27.5 Hz` (A0 -- lowest bin, root note)
- `130.81 Hz` (C3 -- low-mid, the polyphonic bug frequency)
- `440.0 Hz` (A4 -- standard tuning reference)
- `1000.0 Hz` (1 kHz -- mid range)
- `4186.01 Hz` (C8 -- high piano)
- `10000.0 Hz` (10 kHz -- high range where FFT mismatch was worst)

**Config variations:**
- Default config (12 notes/octave, root=27.5 Hz)
- Fine config (48 notes/octave, root=200 Hz) -- chirp-style

### 2. Cross-Implementation Frequency Agreement
**What it catches:** Implementation divergence between PyWavelet, NumPyWavelet, and CuPyWavelet.

For each test tone:
- Run all available implementations on the same input
- Assert all peak at the **same bin index**
- Assert `freqs` arrays are bitwise identical across implementations (same config -> same frequencies)

### 3. Chirp Sweep Monotonicity
**What it catches:** Frequency shifts, non-monotonic bin mapping, convolution artifacts.

- Generate a linear chirp from `f_start` to `f_end`
- Split into overlapping chunks matching the pipeline's chunk_size/overlap
- Run CWT on each chunk
- For each chunk, find the bin with peak energy
- Assert the peak bin is **monotonically non-decreasing** across chunks (chirp only goes up)
- Assert first chunk's peak is near `f_start` and last chunk's peak is near `f_end`

### 4. Kernel FFT Bank Integrity
**What it catches:** The exact FFT length mismatch bug -- directly.

After constructing an `AntsWavelet` (NumPyWavelet or CuPyWavelet):
- For each wavelet kernel `w` at index `i`:
  - Compute `expected = fft(w.kernel_t, wavelet.max_conv_n)`
  - Compare with `wavelet.kernel_f_bank[i]`
  - Assert `np.allclose(expected, wavelet.kernel_f_bank[i], atol=1e-5)`
- This directly tests that all kernels are FFT'd at the shared length

### 5. Impulse Response Center Frequency
**What it catches:** Wavelet kernel construction bugs, Gaussian shaping errors, center frequency drift.

For each WaveletKernel in the bank:
- FFT the kernel at a sufficient length (e.g., `max_conv_n`)
- Find the peak frequency in the magnitude spectrum
- Assert it matches `kernel.freq` within tolerance (+/-1 frequency bin)

### 6. Normalization Behavior (Scale-Dependent)
**What it catches:** Undocumented normalization changes, regression in scale normalization, high-frequency dimming.

**6a. White noise flatness test:**
- Generate white noise (broadband, equal energy at all frequencies)
- Run CWT, then average output across time dimension
- **PyWavelet (no normalization):** expect low-frequency bins to be brighter (1/sqrt(f) bias)
- **AntsWavelet (sqrt(f) normalization):** expect approximately flat response across bins
- Assert the ratio between lowest and highest frequency bin mean energy:
  - PyWavelet: ratio > 3x (low freqs naturally louder)
  - AntsWavelet: ratio < 2x (normalization flattens it)

**6b. Equal-amplitude multi-tone test (NEW):**
- Generate a sum of equal-amplitude sine tones at known frequencies (e.g., 100 Hz, 1 kHz, 10 kHz)
- Run CWT on each implementation
- After normalization, all three peaks should have **comparable magnitude** (within 6 dB / factor of 2)
- This directly tests whether the normalization makes equal-energy signals appear equally bright

**6c. Normalization sufficiency test (NEW):**
- Generate a pure tone at 100 Hz and another at 10 kHz, same amplitude
- Run NumPyWavelet (which applies sqrt(f)) on each
- Compare peak magnitudes: the 10 kHz tone should NOT be more than 2x dimmer than 100 Hz
- If it IS dimmer, the sqrt(f) correction is insufficient and a stronger correction (or L2-normalized kernels) is needed

### 7. Reliable Region Consistency
**What it catches:** Divergence between PyWavelet and AntsWavelet center-keep slicing.

- Construct both PyWavelet and NumPyWavelet with the same config
- For PyWavelet: extract the hardcoded keep size (8192)
- For AntsWavelet: extract `reliable_slice` width
- Assert both keep the same number of samples (or document the difference)
- Assert the output shape of `.cwt()` is identical for both

### 8. Output Shape Regression
**What it catches:** Shape mismatches that would break plotting code.

For each implementation:
- Assert `wavelet.get_output_shape() == (wavelet.num_freqs, wavelet.config.target_width)`
- Assert `wavelet.cwt(input).shape == wavelet.get_output_shape()`
- Assert `len(wavelet.freqs) == wavelet.num_freqs`

### 9. Wavelet Kernel Energy per Scale (NEW)
**What it catches:** Whether the Gaussian envelope in `wavelet_kernel.py` introduces frequency-dependent energy bias that `sqrt(f)` normalization doesn't fully compensate.

- For each WaveletKernel in the bank:
  - Compute L2 norm: `np.sqrt(np.sum(np.abs(kernel.kernel_t)**2))`
  - Record `(freq, l2_norm)`
- Plot or assert: L2 norm should scale as `1/sqrt(f)` (since wider wavelets have more samples)
- If L2 norm does NOT follow `1/sqrt(f)`, the `sqrt(f)` correction in `normalize_by_scale` is mismatched
- This is the root diagnostic: if kernel L2 norm = `C / sqrt(f)`, then multiplying CWT output by `sqrt(f)` is the correct compensation

### 10. Buffer/Visualization Normalization (NEW)
**What it catches:** Whether `CircularFrameBuffer` + `IntensityTracker` introduces artifacts.

- Push synthetic frames with known intensity profiles into `CircularFrameBuffer`
- Verify `get_intensity_max()` returns the expected global max
- Verify `get_flattened_buffer()` preserves relative intensities (no per-frame normalization that would distort cross-frequency comparison)
- Test with `ColorNormalizationConfig(log_mapping=True)` and `False` to verify both paths

---

## Module Dependency Map (for test coverage planning)

```
config.py (WaveletConfig, ColorNormalizationConfig)
    |
    v
wavelet_kernel.py (WaveletKernel, Gaussian)     <-- Test 5, 9
    |
    v
wavelet.py                                       <-- Tests 1-4, 6-8
  |- Wavelet (abstract base)
  |- PyWavelet (pywt backend)
  |- AntsWavelet (frequency-domain convolution)
  |    |- NumPyWavelet
  |    |- CuPyWavelet
    |
    v
viz/plot_normalizer.py (IntensityTracker)        <-- Test 10
    |
    v
viz/plotter.py (CircularFrameBuffer,             <-- Test 10
                AudioFrameBuffer)
    |
    v
research/benchmark.py (figure generation)        <-- Integration-level visual check
```

---

## Implementation Notes

### File Location
`research/unit_tests.py` -- already exists and is invoked via `--unit-tests` flag.

### Test Runner
Tests should use plain `assert` statements with descriptive messages (matching the existing style in `unit_tests.py`), or optionally `pytest` if preferred. Each test category should be a function that can run independently.

### Helper: Tone Generator
```python
def generate_tone(freq_hz, sample_rate, num_samples):
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)
```

### Helper: Find Peak Bin
```python
def find_peak_bin(cwt_output, freqs):
    mean_energy = np.mean(cwt_output, axis=1)
    peak_bin = np.argmax(mean_energy)
    return peak_bin, freqs[peak_bin]
```

### Expected Runtime
- Tests 1-3, 5-8: ~10-30 seconds (fast -- pure tone CWT is one chunk)
- Test 4 (kernel bank integrity): ~1 second (just FFT comparisons, no CWT)
- Tests 9-10: ~5 seconds (kernel analysis and buffer tests)
- Total: under 1 minute

---

## Priority Order

If implementing incrementally:
1. **Test 9** (Kernel Energy per Scale) -- root diagnostic for the normalization question; tells us if sqrt(f) is the right correction
2. **Test 6b/6c** (Multi-tone & sufficiency) -- directly answers "is high-freq dimming a normalization bug?"
3. **Test 4** (Kernel FFT Bank Integrity) -- directly catches the FFT length bug, fastest to write
4. **Test 1** (Pure Tone Peak Accuracy) -- highest signal, catches frequency mapping errors
5. **Test 2** (Cross-Implementation Agreement) -- catches divergence between methods
6. **Test 3** (Chirp Monotonicity) -- catches subtle drift across frequency range
7. **Test 10** (Buffer normalization) -- catches visualization-layer issues
8. Tests 5, 7, 8 as time permits

---

## Critical Files
- `research/unit_tests.py` -- add tests here
- `src/subshader/dsp/wavelet.py` -- code under test (Wavelet, PyWavelet, AntsWavelet, NumPyWavelet)
- `src/subshader/dsp/wavelet_kernel.py` -- WaveletKernel construction (tested by Tests 5, 9)
- `src/subshader/dsp/gaussian.py` -- Gaussian envelope (upstream of kernel energy)
- `src/subshader/viz/plotter.py` -- CircularFrameBuffer, AudioFrameBuffer (tested by Test 10)
- `src/subshader/viz/plot_normalizer.py` -- IntensityTracker (tested by Test 10)
- `src/subshader/config.py` -- WaveletConfig, ColorNormalizationConfig (used to create test configurations)
