# Fix SubShader CWT Frequency Accuracy

## Context

The benchmark comparison figures show SubShader CWT producing visibly different (less accurate) results than STFT and PyWavelet CWT:

1. **Chirp figure**: SubShader's chirp energy appears shifted to higher frequencies
2. **Polyphonic figure**: C3 (~131 Hz) root note appears dim/missing in SubShader, starting around 200 Hz instead

Investigation revealed a **bug in the library's FFT-based convolution** that corrupts SubShader's CWT output, plus a secondary issue with frequency resolution for the chirp test case.

---

## Root Cause: FFT Length Mismatch Bug

**Location:** `src/subshader/dsp/wavelet.py`, `AntsWavelet.__init__()` lines 364-370

Each wavelet kernel has a different `conv_n` (= `input_n + time_support_n - 1`) because `time_support_n` varies by frequency. The code:

```python
self.max_conv_n = max(w.get_conv_n() for w in self.wavelets)
self.kernel_f_bank = np.zeros((num_wavelets, max_conv_n), dtype=np.complex64)
for i, w in enumerate(self.wavelets):
    self.kernel_f_bank[i, :w.get_conv_n()] = w.kernel_f   # ← BUG
```

`w.kernel_f` was computed as `fft(kernel_t, w.conv_n)` (wavelet_kernel.py:66). This produces `conv_n` frequency bins with spacing `fs/conv_n`. But the bank stores them in `max_conv_n`-length rows, and the CWT multiplies with `fft(input, max_conv_n)` which has spacing `fs/max_conv_n`. **Bin k represents different frequencies in each array.**

Impact scales with the ratio `conv_n / max_conv_n`:
- Low-freq wavelets (large time support): `conv_n ≈ max_conv_n` → minimal error
- High-freq wavelets (tiny time support): `conv_n << max_conv_n` → severe misalignment
  - e.g. 20 kHz wavelet: bin 100 = 269 Hz in kernel FFT vs 170 Hz in input FFT

This explains both issues:
- Chirp energy shifts upward at higher frequencies (where the misalignment is worst)
- Low-frequency notes (C3) appear attenuated or shifted

**CuPyWavelet has the same bug** — it copies `self.kernel_f_bank` to GPU (wavelet.py:577-580).

---

## Changes Required

### Step 1: Fix kernel_f_bank construction
**File:** `src/subshader/dsp/wavelet.py`, lines 364-370

Replace:
```python
self.max_conv_n: int = max(w.get_conv_n() for w in self.wavelets)
self.kernel_f_bank: np.ndarray[np.complex64] = np.zeros((self.num_wavelets, self.max_conv_n), dtype=np.complex64)
for i, w in enumerate(self.wavelets):
    self.kernel_f_bank[i, :w.get_conv_n()] = w.kernel_f
```
With:
```python
self.max_conv_n: int = max(w.get_conv_n() for w in self.wavelets)
self.kernel_f_bank: np.ndarray[np.complex64] = np.zeros((self.num_wavelets, self.max_conv_n), dtype=np.complex64)
for i, w in enumerate(self.wavelets):
    self.kernel_f_bank[i] = fft(w.kernel_t, self.max_conv_n)
```

The fix: re-FFT each kernel's time-domain signal at the shared `max_conv_n` length. This zero-pads the time-domain kernel before FFT, producing frequency bins at the correct spacing to match the input's FFT. One-line change.

Note: `fft` is already imported at the top of wavelet.py (`from numpy.fft import fft`).

### Step 2: Chirp-specific WaveletConfig for finer resolution
**File:** `research/benchmark.py`

The default chromatic scale (12 notes/octave) is correct for musical signals but too coarse for a continuous chirp sweep. Add a custom config for the chirp figure.

**2a. Add `wavelet_config` parameter to `_generate_comparison_figure()`:**

Add `wavelet_config=None` to the method signature. Inside, use it when constructing wavelets:
```python
wc = wavelet_config if wavelet_config is not None else config.wavelet
pywt = PyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=wc)
npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=wc)
```

**2b. Create chirp config in `chirp_signal_comparison()`:**
```python
from subshader.config import WaveletConfig
chirp_wc = WaveletConfig(
    notes_per_octave=48,      # 4x finer than default 12
    num_octaves=7,            # covers 200 Hz * 2^7 = 25.6 kHz
    root_note_a0_hz=float(CHIRP_F0),  # start from chirp start frequency
)
```
Pass as `wavelet_config=chirp_wc`. This gives ~336 frequency bins across the chirp range with ~3-cent spacing.

Polyphonic and musical figures continue using the default chromatic scale (12 notes/octave from 27.5 Hz) — correct for musical content.

---

## Critical Files
- `src/subshader/dsp/wavelet.py` — one-line fix in AntsWavelet.__init__ (Step 1)
- `research/benchmark.py` — chirp config enhancement (Step 2)

## Verification

```bash
python research/benchmark.py --figures
```

Check all 3 output images in `assets/images/benchmarks/`:
1. **Chirp**: SubShader CWT chirp line should closely match STFT and PyWavelet trajectories, no upward shift
2. **Polyphonic**: C3 (~131 Hz) should appear at the correct position in SubShader, matching PyWavelet
3. **Musical**: Overall spectral content should be consistent across all three methods
