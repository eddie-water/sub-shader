# Phase 2: CWT Pipeline Polish - Research

**Researched:** 2026-03-21
**Domain:** CWT normalization (DSP), pytest test design
**Confidence:** HIGH

## Summary

The root cause of the low-frequency brightness bias is mathematically confirmed: the wavelet
kernel's L1 norm (sum of absolute values) scales as `1/f`. A 100 Hz kernel has ~100x the L1
norm of a 10 kHz kernel. Because CWT is implemented as linear convolution, the peak response
to a unit-amplitude sinusoid at the kernel's center frequency is directly proportional to the
kernel's L1 norm. Dividing `self.kernel_t` by its own L1 norm in `WaveletKernel.__init__`
makes the peak response a constant ~0.5 across all frequencies tested (100 Hz to 10 kHz,
variation < 0.35%).

The existing `normalize_by_scale` method multiplies by `sqrt(f)`, which half-corrects a `1/f`
bias. After kernel normalization at the source, this method becomes a no-op or can be removed —
it is no longer needed.

The test infrastructure is already functional: `pytest tests/` runs 21 passing tests from the
project root with no manual setup. New tests for this phase follow the established fixture
pattern in `conftest.py`.

**Primary recommendation:** Normalize `WaveletKernel.kernel_t` to unit L1 area at construction
time. Add an explanatory comment documenting the energy-bias reasoning. Make `normalize_by_scale`
a no-op. Remove `cwt_out_type` dead code. Write tests at each pipeline stage.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Normalize wavelet kernels to unit area in `WaveletKernel.__init__` — the kernel itself produces balanced output, no post-hoc correction needed
- **D-02:** The normalization code must be explicitly commented explaining the energy-bias reasoning (wider wavelets collect more energy because longer time support integrates more signal) — this scaffolds a future DSP README section on normalization design intuition
- **D-03:** Current `normalize_by_scale` multiplying by `sqrt(freq)` is insufficient — CWT output amplitude scales as `1/f` (proportional to Gaussian width = `num_fwhm_cycles/f`), so `sqrt(f)` only half-corrects. Kernel unit-area normalization addresses this at the source
- **D-04:** `cwt_out_type` config field is dead code (defined but never used) — clean up during this phase
- **D-05:** Use PyWavelet as-is with any built-in normalization options it provides — do not hand-write normalization for it
- **D-06:** If SubShader's normalized output looks better than PyWavelet's, that's a feature worth showing in benchmark figures
- **D-07:** Intensity tracker position is correct — it already receives post-processed CWT output
- **D-08:** Tracker's job is frame-to-frame color consistency: `max(decayed_global_max, frame_99th_percentile)` so the shader's color range doesn't jump between frames
- **D-09:** Evaluate whether tracker parameters need tuning after the kernel normalization fix, but don't over-engineer — current design is sound
- **D-10:** Tests written at each CWT processing stage to guide the fix — not just end-to-end validation
- **D-11:** Test signals: synthetic multi-tone (equal-amplitude sinusoids at spread frequencies, e.g. 100 Hz, 1 kHz, 5 kHz, 10 kHz) + existing chirp signal from project assets
- **D-12:** Tests persist as the incremental test suite (QUAL-02) for future milestones
- **D-13:** "Visually correct" = equal-amplitude input tones produce comparable CWT magnitudes. DAW spectrogram and STFT are the correctness reference

### Claude's Discretion

- Which frequencies to use in multi-tone test signal
- Tolerance bounds for "comparable magnitude" assertion
- Whether `normalize_by_scale` method is removed entirely or kept as a no-op after kernel normalization
- Downsampling method assessment (current fractional-hop approach)
- Benchmark figure regeneration after fix

### Deferred Ideas (OUT OF SCOPE)

- DSP README section on normalization design intuition — Phase 5 (Documentation)
- Per-band intensity tracking (normalize each frequency row independently) — revisit if needed
- EGL headless rendering — v2 milestone
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-01 | CWT normalization produces consistent brightness across frequency bands (investigate low-frequency brightness bias) | Root cause confirmed: kernel L1 norm scales as 1/f. Fix location identified: `wavelet_kernel.py` line 59. Exact normalization factor verified by simulation. |
| QUAL-02 | Pytest unit tests built incrementally as issues surface (not comprehensive upfront suite) | Existing 21-test suite confirmed passing. conftest.py fixtures available. No pytest config needed — pytest discovers tests without it. Test signal design and tolerance bounds calibrated. |
</phase_requirements>

---

## Standard Stack

### Core (No new dependencies needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | installed | FFT, array ops, norm computation | Already in pipeline; L1 norm is `np.sum(np.abs(...))` |
| pytest | >=7.0 (dev dep) | Test runner | Already installed and working; 21 tests pass |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy.fft | (numpy) | Convolution for test signal generation | Unit test: synthesize reference sinusoids |
| soundfile | installed | Load chirp/polyphonic audio for integration tests | Test with real audio assets |

**No new packages required.** The fix and tests use only what is already installed.

## Architecture Patterns

### Pipeline Flow (after fix)

```
wavelet_kernel.py:
  WaveletKernel.__init__()
    self.kernel_t = self.sin_t * self.gauss_t
    # Normalize to unit area: equalizes CWT response across frequencies.
    # Without this, L1 norm ∝ 1/f (wider Gaussian = more samples = more total energy).
    self.kernel_t /= np.sum(np.abs(self.kernel_t))

wavelet.py:
  AntsWavelet.normalize_by_scale()  →  no-op (return cwt_coefs unchanged)
```

### Pattern 1: L1 Kernel Normalization

**What:** Divide `kernel_t` by `np.sum(np.abs(kernel_t))` immediately after constructing it.

**Why L1 (not L2):** The CWT operation is linear convolution. Peak response to a unit sinusoid
at the kernel's center frequency equals `(1/2) * L1_norm` (verified by simulation). L1
normalization makes this constant across all frequencies. L2 normalization does NOT equalize
(post-L1-normalization, L2 = `L2/L1 ∝ sqrt(f)`, still frequency-dependent).

**Verified result:** After L1 normalization, peak response to unit-amplitude sinusoid is
`0.500 ± 0.002` across 100 Hz to 10 kHz (variation < 0.35%).

```python
# Source: verified by direct simulation in wavelet_kernel.py
# After: self.kernel_t = self.sin_t * self.gauss_t
self.kernel_t /= np.sum(np.abs(self.kernel_t))
```

### Pattern 2: Synthetic Multi-Tone Test Signal

**What:** Construct a signal with equal-amplitude sinusoids at widely-spaced frequencies, run
through the full CWT pipeline, assert that maximum magnitude per frequency band is comparable.

**Test frequencies (Claude's discretion):** 100 Hz, 500 Hz, 1000 Hz, 5000 Hz, 10000 Hz —
three decades of range, all well below Nyquist (22050 Hz), matching chromatic scale coverage.

**Tolerance bounds (Claude's discretion):** After normalization, max CWT magnitude per tone
should be within 2x of each other (ratio < 2.0). This is deliberately loose for unit test
robustness while still catching the pre-fix behavior (which showed ~100x bias 100 Hz vs 10 kHz).

```python
# Source: verified by simulation
import numpy as np

sample_rate = 44100
input_n = 16384
test_freqs = [100.0, 500.0, 1000.0, 5000.0, 10000.0]

t = np.arange(input_n) / sample_rate
signal = sum(np.sin(2 * np.pi * f * t) for f in test_freqs) / len(test_freqs)
```

### Pattern 3: Existing Fixture Reuse

```python
# Source: tests/conftest.py (existing)
@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def valid_audio_path():
    return "assets/audio/daw/a2a3_a4_minor_scale.wav"
```

New tests add fixtures to `conftest.py` for `wavelet_config`, `numpy_wavelet` and the synthetic
multi-tone signal. Do not inline these in test files — follow existing fixture pattern.

### Anti-Patterns to Avoid

- **Post-hoc per-frame normalization for bias correction:** The intensity tracker (`ColorNormalizationConfig`) is for frame-to-frame color stability, not frequency bias. Do not add per-frequency normalization in the shader or renderer.
- **Touching `kernel_f_bank` after construction:** The frequency-domain bank is built from `kernel_t` in `AntsWavelet.__init__`. Normalize `kernel_t` before the bank is built, or rebuild the bank after. The cleanest approach normalizes in `WaveletKernel.__init__` so `kernel_t` is always normalized before any caller sees it.
- **L2 normalization of kernel:** L2 does NOT equalize CWT response. Only L1 normalization achieves `constant peak response / amplitude` across frequencies (verified).
- **Adding normalization in `normalize_by_scale` instead of kernel construction:** This would require per-frequency norm values to be stored and passed through — more complexity, same result. Fix at source.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Synthesis of test sinusoids | Custom tone generator class | `np.sin(2 * np.pi * f * t)` inline | One-liner; no abstraction needed |
| CWT correctness reference | STFT comparison in test suite | DAW/STFT figures for visual validation (not automated) | Automated test just checks magnitude ratios, not waveform match |
| Tolerance calibration | Empirical search for magic numbers | Simulation result: ratio should be < 2.0 after fix (pre-fix ratio is ~100) | Research verified bounds |

## Common Pitfalls

### Pitfall 1: Normalizing `kernel_f` Instead of `kernel_t`

**What goes wrong:** Normalizing the frequency-domain kernel doesn't equalize the time-domain
convolution response the same way, because the FFT spreads the norm differently.

**Why it happens:** The frequency-domain bank `kernel_f_bank` is built from `kernel_t` in
`AntsWavelet.__init__` (line 375: `self.kernel_f_bank[i] = fft(w.kernel_t, self.max_conv_n)`).
If `kernel_t` is already normalized before this line, the bank inherits the normalized kernels
automatically. No changes needed in `AntsWavelet`.

**How to avoid:** Normalize `kernel_t` in `WaveletKernel.__init__` after line 59. The bank
is built after, so it inherits the normalized values.

### Pitfall 2: `normalize_by_scale` Left Active After Kernel Fix

**What goes wrong:** If `sqrt(f)` correction is left active alongside kernel L1 normalization,
the output will be over-corrected: high frequencies will be amplified relative to low frequencies.

**Why it happens:** `normalize_by_scale` was the old (insufficient) correction. After kernel
normalization, the CWT output is already equalized. Applying `sqrt(f)` on top introduces a new
bias in the opposite direction.

**How to avoid:** Make `AntsWavelet.normalize_by_scale` a no-op (`return cwt_coefs`).
`PyWavelet.normalize_by_scale` is already a no-op — no change needed there.

**Warning sign:** In the test, high-frequency bands are brighter than low-frequency bands after
fix (inverted bias indicates `sqrt(f)` correction was left on).

### Pitfall 3: Test Running at Full `input_n = 16384` (slow)

**What goes wrong:** Building a full `AntsWavelet` instance for test scaffolding takes seconds
due to generating all chromatic scale kernels and uploading to the kernel bank.

**How to avoid:** For unit tests, test `WaveletKernel` directly (single-kernel instantiation,
fast) or use a `WaveletConfig` with `num_octaves=2` to limit kernel count. The normalization
property can be verified on a single kernel without instantiating the full bank.

### Pitfall 4: `cwt_out_type` Dead Code Removal — Wrong Deletion Target

**What goes wrong:** `cwt_out_type` is defined in `WaveletConfig` (config.py:106) but never
read anywhere. It is safe to delete the dataclass field. Verify with a grep before deleting —
confirm zero references.

**How to avoid:** Run `grep -r "cwt_out_type" src/` before removing. If zero hits outside
`config.py`, delete the field definition. If any hits appear, remove those usages first.

### Pitfall 5: Test Depends on chirp Audio File Path Being Absolute

**What goes wrong:** Tests run from the project root. The `valid_audio_path` fixture returns a
relative path (`assets/audio/daw/a2a3_a4_minor_scale.wav`). If a new test uses a raw string
path without going through `project_root`, the path won't resolve when `pytest` is run from a
subdirectory.

**How to avoid:** Use the `project_root` fixture for any file-based tests. Or use purely
synthetic signals (no file I/O) for unit tests — preferred for speed.

## Code Examples

### Kernel Normalization (the one-line fix)

```python
# Source: wavelet_kernel.py — add immediately after line 59
# self.kernel_t: np.ndarray[np.complex64] = self.sin_t * self.gauss_t

# Normalize to unit area: equalizes CWT response magnitude across frequencies.
# Without this, Gaussian width = num_fwhm_cycles / f, so L1 norm ∝ 1/f.
# A 100 Hz kernel integrates ~100x more energy than a 10 kHz kernel,
# producing disproportionately bright low-frequency bands in the output.
self.kernel_t /= np.sum(np.abs(self.kernel_t))
```

### normalize_by_scale No-Op (AntsWavelet)

```python
# Source: wavelet.py — AntsWavelet.normalize_by_scale
def normalize_by_scale(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.complexfloating]:
    """
    Kernel normalization in WaveletKernel.__init__ corrects the energy bias at
    the source. This method is retained for interface compatibility but is now
    a no-op.
    """
    return cwt_coefs
```

### Test: Kernel Has Unit L1 Norm After Construction

```python
# tests/test_cwt_normalization.py
import numpy as np
import pytest
from subshader.dsp.wavelet_kernel import WaveletKernel

class TestWaveletKernelNormalization:
    def test_kernel_has_unit_l1_norm(self):
        """Kernel L1 norm must be 1.0 after construction."""
        kernel = WaveletKernel(f=1000.0, sample_rate=44100, num_cycles=6, num_fwhm_cycles=3, input_n=16384)
        l1_norm = np.sum(np.abs(kernel.kernel_t))
        assert abs(l1_norm - 1.0) < 1e-5, f"Expected L1 norm ~1.0, got {l1_norm}"
```

### Test: Equal-Amplitude Tones Produce Comparable CWT Magnitude

```python
class TestCwtBrightnessBias:
    def test_equal_amplitude_tones_produce_comparable_magnitudes(self, numpy_wavelet):
        """
        Equal-amplitude sinusoids at spread frequencies must produce CWT magnitudes
        within 2x of each other after normalization (PIPE-01).
        Pre-fix bias was ~100x between 100 Hz and 10 kHz.
        """
        sample_rate = 44100
        input_n = 16384
        test_freqs = [100.0, 500.0, 1000.0, 5000.0, 10000.0]

        t = np.arange(input_n) / sample_rate
        signal = sum(np.sin(2 * np.pi * f * t) for f in test_freqs) / len(test_freqs)

        cwt_output = numpy_wavelet.cwt(signal.astype(np.float64))

        # Find the frequency rows closest to each test tone
        magnitudes = []
        for f in test_freqs:
            freq_idx = np.argmin(np.abs(numpy_wavelet.freqs - f))
            magnitudes.append(np.max(cwt_output[freq_idx, :]))

        max_mag = max(magnitudes)
        min_mag = min(magnitudes)
        ratio = max_mag / (min_mag + 1e-10)

        assert ratio < 2.0, (
            f"Brightness bias too large: max/min magnitude ratio = {ratio:.2f} "
            f"(magnitudes: {[f'{m:.4f}' for m in magnitudes]}). "
            f"Expected < 2.0 after kernel normalization."
        )
```

### Fixture: NumpyWavelet with Reduced Octave Count for Speed

```python
# tests/conftest.py addition
@pytest.fixture
def wavelet_config_small():
    """WaveletConfig with reduced octave count for fast test execution."""
    from subshader.config import WaveletConfig
    config = WaveletConfig()
    config.num_octaves = 4  # 48 notes instead of 120 — covers 100 Hz to ~2 kHz
    return config

@pytest.fixture
def numpy_wavelet():
    """NpWavelet instance covering full chromatic range for normalization tests."""
    from subshader.config import WaveletConfig
    from subshader.dsp.wavelet import NpWavelet
    config = WaveletConfig()
    return NpWavelet(sample_rate=44100, input_n=16384, config=config)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Post-hoc `sqrt(f)` correction in `normalize_by_scale` | Kernel L1 normalization at construction time | Phase 2 | Eliminates root cause instead of partially compensating |
| `cwt_out_type` config field (dead code) | Field removed | Phase 2 | Config is clean; no misleading parameters |

**Deprecated/outdated:**
- `AntsWavelet.normalize_by_scale` applying `* np.sqrt(self.freqs[:, None])`: insufficient — corrects half the bias; becomes no-op after kernel fix.

## Open Questions

1. **Whether to keep `normalize_by_scale` as a no-op or remove the method entirely**
   - What we know: It is an abstract method on `Wavelet` base class; `PyWavelet` already implements it as a no-op; removing it from the abstract interface would require removing from both subclasses
   - What's unclear: Is the abstract method contract valuable for future implementations, or is it clutter?
   - Recommendation: Keep as no-op in `AntsWavelet` (don't remove the abstract method); the interface documents that subclasses must consider scale normalization. Removing the abstract method is a larger refactor with no immediate benefit.

2. **Intensity tracker parameter tuning (D-09)**
   - What we know: `decay_rate=0.001`, `percentile=99`, `warmup=10` are the current values
   - What's unclear: After kernel normalization lowers the absolute magnitude of low-frequency outputs, the tracker's global max may shift; warmup behavior may look different
   - Recommendation: Run manually after fix and observe visually. No automated test for this — it's a subjective assessment.

3. **Benchmark figure regeneration**
   - What we know: `research/benchmark.py` has a `ReadmeFigures` class that generates comparison PNGs
   - What's unclear: Whether benchmark runs cleanly after normalization or requires parameter updates
   - Recommendation: Run `python research/benchmark.py --figures` after fix and inspect output. If chirp figure shows uniform brightness across the frequency sweep, the fix is visually confirmed.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 (installed as dev dependency) |
| Config file | None — pytest discovers `tests/` without config |
| Quick run command | `pytest tests/test_cwt_normalization.py -x` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | Kernel L1 norm equals 1.0 after construction | unit | `pytest tests/test_cwt_normalization.py::TestWaveletKernelNormalization -x` | Wave 0 |
| PIPE-01 | Equal-amplitude tones produce magnitudes within 2x | unit | `pytest tests/test_cwt_normalization.py::TestCwtBrightnessBias -x` | Wave 0 |
| PIPE-01 | `normalize_by_scale` is a no-op (returns input unchanged) | unit | `pytest tests/test_cwt_normalization.py::TestNormalizeByScale -x` | Wave 0 |
| QUAL-02 | All existing tests still pass after normalization fix | regression | `pytest tests/ -q` | Yes (21 tests) |

### Sampling Rate

- **Per task commit:** `pytest tests/test_cwt_normalization.py -x`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green (`pytest tests/ -q` shows 0 failures) before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_cwt_normalization.py` — covers PIPE-01 (kernel norm, magnitude ratio, no-op)
- [ ] `tests/conftest.py` additions — `numpy_wavelet` fixture needed by PIPE-01 tests

*(Existing `conftest.py` exists but lacks wavelet fixtures; additions required, not a new file)*

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `src/subshader/dsp/wavelet_kernel.py` — kernel construction, no normalization at line 59
- Direct code inspection: `src/subshader/dsp/wavelet.py` — `normalize_by_scale` at line 440, pipeline flow lines 150-165
- Direct code inspection: `src/subshader/dsp/gaussian.py` — FWHM formula: `fwhm_support_s = num_fwhm_cycles / f`
- Direct simulation (verified): CWT peak response to unit sinusoid measured across 100 Hz–10 kHz; L1 normalization produces constant 0.500 ± 0.002 response (< 0.35% variation)
- Direct simulation (verified): L1 norm scales exactly as `1/f` (ratio 100 Hz / 10 kHz = 100.3, matching the frequency ratio)
- Test run: `pytest tests/ -q` — 21 tests pass, no config required

### Secondary (MEDIUM confidence)

- CONTEXT.md decisions D-01 through D-13 — user decisions from `/gsd:discuss-phase`, confirmed consistent with code investigation

### Tertiary (LOW confidence)

None — all findings verified against source code and simulation.

---

## Metadata

**Confidence breakdown:**

- Root cause and fix: HIGH — mathematically verified by simulation, exact normalization factor computed
- Test design and tolerance bounds: HIGH — calibrated from simulation (0.35% variation, < 2.0 ratio is safe)
- Infrastructure (pytest, fixtures): HIGH — confirmed by running `pytest tests/ -q` (21 passed)
- Intensity tracker tuning: LOW — observation-dependent, deferred to post-fix manual check

**Research date:** 2026-03-21
**Valid until:** 2026-06-21 (stable domain; numpy/pytest APIs are stable; 90 days conservative)
