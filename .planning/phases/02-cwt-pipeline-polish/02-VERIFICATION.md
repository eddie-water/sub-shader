---
phase: 02-cwt-pipeline-polish
verified: 2026-03-21T23:09:37Z
status: human_needed
score: 7/8 must-haves verified
human_verification:
  - test: "Open assets/images/benchmarks/chirp_signal_comparison.png"
    expected: "SubShader CWT panel shows approximately uniform brightness across the full 200 Hz to 20 kHz sweep — no decay at high frequencies"
    why_human: "Visual quality of a PNG cannot be verified programmatically; the file exists and was generated from normalized kernels, but the perceptual uniformity requires human inspection"
  - test: "Open assets/images/benchmarks/polyphonic_signal_comparison.png"
    expected: "Equal-amplitude A3/A4/A5 tones appear at comparable brightness levels across frequency bands"
    why_human: "Same as above — perceptual brightness balance in a figure requires human sign-off"
---

# Phase 02: CWT Pipeline Polish Verification Report

**Phase Goal:** CWT output looks visually correct across all frequency bands and the fix is covered by a test
**Verified:** 2026-03-21T23:09:37Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

Plan 01 truths (from must_haves frontmatter):

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | WaveletKernel.kernel_t has unit L1 norm after construction at any frequency | VERIFIED | `wavelet_kernel.py` line 65: `self.kernel_t /= np.sum(np.abs(self.kernel_t))`; `test_kernel_has_unit_l1_norm` and 3 parametrized frequency tests all PASS |
| 2  | Equal-amplitude sinusoids at 100 Hz through 10 kHz produce CWT magnitudes within 2x of each other | VERIFIED | `test_equal_amplitude_tones_produce_comparable_magnitudes` PASSES — 6/6 normalization tests green |
| 3  | AntsWavelet.normalize_by_scale returns its input unchanged (no-op) | VERIFIED | `wavelet.py` line 437: `return cwt_coefs` with no multiplication; `test_normalize_by_scale_is_noop` PASSES |
| 4  | cwt_out_type field no longer exists in WaveletConfig | VERIFIED | `grep -r "cwt_out_type" src/` returns zero matches |
| 5  | All existing tests still pass after the normalization fix | VERIFIED | `pytest tests/ -q` — 27/27 passed (21 pre-existing + 6 new) |

Plan 02 truths (from must_haves frontmatter):

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 6  | Chirp signal comparison figure shows uniform brightness across the frequency sweep (no high-frequency decay) | NEEDS HUMAN | `chirp_signal_comparison.png` exists (793 KB, 2026-03-21 18:58) and was generated from L1-normalized kernels; visual uniformity cannot be confirmed programmatically |
| 7  | Polyphonic signal comparison figure shows comparable intensity across tones | NEEDS HUMAN | `polyphonic_signal_comparison.png` exists (1.4 MB, 2026-03-21 19:06) and was generated from normalized kernels with a real polyphonic audio file; perceptual balance requires human sign-off |
| 8  | Benchmark figure generation completes without errors | VERIFIED | Commit `84175d5` exists and documents successful generation; both PNGs present on disk at expected sizes |

**Score:** 6/8 truths verified automatically, 2/8 require human visual inspection

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_cwt_normalization.py` | CWT normalization test suite | VERIFIED | 121 lines, 3 test classes, 6 test methods, all PASS |
| `src/subshader/dsp/wavelet_kernel.py` | L1-normalized kernel construction | VERIFIED | Line 65: `self.kernel_t /= np.sum(np.abs(self.kernel_t))`; comment block lines 61-64 present |
| `src/subshader/dsp/wavelet.py` | No-op normalize_by_scale for AntsWavelet | VERIFIED | Lines 425-437: method present, returns `cwt_coefs` unchanged, docstring states "no-op" |
| `assets/images/benchmarks/chirp_signal_comparison.png` | Updated chirp comparison with normalized kernels | VERIFIED (exists) | 793 KB, timestamp 2026-03-21 18:58 |
| `assets/images/benchmarks/polyphonic_signal_comparison.png` | Updated polyphonic comparison with normalized kernels | VERIFIED (exists) | 1.4 MB, timestamp 2026-03-21 19:06 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/subshader/dsp/wavelet_kernel.py` | `src/subshader/dsp/wavelet.py` | AntsWavelet builds `kernel_f_bank` from `WaveletKernel.kernel_t` | WIRED | `wavelet.py` line 376: `self.kernel_f_bank[i] = fft(w.kernel_t, self.max_conv_n)` — normalized `kernel_t` is consumed at bank construction, so all convolutions inherit normalization |
| `tests/test_cwt_normalization.py` | `src/subshader/dsp/wavelet_kernel.py` | imports WaveletKernel and asserts L1 norm | WIRED | Line 12: `from subshader.dsp.wavelet_kernel import WaveletKernel`; line 33: asserts `abs(l1_norm - 1.0) < 1e-5` |
| `research/benchmark.py` | `src/subshader/dsp/wavelet.py` | imports NumPyWavelet for figure generation | WIRED | Line 43: `from subshader.dsp.wavelet import PyWavelet, NumPyWavelet` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PIPE-01 | 02-01, 02-02 | CWT normalization produces consistent brightness across frequency bands | SATISFIED | L1 normalization in `wavelet_kernel.py` + `test_equal_amplitude_tones_produce_comparable_magnitudes` (ratio < 2.0 PASSES) + updated benchmark figures |
| QUAL-02 | 02-01, 02-02 | Pytest unit tests built incrementally as issues surface | SATISFIED | 6 new tests in `test_cwt_normalization.py`; full suite 27/27; TDD workflow followed (RED commit `7882f0d`, GREEN commit `4851c3c`) |

No orphaned requirements found — REQUIREMENTS.md lists both PIPE-01 and QUAL-02 as Phase 2 / Complete.

---

## Anti-Patterns Found

No blockers or warnings. Scan of modified files:

- `tests/test_cwt_normalization.py` — no TODOs, no stubs, no placeholder returns; all test logic substantive
- `src/subshader/dsp/wavelet_kernel.py` — `_plot_kernel` method body is `pass` (lines 79-94), but the entire method body is commented-out matplotlib calls followed by `pass`. This is a pre-existing debug visualization stub that predates this phase and does not affect pipeline correctness.
- `src/subshader/dsp/wavelet.py` — no new stubs introduced; `normalize_by_scale` intentionally returns input unchanged (documented no-op, not an accidental empty implementation)
- `src/subshader/config.py` — dead `cwt_out_type` field confirmed absent

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `wavelet_kernel.py` | 94 | `_plot_kernel` is `pass` after commented code | Info | Pre-existing; private debug method; zero pipeline impact |

---

## Human Verification Required

### 1. Chirp comparison: uniform brightness across frequency sweep

**Test:** Open `/home/eddie-water/dev/python/sub-shader/assets/images/benchmarks/chirp_signal_comparison.png`
**Expected:** The SubShader CWT panel shows approximately uniform brightness as the sweep moves from 200 Hz to 20 kHz — no decay or dimming at high frequencies. Brightness distribution should be comparable to the STFT reference panel.
**Why human:** Visual uniformity in a rendered figure is a perceptual judgment. The file exists and was generated from L1-normalized kernels (verified), but whether the result looks correct requires human inspection.

### 2. Polyphonic comparison: comparable intensity across tones

**Test:** Open `/home/eddie-water/dev/python/sub-shader/assets/images/benchmarks/polyphonic_signal_comparison.png`
**Expected:** A3, A4, and A5 tones appear at comparable brightness levels. No single frequency band dominates.
**Why human:** Same reason — perceptual balance of rendered frequency bands requires visual sign-off.

---

## Gaps Summary

No gaps. All automatable must-haves are verified. The two items requiring human review are benchmark figure aesthetics — the underlying DSP correctness (kernel normalization, test coverage, dead code removal) is fully confirmed.

Commits documented in SUMMARY.md all exist in the repository:
- `7882f0d` — TDD RED: failing normalization tests
- `4851c3c` — TDD GREEN: normalization fix applied
- `84175d5` — benchmark figures regenerated

27/27 tests pass. PIPE-01 and QUAL-02 satisfied.

---

_Verified: 2026-03-21T23:09:37Z_
_Verifier: Claude (gsd-verifier)_
