---
phase: 02-cwt-pipeline-polish
plan: "01"
subsystem: dsp
tags: [cwt, normalization, tdd, bug-fix, wavelet]
dependency_graph:
  requires: []
  provides: [normalized-cwt-kernels, cwt-normalization-tests]
  affects: [src/subshader/dsp/wavelet_kernel.py, src/subshader/dsp/wavelet.py, src/subshader/config.py]
tech_stack:
  added: []
  patterns: [TDD red-green, L1 normalization, no-op interface compatibility]
key_files:
  created:
    - tests/test_cwt_normalization.py
  modified:
    - tests/conftest.py
    - src/subshader/dsp/wavelet_kernel.py
    - src/subshader/dsp/wavelet.py
    - src/subshader/config.py
decisions:
  - "L1 kernel normalization applied at WaveletKernel construction, not post-hoc in normalize_by_scale"
  - "normalize_by_scale retained as no-op for interface compatibility with PyWavelet and future backends"
  - "cwt_out_type field removed — confirmed zero references outside config.py"
metrics:
  duration: "~8 minutes"
  completed: "2026-03-21"
  tasks_completed: 2
  files_modified: 5
---

# Phase 02 Plan 01: CWT Kernel Normalization Summary

**One-liner:** L1-normalized Morlet wavelet kernels eliminate low-frequency brightness bias; verified by 6 new tests; dead config field removed.

## What Was Done

Fixed the root cause of disproportionately bright low-frequency CWT bands. Without normalization, the Gaussian width of each wavelet kernel is `num_fwhm_cycles / f`, so the L1 norm scales as `1/f`. A 100 Hz kernel integrates ~100x more energy than a 10 kHz kernel, flooding low-frequency rows in the visualization output.

The fix divides each `kernel_t` by its own L1 norm immediately after construction in `WaveletKernel.__init__`. This produces unit-area kernels at every frequency, making CWT response constant across the full range.

The `normalize_by_scale` post-processing step in `AntsWavelet` was previously compensating for this with `* sqrt(freq)`, which was an approximate correction that didn't fully cancel the bias. It is now a no-op, and the interface is preserved for downstream subclasses.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Write failing tests (TDD RED) | 7882f0d |
| 2 | Apply L1 normalization, make normalize_by_scale no-op, remove dead code | 4851c3c |

## Verification Results

All plan verification checks passed:

1. `pytest tests/test_cwt_normalization.py -v` — 6/6 normalization tests pass
2. `pytest tests/ -q` — 27/27 total tests pass (21 existing + 6 new)
3. `grep -r "cwt_out_type" src/` — no matches (dead code removed)
4. `grep "np.sqrt(self.freqs" src/subshader/dsp/wavelet.py` — no matches (sqrt correction removed)
5. `grep "np.sum(np.abs(self.kernel_t))" src/subshader/dsp/wavelet_kernel.py` — match present

## Decisions Made

**L1 normalization at construction, not post-hoc:** The bias is structural — it comes from the kernel shape. Fixing it at the source in `WaveletKernel.__init__` is cleaner than a post-processing scale factor that only approximately cancels it. Post-hoc approaches also require the caller to know the correction was needed.

**normalize_by_scale retained as no-op:** Removing the method would break the abstract interface and force changes in `PyWavelet` and any future backend. Keeping it as a no-op with a clear docstring costs nothing and preserves the interface contract.

**cwt_out_type removed:** Confirmed by grep that this field was never read outside of its definition in `config.py`. It was dead code from an earlier design that was never wired up.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all modified components are fully functional. No placeholder values introduced.
