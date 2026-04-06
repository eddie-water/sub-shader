---
phase: 08-codebase-refactoring-and-module-cleanup
plan: 02
subsystem: dsp
tags: [refactoring, abc, dsp, cwt, stft, pywavelet, module-structure]

requires:
  - phase: 08-01
    provides: CWTConfig/PipelineConfig hierarchy; WaveletConfig removed

provides:
  - DSP ABC in dsp.py with pre/transform/post abstract methods and process() orchestrator
  - CWT base class (ANTS algorithm, chromatic scale, reliable slice) in cwt.py
  - CpuCWT(CWT) — NumPy FFT convolution backend
  - GpuCWT(CWT) — CuPy FFT convolution backend with GPU memory lifecycle
  - PywaveletCWT(DSP) — pywt.cwt() wrapper with sqrt(scale) normalization
  - STFT(DSP) — scipy.signal.stft with log-frequency interpolation to CWT grid
  - dsp/__init__.py — exports all backends plus NpWavelet/CuWavelet/NumPyWavelet/CuPyWavelet aliases
  - WaveletConfig = CWTConfig alias in config.py for wavelet.py import-time compat

affects:
  - 08-03 (renderer — output_shape from CpuCWT/GpuCWT used to size frame buffer)
  - 08-05 (__main__.py migration — switches from old CuWavelet to GpuCWT)
  - research/ comparison harness (timing.py uses CpuCWT/GpuCWT/PywaveletCWT/STFT)

tech-stack:
  added: []
  patterns:
    - DSP ABC with pre/transform/post pipeline — all backends composable via process()
    - CWT base consolidates 4 old classes (Wavelet, AntsWavelet, NumPyWavelet, CuPyWavelet) into 3 (CWT, CpuCWT, GpuCWT)
    - Deprecated aliases (NpWavelet=CpuCWT, CuWavelet=GpuCWT) enable gradual caller migration
    - WaveletConfig=CWTConfig shim preserves wavelet.py at import-time without modifying it

key-files:
  created:
    - src/subshader/dsp/dsp.py (DSP ABC)
    - src/subshader/dsp/cwt.py (CWT + CpuCWT + GpuCWT)
    - src/subshader/dsp/pywavelet.py (PywaveletCWT)
    - src/subshader/dsp/stft.py (STFT)
    - src/subshader/dsp/__init__.py (module exports + deprecated aliases)
  modified:
    - src/subshader/config.py (WaveletConfig = CWTConfig alias added)

key-decisions:
  - "CWT base class absorbs Wavelet + AntsWavelet shared logic — 7 wavelet.py classes flattened to 3 in cwt.py"
  - "PywaveletCWT and STFT pre/post are stubs (D-14) — comparison backends need transform() only"
  - "WaveletConfig=CWTConfig alias added to config.py to keep wavelet.py importable without modification"

patterns-established:
  - "DSP ABC: pre/transform/post pipeline with process() orchestrator — all backends use this interface"
  - "Deprecated alias pattern: old class names resolve to new names, removed after Plan 08-05 migration"

requirements-completed: [D-08, D-09, D-10, D-11, D-12, D-13, D-14]

duration: 3min
completed: 2026-04-06
---

# Phase 8 Plan 2: DSP Module Restructure Summary

**DSP ABC hierarchy created — 7-class wavelet.py flattened to 3 classes in cwt.py, PywaveletCWT and STFT extracted as standalone backends, all inheriting from DSP with pre/transform/post pipeline.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-04-06T23:11:19Z
- **Completed:** 2026-04-06T23:14:34Z
- **Tasks:** 2
- **Files modified:** 6 files created/modified

## Accomplishments

- Created `dsp.py` with `DSP(ABC)` defining `pre()`, `transform()`, `post()` as abstract methods and `process()` as the concrete orchestrator
- Created `cwt.py` flattening the 4-level hierarchy (Wavelet → AntsWavelet → NumPyWavelet/CuPyWavelet) into `CWT` base + `CpuCWT`/`GpuCWT` concrete classes — all ANTS algorithm logic consolidated in CWT
- Created `pywavelet.py` extracting the pywt.cwt() scale construction and sqrt(scale) normalization from old PyWavelet class into `PywaveletCWT(DSP)`
- Created `stft.py` with `STFT(DSP)` implementing the compute_stft_frame logic from research/utilities/dsp_helpers.py with log-frequency interpolation
- Created `dsp/__init__.py` exporting all backends with `NpWavelet=CpuCWT`, `CuWavelet=GpuCWT`, `NumPyWavelet=CpuCWT`, `CuPyWavelet=GpuCWT` deprecated aliases
- Added `WaveletConfig = CWTConfig` alias to `config.py` so old `wavelet.py` remains importable

## Task Commits

1. **Task 1: Create DSP ABC + cwt.py + pywavelet.py + stft.py** — `49597e3` (feat)
2. **Task 2: Update dsp/__init__.py exports and add backward-compat aliases** — `d131e24` (feat)

## Files Created/Modified

- `/home/eddie-water/dev/python/sub-shader/src/subshader/dsp/dsp.py` — DSP ABC with process() orchestrator
- `/home/eddie-water/dev/python/sub-shader/src/subshader/dsp/cwt.py` — CWT base + CpuCWT + GpuCWT (flattened from 4 old classes)
- `/home/eddie-water/dev/python/sub-shader/src/subshader/dsp/pywavelet.py` — PywaveletCWT with pywt.cwt() + sqrt(scale) normalization
- `/home/eddie-water/dev/python/sub-shader/src/subshader/dsp/stft.py` — STFT backend with log-frequency interpolation
- `/home/eddie-water/dev/python/sub-shader/src/subshader/dsp/__init__.py` — module exports + deprecated class name aliases
- `/home/eddie-water/dev/python/sub-shader/src/subshader/config.py` — WaveletConfig alias added

## Decisions Made

- `WaveletConfig=CWTConfig` alias rather than touching wavelet.py — Plan 08-05 handles the full wavelet.py migration; keeping it importable without modification reduces diff scope
- STFT backend inlines the `compute_stft_frame` logic from `research/utilities/dsp_helpers.py` rather than importing it — production code must not depend on research utilities
- `pre()` and `post()` in PywaveletCWT and STFT are pass-through stubs per D-14 — these comparison backends only need `transform()`; full pre/post pipeline is CWT-specific

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed wavelet.py backward-compat import failure**
- **Found during:** Task 2 verification (`from subshader.dsp.wavelet import CuWavelet`)
- **Issue:** `wavelet.py` imports `WaveletConfig` from `config.py` which was removed in Plan 08-01. Module fails to import, breaking the backward-compat requirement stated in the verification criteria.
- **Fix:** Added `WaveletConfig = CWTConfig` alias to `config.py` alongside the existing `ProcessingConfig = CWTConfig` alias. wavelet.py imports without modification; runtime field access (`config.typical_sampling_freq`, `config.root_note_a0_hz`) will fail on instantiation, which is acceptable — Plan 08-05 completes the migration.
- **Files modified:** `src/subshader/config.py`
- **Commit:** `d131e24` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — broken import caused by Plan 08-01 removing WaveletConfig)
**Impact on plan:** Necessary fix. The backward-compat import check is in the plan's verification criteria.

## Known Stubs

- `PywaveletCWT.pre()` — pass-through; intentional per D-14. PywaveletCWT is comparison-only.
- `PywaveletCWT.post()` — pass-through; intentional per D-14.
- `STFT.pre()` — pass-through; intentional per D-14. STFT is comparison-only.
- `STFT.post()` — pass-through; intentional per D-14.

These stubs are intentional. The plan explicitly specifies pre/post as stubs for comparison backends (D-14). The comparison harness calls `transform()` directly. Full pre/post pipelines are defined only for CWT backends.

## Issues Encountered

None beyond the auto-fixed deviation above.

## User Setup Required

None.

## Next Phase Readiness

- `GpuCWT(config)` and `CpuCWT(config)` can be instantiated directly with a `CWTConfig`
- `from subshader.dsp import GpuCWT, NpWavelet` works for any caller migrating incrementally
- `from subshader.dsp.wavelet import CuWavelet` still works (import-time compat only)
- Plan 08-03 (renderer) can read `dsp.output_shape` from any CWT backend instance
- Plan 08-05 (__main__.py) will migrate `CuWavelet` → `GpuCWT` and remove deprecated aliases

---
*Phase: 08-codebase-refactoring-and-module-cleanup*
*Completed: 2026-04-06*
