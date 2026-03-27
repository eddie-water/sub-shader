---
phase: 07-visual-style-system-and-frequency-range-configuration
plan: 02
subsystem: dsp
tags: [python, decorator, timing, profiling, wavelet, cwt]

requires:
  - phase: 05.2-benchmark-timing-profiling-and-comparison-grid-polish
    provides: cwt_timed() duplicate code path that is now being replaced

provides:
  - "@timed decorator in src/subshader/utils/timing.py"
  - "All 6 wavelet pipeline stages decorated with @timed (timing always available)"
  - "cwt_timed() removed from wavelet.py"
  - "research/timing.py is a thin reporting layer reading _timing_*_ms instance attributes"

affects:
  - research/timing.py
  - any future profiling code reading pipeline timing

tech-stack:
  added: []
  patterns:
    - "@timed decorator pattern: wraps instance methods, stores elapsed ms as _timing_{name}_ms on self"
    - "Timing-always-available pattern: no separate timed code path needed"

key-files:
  created:
    - src/subshader/utils/timing.py
  modified:
    - src/subshader/dsp/wavelet.py
    - src/subshader/utils/__init__.py
    - research/timing.py

key-decisions:
  - "@timed placed on concrete overrides (AntsWavelet, PyWavelet subclasses), not abstract declarations — decorating abstract methods doesn't wrap the overrides"
  - "import time removed from wavelet.py after cwt_timed() deletion — timed decorator owns the time.perf_counter calls"
  - "timed exported from src/subshader/utils/__init__.py for consistent import pattern"

patterns-established:
  - "Pipeline timing via @timed decorator: decorate the method, read _timing_{name}_ms after the call — no parallel timed code path needed"

requirements-completed: [TIM-01, TIM-02, TIM-03, TIM-04]

duration: 4min
completed: 2026-03-27
---

# Phase 07 Plan 02: @timed Decorator and cwt_timed() Removal Summary

**@timed decorator added to src/subshader/utils/timing.py; applied to all 6 wavelet pipeline stages; cwt_timed() duplicate code path deleted; research/timing.py refactored to read _timing_*_ms instance attributes**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-27T04:55:09Z
- **Completed:** 2026-03-27T04:59:13Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `src/subshader/utils/timing.py` with the `@timed` decorator — ~1 microsecond overhead, stores elapsed ms as `self._timing_{method_name}_ms`
- Decorated `cwt()` and all 6 pipeline stage methods (`class_specific_cwt`, `normalize_by_scale`, `compute_mag`, `discard_unreliable_coefs`, `extract_hop_center`, `downsample`) across all concrete subclasses
- Deleted `cwt_timed()` from `wavelet.py` (44 lines of duplicate pipeline code gone)
- Refactored `research/timing.py`: `TimedSubShader.run()` now calls `wavelet.cwt()` and reads `_timing_*_ms` attributes — thin reporting layer, no parallel pipeline variant needed

## Task Commits

Each task was committed atomically:

1. **Task 1: Create @timed decorator and apply to wavelet pipeline stages** - `e05aa45` (feat)
2. **Task 2: Refactor research/timing.py to read timing from decorated method attributes** - `87100fd` (refactor)

**Plan metadata:** (docs commit pending)

## Files Created/Modified

- `src/subshader/utils/timing.py` — New: `@timed` decorator using `functools.wraps` + `time.perf_counter`
- `src/subshader/dsp/wavelet.py` — Added `from subshader.utils.timing import timed`; applied `@timed` to 11 method definitions across `Wavelet`, `AntsWavelet`, `PyWavelet`, `NumPyWavelet`, `CuPyWavelet`; removed `cwt_timed()` and `import time`
- `src/subshader/utils/__init__.py` — Added `timed` to imports and `__all__`
- `research/timing.py` — Replaced `wavelet.cwt_timed(audio_data)` + dict iteration with `wavelet.cwt(audio_data)` + 6 direct attribute reads

## Decisions Made

- `@timed` applied to concrete overrides, not abstract base class declarations — decorating abstract methods in Python ABCs does not wrap concrete override implementations; each subclass that implements the method needs its own `@timed`
- `import time` removed from `wavelet.py` since `cwt_timed()` was its only caller; the decorator owns all `perf_counter` calls now
- `timed` exported from `src/subshader/utils/__init__.py` following the existing export pattern for utility functions

## Deviations from Plan

None — plan executed exactly as written, with one implementation detail the plan noted correctly: decorating abstract methods would not affect concrete overrides, so `@timed` was applied to the concrete implementations in `AntsWavelet`, `PyWavelet`, `NumPyWavelet`, and `CuPyWavelet`.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `@timed` decorator available for any future pipeline method profiling
- Timing data always on the instance after each `cwt()` call — no separate invocation needed
- `cwt_timed()` fully removed; callers using it elsewhere would break (none found in the codebase)

---
*Phase: 07-visual-style-system-and-frequency-range-configuration*
*Completed: 2026-03-27*
