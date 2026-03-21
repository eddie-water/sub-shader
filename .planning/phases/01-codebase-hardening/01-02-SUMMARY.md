---
phase: 01-codebase-hardening
plan: 02
subsystem: dsp, viz, testing
tags: [cupy, gpu-fallback, opengl, pytest, wavelet, plotter]

# Dependency graph
requires:
  - phase: 01-codebase-hardening
    plan: 01
    provides: "gpu_available() in utils/gpu.py; canonical exceptions; AudioConfig default path fix"
provides:
  - "Guarded CuPy imports with _CUPY_AVAILABLE flag in wavelet.py"
  - "GPU fallback wired into SubShader.__init__ via gpu_available()"
  - "Silent render failure eliminated — render_graphic re-raises exceptions"
  - "Texture validation raises ValueError instead of returning None"
  - "Cleanup safe on partial init via hasattr guards"
  - "Console output enabled for startup errors"
  - "Tests for GPU fallback and plotter validation"
affects: [02-audio-sync, 03-documentation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "try/except guard pattern for optional GPU imports with _CUPY_AVAILABLE flag"
    - "SubShader.__init__ takes explicit config parameter — no global state mutation"
    - "validate-and-raise pattern: validation methods raise ValueError, callers don't check return values"
    - "hasattr guards in cleanup() for partial initialization safety"

key-files:
  created:
    - tests/test_gpu_fallback.py
    - tests/test_plotter.py
  modified:
    - src/subshader/dsp/wavelet.py
    - src/subshader/__main__.py
    - src/subshader/viz/plotter.py

key-decisions:
  - "_validate_texture_data returns None on success (raises on failure) — callers no longer need to check return value"
  - "render_graphic logs then re-raises so frame failures crash loudly instead of being silently dropped"
  - "CuPyWavelet.__init__ raises RuntimeError on CuPy absence — prevents misconfigured GPU runs"
  - "SubShader.__init__ takes ProcessingConfig parameter — removes implicit global config dependency"

patterns-established:
  - "Guard optional GPU imports: try/except at module level with _CUPY_AVAILABLE flag"
  - "Validate-and-raise: error branches raise ValueError, not bare return; callers skip null checks"
  - "hasattr guards in cleanup: safe for partial init across all resource holders"

requirements-completed: [PIPE-02, PIPE-03, QUAL-01, QUAL-03]

# Metrics
duration: 4min
completed: 2026-03-21
---

# Phase 01 Plan 02: Codebase Hardening — GPU Fallback and Silent Failure Elimination

**GPU fallback in SubShader.__init__ via gpu_available(), CuPy imports guarded with _CUPY_AVAILABLE flag, and all silent failure paths in plotter.py replaced with explicit raises — 21 tests passing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-21T15:20:08Z
- **Completed:** 2026-03-21T15:24:08Z
- **Tasks:** 2
- **Files modified:** 5 (3 source, 2 test)

## Accomplishments

- GPU detection wired into SubShader.__init__: selects CuWavelet or NpWavelet at startup, logs clear warning on CPU-only fallback
- CuPy imports guarded at module level — wavelet.py imports cleanly even without CuPy installed
- All silent failure paths in plotter.py eliminated: _validate_texture_data raises ValueError, render_graphic re-raises exceptions
- Removed duplicate WindowCloseException from plotter.py — now imports from subshader.exceptions
- Console logging enabled so startup errors are visible to users
- Full test suite: 21 tests passing (test_exceptions, test_gpu_fallback, test_plotter)

## Task Commits

Each task was committed atomically:

1. **Task 1: Guard CuPy imports, wire GPU fallback, fix cleanup** - `044bf2c` (feat)
2. **Task 2 RED: Failing tests for plotter and GPU fallback** - `4079876` (test)
3. **Task 2 GREEN: Fix plotter silent failures and pass tests** - `3021a2f` (feat)

_Note: TDD tasks have multiple commits (test RED → feat GREEN)_

## Files Created/Modified

- `src/subshader/dsp/wavelet.py` — _CUPY_AVAILABLE guard, CuPyWavelet.__init__ guard, cleanup uses log.warning
- `src/subshader/__main__.py` — GPU fallback selection, config param, console logging, hasattr cleanup guards
- `src/subshader/viz/plotter.py` — _validate_texture_data raises ValueError, render_graphic re-raises, removed local WindowCloseException
- `tests/test_gpu_fallback.py` — TestGpuAvailable, TestGpuFallback (3 tests)
- `tests/test_plotter.py` — TestValidateTextureData, TestRenderGraphic (5 tests)

## Decisions Made

- `_validate_texture_data` changed from `-> bool` to `-> None` (raises on error) — callers now call it unconditionally without checking return value
- `render_graphic` keeps `log.error` before re-raising — log for context, crash for correctness
- `CuPyWavelet.__init__` raises `RuntimeError` on CuPy absence — blocks misconfigured GPU instantiation at init time
- `SubShader.__init__` now takes `config: ProcessingConfig` param — removes implicit global state dependency, enables testing without global mutation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test helper bound _validate_texture_data to Renderer, not ShaderPlot**
- **Found during:** Task 2 RED (test writing)
- **Issue:** Plan's test code referenced `ShaderPlot._validate_texture_data` but the method lives on `Renderer`
- **Fix:** Updated test helper to bind from `Renderer._validate_texture_data`, added `ctx` and `TEXTURE_SLOT` to render test mock
- **Files modified:** tests/test_plotter.py
- **Verification:** All 5 plotter tests pass
- **Committed in:** `3021a2f` (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug in test code from plan)
**Impact on plan:** Required fix to test code — production code was correct as specified. No scope creep.

## Issues Encountered

None — all production code changes proceeded as planned. One test code correction needed (plotter method on Renderer not ShaderPlot).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Pipeline hardening complete: GPU fallback, clean imports, no silent failures, safe cleanup
- 21 tests passing: exceptions, GPU fallback, texture validation
- Ready for Phase 02 (audio sync work) — pipeline is now loud on failures, making audio sync debugging easier

---
*Phase: 01-codebase-hardening*
*Completed: 2026-03-21*
