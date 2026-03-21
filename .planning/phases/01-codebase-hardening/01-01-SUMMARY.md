---
phase: 01-codebase-hardening
plan: 01
subsystem: testing
tags: [pytest, exceptions, cupy, gpu, config]

# Dependency graph
requires: []
provides:
  - "Single source of truth for exception hierarchy in exceptions.py"
  - "gpu_available() utility in src/subshader/utils/gpu.py"
  - "Correct AudioConfig default file path"
  - "Clean wavelet_kernel.py without dead cupy import"
  - "Test infrastructure with pytest and 11 passing exception tests"
affects: [02-codebase-hardening, dsp, audio, config]

# Tech tracking
tech-stack:
  added: ["pytest>=7.0"]
  patterns:
    - "Canonical exception imports: all modules import from subshader.exceptions"
    - "Lazy GPU detection: cupy imported inside try/except in gpu_available()"
    - "TDD for infrastructure: tests written alongside code changes"

key-files:
  created:
    - "src/subshader/utils/gpu.py"
    - "tests/__init__.py"
    - "tests/conftest.py"
    - "tests/test_exceptions.py"
  modified:
    - "src/subshader/exceptions.py"
    - "src/subshader/audio/audio_input.py"
    - "src/subshader/config.py"
    - "src/subshader/dsp/wavelet_kernel.py"
    - "src/subshader/utils/__init__.py"
    - "pyproject.toml"

key-decisions:
  - "RuntimeError removed from GRACEFUL_EXCEPTIONS — was masking real errors; SubShaderException + KeyboardInterrupt is the correct scope"
  - "gpu_available() uses lazy cupy import inside try/except to avoid import-time failures on CPU-only machines"
  - "AudioConfig default path changed to assets/audio/daw/a2a3_a4_minor_scale.wav — matches the file the override in __main__.py was forcing"

patterns-established:
  - "Exception imports: from subshader.exceptions import ExceptionClass (never define locally)"
  - "GPU detection: from subshader.utils import gpu_available (never inline cupy try/except)"

requirements-completed: [QUAL-01, QUAL-03]

# Metrics
duration: 2min
completed: 2026-03-21
---

# Phase 01 Plan 01: Codebase Foundation Cleanup Summary

**Exception hierarchy deduplicated to single source of truth, gpu_available() utility created with lazy CuPy import, config default fixed, and 11-test pytest suite passing green**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-21T15:55:53Z
- **Completed:** 2026-03-21T15:57:52Z
- **Tasks:** 2 completed
- **Files modified:** 10

## Accomplishments

- Removed duplicate `AudioFileNotFoundError` and `EndOfAudioException` classes from `audio_input.py`; now imports canonically from `subshader.exceptions`
- Created `src/subshader/utils/gpu.py` with `gpu_available()` using lazy CuPy import — safe on CPU-only machines
- Narrowed `GRACEFUL_EXCEPTIONS` by removing `RuntimeError`, which was silently swallowing real errors
- Fixed `AudioConfig.file_path` default to the correct test file without needing `__main__.py` override
- Removed dead `import cupy as cp` from `wavelet_kernel.py` (was never used in that file)
- Set up pytest infrastructure with `conftest.py` fixtures and 11 exception hierarchy tests all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Consolidate exceptions, fix config default, remove dead code, create gpu utility** - `3722576` (feat)
2. **Task 2: Set up test infrastructure and write exception hierarchy tests** - `0c15422` (test)

## Files Created/Modified

- `src/subshader/utils/gpu.py` — New GPU detection utility with `gpu_available()` function
- `src/subshader/exceptions.py` — Removed RuntimeError from GRACEFUL_EXCEPTIONS and ExceptionReporter
- `src/subshader/audio/audio_input.py` — Removed local exception classes, added canonical import
- `src/subshader/config.py` — Fixed AudioConfig default file_path
- `src/subshader/dsp/wavelet_kernel.py` — Removed dead `import cupy as cp`
- `src/subshader/utils/__init__.py` — Added `gpu_available` export
- `pyproject.toml` — Added `[project.optional-dependencies] dev = ["pytest>=7.0"]`
- `tests/__init__.py` — Package marker (empty)
- `tests/conftest.py` — Shared fixtures: `project_root`, `valid_audio_path`
- `tests/test_exceptions.py` — 11 tests across 4 test classes

## Decisions Made

- Removed `RuntimeError` from `GRACEFUL_EXCEPTIONS`: was masking unexpected errors. The else branch in `ExceptionReporter.report()` already handles unknown exceptions. Keeping the tuple narrow prevents silent failures.
- Lazy CuPy import in `gpu_available()`: importing at module level would crash the entire utils package on CPU-only machines. Lazy import inside try/except is the correct pattern for optional GPU dependencies.
- `AudioConfig` default path fixed to `assets/audio/daw/a2a3_a4_minor_scale.wav`: the `__main__.py` override existed solely to compensate for the wrong default. Fixing the default is the right fix; the override will be cleaned up in Plan 02.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02 can now import `gpu_available` from `subshader.utils` and use it in DSP init
- `__main__.py` file_path override is now redundant and can be removed in Plan 02
- Test infrastructure is in place; new tests can be added to `tests/` without setup work
- All 11 tests pass: `python -m pytest tests/test_exceptions.py -v`

---
*Phase: 01-codebase-hardening*
*Completed: 2026-03-21*

## Self-Check: PASSED

All created files exist on disk. Both task commits (3722576, 0c15422) verified in git log.
