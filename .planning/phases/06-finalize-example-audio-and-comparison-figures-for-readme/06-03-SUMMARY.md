---
phase: 06-finalize-example-audio-and-comparison-figures-for-readme
plan: 03
subsystem: research
tags: [benchmark, timing, comparison-grid, cwt, stft, pywavelet]

# Dependency graph
requires:
  - phase: 06-01
    provides: generate_comparison_grid function, bouncing chirp, TimingAccumulator infrastructure
provides:
  - Overlap-correct duration_s formula in generate_comparison_grid
  - --comparison flag with per-method timing stats (STFT/PyWavelet/SubShader)
  - _STUB_PYWT filename suffix for stub runs
affects: [06-VERIFICATION, readme-figures]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TimingAccumulator with current_idx for in-loop timing collection alongside DSP

key-files:
  created: []
  modified:
    - research/benchmark.py

key-decisions:
  - "--comparison produces timing stats AND the figure; --comparison-grid stays figure-only (backward compat)"
  - "all_timings dict accumulates TimingAccumulator per signal label; printed after column loop, before rendering"
  - "_STUB_PYWT suffix replaces _STUB so the name communicates exactly which row is stubbed"

patterns-established:
  - "comparison=True branches wrap existing DSP calls in time_call without restructuring the frame loop"

requirements-completed: [FIG-04]

# Metrics
duration: 3min
completed: 2026-03-24
---

# Phase 06 Plan 03: Gap Closure - duration_s Fix and --comparison Flag Summary

**Overlap-aware duration_s formula and --comparison flag with per-method STFT/PyWavelet/SubShader timing stats added to benchmark.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-24T19:55:49Z
- **Completed:** 2026-03-24T19:58:48Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Fixed duration_s formula: `((frames_processed - 1) * hop_size + chunk_size) / sr` — old formula inflated duration by ~2x, causing reference trace to extend past spectrogram energy
- Added `--comparison` flag that runs all 3 DSP methods with per-signal timing tables (avg/min/max) and produces the comparison grid figure
- Updated `--stub-pywt` filename suffix from `_STUB` to `_STUB_PYWT` for clarity
- Backward-compatible: `--comparison-grid` still runs without timing, `--timing` still runs SubShader pipeline timing only

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix duration_s overlap formula** - `17d5315` (fix)
2. **Task 2: Add --comparison flag with per-method timing stats** - `ce1d885` (feat)

## Files Created/Modified

- `research/benchmark.py` - duration_s fix + comparison param + TimingAccumulator wiring + --comparison argparse flag + _STUB_PYWT suffix

## Decisions Made

- `--comparison` produces timing stats AND the comparison grid; `--comparison-grid` remains figure-only for backward compatibility
- `all_timings` dict indexed by signal label accumulates a `TimingAccumulator` per column; timing is printed after the column loop and before figure rendering
- `_STUB_PYWT` suffix replaces `_STUB` to make the stub scope explicit in the filename

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both UAT gaps from 06-01 are now closed
- `--comparison --stub-pywt --dpi 200` can be run to produce the final comparison grid with timing stats
- `--comparison-grid --stub-pywt` now produces `comparison_grid_STUB_PYWT.png`
- Phase 06 verification can proceed

---
*Phase: 06-finalize-example-audio-and-comparison-figures-for-readme*
*Completed: 2026-03-24*
