---
phase: 05-documentation
plan: 01
subsystem: testing
tags: [matplotlib, benchmark, cwt, stft, pywavelet, figures, documentation]

# Dependency graph
requires:
  - phase: 02-cwt-pipeline-polish
    provides: NumPyWavelet, PyWavelet with correct L1 kernel normalization used for figure generation
  - phase: 03-audio-visual-sync
    provides: validated audio pipeline and polyphonic audio file
provides:
  - comparison_grid.png: 3x3 matplotlib figure (Chirp/Polyphonic/Musical x STFT/PyWavelet/SubShader) at 200 DPI
  - benchmark.py --comparison-grid flag: repeatable figure generation command
affects:
  - 05-02 README scaffolding (uses comparison_grid.png in Performance section)
  - 05-03 DSP README scaffolding (uses comparison_grid.png)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - All benchmark figure generation lives in research/benchmark.py via argparse flags
    - generate_comparison_grid() follows same DSP pipeline pattern as _generate_comparison_figure()

key-files:
  created:
    - assets/images/benchmarks/comparison_grid.png
  modified:
    - research/benchmark.py

key-decisions:
  - "Chirp column uses 10-second window computed from sr/chunk_size/overlap ratio — ~215 frames at default settings"
  - "Per-column vmax used for each method's spectrogram so each row's detail is independently scaled"
  - "grid wspace=0.02, hspace=0.05 with tight subplots_adjust for minimal whitespace per CONTEXT.md"

patterns-established:
  - "Figure generation: add new argparse flag + standalone generate_*() function + append to modes list"

requirements-completed: [DOCS-05]

# Metrics
duration: 7min
completed: 2026-03-23
---

# Phase 05 Plan 01: Comparison Grid Figure Summary

**3x3 comparison grid (Chirp/Polyphonic/Musical x STFT/PyWavelet/SubShader) generated via benchmark.py --comparison-grid, saved to assets/images/benchmarks/comparison_grid.png at 200 DPI**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-23T20:26:49Z
- **Completed:** 2026-03-23T20:34:06Z
- **Tasks:** 2 (1 auto + 1 checkpoint:human-verify auto-approved)
- **Files modified:** 2

## Accomplishments
- Added `generate_comparison_grid()` function to benchmark.py implementing full DSP pipeline for 3 audio signals
- Added `--comparison-grid` argparse flag wired into run_modes dispatch
- Generated 3600x2000 px comparison grid figure at 200 DPI with all 9 cells containing real DSP output
- Row labels (STFT, PyWavelet, SubShader) and column labels (Chirp, Polyphonic, Musical) applied

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --comparison-grid flag to benchmark.py** - `cfcb0ce` (feat)
2. **Task 2: User approves comparison grid layout** - auto-approved (checkpoint:human-verify, auto_advance=true)

## Files Created/Modified
- `research/benchmark.py` - Added generate_comparison_grid() function and --comparison-grid argparse flag
- `assets/images/benchmarks/comparison_grid.png` - 3x3 comparison grid figure (3600x2000, 200 DPI, 9 cells)

## Decisions Made
- Chirp column uses ~215 frames (derived from 10s target at default sr/chunk_size/overlap) rather than capping at NUM_FRAMES=128 — gives a complete 10s sweep
- Per-row vmax for each column ensures each representation's dynamic range is independently visible
- Tight spacing: wspace=0.02, hspace=0.05 per CONTEXT.md decision to minimize whitespace
- Grid lines at alpha=0.08 provide subtle structure without obscuring spectrograms

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- comparison_grid.png is ready to embed in README.md Performance section (Plan 05-02)
- --comparison-grid flag is repeatable for layout iteration after user review
- User should review the generated figure at assets/images/benchmarks/comparison_grid.png and provide feedback on layout before Plan 05-02 README scaffolding begins

## Self-Check: PASSED
- [FOUND] assets/images/benchmarks/comparison_grid.png
- [FOUND] commit cfcb0ce in git log
- benchmark.py contains `--comparison-grid` in argparse section
- benchmark.py contains `def generate_comparison_grid`

---
*Phase: 05-documentation*
*Completed: 2026-03-23*
