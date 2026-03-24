---
phase: 06-finalize-example-audio-and-comparison-figures-for-readme
plan: 02
subsystem: dsp
tags: [comparison-grid, timing-bar-chart, readme, dsp-docs, benchmark]

# Dependency graph
requires:
  - phase: 06-01
    provides: DPI=200 selection, bouncing chirp synthesis, --dpi flag in benchmark.py
provides:
  - Final comparison_grid.png at DPI=200 with real PyWavelet
  - Regenerated timing_bar_chart.png with current pipeline data
  - generate_timing_bar_chart() function + --timing-chart flag in benchmark.py
  - README.md Performance section with centered hero image and scaffold captions
  - DSP.md Section 6 with timing bar chart reference
affects: [README.md, DSP.md, assets/images/benchmarks/]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "--timing-chart flag in benchmark.py: runs STFT/PyWavelet/NumPyWavelet timing over NUM_FRAMES, saves bar chart to assets/images/benchmarks/timing_bar_chart.png"
    - "DPI=200 with no --dpi flag produces comparison_grid.png (canonical name); --dpi 200 produces comparison_grid_200dpi.png"

key-files:
  created: []
  modified:
    - research/benchmark.py (generate_timing_bar_chart function, --timing-chart flag)
    - README.md (comparison grid hero image, scaffold captions, benchmark section link)
    - DSP.md (timing bar chart in Section 6)
    - assets/images/benchmarks/comparison_grid.png (final DPI=200, real PyWavelet)
    - assets/images/benchmarks/comparison_grid_200dpi.png (same content, DPI-suffixed name)
    - assets/images/benchmarks/timing_bar_chart.png (regenerated with current pipeline data)

key-decisions:
  - "comparison_grid.png copied from comparison_grid_200dpi.png — canonical name for README reference; both exist"
  - "generate_timing_bar_chart() runs STFT + PyWavelet + NumPyWavelet timing with TimingAccumulator; saves bar chart with error bars at user-selected DPI"
  - "Timing bar chart moved from README.md Benchmark section to DSP.md Section 6 -- detailed timing analysis belongs with implementation docs"
  - "README.md Benchmark section now links to DSP.md#6-implementation-deep-dive for detailed breakdown"

requirements-completed: [FIG-04, FIG-05, FIG-06]

# Metrics
duration: 8min
completed: 2026-03-24
---

# Phase 06 Plan 02: Final Comparison Grid, Timing Bar Chart, README + DSP.md Update Summary

**Generated final comparison_grid.png at DPI=200 with real PyWavelet, regenerated timing_bar_chart.png via new --timing-chart flag, and updated README.md with hero image + scaffold captions while moving timing analysis to DSP.md**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-24T20:00:00Z
- **Completed:** 2026-03-24T20:08:57Z
- **Tasks:** 1 auto + 1 checkpoint (pending user verify)
- **Files modified:** 6

## Accomplishments

- Added `generate_timing_bar_chart()` function to `benchmark.py` and wired `--timing-chart` flag — timing chart can now be regenerated programmatically
- Generated final `comparison_grid.png` at DPI=200 with real PyWavelet (no --stub-pywt) — all three rows render actual CWT output
- Regenerated `timing_bar_chart.png` with current pipeline data (128 frames, 4 components: get_chunk, STFT, PyWavelet, NumPyWavelet)
- Updated README.md Performance section:
  - Replaced PLACEHOLDER with `<img src="assets/images/benchmarks/comparison_grid.png" width="80%">`
  - Renamed "Chirp Signal (Frequency Sweep)" section to "Bouncing Chirp"
  - Replaced all REWRITE markers with WRITE scaffold captions ready for user authoring
  - Removed STFT/PyWavelet/SubShader bullet list (stream-of-consciousness notes)
  - Removed timing bar chart from Benchmark section; added link to DSP.md
- Updated DSP.md Section 6: added timing bar chart with WRITE scaffold caption

## Task Commits

1. **Task 1: Generate final comparison grid, regenerate timing bar chart, update README + DSP.md** - `096e4de` (feat)
2. **Task 2: Checkpoint — awaiting user verification** — no commit

## Files Created/Modified

- `research/benchmark.py` — `generate_timing_bar_chart()` function, `--timing-chart` flag
- `README.md` — Performance section with hero grid, scaffold captions, benchmark link
- `DSP.md` — Section 6 timing bar chart with scaffold caption
- `assets/images/benchmarks/comparison_grid.png` — Final DPI=200, real PyWavelet (5.4MB)
- `assets/images/benchmarks/comparison_grid_200dpi.png` — Same content, DPI-suffixed name
- `assets/images/benchmarks/timing_bar_chart.png` — Regenerated with current data (97KB)

## Decisions Made

- Copied `comparison_grid_200dpi.png` to `comparison_grid.png` — canonical name README will reference; the `--dpi 200` flag always produces the `_200dpi` suffix variant
- Added `generate_timing_bar_chart()` as a standalone function (not a method of ReadmeFigures) — it's a different class of output (bar chart vs spectrogram grid)
- Timing bar chart moved to DSP.md Section 6 (Implementation Deep Dive) — closest section to a computational cost section in the current scaffold

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added generate_timing_bar_chart() to benchmark.py**
- **Found during:** Task 1 (investigating timing bar chart regeneration)
- **Issue:** benchmark.py had no code to regenerate `timing_bar_chart.png` — chart was a static file with no generation pathway
- **Fix:** Added `generate_timing_bar_chart()` function (runs STFT/PyWavelet/NumPyWavelet timing over NUM_FRAMES, saves bar chart with min/max error bars) and wired `--timing-chart` CLI flag
- **Files modified:** research/benchmark.py
- **Commit:** 096e4de

**2. [Rule 2 - Missing Critical Functionality] Copied comparison_grid_200dpi.png to comparison_grid.png**
- **Found during:** Task 1 (after generating with --dpi 200)
- **Issue:** `--dpi 200` produces `comparison_grid_200dpi.png` but README references `comparison_grid.png`
- **Fix:** Copied `comparison_grid_200dpi.png` to `comparison_grid.png` to satisfy README's canonical path; both files committed
- **Files modified:** assets/images/benchmarks/comparison_grid.png
- **Commit:** 096e4de

---

**Total deviations:** 2 auto-fixed (both missing critical functionality)
**Impact on plan:** Both were necessary for all acceptance criteria to pass. No scope creep.

## Checkpoint Pending

**Task 2 (checkpoint:human-verify)** is pending user review. The orchestrator will present this checkpoint to the user.

**What to verify:**
1. Open `assets/images/benchmarks/comparison_grid.png` — verify all 3 signal columns render with real PyWavelet (not stub/noise)
2. Preview README.md — verify comparison grid displays at ~80% centered width in Performance section; per-signal sections have scaffold captions; timing bar chart is gone; Benchmark section links to DSP.md
3. Preview DSP.md — verify timing bar chart appears in Section 6 with updated data
4. Open `assets/images/benchmarks/timing_bar_chart.png` — verify it reflects current pipeline timing (4 bars: get_chunk, STFT, PyWavelet, NumPyWavelet)
5. Note any scaffold caption wording to adjust

## Known Stubs

- README.md and DSP.md contain `[WRITE: ...]` scaffold markers — these are intentional placeholders for user prose authoring. Not blocking stubs; they are the intended output of this plan.

---
*Phase: 06-finalize-example-audio-and-comparison-figures-for-readme*
*Completed: 2026-03-24*

## Self-Check: PASSED

All created/modified files verified present. Task 1 commit 096e4de confirmed in git log.
