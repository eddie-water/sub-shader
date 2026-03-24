---
phase: 06-finalize-example-audio-and-comparison-figures-for-readme
plan: 01
subsystem: dsp
tags: [chirp-synthesis, benchmark, comparison-grid, audio-export, dpi]

# Dependency graph
requires:
  - phase: 02-cwt-pipeline-polish
    provides: NumPyWavelet, comparison grid infrastructure, benchmark.py scaffold
provides:
  - build_bouncing_chirp() synthesis function spanning 20Hz-20kHz with parabolic dips
  - build_bouncing_chirp_chunks() wrapper for benchmark pipeline
  - bouncing_chirp.wav audio file for Edison/DAW import
  - comparison_grid_{150,200,250,300}dpi.png for user DPI selection
  - --dpi flag in benchmark.py for DPI-parameterized grid generation
affects: [06-02, README.md, DSP.md]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DPI sentinel: --dpi 0 means default filename (comparison_grid.png); --dpi N produces comparison_grid_Ndpi.png"
    - "Bouncing chirp: CubicSpline in log-frequency space, integrated to phase via cumsum"

key-files:
  created:
    - research/utilities/dsp_helpers.py (build_bouncing_chirp, build_bouncing_chirp_chunks functions)
    - assets/audio/daw/bouncing_chirp.wav
    - assets/images/benchmarks/comparison_grid_150dpi.png
    - assets/images/benchmarks/comparison_grid_200dpi.png
    - assets/images/benchmarks/comparison_grid_250dpi.png
    - assets/images/benchmarks/comparison_grid_300dpi.png
  modified:
    - research/utilities/constants.py (AUDIO_BOUNCING_CHIRP constant)
    - research/utilities/__init__.py (exports for new functions/constant)
    - research/benchmark.py (bouncing chirp wiring, --dpi flag, filename logic)

key-decisions:
  - "Bouncing chirp uses CubicSpline in log-frequency space with peak/dip waypoints — matches user sketch naturally without manual parabola math"
  - "DPI=0 sentinel in generate_comparison_grid() means use default naming; dpi>0 always produces _Ndpi.png suffix regardless of stub_pywt flag"
  - "Stub suffix (_STUB) only applies at default DPI=0 path; explicit --dpi N always gets clean Ndpi name for user comparison"
  - "Checkpoint auto-approved: chirp shape verified visually from generated grid — ascending with clear parabolic dips matching sketch"
  - "DPI selection left to user: 150dpi=5.4MB, 200dpi=8.9MB, 250dpi=13MB, 300dpi=17.7MB — user picks in plan 02"

patterns-established:
  - "Bouncing chirp pattern: total_bounces waypoints, each split 70% peak / 97% dip to avoid CubicSpline duplicate-x error"

requirements-completed: [FIG-01, FIG-02, FIG-03]

# Metrics
duration: 10min
completed: 2026-03-24
---

# Phase 06 Plan 01: Bouncing Chirp Synthesis and Multi-DPI Comparison Grid Summary

**Synthesized bouncing chirp (20Hz-20kHz ascending with parabolic dips) via log-space CubicSpline and generated comparison grid at 4 DPI levels (150/200/250/300) using build_bouncing_chirp_chunks in benchmark.py**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-24T18:27:23Z
- **Completed:** 2026-03-24T18:37:47Z
- **Tasks:** 2 auto + 1 auto-approved checkpoint
- **Files modified:** 9

## Accomplishments

- Implemented `build_bouncing_chirp()` with ascending-with-dips frequency contour matching user's hand-drawn sketch — 9 bounces across 3 decades, CubicSpline in log-frequency space
- Wired bouncing chirp into `generate_comparison_grid()`, replacing the old `build_fm_chirp_chunks` call; exported `bouncing_chirp.wav`
- Added `--dpi` flag to benchmark.py; generated grid at 150, 200, 250, 300 DPI for user to select quality/filesize tradeoff

## Task Commits

Each task was committed atomically:

1. **Task 1: Create bouncing chirp synthesis function and add constants** - `a3ef03d` (feat)
2. **Task 2: Wire bouncing chirp into benchmark.py, add --dpi flag, generate grid** - `a25d74c` (feat)
3. **Task 3: Checkpoint auto-approved** — no commit (visual verification)

## Files Created/Modified

- `research/utilities/dsp_helpers.py` - Added `build_bouncing_chirp()` and `build_bouncing_chirp_chunks()`
- `research/utilities/constants.py` - Added `AUDIO_BOUNCING_CHIRP = "assets/audio/daw/bouncing_chirp.wav"`
- `research/utilities/__init__.py` - Exports for `AUDIO_BOUNCING_CHIRP`, `build_bouncing_chirp`, `build_bouncing_chirp_chunks`
- `research/benchmark.py` - Updated `generate_comparison_grid()` to use bouncing chirp, added `--dpi` flag
- `assets/audio/daw/bouncing_chirp.wav` - Synthesized bouncing chirp (6.08s, 44100Hz)
- `assets/images/benchmarks/comparison_grid_150dpi.png` - Grid at 150 DPI (5.4MB)
- `assets/images/benchmarks/comparison_grid_200dpi.png` - Grid at 200 DPI (8.9MB)
- `assets/images/benchmarks/comparison_grid_250dpi.png` - Grid at 250 DPI (13MB)
- `assets/images/benchmarks/comparison_grid_300dpi.png` - Grid at 300 DPI (17.7MB)

## Decisions Made

- Bouncing chirp uses CubicSpline in log-frequency space with peak/dip waypoints per bounce — matches user sketch naturally
- Used `dpi=0` sentinel in `generate_comparison_grid()` to mean "use default naming"; `dpi>0` always produces `_Ndpi.png` regardless of `--stub-pywt` flag
- Stub suffix only applies to DPI=0 (default) path — explicit `--dpi N` gets clean `comparison_grid_Ndpi.png` name for user comparison

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed CubicSpline duplicate x-axis times in bouncing chirp waypoints**
- **Found during:** Task 1 (build_bouncing_chirp implementation)
- **Issue:** Initial waypoint generation produced duplicate times at decade boundaries — the dip at end of bounce N and start of bounce N+1 collided, causing `CubicSpline: x must be strictly increasing`
- **Fix:** Restructured to global bounce indexing (not per-decade); placed peak at 70% and dip at 97% of each bounce slot, ensuring strictly increasing times
- **Files modified:** research/utilities/dsp_helpers.py
- **Verification:** `build_bouncing_chirp(44100, 6.0)` returns f[0]=20.0Hz, f[-1]=19988.8Hz without error
- **Committed in:** a3ef03d (Task 1 commit)

**2. [Rule 1 - Bug] Fixed --stub-pywt clobbering DPI-variant filenames**
- **Found during:** Task 2 (generating 4 DPI variants)
- **Issue:** Initial filename logic used `_STUB` suffix always when `stub_pywt=True`, so `--dpi 150 --stub-pywt` produced `comparison_grid_STUB_150dpi.png` instead of `comparison_grid_150dpi.png` as the plan requires
- **Fix:** Changed logic: explicit `--dpi N` always produces `comparison_grid_Ndpi.png`; `_STUB` suffix only applies at default DPI=0 path
- **Files modified:** research/benchmark.py
- **Verification:** All 4 DPI files named correctly
- **Committed in:** a25d74c (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 bug)
**Impact on plan:** Both fixes necessary for correct output. No scope creep.

## Issues Encountered

- Benchmark.py must be run from project root (not `research/` subdirectory) — relative paths for audio assets require project root CWD

## User Setup Required

**DPI selection pending.** The user needs to review the 4 grid variants and select their preferred DPI for plan 02:

- `assets/images/benchmarks/comparison_grid_150dpi.png` — 5.4MB
- `assets/images/benchmarks/comparison_grid_200dpi.png` — 8.9MB
- `assets/images/benchmarks/comparison_grid_250dpi.png` — 13MB
- `assets/images/benchmarks/comparison_grid_300dpi.png` — 17.7MB

Plan 02 will use the selected DPI to generate the final non-stub grid. If no selection is recorded, plan 02 defaults to 200 DPI.

## Known Stubs

- PyWavelet row in all 4 comparison grid PNGs is random noise (--stub-pywt used for generation speed). Plan 02 will generate final grids with real PyWavelet CWT.

## Next Phase Readiness

- bouncing_chirp.wav is ready for Edison import (DAW reference screenshot)
- All 4 DPI grid variants ready for user review and selection
- Plan 02 depends on user's DPI selection from this checkpoint
- build_bouncing_chirp_chunks() is ready for integration into any future benchmark/figure work

---
*Phase: 06-finalize-example-audio-and-comparison-figures-for-readme*
*Completed: 2026-03-24*

## Self-Check: PASSED

All created files verified present. Both task commits (a3ef03d, a25d74c) confirmed in git log.
