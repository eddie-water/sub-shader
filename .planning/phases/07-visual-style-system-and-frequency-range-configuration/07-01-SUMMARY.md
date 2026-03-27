---
phase: 07-visual-style-system-and-frequency-range-configuration
plan: 01
subsystem: research-toolkit
tags: [matplotlib, style-system, plotting, research-figures]

# Dependency graph
requires:
  - phase: 05.1-research-toolkit-restructure
    provides: modular research utilities (plotting.py, figures.py, benchmark.py)
provides:
  - research/utilities/style.py — single source of truth for all visual constants
  - plotting.py stripped of backend toggle, style dicts, and seaborn
  - __init__.py updated to export style module
affects:
  - 07-02 (frequency range configuration)
  - 07-03 (figures.py consumer migration)
  - 07-04 (any further visual polish)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All visual constants as plain module-level names in style.py (no dicts, no dataclasses)"
    - "Import style module directly: from . import style, then use style.BG_COLOR etc."
    - "Single canonical dark style — no backend toggling, no per-call style dict merging"

key-files:
  created:
    - research/utilities/style.py
  modified:
    - research/utilities/plotting.py
    - research/utilities/__init__.py
    - research/figures.py
    - research/benchmark.py

key-decisions:
  - "style.py uses plain module-level names — no dicts, no dataclasses (per D-03)"
  - "SEABORN_STYLE values are killed — one canonical dark style only (per D-04)"
  - "Backend toggle (set_backend/get_backend/get_active_style) removed entirely (per D-05)"
  - "figures.py seaborn loop collapsed to single matplotlib path — no seaborn output"
  - "benchmark.py --seaborn flag removed — no seaborn mode exists any more"

patterns-established:
  - "style.py is the import source for all visual constants in research utilities"
  - "plotting.py primitives use style.* directly — no style= parameter, no dict merging"

requirements-completed: [STY-01, STY-02, STY-03, STY-04, STY-05]

# Metrics
duration: 15min
completed: 2026-03-27
---

# Phase 07 Plan 01: Visual Style System Summary

**Single-file style constants module in research/utilities/style.py plus seaborn backend fully removed from plotting.py, figures.py, and benchmark.py**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-27T04:55:00Z
- **Completed:** 2026-03-27T05:10:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created `research/utilities/style.py` with all visual constants as plain module-level names — the single source of truth for colors, font sizes, line widths, figure dimensions, layout spacing, comparison grid constants, and rendering defaults
- Stripped `research/utilities/plotting.py` of DEFAULT_STYLE, SEABORN_STYLE, seaborn import block, and backend toggle (set_backend/get_backend/get_active_style) — all plotting functions now read from `style.*` directly
- Updated `__init__.py` to export `style` module and remove backend toggle exports
- Updated `figures.py` to remove `set_backend` calls and collapse the seaborn backend loop to a single matplotlib path
- Updated `benchmark.py` to remove `--seaborn` CLI flag and `backends` list

## Task Commits

1. **Task 1: Create research/utilities/style.py** - `b806129` (feat)
2. **Task 2: Strip backend toggle from plotting.py, update consumers** - `9e7fe81` (feat)

## Files Created/Modified

- `research/utilities/style.py` — canonical visual constants (created)
- `research/utilities/plotting.py` — stripped of backend toggle, style dicts, seaborn; uses style.* directly
- `research/utilities/__init__.py` — exports style module, removed backend toggle exports
- `research/figures.py` — removed set_backend calls, seaborn import, collapsed to matplotlib-only path
- `research/benchmark.py` — removed --seaborn flag, backends list, SEABORN_AVAILABLE check

## Decisions Made

- SEABORN_STYLE values are not carried forward — one canonical dark style only
- Backend toggle removed entirely; the abstraction had zero remaining value once seaborn is dropped
- `benchmark.py` `--seaborn` CLI flag removed to avoid dead code paths
- `figures.py` seaborn output directories (`BENCHMARKS_SEABORN_DIR`) no longer imported since the seaborn output path is gone

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed set_backend calls in figures.py and benchmark.py**
- **Found during:** Task 2 (after stripping plotting.py)
- **Issue:** figures.py imported `set_backend` from utilities and called it inside a `for backend in self.backends` loop; benchmark.py constructed backends lists and passed them to ReadmeFigures — both would fail to import after removing set_backend from plotting.py
- **Fix:** Removed set_backend call and seaborn loop from figures.py (collapsed to matplotlib-only path); removed SEABORN_AVAILABLE, backends, and --seaborn from benchmark.py; removed BENCHMARKS_SEABORN_DIR import from figures.py; simplified ReadmeFigures.__init__ signature
- **Files modified:** research/figures.py, research/benchmark.py
- **Verification:** `from figures import ReadmeFigures` succeeds after fix
- **Committed in:** `9e7fe81` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The consumer fix was necessary to keep figures.py importable after plotting.py was modified. No scope creep — all changes directly follow from removing the backend toggle pattern.

## Issues Encountered

None — deviation was straightforward and resolved inline.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `style.py` is in place and importable as `from utilities import style`
- `plotting.py` is clean — no backend toggle, no style dicts, no seaborn
- Plan 02 (frequency range configuration) can proceed
- Plan 03 (migrate all consumers to style.py constants) will need to update hardcoded style values in figures.py generate_comparison_grid() and any other callers

---
*Phase: 07-visual-style-system-and-frequency-range-configuration*
*Completed: 2026-03-27*
