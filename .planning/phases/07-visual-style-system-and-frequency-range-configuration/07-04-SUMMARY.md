---
phase: 07-visual-style-system-and-frequency-range-configuration
plan: "04"
subsystem: research-toolkit
tags: [style-system, refactor, comparison-grid, figures]
dependency_graph:
  requires: ["07-01", "07-02", "07-03"]
  provides: ["comparison.py with COMPARISON_METHODS", "figures.py style-migrated"]
  affects: ["research/test_suite.py", "research/comparison.py", "research/figures.py", "research/utilities/style.py"]
tech_stack:
  added: []
  patterns:
    - "Comparison logic extracted from figures.py into dedicated comparison.py module"
    - "Extensible COMPARISON_METHODS list drives method-vs-method figure generation"
    - "All DPI and cmap rendering values in figures.py now reference style.py constants"
key_files:
  created:
    - research/comparison.py
  modified:
    - research/figures.py
    - research/test_suite.py
    - research/utilities/style.py
decisions:
  - "generate_comparison_grid and generate_timing_bar_chart extracted to comparison.py — separation of concerns between per-signal figures (ReadmeFigures) and method-vs-method comparison (comparison.py)"
  - "STUB_DPI = 100 added to style.py — stub_layouts intentionally renders at lower DPI for fast iteration; value now lives in style rather than hardcoded"
  - "cmap=style.GRID_CMAP passed explicitly in figures.py render calls — makes colormap choice visible and controlled by style module"
metrics:
  duration: "~15 minutes"
  completed_date: "2026-03-27"
  tasks_completed: 2
  files_modified: 4
---

# Phase 07 Plan 04: Style Integration and comparison.py Extraction Summary

Integration plan connecting the style system (Plan 01) to all figure consumers, extracting comparison concern from figures.py into comparison.py, and fixing the comparison grid header visual issue.

## Tasks Completed

### Task 1: Extract comparison.py with extensible method list and header fix

- Created `research/comparison.py` with `generate_comparison_grid()` and `generate_timing_bar_chart()` extracted from `figures.py`
- Added `COMPARISON_METHODS` extensible config list at top of module (4 entries: STFT, PyWavelet, SubShader NumPy, SubShader GPU)
- Fixed grid column header padding: `pad=8` replaced with `pad=style.GRID_TITLE_PAD` (20) — gives visible breathing room above column titles
- Replaced all hardcoded visual values in extracted functions with `style.*` references (16 style references in comparison.py)
- Fixed `wav_export` import: deferred `from wav_export import` (old bare module name) replaced with `from utilities.wav_export import`
- Removed both functions from `figures.py`
- Updated `test_suite.py` import: split from single `from figures import ReadmeFigures, generate_comparison_grid, generate_timing_bar_chart` into two lines

**Commit:** `66dc793`

### Task 2: Migrate figures.py to use style.py constants throughout

- Added `from utilities import style` import to `figures.py`
- Replaced `dpi=150` with `style.DEFAULT_DPI` in `_generate_comparison_figure`
- Replaced `dpi=100` with `style.STUB_DPI` in `stub_layouts` (fast-render intent preserved; value lives in style.py)
- Added `STUB_DPI = 100` constant to `utilities/style.py`
- Added explicit `cmap=style.GRID_CMAP` to all `render_spectrogram_row` calls in `figures.py` so colormap is controlled by style module
- Figures.py now has 4 `style.*` references (acceptance criterion: at least 3)

**Commit:** `806de68`

## Deviations from Plan

### Auto-added constants

**[Rule 2 - Missing] Added STUB_DPI = 100 to style.py**
- **Found during:** Task 2
- **Issue:** `stub_layouts` used `dpi=100` which is intentionally lower than DEFAULT_DPI for fast iteration. Replacing with DEFAULT_DPI would change behavior. No style constant existed for stub rendering DPI.
- **Fix:** Added `STUB_DPI = 100` to `style.py` as a named constant preserving the intent
- **Files modified:** `research/utilities/style.py`
- **Commit:** `806de68`

## Verification Results

All plan verification checks pass:

- `from comparison import generate_comparison_grid, COMPARISON_METHODS` imports successfully; 4 methods in list
- `from figures import ReadmeFigures` imports successfully
- `python test_suite.py --help` shows all expected flags
- `comparison.py` has 16 `style.` references; `figures.py` has 4
- `grep "pad=8" comparison.py` returns nothing (header fix applied)
- `grep -r "SEABORN|seaborn" figures.py test_suite.py` returns nothing

## Known Stubs

None — all comparison and figure generation code is wired to real DSP.

## Self-Check: PASSED

- `research/comparison.py` — FOUND
- `research/figures.py` — FOUND
- `research/utilities/style.py` — FOUND
- commit `66dc793` — FOUND
- commit `806de68` — FOUND
