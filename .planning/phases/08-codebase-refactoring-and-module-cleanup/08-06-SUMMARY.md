---
phase: 08-codebase-refactoring-and-module-cleanup
plan: "06"
subsystem: research
tags: [test-suite, signal-registry, timing, figures, cli]
dependency_graph:
  requires: ["08-05"]
  provides: ["signal-registry", "4-mode-test-suite", "timing-file-output"]
  affects: ["research/test_suite.py", "research/figures.py", "research/timing.py"]
tech_stack:
  added: []
  patterns: ["signal-registry-pattern", "cli-mutually-exclusive-group", "template-file-driven-output"]
key_files:
  created:
    - research/utilities/signals.py
    - research/utilities/timing_template.txt
  modified:
    - research/test_suite.py
    - research/timing.py
    - research/figures.py
    - research/utilities/__init__.py
decisions:
  - "Signal registry uses figures/ not reference/ for DAW images — matches actual asset layout from Phase 06/07"
  - "timing_template.txt matches TimedSubShader's 8-method accumulator keys (raw_cwt, normalize, magnitude, edge_trim, hop_center, downsample)"
  - "generate_method_comparison uses build_bouncing_chirp_chunks for synthetic chirp signals — same builder as comparison_grid"
  - "comparison_grid utility preserved in comparison.py per D-29 — not deleted, just not the --figures default"
metrics:
  duration_minutes: 3
  completed_date: "2026-04-06"
  tasks_completed: 2
  files_modified: 6
---

# Phase 08 Plan 06: Test Suite Restructure Summary

Signal registry, 4-mode CLI dispatcher, per-signal comparison figures, and timing-to-file output using an editable template.

## Tasks Completed

### Task 1: Create signal registry and timing template
**Commit:** `724f382`
**Files:** `research/utilities/signals.py`, `research/utilities/timing_template.txt`, `research/utilities/__init__.py`

Created `SIGNALS` list with 3 entries (chirp, polyphonic, musical) and `get_signal()` lookup helper. Created `timing_template.txt` with Python format string placeholders matching the 8 CWT sub-stages tracked by `TimedSubShader`. Exported both from `utilities/__init__.py`.

Key detail: reference image paths use `assets/images/figures/` (existing asset location from Phase 06) rather than the design doc's `assets/images/reference/` which does not exist in the repo.

### Task 2: Restructure test_suite.py dispatcher and update timing/figure modules
**Commit:** `bce8b66`
**Files:** `research/test_suite.py`, `research/timing.py`, `research/figures.py`, `research/comparison.py`

Rewrote `test_suite.py` with a `mutually_exclusive_group` of 4 modes. Added `run_timing()` entry point to `timing.py` along with template loading, report formatting, and file write to `assets/timing/YYYYMMDD_HHMMSS_timing.txt`. Added `generate_method_comparison()` and `generate_all_figures()` to `figures.py`; both import `SIGNALS` from the new registry. `comparison.py` is unchanged — `generate_comparison_grid()` retained as utility per D-29.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Signal reference image paths corrected to match repo layout**
- **Found during:** Task 1
- **Issue:** The plan and design doc specify `assets/images/reference/` for DAW reference images, but Phase 06/07 placed Edison screenshots in `assets/images/figures/`. Using the plan's paths would cause silent missing-image fallbacks on every run.
- **Fix:** Updated SIGNALS entries to use `assets/images/figures/` paths (e.g., `bouncing_chirp_edison.png`, `midi_sine_wave_edison.png`, `beltran_sc_rip_4_bar_edison.png`).
- **Files modified:** `research/utilities/signals.py`
- **Commit:** `724f382`

**2. [Rule 2 - Missing] timing_template.txt uses 8 CWT sub-stages not 5 generic stages**
- **Found during:** Task 1
- **Issue:** Plan's template sketch had 5 generic stage rows. The actual `TimedSubShader` accumulator tracks 8 methods: get_chunk, raw_cwt, normalize, magnitude, edge_trim, hop_center, downsample, push_frame. A 5-stage template could not be filled without data loss.
- **Fix:** Template has 8 rows matching the actual accumulator keys.
- **Files modified:** `research/utilities/timing_template.txt`
- **Commit:** `724f382`

## Known Stubs

None — no placeholder data flows to UI rendering.

## Self-Check: PASSED

Files created:
- `research/utilities/signals.py` — FOUND
- `research/utilities/timing_template.txt` — FOUND

Commits exist:
- `724f382` — FOUND
- `bce8b66` — FOUND
