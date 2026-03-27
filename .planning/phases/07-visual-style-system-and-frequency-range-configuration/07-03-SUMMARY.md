---
phase: 07-visual-style-system-and-frequency-range-configuration
plan: "03"
subsystem: research
tags: [research-toolkit, test-migration, restructure]
dependency_graph:
  requires: ["07-01", "07-02"]
  provides: ["research/test_suite.py", "research/tests/", "research/archive/"]
  affects: ["research/utilities/__init__.py", "pyproject.toml"]
tech_stack:
  added: []
  patterns: ["pytest testpaths config", "conftest shared helpers"]
key_files:
  created:
    - research/test_suite.py
    - research/tests/__init__.py
    - research/tests/conftest.py
    - research/tests/audio/__init__.py
    - research/tests/audio/test_audio_overlap.py
    - research/tests/dsp/__init__.py
    - research/tests/dsp/test_wavelet.py
    - research/tests/dsp/test_wavelet_kernel.py
    - research/tests/viz/__init__.py
    - research/tests/viz/test_plotter.py
    - research/archive/ (5 dirs archived)
  modified:
    - research/utilities/__init__.py
    - research/utilities/wav_export.py (moved from research/)
    - pyproject.toml
  deleted:
    - research/benchmark.py
    - research/wav_export.py
    - src/subshader/conftest.py
    - src/subshader/audio/test_audio_overlap.py
    - src/subshader/dsp/test_wavelet.py
    - src/subshader/dsp/test_wavelet_kernel.py
    - src/subshader/viz/test_plotter.py
decisions:
  - "research/tests added to pythonpath so conftest.py plain helpers are importable via from conftest import"
  - "cwt_timed tests rewritten to use @timed _timing_*_ms attributes — cwt_timed() removed in 07-02"
metrics:
  duration_seconds: 217
  completed_date: "2026-03-27"
  tasks_completed: 2
  files_changed: 20
---

# Phase 07 Plan 03: Research Toolkit Restructure Summary

Research toolkit restructured from an ad-hoc collection into a coherent module tree: dispatcher renamed, test files migrated to mirrored src/ structure, old directories archived, wav_export relocated.

## What Was Built

### Task 1: Rename dispatcher, move wav_export, archive old dirs

- **research/test_suite.py** — renamed from benchmark.py with updated docstring and flag changes
  - `--seaborn` flag and all seaborn-related logic removed
  - `--unit-tests` renamed to `--test` (with `--unit-tests` as alias for compatibility)
  - pytest now runs `research/tests/` instead of `src/`
  - import updated to `from utilities.wav_export import export_signal_to_wav`
- **research/utilities/wav_export.py** — moved from research/ root; re-exported from `utilities/__init__.py`
- **research/archive/** — 5 directories relocated: ants, docs, gpu_basics, misc, python
- **research/benchmark.py** deleted

### Task 2: Migrate test files, update pytest config

- **research/tests/** — new directory tree with 4 test modules mirroring src/ structure:
  - `audio/test_audio_overlap.py` — hop-center overlap tests
  - `dsp/test_wavelet.py` — wavelet correctness, normalization, output shape (15 tests)
  - `dsp/test_wavelet_kernel.py` — kernel FFT bank integrity, L1 norm
  - `viz/test_plotter.py` — CircularFrameBuffer normalization
- **research/tests/conftest.py** — shared helpers: generate_tone, find_peak_bin, _make_wavelet
- **pyproject.toml** — `testpaths = ["research/tests"]`, `pythonpath` extended with `"research/tests"`
- **src/subshader/conftest.py** deleted — helpers live in research/tests/ now

### Frequency Range Config Confirmation (D-08 to D-10)

Confirmed as already implemented — no code changes needed:
- `WaveletConfig.root_note_a0_hz = 27.5` (D-08)
- `WaveletConfig.num_octaves = 10` (D-09)
- `_generate_chromatic_scale()` trims frequencies above Nyquist: `freqs[freqs < self.nyquist_freq]` (D-10)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rewrote cwt_timed tests to use new timing API**
- **Found during:** Task 2 (test migration)
- **Issue:** test_wavelet.py contained `test_cwt_timed_returns_six_stages` and `test_cwt_timed_output_matches_cwt` which called `wt.cwt_timed()` — a method removed in Phase 07-02 in favor of `@timed` decorator attributes
- **Fix:** Replaced with `test_timed_attributes_populated_after_cwt` (verifies `_timing_*_ms` instance attributes exist and are non-negative after `cwt()`) and `test_cwt_output_consistent_across_calls` (verifies determinism)
- **Files modified:** research/tests/dsp/test_wavelet.py
- **Commit:** ba81bc4

**2. [Rule 3 - Blocking] Added research/tests to pythonpath**
- **Found during:** Task 2 verification
- **Issue:** `from conftest import generate_tone, find_peak_bin, _make_wavelet` in test_wavelet.py and test_wavelet_kernel.py failed — pytest's conftest.py is found by pytest but not by Python's import system unless on the path
- **Fix:** Added `"research/tests"` to `pythonpath` in `pyproject.toml`
- **Files modified:** pyproject.toml
- **Commit:** ba81bc4

## Verification Results

- `pytest research/tests/ --co` discovers 15 tests across 4 files
- `pytest research/tests/ -x -q` (non-wavelet subset): 5 passed
- `python research/test_suite.py --help` shows --test, --timing, --comparison-grid flags
- `grep -r "seaborn" research/test_suite.py` returns nothing
- `test -d research/archive/ants` confirms archive move
- `test -f research/utilities/wav_export.py` confirms wav_export relocation

## Commits

- `4232638` — feat(07-03): rename benchmark.py to test_suite.py, move wav_export, archive old dirs
- `ba81bc4` — feat(07-03): migrate test files to research/tests/, update pytest config

## Self-Check: PASSED
