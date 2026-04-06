---
phase: 08-codebase-refactoring-and-module-cleanup
plan: "05"
subsystem: pipeline
tags: [pipeline, orchestrator, refactor, cleanup, imports]
dependency_graph:
  requires: ["08-02", "08-03", "08-04"]
  provides: ["D-18", "D-19", "D-20", "D-21"]
  affects: ["src/subshader/pipeline.py", "src/subshader/__main__.py", "research/tests/"]
tech_stack:
  added: []
  patterns: ["thin CLI entry point", "orchestrator extract", "module alias cleanup"]
key_files:
  created:
    - src/subshader/pipeline.py
    - research/archive/comparison_navigator.py
    - research/archive/GPU_BUFFER_IDEA.md
    - research/archive/pipeline_timing_profile.py
  modified:
    - src/subshader/__main__.py
    - src/subshader/dsp/__init__.py
    - src/subshader/audio/__init__.py
    - src/subshader/dsp/pywavelet.py
    - research/tests/conftest.py
    - research/tests/dsp/test_wavelet.py
    - research/timing.py
    - research/comparison.py
    - research/figures.py
    - research/utilities/dsp_helpers.py
  deleted:
    - src/subshader/dsp/wavelet.py
    - src/subshader/viz/plotter.py
    - src/subshader/viz/plot_normalizer.py
    - src/subshader/viz/__init__.py
    - src/subshader/audio/audio_input.py
    - src/subshader/audio/audio_player.py
decisions:
  - "PywaveletCWT.post() remains a stub (D-14) — tests updated to handle complex output via np.abs()"
  - "pipeline_timing_profile.py archived (not updated) — uses internal wavelet APIs that changed; needs full rewrite"
  - "audio/__init__.py deprecated aliases (AudioInput, AudioPlayer) removed alongside old files"
metrics:
  duration_minutes: ~90
  completed_date: "2026-04-06"
  tasks_completed: 3
  files_changed: 18
  files_deleted: 6
  tests_passing: 15
---

# Phase 08 Plan 05: Module Switchover and Old File Deletion Summary

Complete module switchover: SubShader orchestrator extracted into pipeline.py, all callers migrated from old module paths to new ones, old source files deleted, deprecated aliases removed.

## What Was Built

**Task 1: pipeline.py + thin __main__.py**

`src/subshader/pipeline.py` contains the `SubShader` orchestrator class extracted from the old 222-line `__main__.py`. The `run()` loop reads like pseudocode:

```python
class SubShader:
    def __init__(self, config) -> None:
        self.audio = AudioStream(config)
        self.dsp = GpuCWT(config) if gpu_available() else CpuCWT(config)
        renderer_config = RendererConfig(...)
        self.renderer = Renderer(file_path=config.file_path,
                                  frame_shape=self.dsp.get_output_shape(),
                                  config=renderer_config)

    def run(self) -> None:
        self.audio.start()
        while not self.renderer.should_close():
            chunk = self.audio.next_chunk()
            if chunk is None:
                continue
            coefs = self.dsp.process(chunk)
            self.renderer.update(coefs)
```

`src/subshader/__main__.py` is 39 lines — argparse + `main()` calling `SubShader`.

**Task 2: Import switchover across all callers**

All research scripts and test files migrated:
- `NumPyWavelet` → `CpuCWT`, `CuWavelet` → `GpuCWT`, `PyWavelet` → `PywaveletCWT`
- `AudioInput` → `AudioReader`, `AudioPlayer` → `AudioPlayer` (from new path)
- `config.wavelet.*` / `config.audio.*` → flat `CWTConfig` fields
- `compute_stft_frame()` helper removed; callers use `STFT.process()` directly
- `CircularFrameBuffer` import path updated to `subshader.renderer.frame_buffer`
- All 15 tests pass with updated imports

**Task 3: Old file deletion and alias cleanup**

Deleted via `git rm`: `wavelet.py`, `plotter.py`, `plot_normalizer.py`, `viz/__init__.py`, `audio_input.py`, `audio_player.py`

Removed deprecated aliases from `dsp/__init__.py`: `NpWavelet`, `CuWavelet`, `NumPyWavelet`, `CuPyWavelet`

Removed deprecated aliases from `audio/__init__.py`: `AudioInput`, `AudioPlayer`

Archived to `research/archive/`: `comparison_navigator.py`, `GPU_BUFFER_IDEA.md`, `pipeline_timing_profile.py`

## Commits

| Hash | Message |
|------|---------|
| `366cb7e` | feat(08-05): create pipeline.py and slim __main__.py to thin CLI entry point |
| `e61f63e` | fix(08-05): fix PywaveletCWT stub compat in tests — add get_output_shape(), abs() for complex output |
| `b8b2327` | chore(08-05): delete old source files and remove deprecated aliases |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] `PywaveletCWT.get_output_shape()` not implemented**
- **Found during:** Task 2 test execution — `test_reliable_region_consistency` and `test_output_shape_regression` call `get_output_shape()` on `PywaveletCWT` which didn't have it
- **Issue:** DSP base class has no `get_output_shape()` abstract requirement; PywaveletCWT was missing the method
- **Fix:** Added `get_output_shape()` returning `(len(self.freqs), self.config.chunk_size)` per stub design (D-14)
- **Files modified:** `src/subshader/dsp/pywavelet.py`
- **Commit:** `e61f63e`

**2. [Rule 1 - Bug] `find_peak_bin()` fails for complex output**
- **Found during:** Task 2 — `test_pure_tone_peak_accuracy` incorrect peak for `PywaveletCWT`
- **Issue:** `PywaveletCWT.process()` returns complex CWT coefficients (stub post()); `np.mean()` of complex array gives complex mean, not energy — peak detection was wrong
- **Fix:** Updated `find_peak_bin()` in conftest to use `np.mean(np.abs(cwt_output), axis=1)` — handles both real and complex inputs
- **Files modified:** `research/tests/conftest.py`
- **Commit:** `e61f63e`

**3. [Rule 1 - Bug] `test_reliable_region_consistency` asserts cross-backend shape equality**
- **Found during:** Task 2 — `CpuCWT` outputs `(num_freqs, target_width)` after downsampling; `PywaveletCWT` outputs `(num_freqs, chunk_size)` — shapes differ by design
- **Issue:** The test required `np_out.shape == py_out.shape` but backends have intentionally different output shapes
- **Fix:** Rewrote test to verify each backend's self-consistency (`process() shape == get_output_shape()`) rather than cross-backend equality
- **Files modified:** `research/tests/dsp/test_wavelet.py`
- **Commit:** `e61f63e`

**4. [Rule 1 - Bug] `_timing__normalize_by_scale_ms` attribute does not exist**
- **Found during:** Task 2 — `test_timed_attributes_populated_after_process` listed a nonexistent timing attr
- **Issue:** `_normalize_by_scale` method is not decorated with `@timed`; actual timing attrs include `_extract_hop_center_ms` which was missing from the test's expected list
- **Fix:** Corrected expected attrs list to match actual `@timed` decorated methods
- **Files modified:** `research/tests/dsp/test_wavelet.py`
- **Commit:** `e61f63e`

**5. [Rule 3 - Blocking] `audio/__init__.py` imported deleted files**
- **Found during:** Task 3 — after deleting `audio_input.py`, test `test_numpy_vs_cupy` failed with `ModuleNotFoundError` because `audio/__init__.py` still had deprecated import aliases
- **Fix:** Removed `from .audio_input import AudioInput` and `from .audio_player import AudioPlayer` from `audio/__init__.py`; updated `__all__` to export only `AudioStream`
- **Files modified:** `src/subshader/audio/__init__.py`
- **Commit:** `b8b2327`

**6. [Rule 4 scope — archived, not rewritten] `pipeline_timing_profile.py` used old APIs throughout**
- **Found during:** Task 3 pre-deletion import check
- **Issue:** `research/pipeline_timing_profile.py` imports from `subshader.dsp.wavelet`, `subshader.viz.plotter`, `subshader.audio.audio_input` — all deleted. It also calls `wavelet.cwt()`, `wavelet.normalize_by_scale()`, `wavelet.compute_mag()`, and old constructor signatures — none of which exist in the new API
- **Decision:** Archived to `research/archive/` rather than rewriting. This profiler needs a full rewrite using the new `@timed` decorator pattern already present in `CpuCWT`. Tracked as deferred work.
- **Commit:** `b8b2327`

## Known Stubs

None that affect plan goal. `PywaveletCWT.post()` is intentionally a stub per D-14 — it's a comparison backend, not a production backend. Tests handle this correctly via `np.abs()` on complex output.

## Self-Check: PASSED

- `src/subshader/pipeline.py` exists: FOUND
- `src/subshader/__main__.py` is < 40 lines: 39 lines
- `src/subshader/dsp/wavelet.py` deleted: CONFIRMED
- `src/subshader/viz/plotter.py` deleted: CONFIRMED
- `src/subshader/audio/audio_input.py` deleted: CONFIRMED
- `research/archive/comparison_navigator.py` exists: FOUND
- No NpWavelet in `dsp/__init__.py`: CONFIRMED
- All 15 tests pass: CONFIRMED
- `from subshader.pipeline import SubShader` imports cleanly: CONFIRMED
