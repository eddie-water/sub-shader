---
phase: quick
plan: 260409-uan
subsystem: renderer
tags: [intensity, normalization, pre-scan, fixed-reference]
dependency_graph:
  requires: []
  provides: [fixed intensity normalization reference via pre-scan]
  affects: [src/subshader/renderer/intensity.py, src/subshader/renderer/frame_buffer.py, src/subshader/renderer/renderer.py, src/subshader/pipeline.py, src/subshader/config.py]
tech_stack:
  added: []
  patterns: [pre-scan percentile, fixed shader uniform]
key_files:
  created: []
  modified:
    - src/subshader/renderer/intensity.py
    - src/subshader/renderer/frame_buffer.py
    - src/subshader/renderer/renderer.py
    - src/subshader/pipeline.py
    - src/subshader/config.py
    - research/tests/viz/test_intensity_tracker.py
    - research/tests/viz/test_plotter.py
decisions:
  - "IntensityTracker simplified to a fixed-value container — pre-scan drives the reference, no runtime adaptation"
  - "CircularFrameBuffer drops IntensityTracker entirely — intensity tracking is pipeline-level, not buffer-level"
  - "set_fixed_intensity_max() on Renderer sets shader uniform once before run() — uniform persists across all draw calls"
  - "PRESCAN_NUM_CHUNKS=10 hardcoded as local constant in pipeline.py — tunable without a config field"
metrics:
  duration: ~15 minutes
  completed: 2026-04-10
  tasks_completed: 2
  files_modified: 7
---

# Quick Task 260409-uan: Fix IntensityTracker Normalization Strategy

Replace dynamic global_max tracking in IntensityTracker with a fixed normalization reference computed by pre-scanning 10 evenly-spaced CWT frames before the visualization loop starts.

## What Was Done

### Task 1 — Simplify IntensityTracker and clean ColorNormalizationConfig (TDD)

**IntensityTracker** (`src/subshader/renderer/intensity.py`) was rewritten from an adaptive tracker to a simple fixed-value container:
- Constructor takes `fixed_max: float` (plus `floor_value: float = 1e-8`)
- Sets `self.global_max = max(fixed_max, floor_value)` at construction
- `update()` is a no-op that returns `self.global_max` unchanged
- Removed: `percentile`, `retention_rate`, `warmup_frames`, `frame_count`, `is_ready`

**ColorNormalizationConfig** (`src/subshader/config.py`) was trimmed to two fields:
- Kept: `gamma` (used by shader), `global_intensity_percentile` (used by pre-scan)
- Removed: `retention_rate`, `global_intensity_smoothing_weight`, `initial_intensity`, `frame_intensity_percentile`, `frame_brightness_percentile`
- `validate()` updated to remove checks for removed fields

**Tests** (`research/tests/viz/test_intensity_tracker.py`) fully replaced with 6 tests for the new behavior: construction, no-drift on loud/quiet frames, floor clamp, absence of `retention_rate`, and validate() passing.

### Task 2 — Add pre-scan to pipeline and wire fixed intensity_max through renderer

**SubShader** (`src/subshader/pipeline.py`):
- `_prescan_intensity(renderer_config)` added: reads 10 evenly-spaced chunks across the audio file, runs each through CWT, takes `np.percentile(np.abs(coefs), percentile)`, returns max across all sampled frames. Resets `reader.file_pos = 0` after scan.
- `__init__` calls `_prescan_intensity()` after all stages constructed, then `renderer.set_fixed_intensity_max()`.

**Renderer** (`src/subshader/renderer/renderer.py`):
- `set_fixed_intensity_max(value)` added: calls `gpu_renderer.set_intensity_max(value)` once and stores `self._fixed_intensity_max = value`.
- `update()`: removed `self.gpu_renderer.set_intensity_max(self.frame_buffer.get_intensity_max())` call. Shader uniform persists across draw calls without re-setting each frame.
- Removed `IntensityTracker` import.
- `CircularFrameBuffer` construction simplified (no `color_norm_config`).

**CircularFrameBuffer** (`src/subshader/renderer/frame_buffer.py`):
- Constructor signature changed from `(frame_shape, num_frames, color_norm_config)` to `(frame_shape, num_frames)`.
- Removed `IntensityTracker` construction, `intensity_tracker.update()` call in `push_frame()`, and `get_intensity_max()` method.
- Removed `IntensityTracker` import.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_plotter.py used the old CircularFrameBuffer API**

- **Found during:** Post-task regression run
- **Issue:** `test_buffer_normalization` called `CircularFrameBuffer` with `color_norm_config=color_norm` and then called `buf.get_intensity_max()` — both removed in Task 2.
- **Fix:** Updated `test_plotter.py` to use the new two-argument constructor and removed the `get_intensity_max()` assertion. Renamed test to `test_buffer_preserves_frame_data`.
- **Files modified:** `research/tests/viz/test_plotter.py`
- **Commit:** `60e64bf`

### Out-of-scope Discoveries

`src/subshader/renderer/RENDERER.md` contains stale references to `retention_rate` in its scaffold content. This is a documentation file, not Python source, and was pre-existing — deferred per scope boundary rule.

## Commits

| Hash | Message |
|------|---------|
| `02ddce1` | test(260409-uan): add failing tests for fixed-reference IntensityTracker |
| `ac2db97` | feat(260409-uan): simplify IntensityTracker to fixed-reference and clean ColorNormalizationConfig |
| `9115c26` | feat(260409-uan): add pre-scan and wire fixed intensity_max through renderer |
| `60e64bf` | fix(260409-uan): update test_plotter to match CircularFrameBuffer API change |

## Self-Check

- `src/subshader/renderer/intensity.py` — exists, holds fixed_max
- `src/subshader/pipeline.py` — exists, `_prescan_intensity` method present
- `research/tests/viz/test_intensity_tracker.py` — 6 tests, all pass
- No `retention_rate` in `src/subshader/` Python files (verified via grep)
- No `IntensityTracker` in `frame_buffer.py` (verified via grep)
- `from subshader.pipeline import SubShader` — imports without error

## Self-Check: PASSED
