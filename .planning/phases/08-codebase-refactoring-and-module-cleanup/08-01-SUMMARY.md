---
phase: 08-codebase-refactoring-and-module-cleanup
plan: 01
subsystem: config
tags: [dataclass, inheritance, config, assets, refactoring]

requires:
  - phase: 07-visual-style-system-and-frequency-range-configuration
    provides: Completed research toolkit with style.py, comparison.py, figures.py

provides:
  - PipelineConfig base dataclass with hop_size/nyquist_freq @property derivations
  - CWTConfig(PipelineConfig) with full wavelet parameters
  - RendererConfig(PipelineConfig) with num_frames and color_norm
  - ColorNormalizationConfig with gamma, percentiles, decay_rate, initial_intensity
  - ProcessingConfig = CWTConfig deprecated alias for backward compat
  - assets/audio/reference/ and assets/audio/generated/ lifecycle directories
  - assets/images/reference/ and assets/images/generated/ lifecycle directories
  - assets/timing/ placeholder directory
  - Updated research/utilities/constants.py with new path constants (AUDIO_REFERENCE_DIR, TIMING_DIR, etc.)

affects:
  - 08-02 (DSP module — CWT backends must accept PipelineConfig)
  - 08-03 (renderer — RendererConfig is the renderer's config type)
  - 08-04 (audio — AudioStream discovers sample_rate and writes to PipelineConfig)
  - 08-05 (orchestrator — __main__.py migration from config.audio.* to flat config)
  - research/ modules (all use IMAGES_GENERATED_DIR for output paths)

tech-stack:
  added: []
  patterns:
    - PipelineConfig inheritance hierarchy replaces nested composition (AudioConfig + WaveletConfig + VisualizationConfig → flat CWTConfig)
    - Runtime-discovered values (sample_rate, total_samples) are fields on PipelineConfig, set by AudioStream after file open
    - Derived values (hop_size, nyquist_freq) are @property on PipelineConfig, computed from user-settable fields
    - Deprecated alias (ProcessingConfig = CWTConfig) enables gradual migration of callers

key-files:
  created:
    - assets/audio/reference/ (directory with beltran, a2a3, overlapping_A3_A4_A5, midi_sine_waves, chirp variants)
    - assets/audio/generated/ (directory with bouncing_chirp, chirp_comparison_grid, chirp_random_walk)
    - assets/images/reference/ (numpy_vs_cupy_diff.png)
    - assets/images/generated/ (comparison_grid, signal comparisons, timing_bar_chart, dpi/, stubs/)
    - assets/timing/.gitkeep
  modified:
    - src/subshader/config.py (complete rewrite: PipelineConfig, CWTConfig, RendererConfig hierarchy)
    - research/utilities/constants.py (BENCHMARKS_DIR removed, new lifecycle dir constants added)
    - research/utilities/__init__.py (exports updated to new constant names)
    - research/comparison.py (BENCHMARKS_DIR → IMAGES_GENERATED_DIR)
    - research/figures.py (BENCHMARKS_DIR/BENCHMARKS_STUBS_DIR → IMAGES_GENERATED_DIR)

key-decisions:
  - "CWTConfig chosen as default return type from get_default_config() — most param-rich subclass, correct for full pipeline"
  - "ProcessingConfig = CWTConfig alias defers __main__.py migration to Plan 08-05 — callers still import without error"
  - "bouncing_chirp.wav placed in assets/audio/generated/ not reference/ — it is a synthesized waveform, not a recorded/curated source"
  - "Default file_path changed from assets/audio/daw/a2a3_a4_minor_scale.wav to assets/audio/reference/beltran_sc_rip.wav per D-04"
  - "ColorNormalizationConfig fields replaced with new semantics: gamma, smoothing_weight, percentiles, decay_rate, initial_intensity"

patterns-established:
  - "Config inheritance: PipelineConfig → CWTConfig or RendererConfig; base holds shared pipeline fields"
  - "Asset lifecycle: reference/ for committed input files, generated/ for test-suite outputs"

requirements-completed: [D-01, D-02, D-03, D-04, D-33, D-34, D-35]

duration: 7min
completed: 2026-04-06
---

# Phase 8 Plan 1: Config Redesign and Asset Reorganization Summary

**Flat PipelineConfig inheritance hierarchy replacing nested AudioConfig/WaveletConfig/VisualizationConfig composition, assets reorganized into reference/ and generated/ lifecycle directories with updated path constants throughout research toolkit.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-06T23:02:10Z
- **Completed:** 2026-04-06T23:08:24Z
- **Tasks:** 2
- **Files modified:** 7 source files + 40 asset files moved/created

## Accomplishments

- Rewrote `config.py` from nested-composition to flat-inheritance: `PipelineConfig` base with `CWTConfig` and `RendererConfig` subclasses, each with `validate()` chaining via `super()`
- Added `hop_size` and `nyquist_freq` as `@property` on `PipelineConfig`, eliminating derived-value duplication across call sites
- Reorganized `assets/` into `reference/` (committed source files) and `generated/` (test-suite outputs) lifecycle directories, creating a clear ownership model for all assets
- Updated `research/utilities/constants.py` with `AUDIO_REFERENCE_DIR`, `AUDIO_GENERATED_DIR`, `IMAGES_REFERENCE_DIR`, `IMAGES_GENERATED_DIR`, `TIMING_DIR` and corrected all audio/image path constants

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite config.py with PipelineConfig inheritance hierarchy** - `aa26b4b` (feat)
2. **Task 2: Reorganize assets into reference/generated lifecycle dirs and update constants.py** - `132b25e` (feat)

## Files Created/Modified

- `/home/eddie-water/dev/python/sub-shader/src/subshader/config.py` - Complete rewrite: PipelineConfig base, CWTConfig, RendererConfig, ColorNormalizationConfig (new field semantics), get_default_config() returning CWTConfig, ProcessingConfig = CWTConfig alias
- `/home/eddie-water/dev/python/sub-shader/research/utilities/constants.py` - Replaced BENCHMARKS_DIR/BENCHMARKS_SEABORN_DIR/BENCHMARKS_STUBS_DIR with AUDIO_REFERENCE_DIR, AUDIO_GENERATED_DIR, IMAGES_REFERENCE_DIR, IMAGES_GENERATED_DIR, TIMING_DIR; updated all path constants
- `/home/eddie-water/dev/python/sub-shader/research/utilities/__init__.py` - Updated exports to match new constant names
- `/home/eddie-water/dev/python/sub-shader/research/comparison.py` - BENCHMARKS_DIR → IMAGES_GENERATED_DIR
- `/home/eddie-water/dev/python/sub-shader/research/figures.py` - BENCHMARKS_DIR/BENCHMARKS_STUBS_DIR → IMAGES_GENERATED_DIR pattern with os.path.join for stubs subdir
- `assets/audio/reference/` - 7 committed reference audio files (beltran variants, a2a3, overlapping, midi_sine_waves)
- `assets/audio/generated/` - 3 synthesized audio files (bouncing_chirp, chirp_comparison_grid, chirp_random_walk)
- `assets/images/reference/` - numpy_vs_cupy_diff.png
- `assets/images/generated/` - All test-suite output PNGs, dpi/ and stubs/ subdirs

## Decisions Made

- `get_default_config()` returns `CWTConfig` (not `RendererConfig`) because the full pipeline needs CWT params; callers importing `ProcessingConfig` still get `CWTConfig` via alias
- `ProcessingConfig = CWTConfig` is a flat alias — callers using the old `config.audio.file_path` composition pattern will fail at runtime until Plan 08-05 migrates `__main__.py`; this is intentional — the branch is a full refactor
- `bouncing_chirp.wav` goes to `generated/` because it is synthesized by the research toolkit, not a curated recording
- `assets/plots/` left untouched per D-35 — contains architecture drawio/mermaid diagrams, not media outputs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed broken imports in research/utilities/__init__.py, comparison.py, figures.py**
- **Found during:** Task 2 (constants.py update)
- **Issue:** `research/utilities/__init__.py` imported `BENCHMARKS_DIR`, `BENCHMARKS_SEABORN_DIR`, `BENCHMARKS_STUBS_DIR` from constants.py; `comparison.py` and `figures.py` imported `BENCHMARKS_DIR` from utilities — all removed by constants redesign
- **Fix:** Updated `__init__.py` to export new dir constants; replaced all `BENCHMARKS_DIR` references in `comparison.py` and `figures.py` with `IMAGES_GENERATED_DIR`; inlined `stubs` dir as `os.path.join(IMAGES_GENERATED_DIR, "stubs")` where `BENCHMARKS_STUBS_DIR` was used
- **Files modified:** research/utilities/__init__.py, research/comparison.py, research/figures.py
- **Verification:** `python -c "from research.utilities import *"` exits 0
- **Committed in:** 132b25e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — broken imports caused by removing old directory constants)
**Impact on plan:** Necessary fix. The constants change was in scope; the callers had to follow.

## Issues Encountered

None beyond the auto-fixed deviation above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `PipelineConfig`, `CWTConfig`, `RendererConfig` are ready for Plan 08-02 (DSP module migration)
- `assets/audio/reference/beltran_sc_rip.wav` is now the default audio path in `PipelineConfig.file_path`
- `get_default_config()` returns a valid `CWTConfig` with the audio file present and validating correctly
- `__main__.py` still uses old `config.audio.*` composition pattern — intentionally deferred to Plan 08-05

---
*Phase: 08-codebase-refactoring-and-module-cleanup*
*Completed: 2026-04-06*
