# Phase 7: Visual Style System and Frequency Range Configuration - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Centralize all plot styling into a single constants module, fix comparison grid header margins/centering, add configurable frequency range bounds, and restructure the research toolkit into a coherent architecture with clear separation of concerns for testing, timing, comparison, and figure generation.

**Deliverables:**
- `research/utilities/style.py` — single source of truth for all visual constants
- Comparison grid header fix (top margin + horizontal centering)
- Configurable `root_note_a0_hz` and `num_octaves` in WaveletConfig (Nyquist trimming already exists)
- `@timed` decorator on pipeline methods for always-available profiling
- Research toolkit restructured: `test_suite.py` dispatcher, `comparison.py`, `timing.py`, `utilities/`, `tests/`, `archive/`

</domain>

<decisions>
## Implementation Decisions

### Style Consolidation
- **D-01:** Create `research/utilities/style.py` as the single source of truth for all visual constants. Every color, fontsize, linewidth, alpha, figsize, spacing value lives here.
- **D-02:** All figure functions import from style.py — no hardcoded visual values anywhere in figure code. Changing a value in style.py updates all figures.
- **D-03:** Style values exposed as module-level constants (e.g., `style.BG_COLOR`, `style.FONT_SIZE`). Not dicts, not dataclasses.
- **D-04:** One canonical dark style. Kill the matplotlib/seaborn backend toggle and the `DEFAULT_STYLE`/`SEABORN_STYLE` dict pattern in `plotting.py`.
- **D-05:** The style system is designed for reusability — plotting mechanisms should work for future documentation figures, not just the current comparison grid.

### Comparison Grid Header Fix
- **D-06:** Fix column title top margin — titles too close to top edge, need more breathing room.
- **D-07:** Fix column title horizontal centering — titles misaligned relative to spectrogram columns, especially with the label column offset.

### Frequency Range Configuration
- **D-08:** Keep A0 (27.5Hz) as the default root note. Sub-bass is audible on proper audio systems and contains real content in bass-heavy music.
- **D-09:** `root_note_a0_hz` and `num_octaves` remain the configurable parameters in WaveletConfig. Users who need speed can raise root to A1 (55Hz) or reduce octaves.
- **D-10:** No new Nyquist clamping code needed — `_generate_chromatic_scale()` already trims frequencies above Nyquist at line 127 of `wavelet.py`.

### Pipeline Timing Architecture
- **D-11:** Add `@timed` decorator to pipeline methods (audio chunk, DSP stages, render). Decorator lives in `research/utilities/timing.py` (or `src/subshader/utils/`).
- **D-12:** Overhead is ~1 microsecond per decorated call (~0.2% on a 0.5ms STFT). Negligible at SubShader's pipeline scale (10-30 fps).
- **D-13:** Timing data always available — any code path can access timing results without needing a special timed version of the pipeline. Eliminates drift between timed and untimed paths.
- **D-14:** `research/timing.py` becomes a thin reporting layer that reads timing data from decorated methods, not a parallel reimplementation of the pipeline.

### Research Toolkit Reorganization
- **D-15:** Dispatcher renamed from `benchmark.py` to `test_suite.py`. Single entry point for: `--test`, `--timing`, `--comparison`, `--figures`.
- **D-16:** `research/timing.py` — thin pipeline profiler measuring: audio get_chunk → DSP stages (raw_cwt, normalize, magnitude, edge_trim, hop_center, downsample) → render.
- **D-17:** `research/comparison.py` — method-vs-method figures and timing table. Methods defined as a config list: `[{name, function, label}]`. Adding a new method = append to the list.
- **D-18:** `research/utilities/` — reusable library: style.py (visual constants), plotting.py (grid/column/row layouts, heatmaps, time series), signals.py (chirp generators, test signal library), wav_export.py (moved from research/ root), printing.py (terminal formatting), timing.py (@timed decorator, TimingAccumulator), dsp_helpers.py.
- **D-19:** `wav_export.py` moves from research/ root into `research/utilities/`.
- **D-20:** `ants/`, `docs/`, `gpu_basics/`, `misc/`, `python/` move to `research/archive/`. Historical reference, not active tooling.

### Unit Test Organization
- **D-21:** All tests move from colocated positions in `src/` to `research/tests/`. Centralized test directory.
- **D-22:** `research/tests/` mirrors src/ structure: `research/tests/audio/`, `research/tests/dsp/`, `research/tests/viz/`.
- **D-23:** `test_suite.py` discovers and runs tests via pytest. Boilerplate framework set up in this phase, tests added incrementally over time.

### Comparison Method Extensibility
- **D-24:** Comparison methods defined as a config list in `comparison.py`. Each entry: `{name, compute_function, label}`. Grid and timing table iterate over this list.
- **D-25:** Adding a new method (e.g., a new wavelet backend) = define the compute function + append one entry to the list. No other files need changes.

### Claude's Discretion
- Exact `@timed` decorator implementation details (where timing data is stored, thread safety)
- Internal layout of style.py constant groupings (by concern vs alphabetical)
- Exact gridspec adjustments for header margin/centering fix
- test_suite.py CLI argument parsing approach (argparse, click, or manual)
- How to handle the seaborn import and backend removal (deprecation warning vs clean removal)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Style System
- `research/utilities/plotting.py` — Current `DEFAULT_STYLE` and `SEABORN_STYLE` dicts (lines 29-76), backend toggle, `get_active_style()`. All being replaced.
- `research/figures.py` — `GRID_*` constants (lines 634-637), `LABEL_*` constants (lines 844-846), hardcoded style values throughout. All moving to style.py.

### Comparison Grid
- `research/figures.py` — `generate_comparison_grid()` function, gridspec layout, header title rendering (line 975)

### Frequency Range
- `src/subshader/config.py` — `WaveletConfig` dataclass (lines 91-124): `root_note_a0_hz`, `num_octaves`, `notes_per_octave`
- `src/subshader/dsp/wavelet.py` — `_generate_chromatic_scale()` (lines 105-127) with existing Nyquist trim

### Pipeline Timing
- `research/timing.py` — Current `TimedSubShader` wrapper class
- `research/pipeline_timing_profile.py` — Detailed sub-stage profiling (26KB)
- `research/utilities/timing.py` — `time_call`, `TimingAccumulator`

### Research Toolkit Structure
- `research/benchmark.py` — Current CLI dispatcher (being renamed to test_suite.py)
- `research/utilities/__init__.py` — Current exports
- `research/utilities/constants.py` — Timing/comparison constants
- `research/utilities/dsp_helpers.py` — DSP computation helpers

### Existing Tests
- `src/subshader/audio/test_audio_overlap.py` — Moving to research/tests/audio/
- `src/subshader/dsp/test_wavelet.py` — Moving to research/tests/dsp/
- `src/subshader/dsp/test_wavelet_kernel.py` — Moving to research/tests/dsp/
- `src/subshader/viz/test_plotter.py` — Moving to research/tests/viz/

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `research/utilities/plotting.py` — `create_figure_scaffold()`, `render_top_row()`, `render_cwt_row()` are reusable figure primitives that should survive the restructure
- `research/utilities/dsp_helpers.py` — CWT/STFT/PyWavelet computation helpers used by comparison code
- `research/utilities/printing.py` — Terminal table formatting for timing results
- `research/utilities/constants.py` — Audio file paths, timing iteration counts

### Established Patterns
- Section comments (`# ===== SECTION =====`) used throughout codebase for file organization
- Dataclasses for configuration (WaveletConfig, AudioConfig)
- `__init__.py` explicit exports in utilities package

### Integration Points
- `test_suite.py` dispatcher integrates with all research modules (timing, comparison, tests, figures)
- `@timed` decorator touches production code in `src/subshader/` — careful placement needed
- Style constants consumed by both `research/figures.py` and `research/utilities/plotting.py`
- `comparison.py` imports from `utilities/dsp_helpers.py` for compute functions

</code_context>

<specifics>
## Specific Ideas

- User wants style system designed for reusability: "I want all these things to be easily configurable wherever they're used" — future documentation figures, new plot types should automatically get the canonical style
- Research toolkit should feel like a proper Python module: clear concerns, one entry point, extensible comparison list
- Timing should be "thin" — decorators on methods, not parallel reimplementations of the pipeline
- Sub-bass (20-60Hz) deliberately kept in default range — audible on subwoofers and relevant for bass-heavy music visualization

</specifics>

<deferred>
## Deferred Ideas

- pytest-benchmark integration for CI-friendly rigorous benchmarking — separate phase if needed
- `.mplstyle` file approach (matplotlib native theming) — worth considering if style needs to span multiple projects
- Auto-detection of audio frequency content to dynamically adjust range — too complex for this phase
- Detailed DSP.md foundation figures — Phase 5 plan 05-04 scope, not Phase 7

</deferred>

---

*Phase: 07-visual-style-system-and-frequency-range-configuration*
*Context gathered: 2026-03-27*
