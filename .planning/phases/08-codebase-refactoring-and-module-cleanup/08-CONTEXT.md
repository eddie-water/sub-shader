# Phase 8: Codebase Refactoring and Module Cleanup - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor all core modules for clean separation of concerns, readable flow, and professional naming. Main orchestrator simplified, AudioInput+AudioPlayer merged into AudioStream facade, DSP module reorganized with ABC facade, viz/ renamed to renderer/ with files split by concern, research/test suite restructured for clarity, assets reorganized by lifecycle (reference vs generated). No performance regressions. Nothing deleted — unused files archived.

</domain>

<decisions>
## Implementation Decisions

### Config System
- **D-01:** Single mutable `PipelineConfig` dataclass flows through entire pipeline. AudioStream discovers `sample_rate` from file and writes it back into config. All modules read what they need from the same config object.
- **D-02:** Module-specific configs inherit from `PipelineConfig`. `CWTConfig` adds wavelet params, `RendererConfig` adds rendering params. Shared values (sample_rate, chunk_size, overlap_factor, hop_size, nyquist_freq) live on the base.
- **D-03:** Derived values (`hop_size`, `nyquist_freq`) computed as `@property` on `PipelineConfig` — single source of truth, not recomputed in each module.
- **D-04:** Default runtime audio changed to `assets/audio/reference/beltran_sc_rip.wav`.

### AudioStream Facade
- **D-05:** New `AudioStream` class wraps file I/O (reader) and playback (player) as a facade. Two internal submodules: `reader.py` (was `audio_input.py`) and `player.py` (was `audio_player.py`).
- **D-06:** AudioStream takes `PipelineConfig`, discovers `sample_rate` and `total_samples` from file, writes them back into config.
- **D-07:** AudioStream exposes: `start()`, `get_chunk()`, `next_chunk()` (blocking), `get_playback_sample()`, `has_looped()`, `cleanup()`.

### DSP Module
- **D-08:** `dsp/` directory name stays.
- **D-09:** New `dsp/dsp.py` contains `DSP` ABC with `pre()`, `transform()`, `post()` abstract methods. Each backend inherits and defines its own pre-processing, core transform, and post-processing.
- **D-10:** Backends instantiated directly (Option 1, not factory pattern): `GpuCWT(config)`, `CpuCWT(config)`, `STFT(config)`, `PywaveletCWT(config)`. All inherit from DSP ABC.
- **D-11:** `wavelet.py` (700 lines, 7 classes) refactored into `cwt.py` (CWT base + CpuCWT + GpuCWT). NpWavelet/CuWavelet aliases removed. AntsWavelet intermediate flattened — shared FFT convolution logic moves into CWT base or stays in CpuCWT/GpuCWT.
- **D-12:** STFT extracted from `research/utilities/dsp_helpers.py` into `dsp/stft.py` with same DSP ABC API.
- **D-13:** PyWavelet extracted from `wavelet.py` into `dsp/pywavelet.py` with same DSP ABC API.
- **D-14:** STFT and PyWavelet `pre()`/`post()` can be stubs for now — only used in test suite, not runtime.

### Renderer Module
- **D-15:** `viz/` renamed to `renderer/`.
- **D-16:** `plotter.py` (812 lines, 7 classes) split into: `renderer.py` (Renderer + GLContext), `frame_buffer.py` (CircularFrameBuffer, AudioFrameBuffer), `intensity.py` (IntensityTracker, was plot_normalizer.py).
- **D-17:** Shader files renamed: `vertex_shader.glsl` → `vertex.glsl`, `fragment_shader.glsl` → `fragment.glsl`. `shaders/__init__.py` removed.

### Orchestrator
- **D-18:** `__main__.py` becomes thin CLI entry point (~15 lines): argparse + main(). SubShader class extracted to `pipeline.py`.
- **D-19:** `pipeline.py` SubShader class: `__init__` creates AudioStream, CWT, Renderer from config. `run()` is the main loop. `cleanup()` cleans all three modules.
- **D-20:** Orchestrator sees three modules and a config — no knowledge of readers, players, kernels, frame buffers, or shaders.

### Timing
- **D-21:** `@timed` decorator on ALL pipeline stages: AudioStream methods, DSP stages, Renderer.update(). Not just DSP.
- **D-22:** Timing output format driven by editable template file (`research/utilities/timing_template.txt`). Edit the template to change display format — no code changes needed.
- **D-23:** `--timing` writes results to `assets/timing/timing_YYYY-MM-DD_HH-MM-SS.txt` in addition to terminal output.
- **D-24:** `--timing` should also time the renderer, not just audio + DSP.

### Test Suite
- **D-25:** `test_suite.py` dispatcher has four modes: `--timing`, `--test`, `--compare-methods`, `--figures`.
- **D-26:** `--compare-methods` produces one figure per signal with rows: time series, DAW reference (graceful stub if missing), STFT, PyWavelet, SubShader CWT. Left-hand column with row labels and padding.
- **D-27:** `--compare-methods` interface: `--input-signal "path/to/file.wav"`. Default runs all registry signals. Titles derived from filename or registry label.
- **D-28:** `--figures` runs `--compare-methods` for all signals plus timing bar chart. README uses three per-signal figures side by side instead of monolithic 5×3 grid.
- **D-29:** 5×3 grid utility function kept in `comparison.py` — not deleted, just not the README default.
- **D-30:** `--stub` stubs PyWavelet calls (random image). Output goes to `stubs/` subdirectory with `_STUB` suffix next to where the real result would go.
- **D-31:** `--compare-methods` prints where generated figures are stored.
- **D-32:** Signal registry in `research/utilities/signals.py` — SIGNALS list with name, label, audio path, reference image path, type. Adding a signal = one list append.

### Assets
- **D-33:** Assets organized by lifecycle: `reference/` (committed inputs) and `generated/` (created by test_suite.py) as base subdirs for both audio and images.
- **D-34:** `assets/timing/` directory for timestamped timing results.
- **D-35:** `assets/plots/` stays for architecture diagrams (drawio/mermaid).
- **D-36:** All unused/old files moved to `assets/archive/` (subdirs: audio/, claude/, diagnostics/, images/). Nothing deleted.

### Archiving (not deletion)
- **D-37:** All removed source files go to `research/archive/`: comparison_navigator.py, WaveletDesign.md, frame_counter_pyqt5.py, gl_diagnostics.py, quick_plot.py, pipeline_timing_profile.py, benchmark_timing_template.txt, benchmark_results.csv, branch_goals.md.
- **D-38:** All unused asset files go to `assets/archive/`. Cherry-pick from archive for documentation later.

### Claude's Discretion
- How to handle the AntsWavelet intermediate layer during flattening (merge up vs merge down)
- Exact `@timed` attribute naming and thread safety details
- `timing_template.txt` exact format and placeholder syntax
- `__init__.py` barrel export decisions for each module
- Whether `font_showcase_20.py` goes to archive or gets deleted
- Internal implementation of `AudioStream.next_chunk()` blocking/waiting strategy

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design Documents (created during this discussion)
- `.planning/phases/08-codebase-refactoring-and-module-cleanup/config-design.md` — PipelineConfig pattern, all 7 usage contexts with pseudocode
- `.planning/phases/08-codebase-refactoring-and-module-cleanup/orchestrator-design.md` — Pipeline flow, __main__.py/pipeline.py split, directory structure before/after, open questions
- `.planning/phases/08-codebase-refactoring-and-module-cleanup/test-suite-design.md` — Test suite modes, signal registry, utilities, assets organization
- `.planning/phases/08-codebase-refactoring-and-module-cleanup/dsp-design.md` — DSP ABC design, direct instantiation pattern, directory structure

### Source Files Being Refactored
- `src/subshader/__main__.py` — Current orchestrator (223 lines, splitting into __main__.py + pipeline.py)
- `src/subshader/config.py` — Current config system (ProcessingConfig → PipelineConfig)
- `src/subshader/audio/audio_input.py` — Becomes reader.py inside AudioStream facade
- `src/subshader/audio/audio_player.py` — Becomes player.py inside AudioStream facade
- `src/subshader/dsp/wavelet.py` — 700 lines, 7 classes → cwt.py + pywavelet.py (flattened)
- `src/subshader/viz/plotter.py` — 812 lines, 7 classes → renderer/ split into 3 files
- `src/subshader/viz/plot_normalizer.py` — Becomes renderer/intensity.py
- `src/subshader/viz/comparison_navigator.py` — 1251 lines, archived

### Research Files Being Refactored
- `research/test_suite.py` — CLI dispatcher (simplifying)
- `research/timing.py` — TimedSubShader (adopting config pattern + renderer timing)
- `research/comparison.py` — Comparison grid (keeping utility, not README default)
- `research/figures.py` — ReadmeFigures (reusing --compare-methods code)
- `research/utilities/dsp_helpers.py` — STFT moves to src/subshader/dsp/stft.py
- `research/utilities/constants.py` — Asset paths updating to new structure
- `research/utilities/signals.py` — New: signal registry

### Prior Phase Context
- `.planning/phases/07-visual-style-system-and-frequency-range-configuration/07-CONTEXT.md` — Style system decisions (D-01 through D-25) that constrain this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `research/utilities/style.py` — Visual constants, already centralized in Phase 7. All figure code uses this.
- `research/utilities/plotting.py` — `create_figure_scaffold()`, `render_top_row()`, `render_spectrogram_row()` — reusable for --compare-methods figures
- `src/subshader/utils/timing.py` — `@timed` decorator already exists from Phase 7, needs to extend to audio and renderer
- `research/utilities/printing.py` — Terminal table formatting, reusable for --timing output

### Established Patterns
- Dataclasses for configuration with `validate()` methods
- `@timed` decorator on pipeline methods storing `_timing_*_ms` instance attributes
- Section comments (`# ===== SECTION =====`) for file organization
- Explicit `__init__.py` exports in utility packages

### Integration Points
- `pipeline.py` imports from `audio/`, `dsp/`, `renderer/` — the three module boundaries
- `research/` imports from `src/subshader/` — test suite depends on production code
- `README.md` references `assets/images/` paths — need updating when assets move
- `DSP.md` references `assets/images/benchmarks/timing_bar_chart.png` and `numpy_vs_cupy_diff.png`
- `AUDIO.md` references `assets/audio/daw/a2a3_a4_minor_scale.wav`
- `config.py` default file_path needs updating to new asset location

</code_context>

<specifics>
## Specific Ideas

- "I want main to almost look like pseudocode — keep it high level, no weird names or sections, minimal comments"
- "Readability is important — being able to find where things are in the code is super important"
- "I want the modules to be configurable for different scenarios — for the main use case, but then also for easily plugging in different init configs and params during testing"
- "I like the idea of a base class that stores the common configs/params, and then the modules that need more inherit the base and expand on it intelligently"
- "The test suite utilities should assist with setting up tests, running them, making figures and enforce styles that can be overridden easily enough if desired"
- "Nothing is too too hardcoded"
- "Don't delete anything — archive it. Some of it may be valuable for the DSP READMEs"

</specifics>

<deferred>
## Deferred Ideas

- pytest-benchmark integration for CI-friendly rigorous benchmarking (from Phase 7)
- `.mplstyle` file approach for matplotlib native theming (from Phase 7)
- Auto-detection of audio frequency content to dynamically adjust range (from Phase 7)
- Update architecture drawio diagrams after refactor is complete (user will do manually)
- Cherry-pick archived files for documentation/README content (user will do after archive)

</deferred>

---

*Phase: 08-codebase-refactoring-and-module-cleanup*
*Context gathered: 2026-04-06*
