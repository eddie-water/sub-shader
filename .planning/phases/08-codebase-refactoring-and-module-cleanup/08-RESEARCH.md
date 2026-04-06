# Phase 8: Codebase Refactoring and Module Cleanup - Research

**Researched:** 2026-04-06
**Domain:** Python module refactoring, internal API redesign, asset reorganization
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Config System**
- D-01: Single mutable `PipelineConfig` dataclass flows through entire pipeline. AudioStream discovers `sample_rate` from file and writes it back into config. All modules read what they need from the same config object.
- D-02: Module-specific configs inherit from `PipelineConfig`. `CWTConfig` adds wavelet params, `RendererConfig` adds rendering params. Shared values (sample_rate, chunk_size, overlap_factor, hop_size, nyquist_freq) live on the base.
- D-03: Derived values (`hop_size`, `nyquist_freq`) computed as `@property` on `PipelineConfig` — single source of truth, not recomputed in each module.
- D-04: Default runtime audio changed to `assets/audio/reference/beltran_sc_rip.wav`.

**AudioStream Facade**
- D-05: New `AudioStream` class wraps file I/O (reader) and playback (player) as a facade. Two internal submodules: `reader.py` (was `audio_input.py`) and `player.py` (was `audio_player.py`).
- D-06: AudioStream takes `PipelineConfig`, discovers `sample_rate` and `total_samples` from file, writes them back into config.
- D-07: AudioStream exposes: `start()`, `get_chunk()`, `next_chunk()` (blocking), `get_playback_sample()`, `has_looped()`, `cleanup()`.

**DSP Module**
- D-08: `dsp/` directory name stays.
- D-09: New `dsp/dsp.py` contains `DSP` ABC with `pre()`, `transform()`, `post()` abstract methods. Each backend inherits and defines its own pre-processing, core transform, and post-processing.
- D-10: Backends instantiated directly (Option 1, not factory pattern): `GpuCWT(config)`, `CpuCWT(config)`, `STFT(config)`, `PywaveletCWT(config)`. All inherit from DSP ABC.
- D-11: `wavelet.py` (700 lines, 7 classes) refactored into `cwt.py` (CWT base + CpuCWT + GpuCWT). NpWavelet/CuWavelet aliases removed. AntsWavelet intermediate flattened — shared FFT convolution logic moves into CWT base or stays in CpuCWT/GpuCWT.
- D-12: STFT extracted from `research/utilities/dsp_helpers.py` into `dsp/stft.py` with same DSP ABC API.
- D-13: PyWavelet extracted from `wavelet.py` into `dsp/pywavelet.py` with same DSP ABC API.
- D-14: STFT and PyWavelet `pre()`/`post()` can be stubs for now — only used in test suite, not runtime.

**Renderer Module**
- D-15: `viz/` renamed to `renderer/`.
- D-16: `plotter.py` (812 lines, 7 classes) split into: `renderer.py` (Renderer + GLContext), `frame_buffer.py` (CircularFrameBuffer, AudioFrameBuffer), `intensity.py` (IntensityTracker, was plot_normalizer.py).
- D-17: Shader files renamed: `vertex_shader.glsl` → `vertex.glsl`, `fragment_shader.glsl` → `fragment.glsl`. `shaders/__init__.py` removed.

**Orchestrator**
- D-18: `__main__.py` becomes thin CLI entry point (~15 lines): argparse + main(). SubShader class extracted to `pipeline.py`.
- D-19: `pipeline.py` SubShader class: `__init__` creates AudioStream, CWT, Renderer from config. `run()` is the main loop. `cleanup()` cleans all three modules.
- D-20: Orchestrator sees three modules and a config — no knowledge of readers, players, kernels, frame buffers, or shaders.

**Timing**
- D-21: `@timed` decorator on ALL pipeline stages: AudioStream methods, DSP stages, Renderer.update(). Not just DSP.
- D-22: Timing output format driven by editable template file (`research/utilities/timing_template.txt`). Edit the template to change display format — no code changes needed.
- D-23: `--timing` writes results to `assets/timing/timing_YYYY-MM-DD_HH-MM-SS.txt` in addition to terminal output.
- D-24: `--timing` should also time the renderer, not just audio + DSP.

**Test Suite**
- D-25: `test_suite.py` dispatcher has four modes: `--timing`, `--test`, `--compare-methods`, `--figures`.
- D-26: `--compare-methods` produces one figure per signal with rows: time series, DAW reference (graceful stub if missing), STFT, PyWavelet, SubShader CWT. Left-hand column with row labels and padding.
- D-27: `--compare-methods` interface: `--input-signal "path/to/file.wav"`. Default runs all registry signals. Titles derived from filename or registry label.
- D-28: `--figures` runs `--compare-methods` for all signals plus timing bar chart. README uses three per-signal figures side by side instead of monolithic 5×3 grid.
- D-29: 5×3 grid utility function kept in `comparison.py` — not deleted, just not the README default.
- D-30: `--stub` stubs PyWavelet calls (random image). Output goes to `stubs/` subdirectory with `_STUB` suffix next to where the real result would go.
- D-31: `--compare-methods` prints where generated figures are stored.
- D-32: Signal registry in `research/utilities/signals.py` — SIGNALS list with name, label, audio path, reference image path, type. Adding a signal = one list append.

**Assets**
- D-33: Assets organized by lifecycle: `reference/` (committed inputs) and `generated/` (created by test_suite.py) as base subdirs for both audio and images.
- D-34: `assets/timing/` directory for timestamped timing results.
- D-35: `assets/plots/` stays for architecture diagrams (drawio/mermaid).
- D-36: All unused/old files moved to `assets/archive/` (subdirs: audio/, claude/, diagnostics/, images/). Nothing deleted.

**Archiving (not deletion)**
- D-37: All removed source files go to `research/archive/`: comparison_navigator.py, WaveletDesign.md, frame_counter_pyqt5.py, gl_diagnostics.py, quick_plot.py, pipeline_timing_profile.py, benchmark_timing_template.txt, benchmark_results.csv, branch_goals.md.
- D-38: All unused asset files go to `assets/archive/`. Cherry-pick from archive for documentation later.

### Claude's Discretion

- How to handle the AntsWavelet intermediate layer during flattening (merge up vs merge down)
- Exact `@timed` attribute naming and thread safety details
- `timing_template.txt` exact format and placeholder syntax
- `__init__.py` barrel export decisions for each module
- Whether `font_showcase_20.py` goes to archive or gets deleted
- Internal implementation of `AudioStream.next_chunk()` blocking/waiting strategy

### Deferred Ideas (OUT OF SCOPE)

- pytest-benchmark integration for CI-friendly rigorous benchmarking (from Phase 7)
- `.mplstyle` file approach for matplotlib native theming (from Phase 7)
- Auto-detection of audio frequency content to dynamically adjust range (from Phase 7)
- Update architecture drawio diagrams after refactor is complete (user will do manually)
- Cherry-pick archived files for documentation/README content (user will do after archive)
</user_constraints>

---

## Summary

Phase 8 is a pure structural refactor — no new features, no algorithm changes, no performance work. Every file has been read and understood. The codebase is clean Python with no external framework magic, so the primary planning concerns are: (1) keeping `git mv` traceable, (2) fixing all import chains atomically per wave so nothing breaks mid-phase, and (3) ensuring the research/test suite stays runnable throughout.

The six major work streams are independent enough to plan as separate waves: config redesign, audio facade, DSP restructure, renderer split, orchestrator extraction, and test suite + asset reorganization. The biggest technical risks are the `wavelet.py` class hierarchy flattening (AntsWavelet has shared logic needed by both CpuCWT and GpuCWT), the `viz/__init__.py` re-export of `comparison_navigator.py` classes (must be cleaned up atomically), and the asset path changes that will break `constants.py` and the default config `file_path`.

**Primary recommendation:** Plan waves around import boundaries. Each wave should leave the codebase in a runnable state: fix all callsites within the same wave as the renamed/moved module.

---

## Project Constraints (from CLAUDE.md)

- Python 3.12.3, CuPy, ModernGL — no rewrites, no new dependencies
- Dataclasses with `validate()` for all config objects
- `get_` prefix for getter methods; single `_` for internal helpers
- Section comments (`# ===== SECTION =====`) for file organization
- Explicit `__init__.py` exports in utility packages
- No linter config — follow PEP 8, ~100 char line limit
- `@timed` decorator lives in `src/subshader/utils/timing.py`
- `gpu_available()` lives in `src/subshader/utils/gpu.py`
- GSD workflow required before file edits

---

## Current Codebase Inventory

### Source Files Being Refactored

| File | Lines | Classes | Disposition |
|------|-------|---------|-------------|
| `src/subshader/__main__.py` | 223 | SubShader | Split: `__main__.py` (~15 ln) + `pipeline.py` |
| `src/subshader/config.py` | 411 | 5 (ProcessingConfig + 4 sub-configs) | Redesign → PipelineConfig + CWTConfig + RendererConfig |
| `src/subshader/audio/audio_input.py` | 158 | AudioInput | Rename → `reader.py`, wrap by `audio_stream.py` |
| `src/subshader/audio/audio_player.py` | 151 | AudioPlayer | Rename → `player.py`, wrap by `audio_stream.py` |
| `src/subshader/dsp/wavelet.py` | 697 | 7 (Wavelet, PyWavelet, AntsWavelet, NumPyWavelet, CuPyWavelet, NpWavelet, CuWavelet) | Flatten → `dsp.py` (ABC) + `cwt.py` + `pywavelet.py` |
| `src/subshader/viz/plotter.py` | 799 | 7 (Plotter, ShaderPlot, GLContext, Renderer, CircularFrameBuffer, AudioFrameBuffer + PyQtGraph) | Split → `renderer.py` + `frame_buffer.py` + `intensity.py` |
| `src/subshader/viz/plot_normalizer.py` | 61 | IntensityTracker | Move → `renderer/intensity.py` |
| `src/subshader/viz/comparison_navigator.py` | 1251 | 3 (KernelNavigator, TransformNavigator, TopLevelComparisonNavigator) | Archive → `research/archive/` |

### Research Files Being Refactored

| File | Lines | Disposition |
|------|-------|-------------|
| `research/test_suite.py` | ~100 | Simplify — 4 modes (--timing, --test, --compare-methods, --figures) |
| `research/timing.py` | 141 | Update to new config/module names + renderer timing |
| `research/comparison.py` | 552 | Keep utility function; comparison grid not default README |
| `research/figures.py` | 463 | Refactor to reuse `--compare-methods` code |
| `research/utilities/dsp_helpers.py` | ~332 | STFT moves to `src/subshader/dsp/stft.py`; chirp builders stay in dsp_helpers.py or move to signals.py |
| `research/utilities/constants.py` | 74 | Update all asset paths to new structure |
| `research/utilities/signals.py` | NEW | Signal registry |

### Tests That Will Need Import Updates

| File | Current Imports | Must Update To |
|------|----------------|----------------|
| `research/tests/dsp/test_wavelet.py` | `from subshader.dsp.wavelet import PyWavelet, NumPyWavelet` | `PywaveletCWT`, `CpuCWT` from new modules |
| `research/tests/dsp/test_wavelet_kernel.py` | (no class renames expected) | `wavelet_kernel.py` stays |
| `research/tests/audio/test_audio_overlap.py` | likely imports AudioInput | `AudioStream` or `reader.py` |
| `research/tests/viz/test_plotter.py` | imports CircularFrameBuffer | `from subshader.renderer.frame_buffer import CircularFrameBuffer` |
| `research/tests/conftest.py` | `get_default_config`, `_make_wavelet` with old class names | Update wavelet class references |

---

## Architecture Patterns

### Recommended Final Directory Structure

```
src/subshader/
├── __init__.py
├── __main__.py          # ~15 lines: argparse + main()
├── pipeline.py          # SubShader class: __init__, run(), cleanup()
├── config.py            # PipelineConfig, CWTConfig, RendererConfig, ColorNormalizationConfig
├── exceptions.py        # unchanged
│
├── audio/
│   ├── __init__.py      # exports AudioStream only
│   ├── audio_stream.py  # AudioStream facade
│   ├── reader.py        # was audio_input.py
│   └── player.py        # was audio_player.py
│
├── dsp/
│   ├── __init__.py      # exports DSP, CpuCWT, GpuCWT, STFT, PywaveletCWT
│   ├── dsp.py           # DSP ABC (pre, transform, post)
│   ├── cwt.py           # CpuCWT(DSP), GpuCWT(DSP) — was wavelet.py
│   ├── stft.py          # STFT(DSP) — extracted from dsp_helpers.py
│   ├── pywavelet.py     # PywaveletCWT(DSP) — extracted from wavelet.py
│   ├── wavelet_kernel.py  # unchanged
│   └── gaussian.py      # unchanged
│
├── renderer/            # was viz/
│   ├── __init__.py      # exports Renderer (the ShaderPlot facade)
│   ├── renderer.py      # ShaderPlot renamed Renderer + GLContext
│   ├── frame_buffer.py  # CircularFrameBuffer, AudioFrameBuffer
│   ├── intensity.py     # IntensityTracker
│   └── shaders/
│       ├── vertex.glsl    # was vertex_shader.glsl
│       └── fragment.glsl  # was fragment_shader.glsl
│       # no __init__.py — load via Path(__file__).parent
│
└── utils/
    ├── __init__.py      # unchanged
    ├── logging.py       # unchanged
    ├── timing.py        # unchanged (@timed decorator)
    ├── gpu.py           # unchanged
    ├── loop_timer.py    # unchanged
    └── os_env_setup.py  # unchanged

research/
├── test_suite.py        # simplified: 4 modes only
├── timing.py            # updated to new module names + renderer
├── comparison.py        # keep utility, not README default
├── figures.py           # refactored to reuse compare-methods code
├── utilities/
│   ├── __init__.py      # update exports
│   ├── style.py         # unchanged
│   ├── plotting.py      # unchanged
│   ├── printing.py      # unchanged
│   ├── timing.py        # add timing_template.txt support
│   ├── constants.py     # update all asset paths
│   ├── signals.py       # NEW: signal registry
│   ├── dsp_helpers.py   # chirp builders stay; compute_stft_frame moves to src
│   ├── timing_template.txt  # NEW: editable timing output template
│   └── wav_export.py    # unchanged
└── tests/               # update all import paths

assets/
├── audio/
│   ├── reference/       # was: daw/ + songs/ + figures/
│   └── generated/       # was: figures/ (synthesized signals)
├── images/
│   ├── reference/       # was: images/figures/ (edison screenshots, diffs)
│   └── generated/       # was: images/benchmarks/ (comparison grids, timing)
├── timing/              # NEW: timestamped timing output files
├── plots/               # unchanged (architecture diagrams)
└── archive/             # NEW: old/unused files
    ├── audio/
    ├── images/
    ├── claude/
    └── diagnostics/
```

### Pattern 1: AntsWavelet Flattening (Claude's Discretion)

The current hierarchy is: `Wavelet` → `AntsWavelet` → `NumPyWavelet`/`CuPyWavelet`.

`AntsWavelet` holds shared state that both CPU and GPU backends need:
- `self.wavelets` (list of WaveletKernel objects)
- `self.max_conv_n` (int)
- `self.kernel_f_bank` (np.ndarray, complex64)
- `self.reliable_slice` (slice)
- `_create_reliable_slice()`, `_create_coi_mask()` helpers

**Recommendation:** Merge `AntsWavelet` logic into a new `CWT` base class that sits between `DSP` and the concrete `CpuCWT`/`GpuCWT`. The `CWT` class holds all kernel construction, reliable-slice logic, and shared pipeline stages (`normalize_by_scale` no-op, `discard_unreliable_coefs`, `extract_hop_center`, `downsample`). `CpuCWT` and `GpuCWT` each implement only `class_specific_cwt` (renamed to `transform`). This keeps the hierarchy flat at two levels.

```
DSP (ABC: pre, transform, post)
└── CWT (base: kernel construction, reliable slice, discard, hop center, downsample)
    ├── CpuCWT (implements: transform = NumPy FFT convolution)
    └── GpuCWT (implements: transform = CuPy FFT convolution + GPU upload)
```

`PywaveletCWT` inherits directly from `DSP` — it has its own scale logic and doesn't share the WaveletKernel machinery.

### Pattern 2: Config Inheritance

Current: nested composition (`ProcessingConfig.audio`, `.wavelet`, `.viz`).
New: flat inheritance (`CWTConfig(PipelineConfig)`, `RendererConfig(PipelineConfig)`).

Key constraint: `validate()` methods must remain on each config class. The base `PipelineConfig.validate()` covers file path and shared params; `CWTConfig.validate()` extends it with wavelet-specific checks; `RendererConfig.validate()` extends it with rendering checks.

```python
@dataclass
class PipelineConfig:
    file_path: str = "assets/audio/reference/beltran_sc_rip.wav"
    chunk_size: int = 1 << 14
    overlap_factor: float = 0.5
    sample_rate: float = 44100.0       # written back by AudioStream
    total_samples: int = 0             # written back by AudioStream

    @property
    def hop_size(self) -> int:
        return int(self.chunk_size * (1.0 - self.overlap_factor))

    @property
    def nyquist_freq(self) -> float:
        return self.sample_rate / 2.0

    def validate(self) -> list[str]: ...

@dataclass
class CWTConfig(PipelineConfig):
    notes_per_octave: int = 12
    num_octaves: int = 10
    root_note_hz: float = 27.5
    target_width: int = 64
    num_cycles: int = 6
    num_fwhm_cycles: int = 3

@dataclass
class RendererConfig(PipelineConfig):
    num_frames: int = 256
    gamma: float = 0.5
    color_norm: ColorNormalizationConfig = field(default_factory=ColorNormalizationConfig)
```

The pipeline.py `SubShader` uses a single `CWTConfig` (or `RendererConfig`) instance that all three modules share. Since both are subclasses of PipelineConfig, one concrete config object that has all the fields works for passing to `AudioStream`, `GpuCWT`, and `Renderer`.

**Practical approach for the main pipeline:** Create one `CWTConfig` object (it's the most param-rich), pass it everywhere. `Renderer` only reads `PipelineConfig` fields + the `color_norm` param, so it works with any subclass. This avoids creating separate config instances.

### Pattern 3: AudioStream Facade

The critical behavioral detail: `AudioStream` must handle the audio-clock sync logic currently in `__main__.py`. The existing render loop polls `audio_player.get_playback_sample()`, compares to `next_expected_sample`, sleeps 1ms when not ready, and seeks `audio_input.file_pos` on each iteration.

`AudioStream.next_chunk()` (blocking variant) should internalize this polling loop so `pipeline.py` sees a clean blocking call. The `time.sleep(0.001)` yield stays inside `next_chunk()`.

```python
class AudioStream:
    def next_chunk(self) -> np.ndarray | None:
        """Block until audio clock advances, then return next chunk."""
        while True:
            if self.player.has_looped():
                self._handle_loop_reset()
            playback_pos = self.player.get_playback_sample()
            if playback_pos >= self._next_expected_sample:
                return self._seek_and_read(playback_pos)
            time.sleep(0.001)
```

The `get_chunk()` non-blocking variant remains for the research/timing harness which controls its own loop.

### Pattern 4: Shaders Without __init__.py

Current: `shaders/__init__.py` with `get_vertex_shader_source()` / `get_fragment_shader_source()` functions.
New: Direct path loading in `renderer.py` using `Path(__file__).parent / "shaders"`:

```python
def _load_shader(name: str) -> str:
    shader_path = Path(__file__).parent / "shaders" / name
    return shader_path.read_text()
```

The `shaders/__init__.py` is removed (D-17). The two GLSL files are renamed. `renderer.py` loads them directly.

### Anti-Patterns to Avoid

- **Import cycles:** `pipeline.py` imports from `audio/`, `dsp/`, `renderer/`. None of these packages should import from `pipeline.py`. Keep the dependency arrow one-way downward.
- **Module-level side effects during import:** Current `__main__.py` calls `logger_init()` and `get_default_config()` at module level. These must move inside `main()`. The new `__main__.py` should have zero module-level side effects.
- **config.py importing GPU/audio:** Current `config.py` imports `tkinter` for display detection and has GPU/CPU memory validation that estimates GPU usage. These validations reference `self.audio.chunk_size` etc. In the new design, `ProcessingConfig.validate()` becomes `PipelineConfig.validate()` plus subclass validators. The GPU memory estimate logic can stay in `CWTConfig.validate()` since it references CWT-specific params.
- **Breaking research/tests mid-wave:** Tests import `from subshader.dsp.wavelet import PyWavelet, NumPyWavelet`. Any wave that moves these classes must update test imports in the same commit.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio clock blocking wait | Custom event loop | `time.sleep(0.001)` poll (already proven) | Pattern works, low latency, simple |
| Shader file loading | Custom resource resolver | `pathlib.Path(__file__).parent / "shaders"` | Standard Python path |
| `@timed` decorator | New implementation | Existing `src/subshader/utils/timing.py` | Already handles all stages, just extend to new method names |
| Signal registry | Database or YAML | Simple Python list of dicts in `signals.py` | One-append extensibility, no dependency |
| Timing output template | Jinja2 or custom templating | Simple `str.format()` or `str.replace()` with named placeholders | Minimal, editable text file |

---

## Common Pitfalls

### Pitfall 1: Changing Constructor Signatures Breaks Test Conftest

**What goes wrong:** `conftest.py` uses `_make_wavelet(cls, config)` which calls `cls(sample_rate=sr, input_n=chunk, config=config.wavelet)`. After the config redesign, `CpuCWT(config)` takes a single `PipelineConfig` — the constructor signature changes entirely.

**Why it happens:** The old `Wavelet.__init__` took `(sample_rate, input_n, config, overlap_factor)` explicitly. The new DSP ABC takes `(config)` only, with all values read from config.

**How to avoid:** Update `conftest.py` and all test files in the same wave as the DSP refactor. Plan this as one atomic task, not two separate ones.

**Warning signs:** `TypeError: __init__() got an unexpected keyword argument` in pytest.

### Pitfall 2: viz/__init__.py Re-exports comparison_navigator Classes

**What goes wrong:** Current `src/subshader/viz/__init__.py` exports `KernelNavigator, TransformNavigator, TopLevelComparisonNavigator` from `comparison_navigator.py`. If `viz/` is renamed to `renderer/` and `comparison_navigator.py` is archived before the `__init__.py` is updated, any code that does `from subshader.viz import ...` will fail.

**Why it happens:** The `viz/__init__.py` re-export exists, and the comparison_navigator is being archived.

**How to avoid:** When archiving `comparison_navigator.py`, replace `viz/__init__.py` with a minimal `renderer/__init__.py` that only exports `Renderer` (the renamed ShaderPlot). Do this in one commit.

**Warning signs:** `ImportError: cannot import name 'KernelNavigator' from 'subshader.viz'`

### Pitfall 3: WaveletConfig.typical_sampling_freq vs PipelineConfig.sample_rate

**What goes wrong:** Current `Wavelet.__init__` checks `if sample_rate != self.config.typical_sampling_freq` and raises `ValueError` if they differ. In the new design, `sample_rate` lives on `PipelineConfig` (written by AudioStream). The old `WaveletConfig.typical_sampling_freq` field disappears. If this validation is carried forward, it needs to compare against `config.sample_rate` instead.

**Why it happens:** The old architecture had separate configs; the new one unifies them. The validation logic must be updated to reference the correct field.

**How to avoid:** In `cwt.py`, validate `config.sample_rate > 0` and that it's been set (not the default 0 from a fresh config before AudioStream runs). Do NOT replicate the old sample-rate mismatch check — AudioStream is now the sole writer of sample_rate.

### Pitfall 4: Asset Paths in constants.py and Default config file_path

**What goes wrong:** `research/utilities/constants.py` has hardcoded paths like `"assets/audio/figures/bouncing_chirp.wav"` and `"assets/audio/songs/beltran_sc_rip.wav"`. After asset reorganization to `reference/` and `generated/`, every one of these constants breaks.

Current mapping:
- `assets/audio/figures/` → `assets/audio/generated/`
- `assets/audio/songs/` → `assets/audio/reference/`
- `assets/audio/daw/` → `assets/audio/reference/` (for the ones being kept)
- `assets/images/figures/` → `assets/images/reference/`
- `assets/images/benchmarks/` → `assets/images/generated/`

**How to avoid:** Plan the asset reorganization wave to include constants.py and config.py default path updates in the same commit as the git mv operations.

### Pitfall 5: @timed on AudioStream Methods — Thread Safety

**What goes wrong:** `@timed` stores timing as `self._timing_{method_name}_ms`. `AudioStream.get_chunk()` and `AudioStream.next_chunk()` are called from the main thread, but `AudioPlayer._callback()` runs on the sounddevice OS thread. If `@timed` is accidentally applied to `_callback`, the attribute write from the audio thread and read from the main thread is unsynchronized.

**Why it happens:** Blanket application of `@timed` to all pipeline stages without checking which methods run on which thread.

**How to avoid:** Apply `@timed` only to `get_chunk()`, `next_chunk()`, and `start()` on `AudioStream` — not to `AudioPlayer._callback()` or any internal player method. The `@timed` decorator is not thread-safe (no lock on the attribute write).

### Pitfall 6: Shaders Path After rename

**What goes wrong:** After renaming `viz/shaders/vertex_shader.glsl` → `renderer/shaders/vertex.glsl`, any hardcoded path string breaks. The current `shaders/__init__.py` uses `Path(__file__).parent / filename` which is relative to the shaders package — this self-heals after the directory move. The GLSL filename strings must be updated too.

**How to avoid:** In the same commit that renames the GLSL files and removes `shaders/__init__.py`, update `renderer.py` to call `_load_shader("vertex.glsl")` and `_load_shader("fragment.glsl")`.

---

## Code Examples

### New pipeline.py (target form)

```python
# Source: orchestrator-design.md
class SubShader:
    def __init__(self, config: PipelineConfig):
        self.audio    = AudioStream(config)
        self.dsp      = GpuCWT(config) if gpu_available() else CpuCWT(config)
        self.renderer = Renderer(config, self.dsp.output_shape)

    def run(self) -> None:
        self.audio.start()
        while not self.renderer.should_close():
            chunk = self.audio.next_chunk()
            coefs = self.dsp.process(chunk)
            self.renderer.update(coefs)

    def cleanup(self) -> None:
        self.audio.cleanup()
        self.dsp.cleanup()
        self.renderer.cleanup()
```

### New __main__.py (target form)

```python
from subshader.pipeline import SubShader
from subshader.config import CWTConfig
from subshader import exceptions

def main() -> None:
    args = _parse_args()
    config = CWTConfig(file_path=args.audio_file) if args.audio_file else CWTConfig()
    pipeline = SubShader(config)
    try:
        pipeline.run()
    except exceptions.GRACEFUL_EXCEPTIONS as e:
        exceptions.reporter.report(e)
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()
```

### DSP ABC (target form)

```python
# Source: dsp-design.md
class DSP(ABC):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(self, chunk: np.ndarray) -> np.ndarray:
        data = self.pre(chunk)
        raw  = self.transform(data)
        return self.post(raw)

    @abstractmethod
    def pre(self, chunk: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def post(self, raw: np.ndarray) -> np.ndarray: ...
```

### AudioStream facade (target form)

```python
class AudioStream:
    def __init__(self, config: PipelineConfig) -> None:
        self._reader = Reader(config)
        # Write discovered values back into shared config object (D-06)
        config.sample_rate    = float(self._reader.sample_rate)
        config.total_samples  = self._reader.total_samples
        self._player = Player(config)
        self._config = config
        self._next_expected_sample = 0

    def start(self) -> None:
        self._player.start()

    def get_chunk(self) -> np.ndarray | None:
        return self._reader.get_chunk()

    def next_chunk(self) -> np.ndarray | None:
        """Block until audio clock advances, seek to match, return chunk."""
        while True:
            if self._player.has_looped():
                self._handle_loop_reset()
            pos = self._player.get_playback_sample()
            if pos >= self._next_expected_sample:
                return self._seek_and_read(pos)
            time.sleep(0.001)
```

### Signal registry (target form)

```python
# research/utilities/signals.py
SIGNALS = [
    {
        "name": "chirp",
        "label": "Bouncing Chirp",
        "audio": "assets/audio/generated/bouncing_chirp.wav",
        "reference": "assets/images/reference/bouncing_chirp_edison.png",
        "type": "synthetic",
    },
    {
        "name": "polyphonic",
        "label": "MIDI Sine Waves",
        "audio": "assets/audio/reference/midi_sine_waves.wav",
        "reference": "assets/images/reference/midi_sine_wave_edison.png",
        "type": "file",
    },
    {
        "name": "musical",
        "label": "Beltran (4 Bars)",
        "audio": "assets/audio/reference/beltran_sc_rip_4_bar.wav",
        "reference": "assets/images/reference/beltran_4_bar_edison.png",
        "type": "file",
    },
]
```

---

## Runtime State Inventory

This is a rename/refactor phase — checking all five runtime state categories.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | None — no database, no Mem0, no Redis in this project | None |
| Live service config | None — desktop app, no external services, no n8n | None |
| OS-registered state | None — no Task Scheduler tasks, no systemd units, no pm2 processes | None |
| Secrets/env vars | `SUBSHADER_DEBUG`, `DISPLAY`, `LIBGL_ALWAYS_SOFTWARE`, `MESA_GL_VERSION_OVERRIDE` — env var names unchanged; code reads them by name in `os_env_setup.py` which is not being modified | None |
| Build artifacts | `venv/` install of `subshader` package — `pyproject.toml` package name stays `subshader`, entry point `subshader.__main__:main` stays the same. Re-install not needed unless pyproject.toml entry point changes | None — entry point unchanged |

**Nothing runtime-stateful is changing** — only Python file locations and class names within the package.

One important note: `pyproject.toml` references the entry point as `subshader.__main__:main`. After the refactor, `main()` still lives in `__main__.py` (just much shorter). No pyproject.toml changes needed.

---

## Environment Availability Audit

Step 2.6: SKIPPED — this is a code/config-only refactor. No new external tools, services, CLIs, or runtimes are being added. All existing dependencies (CuPy, ModernGL, sounddevice, etc.) are already installed and verified working from Phase 7.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (version from venv) |
| Config file | `pyproject.toml` (pythonpath includes `src/subshader` and `research/tests`) |
| Quick run command | `pytest research/tests/ -v` |
| Full suite command | `pytest research/tests/ -v` |

### Phase Requirements → Test Map

This phase has no formal REQ-IDs. The implicit requirements are behavioral: all existing tests pass after the refactor, and the pipeline runs correctly.

| Behavior | Test Type | Automated Command | File Exists? |
|----------|-----------|-------------------|-------------|
| CWT output shape unchanged after class rename | unit | `pytest research/tests/dsp/test_wavelet.py -x` | Yes (needs import updates) |
| CWT normalization unchanged | unit | `pytest research/tests/dsp/test_wavelet.py::test_kernel_energy_per_scale -x` | Yes |
| AudioInput chunk reading unchanged | unit | `pytest research/tests/audio/test_audio_overlap.py -x` | Yes |
| Renderer frame buffer unchanged | unit | `pytest research/tests/viz/test_plotter.py -x` | Yes |
| Pipeline runs end-to-end | smoke | `python -m subshader` (manual, needs display) | Manual only |

### Wave 0 Gaps

No new test files needed. Existing test files need import path updates, not new tests. The import updates happen in the same wave as the source file moves.

---

## Import Chain Map

Critical: imports must be updated atomically with the source file that moves. This table is the planning dependency map.

| Current Import | Updated Import | Used In |
|---------------|----------------|---------|
| `from subshader.dsp.wavelet import CuWavelet, NpWavelet` | `from subshader.dsp.cwt import GpuCWT, CpuCWT` | `__main__.py` → `pipeline.py` |
| `from subshader.dsp.wavelet import NumPyWavelet` | `from subshader.dsp.cwt import CpuCWT` | `research/timing.py` |
| `from subshader.dsp.wavelet import PyWavelet, NumPyWavelet` | `from subshader.dsp.pywavelet import PywaveletCWT`, `from subshader.dsp.cwt import CpuCWT` | `research/tests/dsp/test_wavelet.py` |
| `from subshader.dsp.wavelet import CuPyWavelet` | `from subshader.dsp.cwt import GpuCWT` | `research/tests/dsp/test_wavelet.py` |
| `from subshader.viz.plotter import ShaderPlot` | `from subshader.renderer.renderer import Renderer` | `__main__.py` → `pipeline.py` |
| `from subshader.viz.plotter import CircularFrameBuffer` | `from subshader.renderer.frame_buffer import CircularFrameBuffer` | `research/timing.py` |
| `from subshader.viz.plotter import CircularFrameBuffer` | same, new path | `research/tests/viz/test_plotter.py` |
| `from subshader.audio.audio_input import AudioInput` | `from subshader.audio.audio_stream import AudioStream` | `__main__.py` → `pipeline.py`, `research/timing.py` |
| `from subshader.audio.audio_input import AudioInput` | `from subshader.audio.reader import Reader` (internal to AudioStream) | `research/tests/audio/test_audio_overlap.py` (may use AudioInput directly) |
| `from subshader.audio.audio_player import AudioPlayer` | internal to `AudioStream` | `__main__.py` → removed (facade hides it) |
| `from subshader.config import get_default_config, ProcessingConfig` | `from subshader.config import CWTConfig` (+ no get_default_config) | everywhere |
| `from subshader.config import WaveletConfig` | `from subshader.config import CWTConfig` | `research/tests/dsp/test_wavelet.py`, conftest.py |
| `from subshader.config import VisualizationConfig` | `from subshader.config import RendererConfig` | `research/tests/viz/test_plotter.py` |
| `from .shaders import get_vertex_shader_source, get_fragment_shader_source` | direct `Path(__file__).parent / "shaders" / "vertex.glsl"` | `renderer/renderer.py` |
| `from ..config import WaveletConfig` | `from ..config import CWTConfig` | `src/subshader/dsp/cwt.py` |

---

## Asset Path Migration Map

Current → New (for updating constants.py and any hardcoded references):

| Current Path | New Path | Used In |
|-------------|---------|---------|
| `assets/audio/daw/a2a3_a4_minor_scale.wav` | `assets/audio/reference/a2a3_a4_minor_scale.wav` | config.py default (was DAW test fixture) |
| `assets/audio/songs/beltran_sc_rip.wav` | `assets/audio/reference/beltran_sc_rip.wav` | constants.py `AUDIO_BELTRAN`, new default (D-04) |
| `assets/audio/songs/beltran_sc_rip_16_bar.wav` | `assets/audio/reference/beltran_sc_rip_16_bar.wav` | constants.py `AUDIO_BELTRAN_16BAR` |
| `assets/audio/songs/beltran_sc_rip_8_bar.wav` | `assets/audio/reference/beltran_sc_rip_8_bar.wav` | constants.py `AUDIO_BELTRAN_8BAR` |
| `assets/audio/figures/beltran_sc_rip_4_bar.wav` | `assets/audio/reference/beltran_sc_rip_4_bar.wav` | constants.py `AUDIO_BELTRAN_4BAR`, signals.py |
| `assets/audio/figures/bouncing_chirp.wav` | `assets/audio/generated/bouncing_chirp.wav` | constants.py `AUDIO_BOUNCING_CHIRP`, signals.py |
| `assets/audio/figures/midi_sine_waves.wav` | `assets/audio/reference/midi_sine_waves.wav` | constants.py `AUDIO_MIDI_SINE_WAVES`, signals.py |
| `assets/images/figures/bouncing_chirp_edison.png` | `assets/images/reference/bouncing_chirp_edison.png` | constants.py, signals.py |
| `assets/images/figures/midi_sine_wave_edison.png` | `assets/images/reference/midi_sine_wave_edison.png` | constants.py, signals.py |
| `assets/images/figures/beltran_sc_rip_4_bar_edison.png` | `assets/images/reference/beltran_4_bar_edison.png` | constants.py, signals.py |
| `assets/images/benchmarks/` (generated outputs) | `assets/images/generated/` | constants.py `BENCHMARKS_DIR` |
| `assets/images/benchmarks/stubs/` | `assets/images/generated/stubs/` | constants.py `BENCHMARKS_STUBS_DIR` |
| `assets/images/benchmarks/numpy_vs_cupy_diff.png` | `assets/images/reference/numpy_vs_cupy_diff.png` | DSP.md reference |

Files going to archive (not moved to new structure):
- `assets/audio/daw/` files not listed above → `assets/archive/audio/`
- `assets/images/diagnostics/` → `assets/archive/diagnostics/`
- `assets/images/claude/` → `assets/archive/claude/`
- `assets/images/benchmarks/dpi/` comparison_grid_*dpi.png → `assets/archive/images/`

---

## Files to Archive (Research)

Per D-37, these go to `research/archive/`:

| File | Current Location | Reason |
|------|-----------------|--------|
| `comparison_navigator.py` | `src/subshader/viz/` | Legacy PyQtGraph, not in pipeline |
| `WaveletDesign.md` | `src/subshader/dsp/` | Superseded by dsp-design.md |
| `GPU_BUFFER_IDEA.md` | `src/subshader/viz/` | Design note, not active |
| `frame_counter_pyqt5.py` | unknown | Per D-37 |
| `gl_diagnostics.py` | unknown | Per D-37 |
| `quick_plot.py` | unknown | Per D-37 |
| `pipeline_timing_profile.py` | `research/` | Per D-37 |
| `benchmark-timing-template.txt` | `research/` | Per D-37 |
| `benchmark_results.csv` | `research/` | Per D-37 |
| `branch_goals.md` | `research/` | Per D-37 |
| `font_showcase_20.py` | `research/utilities/` | Claude's discretion: archive (not production code) |
| `README_pipeline_timing.md` | `research/` | Old doc |

---

## Open Questions

1. **`get_default_config()` removal**
   - What we know: Currently called at module level in `__main__.py` and in `research/timing.py`
   - What's unclear: Does anything else call `get_default_config()`? Tests use it via conftest.py.
   - Recommendation: Search all call sites before removing. Replace with `CWTConfig()` direct instantiation. Update conftest.py to use `CWTConfig()`.

2. **`AudioFrameBuffer` in plotter.py (lines 591+)**
   - What we know: Defined in `plotter.py` as a circular buffer for 1D audio chunks. Not obviously used in the main pipeline or timing harness.
   - What's unclear: Is it used anywhere in current tests or figures code?
   - Recommendation: Grep for usage before planning. If used, move to `frame_buffer.py`. If unused, archive.

3. **pyqtgraph import in plotter.py**
   - What we know: `import pyqtgraph as pg` is at the top of `plotter.py`. There are PyQtGraph classes in the lower portion of the file.
   - What's unclear: Whether removing the import causes an import-time error on systems without PyQtGraph installed.
   - Recommendation: When splitting plotter.py, the PyQtGraph classes go to `research/archive/`. The `import pyqtgraph` line in `renderer.py` should be removed entirely. Verify no runtime dependency on pyqtgraph remains in the main pipeline.

4. **`typical_sampling_freq` in WaveletConfig**
   - What we know: Used for sample rate validation in `Wavelet.__init__`. In the new design, `sample_rate` lives on `PipelineConfig` and is written by `AudioStream`.
   - What's unclear: Should `CWTConfig` keep `typical_sampling_freq` for documentation/assertion purposes, or remove it entirely?
   - Recommendation: Remove `typical_sampling_freq`. The sample rate check in `CWT.__init__` should verify `config.sample_rate > 0` (i.e., AudioStream has run before CWT is called).

---

## Sources

### Primary (HIGH confidence)

- Direct source code inspection: all files above read at actual line counts and content
- `08-CONTEXT.md` — all decisions locked, verified against current code
- Design documents (`config-design.md`, `orchestrator-design.md`, `test-suite-design.md`, `dsp-design.md`) — canonical pseudocode for target state

### Secondary (MEDIUM confidence)

- Python dataclass inheritance behavior — standard Python, no version quirks at 3.12
- `pathlib.Path(__file__).parent` for relative resource loading — well-established pattern

---

## Metadata

**Confidence breakdown:**
- Current codebase inventory: HIGH — all files read directly
- Target structure: HIGH — locked in design documents
- Import chain mapping: HIGH — derived from actual imports in source files
- Asset path mapping: HIGH — derived from actual directory listing + constants.py
- Pitfalls: HIGH — derived from actual code patterns observed

**Research date:** 2026-04-06
**Valid until:** N/A — pure code refactor, no external dependencies to go stale
