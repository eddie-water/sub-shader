# Phase 7: Visual Style System and Frequency Range Configuration - Research

**Researched:** 2026-03-27
**Domain:** Python/matplotlib style consolidation, argparse refactor, decorator instrumentation, pytest test reorganization
**Confidence:** HIGH

## Summary

Phase 7 is a structured refactor across four independent concerns: (1) consolidating all matplotlib visual constants into a single `research/utilities/style.py` module, (2) fixing the comparison grid column header top-margin and centering, (3) confirming that `WaveletConfig.root_note_a0_hz` and `num_octaves` are already wired as the correct frequency-range controls with Nyquist trimming already in place, and (4) restructuring the research toolkit into a coherent dispatcher/module tree with centralized tests.

No new dependencies are introduced. This phase works entirely within the existing Python/matplotlib/pytest stack the project already uses. The scope is well-scoped refactoring: every change is isolated to `research/` except the `@timed` decorator (which may optionally touch `src/subshader/dsp/wavelet.py` if the approach shifts from `cwt_timed()` to a decorator) and the `WaveletConfig` additions.

**Primary recommendation:** Treat the five deliverables as independent plans that can be executed in any order. The style system (D-01 through D-05) and the research toolkit reorganization (D-15 through D-25) are the highest-effort items and should be first to unblock all other figure-generating work.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Style Consolidation**
- D-01: Create `research/utilities/style.py` as the single source of truth for all visual constants. Every color, fontsize, linewidth, alpha, figsize, spacing value lives here.
- D-02: All figure functions import from style.py — no hardcoded visual values anywhere in figure code.
- D-03: Style values exposed as module-level constants (e.g., `style.BG_COLOR`, `style.FONT_SIZE`). Not dicts, not dataclasses.
- D-04: One canonical dark style. Kill the matplotlib/seaborn backend toggle and the `DEFAULT_STYLE`/`SEABORN_STYLE` dict pattern in `plotting.py`.
- D-05: Style system designed for reusability — plotting mechanisms should work for future documentation figures.

**Comparison Grid Header Fix**
- D-06: Fix column title top margin — titles too close to top edge, need more breathing room.
- D-07: Fix column title horizontal centering — titles misaligned relative to spectrogram columns.

**Frequency Range Configuration**
- D-08: Keep A0 (27.5Hz) as default root note. Sub-bass deliberately kept.
- D-09: `root_note_a0_hz` and `num_octaves` remain the configurable parameters in WaveletConfig.
- D-10: No new Nyquist clamping code needed — `_generate_chromatic_scale()` already trims at line 127.

**Pipeline Timing Architecture**
- D-11: Add `@timed` decorator to pipeline methods. Decorator lives in `research/utilities/timing.py` (or `src/subshader/utils/`).
- D-12: ~1 microsecond overhead per decorated call. Negligible.
- D-13: Timing data always available — no special timed version of the pipeline needed.
- D-14: `research/timing.py` becomes a thin reporting layer.

**Research Toolkit Reorganization**
- D-15: Dispatcher renamed from `benchmark.py` to `test_suite.py`. Single entry point: `--test`, `--timing`, `--comparison`, `--figures`.
- D-16: `research/timing.py` — thin pipeline profiler.
- D-17: `research/comparison.py` — method-vs-method figures and timing table. Methods as config list.
- D-18: `research/utilities/` — reusable library: style.py, plotting.py, signals.py, wav_export.py, printing.py, timing.py, dsp_helpers.py.
- D-19: `wav_export.py` moves from research/ root into `research/utilities/`.
- D-20: `ants/`, `docs/`, `gpu_basics/`, `misc/`, `python/` move to `research/archive/`.

**Unit Test Organization**
- D-21: All tests move from colocated positions in `src/` to `research/tests/`.
- D-22: `research/tests/` mirrors src/ structure: `research/tests/audio/`, `research/tests/dsp/`, `research/tests/viz/`.
- D-23: `test_suite.py` discovers and runs tests via pytest.

**Comparison Method Extensibility**
- D-24: Comparison methods defined as config list in `comparison.py`. Each entry: `{name, compute_function, label}`.
- D-25: Adding a new method = define compute function + append one entry.

### Claude's Discretion
- Exact `@timed` decorator implementation details (where timing data is stored, thread safety)
- Internal layout of style.py constant groupings (by concern vs alphabetical)
- Exact gridspec adjustments for header margin/centering fix
- test_suite.py CLI argument parsing approach (argparse, click, or manual)
- How to handle the seaborn import and backend removal (deprecation warning vs clean removal)

### Deferred Ideas (OUT OF SCOPE)
- pytest-benchmark integration for CI-friendly rigorous benchmarking
- `.mplstyle` file approach (matplotlib native theming)
- Auto-detection of audio frequency content to dynamically adjust range
- Detailed DSP.md foundation figures — Phase 5 plan 05-04 scope
</user_constraints>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | installed (project dep) | Figure generation, gridspec layout, subplots_adjust | Already the project's only figure backend after seaborn removal |
| pytest | >=7.0 (dev dep) | Test discovery and execution | Already configured in pyproject.toml with pythonpath |
| Python stdlib: functools, time | — | `@timed` decorator (functools.wraps), perf_counter | Zero new dependencies |
| argparse | stdlib | test_suite.py CLI dispatcher | Already used in benchmark.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | project dep | Style-agnostic: timing arrays, stats in tests | Always present |
| seaborn | optional (removable) | Currently drives backend toggle; removing in D-04 | Phase 7 kills the dependency path |

**Installation:** No new packages required. This phase adds zero dependencies.

---

## Architecture Patterns

### Recommended Project Structure After Phase 7
```
research/
├── test_suite.py          # Renamed from benchmark.py — single CLI entry point
├── timing.py              # Thin profiler: reads @timed data from decorated methods
├── comparison.py          # Method-vs-method figures + timing table
├── figures.py             # Retained: generate_comparison_grid(), ReadmeFigures
├── pipeline_timing_profile.py  # Retained as-is (detailed sub-stage profiler)
├── utilities/
│   ├── __init__.py        # Updated exports
│   ├── style.py           # NEW: all visual constants as module-level names
│   ├── plotting.py        # STRIPPED: DEFAULT_STYLE/SEABORN_STYLE/backend toggle removed; primitives retained
│   ├── timing.py          # EXTENDED: add @timed decorator + MethodTimingStore
│   ├── constants.py       # Retained: audio paths, DSP params
│   ├── dsp_helpers.py     # Retained as-is
│   ├── printing.py        # Retained as-is
│   ├── signals.py         # NEW (or rename of dsp_helpers chirp section): chirp generators, test signals
│   └── wav_export.py      # MOVED from research/ root
├── tests/
│   ├── __init__.py
│   ├── audio/
│   │   └── test_audio_overlap.py    # MOVED from src/subshader/audio/
│   ├── dsp/
│   │   ├── test_wavelet.py          # MOVED from src/subshader/dsp/
│   │   └── test_wavelet_kernel.py   # MOVED from src/subshader/dsp/
│   └── viz/
│       └── test_plotter.py          # MOVED from src/subshader/viz/
└── archive/
    ├── ants/
    ├── docs/
    ├── gpu_basics/
    ├── misc/
    └── python/
```

### Pattern 1: Module-Level Style Constants
**What:** All visual values live in `research/utilities/style.py` as plain module-level names. Consumer code does `import research.utilities.style as style` then uses `style.BG_COLOR`, `style.LABEL_FONT_SIZE`, etc.
**When to use:** Any figure-generating function in research/ — figures.py, comparison.py, plotting.py primitives, future documentation figures.
**Example:**
```python
# research/utilities/style.py
BG_COLOR = "#1A1A1A"
WAVEFORM_COLOR = "#ffffcf"
GRID_CMAP = "inferno"
LABEL_FONT_SIZE = 24
TICK_LABEL_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 32
SUBTITLE_FONT_SIZE = 24
FIGURE_WIDTH = 20
ROW_HEIGHT = 4
HSPACE = 0.22
LEFT_MARGIN = 0.06
RIGHT_MARGIN = 0.94
BOTTOM_MARGIN = 0.04
TOP_MARGIN = 0.90
SUPTITLE_Y = 0.975
SUBTITLE_Y = 0.925
DEFAULT_DPI = 150
WAVEFORM_ALPHA = 0.75
FREQ_LINE_COLOR = "#AAAAAA"
FREQ_LINE_WIDTH = 2
```

### Pattern 2: @timed Decorator with Thread-Local Storage
**What:** A function decorator stored in `research/utilities/timing.py` (or `src/subshader/utils/timing.py`) that wraps a method with `time.perf_counter` and stores the last call's elapsed time on the instance.
**When to use:** Pipeline methods where timing should always be available without a separate code path. The key constraint from D-13 is that timing data is accessible from any call, not just from a special `cwt_timed()` variant.
**Recommendation (Claude's discretion):** Store last elapsed time as an instance attribute (`_timing_<method_name>_ms`). This is thread-safe for single-threaded use (SubShader's current model), zero-friction to access, and requires no external state. No lock needed since SubShader pipeline is single-threaded.

```python
# research/utilities/timing.py  (or src/subshader/utils/timing.py)
import time
import functools

def timed(method):
    """Decorator: wraps a method, stores last elapsed ms as instance attr."""
    attr = f"_timing_{method.__name__}_ms"
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = method(self, *args, **kwargs)
        setattr(self, attr, (time.perf_counter() - t0) * 1000.0)
        return result
    return wrapper
```

**Access pattern:**
```python
wavelet = CuWavelet(...)
result = wavelet.cwt(data)
# timing always available:
elapsed_ms = wavelet._timing_cwt_ms
```

### Pattern 3: Comparison Method Config List
**What:** In `research/comparison.py`, the set of methods to compare is a plain Python list of dicts. Grid iteration and timing table both loop over this list.
**When to use:** D-24/D-25 — adding a new wavelet backend requires only one list append.
```python
# research/comparison.py
COMPARISON_METHODS = [
    {"name": "stft",      "fn": compute_stft_frame,   "label": "STFT"},
    {"name": "pywavelet", "fn": run_pywavelet,         "label": "PyWavelet"},
    {"name": "numpy_cwt", "fn": run_numpy_cwt,         "label": "SubShader (NumPy)"},
    {"name": "gpu_cwt",   "fn": run_gpu_cwt,           "label": "SubShader (GPU)"},
]
```

### Pattern 4: Comparison Grid Header Fix
**What:** The current header (line 975 of figures.py) uses `set_title(..., pad=8)` on row 0 axes. With `top=1-GRID_MARGIN` (0.95), titles sit very close to the figure edge. Two problems: (1) `pad=8` is in points, not in figure-space — insufficient breathing room at high DPI; (2) horizontal centering uses `set_title` on the gridspec cell, which centers over the cell including any yticklabel space, not over the spectrogram image itself.

**Fix strategy (gridspec-based, Claude's discretion):**
```python
# Option A: Increase pad and use fig.text for centered column titles
# This decouples title placement from the axes bounding box.
for col_idx, col in enumerate(column_data):
    ax = axes[0][col_idx + 1]
    # Get the axes bounding box in figure coordinates, then place text at top
    # Using constrained_layout or tight_layout as alternative
    ax.set_title(col["label"], fontsize=LABEL_FONT_SIZE,
                 fontweight="bold", pad=16)  # increased from 8

# Option B: Replace subplots_adjust(top=0.95) with top=0.92 to leave header breathing room
plt.subplots_adjust(left=GRID_MARGIN, right=1-GRID_MARGIN,
                    top=0.92, bottom=GRID_MARGIN)
```

The exact gridspec value is Claude's discretion per CONTEXT.md. The key insight: `pad` in `set_title` is in points (roughly 1pt = 1/72 inch); at 200 DPI a `pad=8` is about 11px — barely visible. Increasing to `pad=16` or `pad=24` gives visible breathing room. Horizontal centering is only an issue if the label column's width ratio causes the spectrogram cells to not align with the `set_title` text; using `ha="center"` and `loc="center"` (default) in `set_title` on the data axes (col 1-3) should already center over the spectrogram cell.

### Anti-Patterns to Avoid
- **Dict-based style objects:** `DEFAULT_STYLE = {...}` then `s = {**DEFAULT_STYLE, **style}` everywhere. Per D-03, this is being replaced with module-level constants. The style merge pattern means any override silently shadows a constant — hard to trace. Module constants make every value a direct, traceable name.
- **Backend toggle global state:** The `_BACKEND` module global in `plotting.py` with `set_backend()`/`get_backend()` adds mutable global state that makes function behavior dependent on call order. With one canonical dark style, this entire mechanism is removed.
- **Parallel `cwt_timed()` code path:** The current `cwt_timed()` in `wavelet.py` (lines 179-222) duplicates the entire `cwt()` pipeline body. Per D-11 to D-14, the `@timed` decorator approach replaces this with decorators on individual stage methods, and `cwt_timed()` can be removed or retained as a thin wrapper that reads instance timing attributes after calling `cwt()`.
- **`import seaborn` at module level in plotting.py:** The try/except import at lines 19-23 adds a conditional dependency that propagates through get_active_style(). Clean removal: delete the seaborn import block, the `SEABORN_AVAILABLE` flag, the `_render_spectrogram_seaborn()` function, and the backend dispatch in `render_spectrogram_row()`.
- **Moving test files without updating pytest discovery:** pyproject.toml currently has `pythonpath = ["src", "src/subshader"]` which makes `conftest` importable from colocated test files. When tests move to `research/tests/`, the conftest.py files need to move with them or be recreated at `research/tests/conftest.py`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Constant grouping / namespace | Custom `StyleConfig` dataclass | Plain module-level names | D-03 explicitly forbids dicts/dataclasses; module-level names are simpler and directly importable |
| Timing decorator storage | External dict keyed by instance id | Instance attribute `_timing_*_ms` | Thread-safe for single-threaded use; no weak-ref management; directly inspectable |
| CLI argument parsing | Manual sys.argv parsing | argparse (already in benchmark.py) | Already present; copy the pattern, add/rename flags |
| Test discovery | Custom test runner | pytest with `--rootdir` or `testpaths` config | Already configured in pyproject.toml |
| Frequency Nyquist clamping | New clamping code in WaveletConfig | `_generate_chromatic_scale()` line 127 | Already does `freqs[freqs < self.nyquist_freq]` — confirmed, no new code needed |

**Key insight:** The domain is internal refactoring of Python research tooling. The risk is over-engineering. Every decision in CONTEXT.md favors the simplest working structure.

---

## Common Pitfalls

### Pitfall 1: Test Imports Break After Move
**What goes wrong:** `test_wavelet.py` imports `from conftest import generate_tone, find_peak_bin, _make_wavelet` using a bare name. When the file moves from `src/subshader/dsp/` to `research/tests/dsp/`, Python can't find `conftest` unless pytest is invoked from a directory where `conftest.py` is in scope.
**Why it happens:** pytest's conftest discovery is directory-scope based. The current conftest.py lives in `src/subshader/dsp/` (or a parent). Moving test files changes the conftest resolution chain.
**How to avoid:** Create `research/tests/conftest.py` containing (or importing from) the shared fixtures `generate_tone`, `find_peak_bin`, `_make_wavelet`. Update pyproject.toml `pythonpath` to include `research/tests` if needed, or use `research/tests/conftest.py` directly.
**Warning signs:** `ModuleNotFoundError: No module named 'conftest'` when running `pytest research/tests/`.

### Pitfall 2: Style Constants Used Before Style Module Exists
**What goes wrong:** If `plotting.py` primitives are updated to reference `style.BG_COLOR` before `style.py` is created, all figure generation breaks immediately.
**Why it happens:** Incremental migration of a file that has consumers.
**How to avoid:** Create `style.py` first (with all constants), then update consumers one file at a time. The `plotting.py` primitives are consumed by both `figures.py` and `comparison.py` — updating plotting.py before those is safe as long as style.py already exists.

### Pitfall 3: Seaborn Removal Breaks benchmark.py Flags
**What goes wrong:** `benchmark.py` has `--seaborn` and `--seaborn --stub-pywt` flags. If seaborn rendering is removed from `plotting.py` but the argparse flags are kept (even as no-ops), users get silently ignored flags with no error message.
**Why it happens:** Partial removal — removed the implementation but left the CLI surface.
**How to avoid:** When renaming `benchmark.py` to `test_suite.py` and restructuring flags (D-15), remove `--seaborn` from the argument parser. Document the removal in a comment or print warning if old flag is passed.

### Pitfall 4: @timed Decorator Placement in src/ vs research/
**What goes wrong:** D-11 says the decorator can live in `research/utilities/timing.py` OR `src/subshader/utils/`. If the decorator is placed in `research/utilities/timing.py` but applied to `src/subshader/dsp/wavelet.py` methods, that creates an import from research/ into src/ which inverts the dependency direction.
**Why it happens:** The two locations serve different masters — research/ is for tooling, src/ is for production code.
**How to avoid:** If the decorator is applied to production methods in `src/subshader/dsp/wavelet.py`, it must live in `src/subshader/utils/timing.py`. If it's only used in research pipelines (TimedSubShader etc.), it can stay in `research/utilities/timing.py` and be applied via wrapper or subclass without touching production code. The phase 05.2 decision "production code must not import from research/utilities" (STATE.md line 111) confirms this constraint.

### Pitfall 5: Comparison Grid Header Fix Over-Adjusts
**What goes wrong:** Reducing `top=` in `subplots_adjust` gives the titles breathing room but shrinks all 5 spectrogram rows proportionally, making the figures shorter.
**Why it happens:** `subplots_adjust` affects the entire figure layout, not just the top row.
**How to avoid:** Use `pad` parameter on `set_title` first (least-invasive). If that's insufficient, adjust `top` but increase `figsize` height to compensate (e.g., `figsize=(24, 17)` instead of `(24, 16)`). The figure dimensions are moving to `style.py` making this a one-line change.

### Pitfall 6: Test File Move Invalidates Colocated conftest.py References
**What goes wrong:** `src/subshader/dsp/test_wavelet_kernel.py` may have a conftest.py at the same level that's not visible from `research/tests/dsp/`.
**Why it happens:** pytest conftest.py is directory-scoped and walks upward, not sideways.
**How to avoid:** Before moving any test file, run `pytest src/ -v --collect-only` to see exactly which conftest.py files are discovered. Recreate all needed fixtures at `research/tests/conftest.py` or at sub-directory conftest.py files.

---

## Code Examples

Verified from source files:

### Current Style Dict Pattern (to be replaced)
```python
# research/utilities/plotting.py lines 29-51 — CURRENT
DEFAULT_STYLE = {
    "waveform_color": "#606060",
    "freq_bg": "#1A1A1A",
    "title_fontsize": 24,
    "tick_labelsize": 14,
    "axis_label_fontsize": 18,
    "suptitle_fontsize": 32,
    "figsize_w": 20,
    "row_height": 4,
    ...
}
```

```python
# research/utilities/style.py — TARGET pattern per D-03
WAVEFORM_COLOR = "#606060"
FREQ_BG = "#1A1A1A"
TITLE_FONT_SIZE = 24
TICK_LABEL_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 32
FIGURE_WIDTH = 20
ROW_HEIGHT = 4
```

### Current Backend Toggle (to be removed)
```python
# research/utilities/plotting.py lines 82-114 — DELETE all of this
_BACKEND = "matplotlib"
def set_backend(backend): ...
def get_backend(): ...
def get_active_style(): ...  # returns SEABORN_STYLE or DEFAULT_STYLE
```

### Current Header Placement (to be fixed)
```python
# research/figures.py line 975 — CURRENT: pad=8 is too small
axes[0][col_idx + 1].set_title(col["label"], fontsize=LABEL_FONT_SIZE,
                                fontweight="bold", pad=8)
# Also: subplots_adjust top=1-0.05=0.95 leaves minimal headroom
```

### WaveletConfig Frequency Parameters (confirmed correct)
```python
# src/subshader/config.py lines 99-101 — CURRENT state
notes_per_octave: int = 12
num_octaves: int = 10
root_note_a0_hz: np.float64 = 27.5

# src/subshader/dsp/wavelet.py lines 122-127 — Nyquist trim already exists
freqs = np.float64(root_note) * (scale_factor ** i)
return freqs[freqs < self.nyquist_freq]  # D-10: no new code needed
```

### Current cwt_timed Pattern (to be refactored)
```python
# src/subshader/dsp/wavelet.py lines 179-222
# Full duplicate of cwt() pipeline with inline perf_counter calls
# This is replaced by @timed decorators on individual stage methods
```

### @timed Decorator Target Pattern
```python
# src/subshader/utils/timing.py (if in production) OR
# research/utilities/timing.py (if research-only)
import time, functools

def timed(method):
    attr = f"_timing_{method.__name__}_ms"
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = method(self, *args, **kwargs)
        setattr(self, attr, (time.perf_counter() - t0) * 1000.0)
        return result
    return wrapper

# Applied to pipeline methods:
# @timed
# def cwt(self, input_data): ...
```

### test_suite.py CLI Pattern
```python
# research/test_suite.py — renamed from benchmark.py
# Flags: --test, --timing, --comparison, --figures, --dpi N
# Removed: --seaborn, --seaborn --stub-pywt (backend removed with style consolidation)
# Retained: --stub, --stub-pywt, --figures-chirp, --figures-polyphonic, --figures-musical
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic benchmark.py | benchmark.py + figures.py + timing.py + wav_export.py | Phase 5.1 | benchmark.py is now thin dispatcher |
| Unit tests colocated with source | Colocated pytest files in src/ | Phase 5.1 | Moving to research/tests/ in Phase 7 |
| Seaborn optional backend | Seaborn still imported with try/except | Present | Phase 7 removes entirely per D-04 |
| Inline perf_counter in cwt_timed() | Duplicate code path for timing | Phase 5.2 | Phase 7 replaces with decorator pattern |

**Deprecated/outdated after this phase:**
- `DEFAULT_STYLE` and `SEABORN_STYLE` dicts in `plotting.py`: replaced by `style.py` module constants
- `set_backend()`, `get_backend()`, `get_active_style()` in `plotting.py`: removed entirely
- `cwt_timed()` method in `wavelet.py`: superseded by `@timed` decorator; may be retained as thin wrapper or removed
- `benchmark.py` filename: renamed to `test_suite.py`
- `wav_export.py` at `research/` root: moved to `research/utilities/wav_export.py`

---

## Open Questions

1. **Does `@timed` go in `src/` or stay in `research/`?**
   - What we know: Phase 5.2 decision (STATE.md line 111) states "production code must not import from research/utilities." The current `cwt_timed()` uses inline `time.perf_counter` in `wavelet.py` to avoid this constraint.
   - What's unclear: D-11 says the decorator can live in either location. D-13 says timing should be "always available" — this implies it goes on the production methods, which means it must be in `src/subshader/utils/`.
   - Recommendation: Create `src/subshader/utils/timing.py` with the `@timed` decorator. Apply to relevant methods in `wavelet.py`. The `research/utilities/timing.py` retains `time_call` and `TimingAccumulator` for research-layer batch timing. This respects the dependency direction constraint.

2. **Should `wav_export.py` move require updating all callers?**
   - What we know: `benchmark.py` line 28 does `from wav_export import export_signal_to_wav` using bare module name. `figures.py` also uses `export_signal_to_wav` (via `from utilities import`).
   - What's unclear: After moving to `utilities/wav_export.py`, does the `__init__.py` re-export it so existing callers don't break?
   - Recommendation: Add `from .wav_export import export_signal_to_wav` to `research/utilities/__init__.py` so existing callers that do `from utilities import export_signal_to_wav` continue to work. Then update `benchmark.py`/`test_suite.py` directly.

3. **Where should `comparison.py`'s compute functions come from?**
   - What we know: `figures.py` currently holds `generate_comparison_grid()` with inline DSP computation calls. `research/utilities/dsp_helpers.py` holds `compute_stft_frame`, chirp generators, etc. The `comparison.py` method config list (D-24) needs compute functions.
   - What's unclear: Whether `generate_comparison_grid()` moves into `comparison.py` or stays in `figures.py`.
   - Recommendation: Move `generate_comparison_grid()` to `comparison.py` since it is a comparison operation. `figures.py` retains `ReadmeFigures` (single-signal per-method figures). This gives the cleaner separation of concerns D-17 describes.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 7 is purely code/config refactoring with no external dependencies beyond the existing Python environment. All tools (matplotlib, pytest, argparse) are already present.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest research/tests/ -x -q` |
| Full suite command | `pytest research/tests/ -v` |

### Phase Requirements → Test Map

This phase has no formal requirement IDs (null in phase metadata). The testable behaviors are:

| Behavior | Test Type | Automated Command | Notes |
|----------|-----------|-------------------|-------|
| All 4 existing test files pass after move to research/tests/ | unit | `pytest research/tests/ -v` | Core regression — D-21/D-22 |
| `test_suite.py --test` discovers and runs research/tests/ | smoke | `python research/test_suite.py --test` | D-23 |
| `style.py` constants are importable from figures.py | smoke | `python -c "from research.utilities import style; print(style.BG_COLOR)"` (from project root) | D-01/D-02 |
| `WaveletConfig` accepts custom `root_note_a0_hz` and `num_octaves` | unit | Already tested via `test_wavelet.py` | D-09 — parameters already exist |
| Comparison grid generates without error | smoke | `python research/test_suite.py --comparison-grid --stub-pywt` | D-06/D-07 visual output |

### Sampling Rate
- **Per task commit:** `pytest research/tests/ -x -q`
- **Per wave merge:** `pytest research/tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `research/tests/__init__.py` — needed for pytest discovery
- [ ] `research/tests/audio/__init__.py`
- [ ] `research/tests/dsp/__init__.py`
- [ ] `research/tests/viz/__init__.py`
- [ ] `research/tests/conftest.py` — recreate `generate_tone`, `find_peak_bin`, `_make_wavelet` fixtures currently in `src/subshader/dsp/conftest.py` (or wherever they live)
- [ ] `research/utilities/style.py` — must exist before any consumer is updated

---

## Project Constraints (from CLAUDE.md)

| Constraint | Applies To |
|-----------|------------|
| No rewrites — Python, CuPy, ModernGL existing pipeline | D-11 decorator must not change cwt() behavior |
| Descriptive names, helpers, no comment litter, structure over documentation | style.py constant names must be explicit (LABEL_FONT_SIZE not LFS) |
| Dataclasses for configuration; WaveletConfig is a dataclass | D-09: root_note_a0_hz and num_octaves already in WaveletConfig dataclass — no structural change |
| Single underscore for internal methods | @timed attributes `_timing_*_ms` follow the internal convention |
| GSD workflow enforcement: no direct repo edits outside GSD workflow | Implementation follows execute-phase |
| No new dependencies | Confirmed: zero new packages |
| `production code must not import from research/utilities` | @timed must live in src/subshader/utils/ if applied to production methods |

---

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `research/utilities/plotting.py` lines 1-326 — all style constants, backend toggle, seaborn integration confirmed
- Direct source inspection: `research/figures.py` lines 633-637, 844-846, 973-979 — hardcoded grid constants and header rendering confirmed
- Direct source inspection: `src/subshader/config.py` lines 91-128 — WaveletConfig dataclass, `root_note_a0_hz`, `num_octaves` already present
- Direct source inspection: `src/subshader/dsp/wavelet.py` lines 105-127 — `_generate_chromatic_scale()` with existing Nyquist trim confirmed
- Direct source inspection: `src/subshader/dsp/wavelet.py` lines 179-222 — `cwt_timed()` duplicate pipeline confirmed
- Direct source inspection: `research/timing.py` — `TimedSubShader` uses external `time_call` wrappers (not @timed)
- Direct source inspection: `research/utilities/timing.py` — `time_call` and `TimingAccumulator` (no @timed yet)
- Direct source inspection: `research/utilities/__init__.py` — current exports
- Direct source inspection: `research/utilities/constants.py` — audio paths, DSP params
- Direct source inspection: `research/benchmark.py` — CLI flags, argparse structure
- Direct source inspection: `pyproject.toml` — pytest config, pythonpath settings
- Direct source inspection: `.planning/STATE.md` line 111 — "production code must not import from research/utilities" decision

### Secondary (MEDIUM confidence)
- Matplotlib `set_title` pad semantics: pad parameter is in points (matplotlib official docs — standard behavior, HIGH confidence)
- pytest conftest.py directory scoping: standard pytest behavior, HIGH confidence

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies, everything verified in place
- Architecture: HIGH — all source files read directly; patterns derived from actual code
- Pitfalls: HIGH — each pitfall is rooted in observed code structure, not speculation
- Style consolidation approach: HIGH — exact constants extracted from plotting.py source
- @timed decorator placement: MEDIUM — correct location (src/ vs research/) depends on implementation choice left to Claude's discretion; both options are viable

**Research date:** 2026-03-27
**Valid until:** 2026-06-27 (stable internal refactor; no external dependencies)
