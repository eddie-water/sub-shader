# Phase 1: Codebase Hardening - Research

**Researched:** 2026-03-21
**Domain:** Python error handling, conditional imports, exception hierarchy, code quality
**Confidence:** HIGH

## Summary

Phase 1 is a targeted hardening pass on an existing Python pipeline. All issues are already identified — the CONTEXT.md code scout pinpointed the exact files and line numbers. No new libraries are needed; this is pure refactoring of `__main__.py`, `wavelet.py`, `wavelet_kernel.py`, `audio_input.py`, `plotter.py`, `exceptions.py`, and `config.py`.

The two primary concerns are: (1) making GPU fallback live in `SubShader.__init__` rather than benchmark code, and (2) eliminating silent failure paths — swallowed render exceptions, `None`-returning validators, and hardcoded audio paths that produce blank visualizations instead of clear errors. Both follow standard Python patterns and require no new dependencies.

The discretionary scope (exception dedup, dead code, CuPy import guarding, logger timing, cleanup guards) involves judgment calls on how much cleanup to do while staying within the "no unnecessary refactoring" constraint from QUAL-03.

**Primary recommendation:** Work file-by-file in dependency order — `exceptions.py` first (consolidation), then `config.py` (path default), then `wavelet.py` (lazy imports + GPU detection), then `__main__.py` (fallback wiring), then `plotter.py` (silent failure removal).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Frame failures crash the app with a clear error — no silent frame drops, no swallowed render exceptions
**D-02:** Texture validation failures raise an exception instead of silently returning stale data
**D-03:** GPU cleanup errors use the logger, not print()
**D-04:** Audio file path lives in AudioConfig as the config default — remove the hardcoded override in `__main__.py`
**D-05:** Config remains overridable before SubShader instantiation (existing pattern: modify `config.audio.file_path` before calling `SubShader(config)`)
**D-06:** Missing audio file raises AudioFileNotFoundError immediately at init — not a silent blank visualization
**D-07:** GPU unavailability detected at init, logged as a clear console message: "GPU unavailable, running on NumPy — expect slower performance"
**D-08:** Session continues on NumPy path after the message — no crash, no re-check mid-session
**D-09:** GPU fallback logic lives in DSP block instantiation (`SubShader.__init__`), not in benchmark code

### Claude's Discretion
- Exception hierarchy cleanup (dedup classes, fix base classes, narrow RuntimeError in GRACEFUL_EXCEPTIONS)
- Dead code removal (unused COI mask methods, dead CuPy import in wavelet_kernel.py)
- CuPy import guarding strategy (lazy imports, conditional imports, or restructured modules)
- Cleanup guard safety (`hasattr` checks in `SubShader.cleanup()`)
- Logger configuration timing (console_output=False at module level is currently hiding startup errors)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-02 | GPU fallback lives in DSP block instantiation, not benchmark code — auto-detects GPU failure and falls back to NumPy | `gpu_available()` in `research/utilities/constants.py:50` is ready to promote; `CuWavelet` and `NpWavelet` aliases already exist in `wavelet.py` — selection just needs to move to `SubShader.__init__` |
| PIPE-03 | GPU availability checked at init — if unavailable, run on NumPy path for the session | D-07/D-08 — detect once at init, log clear message, assign wavelet class, never re-check |
| QUAL-01 | Clean, readable code — descriptive function names, well-factored helpers, minimal comments | Addressed through exception consolidation, dead code removal, and replacing print() with logger |
| QUAL-03 | Existing readability maintained — no unnecessary refactoring | Constrains discretionary scope: touch only files that have hardening work; don't restructure things that aren't broken |
</phase_requirements>

---

## Standard Stack

### Core (no new dependencies needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `logging` | 3.12.3 | Logger-based error reporting | Already used; replaces stray `print()` calls |
| Python stdlib `os.path` | 3.12.3 | File existence checks | Already used in `AudioInput.__init__` |
| `cupy` | existing | GPU detection via `cp.cuda.runtime.getDevice()` | Pattern from `research/utilities/constants.py:50` — lazy import inside try/except |

**No installation needed.** All required tools are already in the venv or stdlib.

### No Alternatives Needed

This phase adds no new libraries. All patterns use existing infrastructure.

---

## Architecture Patterns

### Pattern 1: Lazy / Guarded CuPy Import

**What:** Import CuPy inside a try/except rather than at module top-level. NumPy-only code paths never trigger the import.

**When to use:** Any module that uses CuPy but should degrade gracefully — `wavelet.py` (line 24-25 is the blocker) and `wavelet_kernel.py` (line 6 is dead weight).

**Example (from `research/utilities/constants.py:50`):**
```python
def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False
```

The same pattern applies to the top-level imports in `wavelet.py`:
```python
# Before (blocks entire module if CuPy absent):
import cupy as cp
from cupyx.scipy import fft as cp_fft

# After (guarded):
try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False
```

`CuPyWavelet.__init__` can then raise a clear `RuntimeError` (or a new `GpuUnavailableError`) if `_CUPY_AVAILABLE` is False, making the failure explicit and catchable.

### Pattern 2: GPU Detection and Wavelet Selection at Init

**What:** Check GPU once in `SubShader.__init__`, select the wavelet class, log the result. Never check again.

**When to use:** Exactly this case — D-07, D-08, D-09.

```python
# In SubShader.__init__:
if gpu_available():
    wavelet_class = CuWavelet
else:
    log.warning("GPU unavailable, running on NumPy — expect slower performance")
    wavelet_class = NpWavelet

self.wavelet = wavelet_class(
    sample_rate=self.audio_input.get_sample_rate(),
    input_n=self.audio_input.get_chunk_size(),
    config=config.wavelet,
)
```

`gpu_available()` should live in `src/subshader/utils/` (promoted from `research/utilities/constants.py`).

### Pattern 3: Exception Hierarchy Consolidation

**What:** Remove duplicate exception classes in `audio_input.py` and `plotter.py`; make them import from `subshader.exceptions` instead.

**Current state:**
- `subshader.exceptions` — authoritative `AudioFileNotFoundError`, `EndOfAudioException`, `WindowCloseException` (correct base: `SubShaderException`)
- `audio_input.py` — duplicate `AudioFileNotFoundError(Exception)`, `EndOfAudioException(Exception)` (wrong base)
- `plotter.py` — duplicate `WindowCloseException(Exception)` (wrong base)

**Fix:** Delete the duplicates in `audio_input.py` and `plotter.py`, add imports from `subshader.exceptions`. The authoritative versions already exist with the correct base class and `log_level` attribute.

### Pattern 4: Raising Instead of Silently Returning

**What:** `_validate_texture_data` currently returns `None` on every error branch (it's a bug — the type annotation says `bool`). `render_graphic` catches the render exception and swallows it.

**Fix for `_validate_texture_data`:**
```python
# Before: every error path just returns (implicitly None)
if texture_data is None:
    log.error("Texture data is None")
    return  # <-- returns None, caller thinks validation passed

# After: raise explicitly
if texture_data is None:
    raise ValueError("Texture data is None — cannot upload to GPU texture")
```

**Fix for `render_graphic`:**
```python
# Before: swallows all exceptions, frame silently drops
except Exception as e:
    log.error(f"Render exception: {e}")

# After: re-raise so the main loop sees the failure (D-01)
except Exception as e:
    log.error(f"Render exception: {e}")
    raise
```

### Pattern 5: hasattr Guards in cleanup()

**What:** `SubShader.cleanup()` checks `if self.plotter:` etc. This fails with `AttributeError` if init raised before the attribute was assigned (partial init).

**Fix:**
```python
# Before:
if self.plotter:
    ...

# After:
if hasattr(self, 'plotter') and self.plotter:
    ...
```

This is the same pattern `AudioInput.cleanup()` already uses for `file_handle` (line 158).

### Pattern 6: AudioConfig Default Path vs. __main__ Override

**What:** `config.py:59` has `file_path: str = "assets/audio/songs/beltran_sc_rip.wav"`. `__main__.py:45` overrides it to a different hardcoded path at module level (before `SubShader` is even constructed).

**Fix per D-04/D-05:**
1. Set the correct default in `AudioConfig.file_path` (the actual demo file)
2. Remove the module-level override from `__main__.py`
3. Config remains overridable before `SubShader(config)` — that pattern is unchanged

### Pattern 7: Logger Timing Fix

**What:** `__main__.py:34` calls `logger_init(console_output=False)`. This hides all startup errors from the console. Any exception raised during `get_default_config()` or `SubShader.__init__()` is invisible to the user.

**Fix:** Enable `console_output=True` (or remove the parameter since it defaults to True in `logging.py`). The existing `logger_init` signature already supports this — no changes to `logging.py` needed.

### Pattern 8: GRACEFUL_EXCEPTIONS Scope Narrowing (discretionary)

**What:** `GRACEFUL_EXCEPTIONS` currently includes bare `RuntimeError`. This catches too broadly — any `RuntimeError` anywhere (including GPU kernel crashes that should surface as bugs) gets silently reported and swallowed.

**Fix:** Replace with specific exceptions or remove `RuntimeError` if it was only there to catch CuPy init failures (which will now be handled explicitly by the GPU detection pattern).

### Anti-Patterns to Avoid

- **Bare except in cleanup:** Don't use `except Exception: pass` in cleanup. Use `except Exception as e: log.warning(...)` to preserve visibility.
- **Module-level side effects on import:** Don't trigger GPU operations (CuPy memory allocation, CUDA context init) at import time. This blocks the entire module if CUDA is unavailable.
- **print() for error reporting:** All error-level events must go through the logger. `print()` bypasses log level filtering and file output.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU detection | Custom CUDA query logic | `gpu_available()` from `research/utilities/constants.py` | Already correct, tested in research context — just promote it |
| Exception base class | New exception base | Existing `SubShaderException` in `exceptions.py` | Already has `log_level` attribute and correct `__init__` signature |
| Lazy import mechanism | Custom import hook | try/except at module or function level | Standard Python idiom, zero overhead |
| Logger setup | Custom handler | Existing `logger_init()` / `get_logger()` | Already configured with file and console handlers |

**Key insight:** Every tool needed for this phase already exists in the codebase. This phase is about wiring them correctly — not building new infrastructure.

---

## Common Pitfalls

### Pitfall 1: Partial Init AttributeError

**What goes wrong:** `SubShader.cleanup()` runs `if self.plotter:` — but if `AudioInput.__init__` raised before `self.plotter` was ever assigned, this raises `AttributeError: 'SubShader' object has no attribute 'plotter'` and cleanup itself crashes.
**Why it happens:** Python doesn't zero-initialize instance attributes; they only exist after assignment.
**How to avoid:** Use `hasattr(self, 'plotter') and self.plotter` in all three cleanup guards.
**Warning signs:** `AttributeError` in cleanup after any init failure.

### Pitfall 2: CuPy Import at Module Level Blocks NumPy Path

**What goes wrong:** `wavelet.py:24-25` runs `import cupy as cp` unconditionally. Even `NumPyWavelet` and `NpWavelet` cannot be imported on a machine without CUDA — the entire `wavelet` module fails to import.
**Why it happens:** Python executes all top-level import statements when the module is first loaded, regardless of which classes will actually be used.
**How to avoid:** Move CuPy imports inside `CuPyWavelet.__init__` or guard them with a try/except at module level (setting `_CUPY_AVAILABLE`). The `NumPyWavelet`/`AntsWavelet`/`Wavelet` classes themselves have zero CuPy usage and should remain importable.
**Warning signs:** `ModuleNotFoundError: No module named 'cupy'` at startup even when GPU is unavailable.

### Pitfall 3: Duplicate Exception Classes With Wrong Base

**What goes wrong:** `audio_input.py` raises `AudioFileNotFoundError` which is `Exception`-based. `exceptions.py` catches `SubShaderException`. The `AudioInput` version is never caught by `GRACEFUL_EXCEPTIONS` and propagates as an unhandled exception.
**Why it happens:** The duplicates were defined locally before the central `exceptions.py` was established (or the central module was created afterward without updating the original files).
**How to avoid:** Delete local duplicates, import from `subshader.exceptions`. Verify GRACEFUL_EXCEPTIONS catches the imported version.
**Warning signs:** Unhandled exception traceback on bad audio path instead of clean error message.

### Pitfall 4: _validate_texture_data Returns None as Success

**What goes wrong:** Every error branch in `_validate_texture_data` returns `None` (implicit). The caller checks `if not self._validate_texture_data(data):` — `None` is falsy, so validation failures DO block the upload. But `True` is also the success case. If someone refactors the caller to `if self._validate_texture_data(data):` the behavior inverts silently.
**Why it happens:** The function is annotated `-> bool` but returns `None` on error paths — a type error. The current caller `update_texture` likely has the check inverted for None to work.
**How to avoid:** Per D-02, raise on invalid data instead of returning. Remove the bool return entirely — success = no exception, failure = exception.
**Warning signs:** Texture upload silently skipped with only a log.error message, no exception in the main loop.

### Pitfall 5: console_output=False Hides Startup Errors

**What goes wrong:** `logger_init(console_output=False)` in `__main__.py:34` means all log output before the window opens goes to file only. If `AudioFileNotFoundError` is raised in `AudioInput.__init__`, the user sees nothing — no console message, just a silent exit (or a cryptic traceback if the exception isn't caught).
**Why it happens:** Likely set during early development to reduce terminal noise. Never changed back.
**How to avoid:** Use `console_output=True` (the default). The file handler still captures everything; console output is additive.
**Warning signs:** App exits silently with no error shown, but `logs/subshader.log` contains the error.

---

## Code Examples

### Promoting gpu_available() to utils/

```python
# src/subshader/utils/gpu.py  (new file)
def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False
```

### Guarded CuPy import in wavelet.py

```python
# At module top-level, replacing unconditional imports:
try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False

# In CuPyWavelet.__init__:
def __init__(self, ...):
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available — use NpWavelet instead")
    super().__init__(...)
    ...
```

### Wavelet selection in SubShader.__init__

```python
from subshader.utils.gpu import gpu_available
from subshader.dsp.wavelet import CuWavelet, NpWavelet

class SubShader:
    def __init__(self, config):
        self.audio_input = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )

        wavelet_class = CuWavelet if gpu_available() else NpWavelet
        if wavelet_class is NpWavelet:
            log.warning("GPU unavailable, running on NumPy — expect slower performance")

        self.wavelet = wavelet_class(
            sample_rate=self.audio_input.get_sample_rate(),
            input_n=self.audio_input.get_chunk_size(),
            config=config.wavelet,
        )
        ...
```

### Cleanup with hasattr guards

```python
def cleanup(self):
    """Idempotent cleanup: safe to call any time, even after partial init."""
    log.info("Cleaning up module resources")

    if hasattr(self, 'plotter') and self.plotter:
        try:
            self.plotter.cleanup()
        finally:
            self.plotter = None

    if hasattr(self, 'wavelet') and self.wavelet:
        try:
            self.wavelet.cleanup()
        finally:
            self.wavelet = None

    if hasattr(self, 'audio_input') and self.audio_input:
        try:
            self.audio_input.cleanup()
        finally:
            self.audio_input = None

    if hasattr(self, 'loop_timer'):
        self.loop_timer = None

    log.info("Cleanup complete")
```

### CuPyWavelet.cleanup() using logger instead of print()

```python
def cleanup(self) -> None:
    try:
        if hasattr(self, 'kernel_f_bank_gpu') and self.kernel_f_bank_gpu is not None:
            del self.kernel_f_bank_gpu
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        log.warning(f"Error during GPU cleanup: {e}")
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Unconditional `import cupy` at module top | Guarded `try: import cupy` with `_CUPY_AVAILABLE` flag | NumPy-only environments can import the module |
| Duplicate exceptions in each module | Single source of truth in `exceptions.py`, imported by consumers | Consistent base class, consistent catch behavior |
| GPU selection in benchmark / hardcoded in `__main__` | `gpu_available()` called once in `SubShader.__init__` | Benchmark code deletable without breaking fallback |

---

## Open Questions

1. **Which file should be the canonical home for `gpu_available()`?**
   - What we know: It currently lives in `research/utilities/constants.py` — outside `src/`, so not importable by the main package.
   - What's unclear: Whether a new `src/subshader/utils/gpu.py` is the right home, or whether it belongs directly in `utils/__init__.py`.
   - Recommendation: Create `src/subshader/utils/gpu.py` — consistent with the existing file-per-concern pattern in `utils/` (see `loop_timer.py`, `logging.py`).

2. **Should GRACEFUL_EXCEPTIONS include RuntimeError after this phase?**
   - What we know: `RuntimeError` is currently in the tuple to catch CuPy init failures. After this phase, those failures are handled explicitly by GPU detection.
   - What's unclear: Whether any other `RuntimeError` paths exist that need graceful handling.
   - Recommendation: Remove `RuntimeError` from `GRACEFUL_EXCEPTIONS` as a discretionary cleanup item. If a `RuntimeError` surfaces, it should be investigated — not silently swallowed.

3. **Dead COI mask methods in AntsWavelet — delete or keep?**
   - What we know: `_create_coi_mask()` and `apply_coi_mask()` are defined but `discard_unreliable_coefs` calls `slice_for_reliable_region` instead. The COI mask path is commented out.
   - What's unclear: Whether these methods are intended to be revived later (see `TODO-37`).
   - Recommendation: Keep as discretionary removal. If the planner includes it, remove `_create_coi_mask` and `apply_coi_mask`. The `TODO-37` comment on `discard_unreliable_coefs` explains the tradeoff.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed) |
| Config file | none — needs Wave 0 setup |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-02 | `SubShader.__init__` selects `NpWavelet` when GPU unavailable | unit | `python -m pytest tests/test_gpu_fallback.py -x` | Wave 0 |
| PIPE-03 | `gpu_available()` returns False when CuPy absent | unit | `python -m pytest tests/test_gpu_fallback.py::test_gpu_available_without_cupy -x` | Wave 0 |
| PIPE-03 | `SubShader.__init__` logs warning when GPU unavailable | unit | `python -m pytest tests/test_gpu_fallback.py::test_gpu_fallback_logs_warning -x` | Wave 0 |
| QUAL-01 | `AudioFileNotFoundError` raised (not silent) on bad path | unit | `python -m pytest tests/test_audio_input.py::test_missing_file_raises -x` | Wave 0 |
| QUAL-01 | `_validate_texture_data` raises instead of returning None | unit | `python -m pytest tests/test_plotter.py::test_validate_texture_raises -x` | Wave 0 |
| QUAL-03 | Exception imports resolve to `subshader.exceptions` (no duplicates) | unit | `python -m pytest tests/test_exceptions.py::test_no_duplicate_classes -x` | Wave 0 |

Note: Tests for GPU fallback will need to mock CuPy unavailability using `unittest.mock.patch`. `gpu_available()` can be patched directly — no real GPU required.

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — package marker
- [ ] `tests/conftest.py` — shared fixtures (mock config, mock audio file)
- [ ] `tests/test_gpu_fallback.py` — covers PIPE-02, PIPE-03
- [ ] `tests/test_audio_input.py` — covers QUAL-01 (missing file error)
- [ ] `tests/test_plotter.py` — covers QUAL-01 (texture validation raise)
- [ ] `tests/test_exceptions.py` — covers QUAL-03 (no duplicate classes)
- [ ] Framework install: `pip install pytest` — not in `pyproject.toml` yet

---

## Sources

### Primary (HIGH confidence)

- Direct source inspection of `src/subshader/__main__.py`, `wavelet.py`, `wavelet_kernel.py`, `audio_input.py`, `plotter.py`, `exceptions.py`, `config.py`, `utils/logging.py` — all findings are first-hand code reading
- `research/utilities/constants.py:50` — canonical `gpu_available()` implementation, read directly

### Secondary (MEDIUM confidence)

- Python docs — lazy import pattern (try/except at module level) is standard Python; `_CUPY_AVAILABLE` flag is the idiomatic approach
- Python docs — `hasattr` guard in cleanup is the documented way to handle partial init in Python classes

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all tools already present in codebase
- Architecture: HIGH — all patterns are first-hand code reading of the actual files; no external sources needed
- Pitfalls: HIGH — all pitfalls are directly observed bugs in the source, not hypothetical

**Research date:** 2026-03-21
**Valid until:** 2026-06-21 (stable Python patterns; no fast-moving dependencies involved)
