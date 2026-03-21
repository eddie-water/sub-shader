# Coding Conventions

**Analysis Date:** 2026-03-21

## Naming Patterns

**Files:**
- Lowercase with underscores: `audio_input.py`, `loop_timer.py`, `plot_normalizer.py`
- Module files use full descriptive names without abbreviation
- Package directories are lowercase: `audio/`, `dsp/`, `viz/`, `utils/`

**Functions:**
- Lowercase with underscores: `get_logger()`, `get_sample_rate()`, `end_loop_and_report()`
- Getter methods use `get_` prefix: `get_chunk()`, `get_output_shape()`, `get_loops_per_second()`
- Internal methods use single leading underscore: `_generate_chromatic_scale()`, `_validate_gpu_memory()`, `_create_reliable_slice()`
- Setter-adjacent methods use present tense: `validate()`, `update_plot()`, `normalize_by_scale()`

**Variables:**
- Lowercase with underscores for multi-word identifiers: `sample_rate`, `chunk_size`, `overlap_factor`, `num_frames`
- Single letter loops acceptable in DSP context: `f` (frequency), `i` (index), `t` (time), `w` (wavelet)
- Constant-like class variables use uppercase with underscores: `GRACEFUL_EXCEPTIONS`, `PI`

**Types:**
- Dataclasses for configuration: `AudioConfig`, `WaveletConfig`, `VisualizationConfig`, `ColorNormalizationConfig`
- Abstract base classes have clear inheritance: `Wavelet`, `Plotter`, inherited by `PyWavelet`, `AntsWavelet`, `ShaderPlot`
- Custom exception classes inherit from parent exceptions: `SubShaderException`, `AudioFileNotFoundError`, `EndOfAudioException`

**Type Annotations:**
- Type hints used consistently with NumPy dtypes: `np.float64`, `np.ndarray[np.float64]`, `np.ndarray[np.complexfloating]`
- Return type annotations on methods: `-> None`, `-> np.ndarray`, `-> bool`, `-> float`
- Parameter annotations include NumPy array shapes where relevant: `np.ndarray[np.float64]`
- Union types use pipes: `object | None`

## Code Style

**Formatting:**
- No linter configuration found; follows PEP 8 style guide
- Line length appears to be around 100 characters (some lines extend slightly beyond)
- Indentation: 4 spaces (standard Python)
- Blank lines: 2 between module-level sections, 1 between method definitions

**Module Organization:**
Consistent structure with section comments separating concerns:
```python
"""Module docstring."""

# =============================================================================
# IMPORTS
# =============================================================================

# =============================================================================
# LOGGING
# =============================================================================

# =============================================================================
# CONSTANTS/CONFIGURATION
# =============================================================================

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================
```

Example from `config.py` and `audio_input.py` — this pattern is used throughout the codebase.

**Linting:**
- No `.flake8`, `.pylintrc`, or ESLint configuration files present
- No configured pre-commit hooks detected
- Project relies on developer discipline for code style

## Import Organization

**Order:**
1. Standard library imports (`os`, `sys`, `time`, `logging`)
2. Third-party imports (`numpy`, `cupy`, `soundfile`, `moderngl`, `pywt`)
3. Relative imports from project (`from subshader.utils.logging import get_logger`)
4. Local module imports (`from .wavelet_kernel import WaveletKernel`)

**Path Aliases:**
No aliases configured; imports use full module paths:
- `from subshader.config import AudioConfig, WaveletConfig`
- `from subshader.utils.logging import get_logger`
- `from ..config import WaveletConfig` (relative when in subdirectory)

**Barrel Files:**
- `src/subshader/utils/__init__.py` exports key utilities: `logger_init`, `get_logger`, `set_log_level`, `LoopTimer`, `env_init`
- `src/subshader/viz/__init__.py` exists but is minimal
- Explicit imports preferred over wildcard `import *`

## Error Handling

**Patterns:**
- Custom exception hierarchy rooted in `SubShaderException` — see `src/subshader/exceptions.py`
- Domain-specific exceptions: `AudioFileNotFoundError`, `EndOfAudioException`, `WindowCloseException`
- Log level attribute on exceptions: `log_level = "error"` or `"info"`
- Graceful exit through exception tuple `GRACEFUL_EXCEPTIONS` in main loop

**Implementation:**
```python
class SubShaderException(Exception):
    """Base exception for SubShader application."""
    log_level = "info"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
```

**Handling Pattern:**
```python
try:
    subshader.loop()
except exceptions.GRACEFUL_EXCEPTIONS as e:
    exceptions.reporter.report(e)
finally:
    subshader.cleanup()
```

Exception reporter in `src/subshader/exceptions.py` uses `log_level` attribute to determine output severity.

## Logging

**Framework:** Standard Python `logging` module

**Setup:** Centralized in `src/subshader/utils/logging.py`
- `logger_init()` configures root logger with handlers (console and file)
- `get_logger(name)` provides module-level loggers
- Default log file: `logs/subshader.log`

**Patterns:**
```python
log = get_logger(__name__)  # At module top

# Usage
log.info(f"Audio file loaded: {self.file_path}")
log.error(f"Input data length mismatch: {input_data.shape[0]}")
log.debug(f"Downsampled to output data: {coefs.shape} -> {downsampled.shape}")
```

**Emoji usage:** Uses checkmarks and X marks for visual status (`✅`, `❌`)

## Comments

**When to Comment:**
- Complex algorithms need explanation: `src/subshader/dsp/wavelet.py` includes extensive docstrings explaining CWT edge effects and normalization
- Configuration and parameter meanings: `src/subshader/config.py` comments explain overlap factors, decay rates, percentiles
- Non-obvious DSP/math concepts: tone support windows, cone of influence, scale-dependent normalization

**DocString/TSDoc:**
Used consistently with Google-style docstrings:
```python
def get_chunk(self) -> np.ndarray[np.float64]:
    """
    Retrieves the next chunk of audio samples from the file.

    Returns:
        np.ndarray: The next chunk of audio data. None indicates EOF.
    """
```

**TODO/FIXME Comments:**
- Format: `TODO-[ISSUE-NUMBER]` or `TODO` with inline explanation
- Examples in codebase:
  - `TODO-45 Fix the overlap and plot overlap relationship` (`config.py:65`)
  - `TODO-36 This check needs to be re-visited...` (`config.py:348`)
  - `TODO ISSUE-36 why 1.5-1.0?` (`wavelet.py:283`)

## Function Design

**Size:** Generally 20-80 lines for public methods, up to 150+ for complex DSP methods
- `AudioInput.get_chunk()`: ~20 lines (straightforward)
- `Wavelet.cwt()`: ~32 lines (orchestrates multiple steps)
- `ProcessingConfig.validate()`: ~46 lines (validation logic)
- `AntsWavelet.discard_unreliable_coefs()`: 82 lines (extensive documentation)

**Parameters:**
- Use positional args for essential parameters; keyword args for optional ones
- Type hints required for all parameters in public APIs
- Default values provided for configuration objects
- Example:
```python
def __init__(self, path: str, chunk_size: int = 4096, overlap_factor: float = 0.5) -> None:
```

**Return Values:**
- Explicit return types via annotations
- Methods returning data structures always annotated: `-> np.ndarray`, `-> bool`, `-> tuple[int, int]`
- None return should be explicit: `-> None` (not omitted)
- Example:
```python
def validate(self) -> List[str]:
    """Returns list of validation errors (empty if valid)."""
```

## Module Design

**Exports:**
- Public methods have no leading underscore
- Single underscore `_` prefix for internal helper methods
- Double underscore `__` reserved for name mangling (not used in codebase)
- Classes export public interface through `__init__` and public methods

**Barrel Files:**
- `src/subshader/utils/__init__.py` explicitly exports key classes and functions
- Minimizes need for deep imports: `from subshader.utils import LoopTimer` rather than `from subshader.utils.loop_timer import LoopTimer`

**Class Hierarchy:**
- Abstract base classes use `ABC` and `@abstractmethod` decorators
- Example hierarchy in DSP: `Wavelet` → `AntsWavelet` → `NumPyWavelet`/`CuPyWavelet`
- Plotter hierarchy: `Plotter` → `ShaderPlot`
- Configuration uses dataclasses with `@dataclass` decorator

---

*Convention analysis: 2026-03-21*
