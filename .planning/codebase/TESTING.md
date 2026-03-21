# Testing Patterns

**Analysis Date:** 2026-03-21

## Test Framework

**Runner:**
- No formal test framework configured (pytest/unittest not detected)
- Manual testing via `research/benchmark.py` script only

**Assertion Library:**
- No assertion library configured
- Manual comparison and validation used in benchmarks

**Run Commands:**
```bash
# Benchmark script (research/benchmark.py) supports multiple modes
python research/benchmark.py --timing              # STFT vs PyWavelet vs NumPy CWT vs CuPy CWT timing
python research/benchmark.py --figures             # Generate comparison PNGs (matplotlib)
python research/benchmark.py --figures --stub      # Generate stubs (fast iteration)
python research/benchmark.py --seaborn             # Generate seaborn heatmap style PNGs
python research/benchmark.py --unit-tests          # Run unit tests (NumPy vs CuPy verification)
python research/benchmark.py --all                 # Run all modes
```

## Test File Organization

**Location:**
- No dedicated `tests/` directory
- Benchmarking and validation performed in `research/benchmark.py`
- Ad-hoc testing via module-level scripts

**Naming:**
- Not applicable (no test files follow standard naming convention)

**Structure:**
Test organization is entirely within `research/benchmark.py`:
- Timing comparisons for different CWT implementations
- Figure generation using matplotlib/seaborn
- Unit test mode for NumPy vs CuPy verification
- GPU availability checking and fallback logic

## Test Structure

**Suite Organization:**
```python
# Modes (from benchmark.py docstring)
# (default) - Run all modes
# --timing - STFT vs PyWavelet vs NumPy CWT vs CuPy CWT timing
# --figures - Generate 3 README comparison PNGs (matplotlib)
# --unit-tests - Run unit tests (NumPy vs CuPy verification)
# --all - Run timing + figures + unit tests
```

**Patterns:**
- Command-line argument parsing with argparse for mode selection
- Helper functions for timing: `time_call()`, `TimingAccumulator()`
- Progress reporting: `live_progress()`, `clear_progress()`
- Result output formatting: `print_results_header()`, `print_results_row()`, `print_total_time()`

**Actual Test Invocation:**
None found in core codebase. Validation occurs through:
1. Configuration validation in `src/subshader/config.py`: `validate()` methods return list of errors
2. Manual assertion in main loop `src/subshader/__main__.py`: checks audio file exists, validates chunk reading
3. Exception handling for edge cases: window close, audio end, file not found

## Mocking

**Framework:** Not detected
- No mocking library imports found (unittest.mock, pytest-mock, etc.)
- Test data stubs used in benchmarks: `--stub` and `--stub-pywt` flags

**Patterns:**
From `research/benchmark.py`:
```python
# Stub modes generate random/synthetic data instead of real DSP processing
--figures --stub           # Fast iteration without real CWT computation
--figures --stub-pywt      # Skip PyWavelet, use random stubs
```

Stub data appears to be randomly generated placeholders for rapid iteration.

**What to Mock:**
- Audio file I/O (use stub audio data generators)
- GPU operations (provide CPU fallback via `gpu_available()` check)
- Window rendering (testing without display)

**What NOT to Mock:**
- Core DSP algorithms: wavelet kernels, FFT operations need genuine computation for correctness
- Configuration validation: must test actual validation logic
- Audio file reading: use small test audio files rather than mocking

## Fixtures and Factories

**Test Data:**
No formal fixture system found. Data handling:
```python
# From research/benchmark.py helper imports
AUDIO_DEFAULT = "path/to/default"
AUDIO_POLYPHONIC = "path/to/polyphonic"
AUDIO_MUSICAL = "path/to/musical"
MIDI_POLYPHONIC = "path/to/midi"
DAW_POLYPHONIC = "path/to/daw"
DAW_MUSICAL = "path/to/daw_musical"
STFT_NPERSEG = 512
NUM_FRAMES = 32
CHIRP_F0 = 100
CHIRP_F1 = 10000
```

Audio files stored in `assets/audio/` directory with subdirectories:
- `assets/audio/daw/` - DAW-generated audio
- `assets/audio/songs/` - Full songs for testing
- `assets/audio/midi/` - MIDI-generated test audio

**Location:**
- Audio fixtures: `assets/audio/` directory
- Helper utilities: `research/utilities.py` (imported by benchmark.py)
- Benchmark output: `assets/images/benchmarks/` for results

## Coverage

**Requirements:** No coverage enforcement configured
- No `.coverage` config file
- No coverage targets specified in pyproject.toml
- Test coverage unknown

**View Coverage:**
No coverage reporting available. Would require:
```bash
pip install pytest pytest-cov
pytest --cov=src tests/
```

## Test Types

**Unit Tests:**
- Scope: Individual function/method validation
- Approach: Manual via `--unit-tests` flag in benchmark.py
- Example from code: Configuration validation methods return error lists
  ```python
  # In config.py
  errors = config.validate()
  if errors:
      for error in errors:
          log.error(f"  - {error}")
  ```

**Integration Tests:**
- Scope: Pipeline end-to-end (audio → CWT → rendering)
- Approach: Run main application with test audio files
- Invocation: `python -m subshader` with test audio configured
- Validation: Check for no exceptions, proper GPU/CPU fallback behavior

**E2E Tests:**
- Framework: Not formally implemented
- Approach: Manual visual verification of rendered output
- Testing: Visual inspection of plots against expected time-frequency representations
- Benchmarks generate comparison PNGs to assess accuracy against STFT/PyWavelet baselines

## Common Patterns

**Async Testing:**
Not applicable (no async code in codebase)

**Error Testing:**
Exception validation through custom exception hierarchy:
```python
# From exceptions.py
class SubShaderException(Exception):
    """Base exception for SubShader application."""
    log_level = "info"

# Catching in main loop
try:
    subshader.loop()
except exceptions.GRACEFUL_EXCEPTIONS as e:
    exceptions.reporter.report(e)
```

**Validation Pattern:**
Configuration objects implement `validate()` returning error list:
```python
# From config.py
def validate(self) -> List[str]:
    """Validate critical audio configuration parameters.

    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []

    if not os.path.exists(self.file_path):
        errors.append(f"Audio file not found: {self.file_path}")

    if self.chunk_size <= 0:
        errors.append(f"chunk_size ({self.chunk_size}) must be positive")

    return errors
```

Used at application startup:
```python
# From __main__.py
config = get_default_config()
errors = config.validate()
if errors:
    error_msg = "Configuration validation failed:\n" + "\n".join(...)
    log.error(error_msg)
    raise ValueError(error_msg)
```

## Testing Gaps

**Known Coverage Gaps:**
1. No unit tests for audio chunking/overlap logic
2. No tests for edge cases in CWT (very short signals, extreme frequencies)
3. No GPU-specific error conditions (out of memory, device not available)
4. No tests for shader compilation failures
5. No tests for circular frame buffer wraparound behavior
6. No multithread/multiprocess safety tests

**Critical Untested Areas:**
- `src/subshader/viz/plotter.py` (812 lines, complex GPU state management)
- `src/subshader/viz/comparison_navigator.py` (1251 lines, GUI logic)
- Shader code in `src/subshader/viz/shaders/`
- Signal generator utilities in `src/subshader/utils/signal_generator.py`

## Recommended Testing Strategy

**Priority 1 - Add pytest + fixtures:**
```bash
pip install pytest pytest-cov numpy
```

Create `tests/` directory with:
- `tests/conftest.py` - shared fixtures for audio data, config, GPU availability
- `tests/test_config.py` - all validation methods
- `tests/test_audio_input.py` - chunk extraction, overlap behavior, EOF handling

**Priority 2 - Add simple unit tests:**
- `tests/test_audio_input.py` - test `get_chunk()` with various overlap factors
- `tests/test_wavelet_kernels.py` - test `WaveletKernel` initialization
- `tests/test_config_validation.py` - test all config validation scenarios

**Priority 3 - Integration tests:**
- `tests/test_pipeline.py` - full pipeline: audio → CWT → rendering
- Test GPU fallback to CPU when GPU unavailable
- Test frame buffer circular behavior

---

*Testing analysis: 2026-03-21*
