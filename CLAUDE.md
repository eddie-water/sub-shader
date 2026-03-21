<!-- GSD:project-start source:PROJECT.md -->
## Project

**SubShader**

SubShader is a real-time audio visualization tool that uses continuous wavelet transforms (CWT) to render frequency content from live audio. It runs a GPU-accelerated pipeline (CuPy + OpenGL) on a three-stage architecture: audio input, DSP processing, and shader-based rendering. The project is targeting a "Demo Ready" milestone where anyone can visit a hosted URL and see real-time audio visualization without installing anything.

**Core Value:** The visualization accurately tracks audio input in real time with minimal latency — if the visual doesn't feel synced to the audio, nothing else matters.

### Constraints

- **GPU:** NVIDIA 4060 Ti 16GB VRAM — server compute budget for hosted demo
- **Tech stack:** Python, CuPy, ModernGL, existing pipeline — no rewrites
- **Documentation voice:** User authors final prose. Claude scaffolds and suggests, doesn't write final copy.
- **Test approach:** Pytest, incremental, not a dedicated phase. Shouldn't require constant user attention.
- **Code style:** Descriptive names, helpers, no comment litter. Structure over documentation.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.12.3 - Core application language
- Python 3.9+ - Minimum required version (specified in `pyproject.toml`)
- GLSL - Fragment and vertex shaders for GPU rendering
## Runtime
- Python 3.12.3 (current environment)
- CUDA-capable GPU (required for CuPy execution)
- OpenGL 3.3+ (required for shader compilation and rendering)
- WSL2 supported with environment detection in `src/subshader/utils/os_env_setup.py`
- pip (via setuptools)
- Lockfile: Not present (using `pyproject.toml` only)
- Virtual environment: Python venv at `/home/eddie-water/dev/python/sub-shader/venv`
## Frameworks
- setuptools (build system) - Project packaging and installation
- CuPy - CUDA-accelerated array computing for GPU-offloaded Continuous Wavelet Transform (CWT)
- cupyx.scipy - CuPy's SciPy compatibility layer, specifically `cupyx.scipy.fft` for FFT operations
- soundfile - Audio file I/O (WAV format support for reading audio files)
- PyWavelets (pywt) - Wavelet transforms (reference implementation in `src/subshader/dsp/wavelet.py`)
- scipy - Scientific computing, includes signal processing tools (STFT, resampling)
- ModernGL 5.6.4+ - Modern OpenGL wrapper for GPU rendering
- GLFW - Window management and OpenGL context creation
- PyOpenGL - Python OpenGL bindings
- PyQt5 - GUI framework for legacy/debug visualization interface
- pyqtgraph - Fast PyQt5-based plotting (includes `CircularFrameBuffer` and `AudioFrameBuffer` classes)
- matplotlib - Plotting and visualization (used in research/benchmark tools)
- NumPy - Numerical array operations, FFT via `numpy.fft`
- Scipy - Scientific computing (signal processing: STFT, resampling)
- tkinter - System display detection in configuration (see `src/subshader/config.py`)
## Key Dependencies
- `cupy` - GPU acceleration for CWT computations; without it, wavelet operations fall back to CPU
- `moderngl>5.6.4` - GPU rendering; incompatible versions may cause shader compilation failures
- `soundfile` - Audio file loading; missing this breaks audio input pipeline
- `PyOpenGL` - OpenGL bindings; required for shader execution
- `numpy` - Core numerical operations for all DSP components
- `scipy` - Signal processing tools (STFT, resampling for comparison/benchmarking)
- `pywavelets` - Reference CWT implementation for comparison; used in research and benchmarking
- `glfw` - Critical for window creation and input handling
- `pyqtgraph` - Provides fast rendering of 2D data (alternative to shader-based rendering in legacy mode)
## Configuration
- Configuration managed via dataclasses in `src/subshader/config.py`:
- Default configuration loaded via `get_default_config()` and overridable in `src/subshader/__main__.py`
- Display dimensions auto-detected from system via tkinter in `_get_system_display_size()`
- `pyproject.toml` - Primary build and dependency configuration
- No setup.cfg, setup.py, or build scripts present
- Package directory: `src/` (configured via `[tool.setuptools]`)
- `SUBSHADER_DEBUG` - Enables OpenGL debug output when set to `'1'`
- `DISPLAY` - WSL display configuration (auto-set to `:0` if not present in WSL)
- `LIBGL_ALWAYS_SOFTWARE` - WSL graphics mode (set to `'1'` for software rendering in WSL)
- `MESA_GL_VERSION_OVERRIDE` - WSL OpenGL version override (set to `'3.3'`)
- Python interpreter path configured in `.vscode/settings.json` (venv path)
## Platform Requirements
- Python 3.9 or higher
- CUDA Toolkit (for CuPy GPU execution)
- GPU with CUDA support (NVIDIA required)
- OpenGL 3.3+ capable hardware
- WSL2 (Windows Subsystem for Linux 2) supported with automatic detection
- pip and venv for dependency management
- Same runtime requirements as development
- No containerization or managed deployment detected
- Standalone desktop application (uses local audio files)
- Graphics output to local display (DISPLAY environment variable)
- setuptools >=61.0 (specified in build-system.requires)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores: `audio_input.py`, `loop_timer.py`, `plot_normalizer.py`
- Module files use full descriptive names without abbreviation
- Package directories are lowercase: `audio/`, `dsp/`, `viz/`, `utils/`
- Lowercase with underscores: `get_logger()`, `get_sample_rate()`, `end_loop_and_report()`
- Getter methods use `get_` prefix: `get_chunk()`, `get_output_shape()`, `get_loops_per_second()`
- Internal methods use single leading underscore: `_generate_chromatic_scale()`, `_validate_gpu_memory()`, `_create_reliable_slice()`
- Setter-adjacent methods use present tense: `validate()`, `update_plot()`, `normalize_by_scale()`
- Lowercase with underscores for multi-word identifiers: `sample_rate`, `chunk_size`, `overlap_factor`, `num_frames`
- Single letter loops acceptable in DSP context: `f` (frequency), `i` (index), `t` (time), `w` (wavelet)
- Constant-like class variables use uppercase with underscores: `GRACEFUL_EXCEPTIONS`, `PI`
- Dataclasses for configuration: `AudioConfig`, `WaveletConfig`, `VisualizationConfig`, `ColorNormalizationConfig`
- Abstract base classes have clear inheritance: `Wavelet`, `Plotter`, inherited by `PyWavelet`, `AntsWavelet`, `ShaderPlot`
- Custom exception classes inherit from parent exceptions: `SubShaderException`, `AudioFileNotFoundError`, `EndOfAudioException`
- Type hints used consistently with NumPy dtypes: `np.float64`, `np.ndarray[np.float64]`, `np.ndarray[np.complexfloating]`
- Return type annotations on methods: `-> None`, `-> np.ndarray`, `-> bool`, `-> float`
- Parameter annotations include NumPy array shapes where relevant: `np.ndarray[np.float64]`
- Union types use pipes: `object | None`
## Code Style
- No linter configuration found; follows PEP 8 style guide
- Line length appears to be around 100 characters (some lines extend slightly beyond)
- Indentation: 4 spaces (standard Python)
- Blank lines: 2 between module-level sections, 1 between method definitions
- No `.flake8`, `.pylintrc`, or ESLint configuration files present
- No configured pre-commit hooks detected
- Project relies on developer discipline for code style
## Import Organization
- `from subshader.config import AudioConfig, WaveletConfig`
- `from subshader.utils.logging import get_logger`
- `from ..config import WaveletConfig` (relative when in subdirectory)
- `src/subshader/utils/__init__.py` exports key utilities: `logger_init`, `get_logger`, `set_log_level`, `LoopTimer`, `env_init`
- `src/subshader/viz/__init__.py` exists but is minimal
- Explicit imports preferred over wildcard `import *`
## Error Handling
- Custom exception hierarchy rooted in `SubShaderException` — see `src/subshader/exceptions.py`
- Domain-specific exceptions: `AudioFileNotFoundError`, `EndOfAudioException`, `WindowCloseException`
- Log level attribute on exceptions: `log_level = "error"` or `"info"`
- Graceful exit through exception tuple `GRACEFUL_EXCEPTIONS` in main loop
## Logging
- `logger_init()` configures root logger with handlers (console and file)
- `get_logger(name)` provides module-level loggers
- Default log file: `logs/subshader.log`
## Comments
- Complex algorithms need explanation: `src/subshader/dsp/wavelet.py` includes extensive docstrings explaining CWT edge effects and normalization
- Configuration and parameter meanings: `src/subshader/config.py` comments explain overlap factors, decay rates, percentiles
- Non-obvious DSP/math concepts: tone support windows, cone of influence, scale-dependent normalization
- Format: `TODO-[ISSUE-NUMBER]` or `TODO` with inline explanation
- Examples in codebase:
## Function Design
- `AudioInput.get_chunk()`: ~20 lines (straightforward)
- `Wavelet.cwt()`: ~32 lines (orchestrates multiple steps)
- `ProcessingConfig.validate()`: ~46 lines (validation logic)
- `AntsWavelet.discard_unreliable_coefs()`: 82 lines (extensive documentation)
- Use positional args for essential parameters; keyword args for optional ones
- Type hints required for all parameters in public APIs
- Default values provided for configuration objects
- Example:
- Explicit return types via annotations
- Methods returning data structures always annotated: `-> np.ndarray`, `-> bool`, `-> tuple[int, int]`
- None return should be explicit: `-> None` (not omitted)
- Example:
## Module Design
- Public methods have no leading underscore
- Single underscore `_` prefix for internal helper methods
- Double underscore `__` reserved for name mangling (not used in codebase)
- Classes export public interface through `__init__` and public methods
- `src/subshader/utils/__init__.py` explicitly exports key classes and functions
- Minimizes need for deep imports: `from subshader.utils import LoopTimer` rather than `from subshader.utils.loop_timer import LoopTimer`
- Abstract base classes use `ABC` and `@abstractmethod` decorators
- Example hierarchy in DSP: `Wavelet` → `AntsWavelet` → `NumPyWavelet`/`CuPyWavelet`
- Plotter hierarchy: `Plotter` → `ShaderPlot`
- Configuration uses dataclasses with `@dataclass` decorator
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Linear data flow: Audio Source → DSP Block → Renderer
- Modular separation of concerns with clear interfaces between stages
- Configuration-driven parameter management across all components
- GPU acceleration for computationally intensive signal processing
- Graceful exception handling with custom exception hierarchy
## Layers
- Purpose: Load audio files and deliver overlapping window frames to the DSP stage
- Location: `src/subshader/audio/audio_input.py`
- Contains: AudioInput class for file reading, chunk extraction with configurable overlap
- Depends on: soundfile, numpy, logging
- Used by: SubShader main orchestrator
- Purpose: Perform Continuous Wavelet Transform (CWT) on raw audio samples across chromatic scale
- Location: `src/subshader/dsp/`
- Contains:
- Depends on: numpy, cupy, pywavelets, scipy, matplotlib
- Used by: SubShader main orchestrator
- Purpose: Store processed DSP results in circular buffer and render via GPU shader
- Location: `src/subshader/viz/`
- Contains:
- Depends on: glfw, moderngl, pyqtgraph, pyqt5, matplotlib, numpy
- Used by: SubShader main orchestrator
- Purpose: Centralized, validated parameter management
- Location: `src/subshader/config.py`
- Contains: Dataclass-based config objects (AudioConfig, WaveletConfig, VisualizationConfig, ColorNormalizationConfig, ProcessingConfig)
- Validation includes: file existence, GPU memory, CPU memory, OpenGL texture limits, performance targets
- Used by: All components during initialization
- Location: `src/subshader/utils/`
- Contains: Logging, environment setup, loop timing, frame counting, GL diagnostics
- Used by: All layers
- Location: `src/subshader/exceptions.py`
- Contains: Custom exception hierarchy (SubShaderException, EndOfAudioException, WindowCloseException, AudioFileNotFoundError)
- Pattern: log_level attribute on exceptions, singleton ExceptionReporter for unified handling
## Data Flow
- Audio position: maintained in AudioInput.file_pos (hop-based seeking)
- Frame history: maintained in CircularFrameBuffer with circular indexing
- Global intensity: maintained in IntensityTracker with exponential decay
- GPU state: maintained in Renderer and GLContext (shader, texture, context)
## Key Abstractions
- Purpose: Define interface and common logic for CWT implementations
- Examples: `src/subshader/dsp/wavelet.py` - Wavelet, PyWavelet, AntsWavelet, NumPyWavelet, CuPyWavelet, CuWavelet
- Pattern: Template method pattern for cwt() with abstract subclass hooks (class_specific_cwt, normalize_by_scale, discard_unreliable_coefs, cleanup)
- Implementations vary by computation location (CPU vs GPU) and algorithm source
- Purpose: Define interface for visualization backends
- Examples: `src/subshader/viz/plotter.py` - Plotter, ShaderPlot
- Pattern: Abstract base with update_plot() and should_window_close() methods
- Allows swapping renderer implementations (currently ShaderPlot with ModernGL)
- Purpose: Centralized, validated parameter containers
- Examples: AudioConfig, WaveletConfig, VisualizationConfig, ColorNormalizationConfig
- Pattern: Dataclass with validate() method returning error list
- Supports composition: ProcessingConfig contains sub-configs
- Purpose: Coordinate pipeline execution and resource lifecycle
- Location: `src/subshader/__main__.py`
- Pattern: Initialization in __init__, main loop in loop(), cleanup in cleanup()
- Manages dependencies: creates AudioInput, CuWavelet, ShaderPlot in order
## Entry Points
- Location: `src/subshader/__main__.py`
- Triggers: `python -m subshader` or direct import
- Responsibilities:
- Before loop starts, config values can be modified programmatically
- Example in __main__.py: `config.audio.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav"`
- Validation happens in get_default_config() before loop
## Error Handling
```python
```
```python
```
- ExceptionReporter.report() inspects exception type and logs at appropriate level
- SubShader.loop() catches GRACEFUL_EXCEPTIONS tuple
- All exceptions propagate to main() finally block for cleanup
- cleanup() is idempotent (safe to call multiple times, handles None attributes)
## Cross-Cutting Concerns
- Centralized via `src/subshader/utils/logging.py`
- logger_init(log_level, console_output, file_output) configures root logger
- All modules import and use get_logger(__name__)
- Formatted with timestamp, module name, level, message
- File output to `logs/subshader.log`
- Config validation at startup via ProcessingConfig.validate()
- Checks: file existence, GPU memory, CPU memory, OpenGL limits, performance targets
- Input validation in each component: AudioInput checks chunk_size > 0, Wavelet checks input_n matches expected shape
- Shape validation throughout pipeline
- Not applicable - local audio processing only
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
