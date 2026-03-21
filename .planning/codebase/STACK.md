# Technology Stack

**Analysis Date:** 2026-03-21

## Languages

**Primary:**
- Python 3.12.3 - Core application language
- Python 3.9+ - Minimum required version (specified in `pyproject.toml`)
- GLSL - Fragment and vertex shaders for GPU rendering

## Runtime

**Environment:**
- Python 3.12.3 (current environment)
- CUDA-capable GPU (required for CuPy execution)
- OpenGL 3.3+ (required for shader compilation and rendering)
- WSL2 supported with environment detection in `src/subshader/utils/os_env_setup.py`

**Package Manager:**
- pip (via setuptools)
- Lockfile: Not present (using `pyproject.toml` only)
- Virtual environment: Python venv at `/home/eddie-water/dev/python/sub-shader/venv`

## Frameworks

**Core:**
- setuptools (build system) - Project packaging and installation

**GPU Computing:**
- CuPy - CUDA-accelerated array computing for GPU-offloaded Continuous Wavelet Transform (CWT)
- cupyx.scipy - CuPy's SciPy compatibility layer, specifically `cupyx.scipy.fft` for FFT operations

**Audio Processing:**
- soundfile - Audio file I/O (WAV format support for reading audio files)
- PyWavelets (pywt) - Wavelet transforms (reference implementation in `src/subshader/dsp/wavelet.py`)
- scipy - Scientific computing, includes signal processing tools (STFT, resampling)

**Graphics Rendering:**
- ModernGL 5.6.4+ - Modern OpenGL wrapper for GPU rendering
- GLFW - Window management and OpenGL context creation
- PyOpenGL - Python OpenGL bindings

**UI & Visualization:**
- PyQt5 - GUI framework for legacy/debug visualization interface
- pyqtgraph - Fast PyQt5-based plotting (includes `CircularFrameBuffer` and `AudioFrameBuffer` classes)
- matplotlib - Plotting and visualization (used in research/benchmark tools)

**Data Processing:**
- NumPy - Numerical array operations, FFT via `numpy.fft`
- Scipy - Scientific computing (signal processing: STFT, resampling)

**Development Utilities:**
- tkinter - System display detection in configuration (see `src/subshader/config.py`)

## Key Dependencies

**Critical:**
- `cupy` - GPU acceleration for CWT computations; without it, wavelet operations fall back to CPU
  - Depends on CUDA toolkit installation and CUDA-capable GPU
- `moderngl>5.6.4` - GPU rendering; incompatible versions may cause shader compilation failures
- `soundfile` - Audio file loading; missing this breaks audio input pipeline
- `PyOpenGL` - OpenGL bindings; required for shader execution

**Signal Processing:**
- `numpy` - Core numerical operations for all DSP components
- `scipy` - Signal processing tools (STFT, resampling for comparison/benchmarking)
- `pywavelets` - Reference CWT implementation for comparison; used in research and benchmarking

**Rendering:**
- `glfw` - Critical for window creation and input handling
- `pyqtgraph` - Provides fast rendering of 2D data (alternative to shader-based rendering in legacy mode)

## Configuration

**Environment:**
- Configuration managed via dataclasses in `src/subshader/config.py`:
  - `AudioConfig` - Audio file path, chunk size, overlap factor
  - `WaveletConfig` - Sampling frequency, scale parameters (notes per octave, octaves, root note)
  - `VisualizationConfig` - Display dimensions, color normalization settings
- Default configuration loaded via `get_default_config()` and overridable in `src/subshader/__main__.py`
- Display dimensions auto-detected from system via tkinter in `_get_system_display_size()`

**Build:**
- `pyproject.toml` - Primary build and dependency configuration
- No setup.cfg, setup.py, or build scripts present
- Package directory: `src/` (configured via `[tool.setuptools]`)

**Environment Variables:**
- `SUBSHADER_DEBUG` - Enables OpenGL debug output when set to `'1'`
- `DISPLAY` - WSL display configuration (auto-set to `:0` if not present in WSL)
- `LIBGL_ALWAYS_SOFTWARE` - WSL graphics mode (set to `'1'` for software rendering in WSL)
- `MESA_GL_VERSION_OVERRIDE` - WSL OpenGL version override (set to `'3.3'`)
- Python interpreter path configured in `.vscode/settings.json` (venv path)

## Platform Requirements

**Development:**
- Python 3.9 or higher
- CUDA Toolkit (for CuPy GPU execution)
- GPU with CUDA support (NVIDIA required)
- OpenGL 3.3+ capable hardware
- WSL2 (Windows Subsystem for Linux 2) supported with automatic detection
- pip and venv for dependency management

**Production:**
- Same runtime requirements as development
- No containerization or managed deployment detected
- Standalone desktop application (uses local audio files)
- Graphics output to local display (DISPLAY environment variable)

**Build Requirements:**
- setuptools >=61.0 (specified in build-system.requires)

---

*Stack analysis: 2026-03-21*
