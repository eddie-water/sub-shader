# Codebase Structure

**Analysis Date:** 2026-03-21

## Directory Layout

```
project-root/
├── .planning/              # GSD planning documents
├── assets/                 # Media assets (audio files, images, benchmarks)
├── logs/                   # Runtime log files (generated)
├── research/               # Benchmark and comparison research
├── .vscode/                # VSCode editor configuration
├── src/                    # Python source code root
│   └── subshader/          # Main package
│       ├── __init__.py     # Version metadata
│       ├── __main__.py     # Entry point orchestrator
│       ├── config.py       # Configuration dataclasses
│       ├── exceptions.py   # Custom exception hierarchy
│       ├── audio/          # Audio input layer
│       │   └── audio_input.py
│       ├── dsp/            # Digital signal processing layer
│       │   ├── gaussian.py
│       │   ├── wavelet.py
│       │   └── wavelet_kernel.py
│       ├── viz/            # Visualization layer
│       │   ├── plotter.py
│       │   ├── plot_normalizer.py
│       │   ├── comparison_navigator.py
│       │   └── shaders/
│       │       ├── __init__.py
│       │       ├── vertex_shader.glsl
│       │       └── fragment_shader.glsl
│       └── utils/          # Utility modules
│           ├── __init__.py
│           ├── logging.py
│           ├── loop_timer.py
│           ├── os_env_setup.py
│           ├── frame_counter_pyqt5.py
│           ├── gl_diagnostics.py
│           ├── signal_generator.py
│           └── quick_plot.py
├── pyproject.toml          # Python package metadata
├── README.md               # Project overview
└── .gitignore             # Git exclusions
```

## Directory Purposes

**src/subshader:**
- Purpose: Main application package
- Contains: Modular components for audio, DSP, visualization, and utilities
- Key files: `__main__.py` (orchestrator), `config.py` (configuration)

**src/subshader/audio:**
- Purpose: Audio file input and chunking
- Contains: AudioInput class for reading files with overlap
- Key files: `audio_input.py` (176 lines)

**src/subshader/dsp:**
- Purpose: Time-frequency signal analysis via Continuous Wavelet Transform
- Contains: Wavelet implementations (PyWavelet, NumPyWavelet, CuPyWavelet), kernel construction
- Key files:
  - `wavelet.py` (628 lines) - Base class, PyWavelet, ANTS implementations, CuPy GPU version
  - `wavelet_kernel.py` (95 lines) - Morlet wavelet kernel generation
  - `gaussian.py` (37 lines) - Gaussian envelope shaping

**src/subshader/viz:**
- Purpose: GPU-accelerated real-time visualization
- Contains: Plotter interface, ShaderPlot with ModernGL, circular frame buffer, intensity tracking
- Key files:
  - `plotter.py` (812 lines) - Main renderer, GLContext, Renderer, CircularFrameBuffer
  - `plot_normalizer.py` (61 lines) - IntensityTracker for global max across frames
  - `comparison_navigator.py` (1251 lines) - PyQtGraph-based comparison tool (legacy)
  - `shaders/` - GLSL shader source code

**src/subshader/utils:**
- Purpose: Cross-cutting utilities and helpers
- Contains: Logging, performance monitoring, environment setup, diagnostics
- Key files:
  - `logging.py` (112 lines) - Centralized logger configuration
  - `loop_timer.py` (66 lines) - Main loop performance monitoring
  - `os_env_setup.py` (59 lines) - Environment variable initialization
  - `frame_counter_pyqt5.py` (165 lines) - PyQt5 FPS display widget
  - `gl_diagnostics.py` (215 lines) - OpenGL capability detection
  - `signal_generator.py` (92 lines) - Synthetic signal generation for testing
  - `quick_plot.py` (68 lines) - Matplotlib plotting helper

## Key File Locations

**Entry Points:**
- `src/subshader/__main__.py`: Main orchestrator - creates SubShader instance, runs main loop, handles exceptions, cleanup

**Configuration:**
- `src/subshader/config.py`: All configuration classes and validation

**Core Logic by Layer:**

*Audio Processing:*
- `src/subshader/audio/audio_input.py`: File reading and chunk extraction with overlap

*DSP:*
- `src/subshader/dsp/wavelet.py`: Wavelet transform implementations and post-processing
- `src/subshader/dsp/wavelet_kernel.py`: Morlet wavelet kernel construction
- `src/subshader/dsp/gaussian.py`: Gaussian envelope generation

*Visualization:*
- `src/subshader/viz/plotter.py`: Rendering pipeline, frame buffer, shader execution
- `src/subshader/viz/plot_normalizer.py`: Intensity normalization across frames
- `src/subshader/viz/shaders/`: GLSL shader source code

**Testing & Experimentation:**
- `research/`: Benchmark scripts and comparison analyses
- `assets/audio/`: Test audio files

## Naming Conventions

**Files:**
- Module files: `snake_case.py` (e.g., `audio_input.py`, `wavelet_kernel.py`)
- Shader files: `{name}_shader.glsl` (e.g., `vertex_shader.glsl`, `fragment_shader.glsl`)
- Config files: Named by function (e.g., `config.py`)
- Package files: `__init__.py` for imports, `__main__.py` for entry point

**Directories:**
- Functional grouping: `audio`, `dsp`, `viz`, `utils`
- All lowercase with underscores for multi-word names
- Abbreviated descriptive names reflecting purpose

**Python Classes:**
- Class names: PascalCase (e.g., AudioInput, CuWavelet, ShaderPlot, IntensityTracker, GLContext)
- Abstract base classes: Prefix with common interface (Wavelet, Plotter)
- Config classes: Suffix with "Config" (AudioConfig, WaveletConfig, VisualizationConfig)
- Exception classes: Suffix with "Exception" or "Error" (EndOfAudioException, AudioFileNotFoundError)

**Python Functions & Methods:**
- Function names: snake_case (e.g., get_chunk, validate, normalize_by_scale)
- Private methods: Leading underscore (e.g., _create_reliable_slice, _validate_gpu_memory)
- Property-like getters: get_ prefix (e.g., get_sample_rate, get_output_shape)
- Boolean checks: is_ or should_ prefix (e.g., is_ready, should_window_close)

**Python Variables:**
- Module-level constants: UPPER_CASE (e.g., PI, GRACEFUL_EXCEPTIONS, CUPY_AVAILABLE)
- Instance attributes: snake_case (e.g., sample_rate, output_shape, global_max)
- Loop variables: Single letters or descriptive (e.g., i, w for wavelet, coefs for coefficients)

## Where to Add New Code

**New Feature (e.g., different audio effect):**
- Primary code: `src/subshader/dsp/` (create new module or extend wavelet.py)
- Tests: Create corresponding test file or test directory structure
- Configuration: Add to WaveletConfig or create new config class in `config.py`
- Integration: Connect in SubShader.__init__() after wavelet initialization

**New Component/Module (e.g., audio visualization filter):**
- Core implementation: `src/subshader/viz/` for visualization, `src/subshader/audio/` for audio
- Create new file or extend existing based on module size
- Follow base class pattern if multiple implementations expected (Plotter, Wavelet pattern)
- Configuration: Add config dataclass if needed, register in ProcessingConfig

**New Utility Function:**
- Shared helpers: `src/subshader/utils/` with appropriate module name
- If cross-cutting concern (logging, timing, etc.): add to existing utils module
- If specialized tool: create new utility module (e.g., `signal_processor.py`)
- Export via `src/subshader/utils/__init__.py` if widely used

**New DSP Post-Processing Step:**
- Add to Wavelet class or create specialized post-processor class
- Location: `src/subshader/dsp/`
- Follow Wavelet pattern: abstract method in base, implement in subclasses
- Integrate into cwt() pipeline between existing steps

**New Configuration Parameter:**
- Location: `src/subshader/config.py`
- Create or update dataclass (AudioConfig, WaveletConfig, VisualizationConfig, ColorNormalizationConfig)
- Add validation method or extend existing validate()
- Document parameter with docstring in dataclass

## Special Directories

**research/:**
- Purpose: Benchmark scripts and research artifacts
- Generated: Partially (outputs)
- Committed: Yes (research code and documentation)
- Contents: Comparative analysis of different signal processing methods

**assets/:**
- Purpose: Media resources
- Generated: Partially (benchmark images)
- Committed: Yes (audio files, design assets)
- Subdirectories:
  - `audio/`: Test audio files (.wav format)
  - `images/`: Visualization outputs and benchmarks

**logs/:**
- Purpose: Runtime logging output
- Generated: Yes (created by logger_init)
- Committed: No (in .gitignore)
- Contains: `subshader.log` with timestamped application logs

**src/subshader.egg-info/:**
- Purpose: Python package metadata (auto-generated by setuptools)
- Generated: Yes
- Committed: No (in .gitignore)

**.__pycache__/:**
- Purpose: Python bytecode cache (auto-generated)
- Generated: Yes
- Committed: No (in .gitignore)

## Import Patterns

**Standard Library:**
```python
import os
import time
from pathlib import Path
from typing import Optional, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
```

**External Dependencies:**
```python
import numpy as np
import cupy as cp
from cupyx.scipy import fft as cp_fft
import soundfile as sf
import glfw
import moderngl
from subshader.utils.logging import get_logger
```

**Internal Imports:**
```python
# Absolute imports from package root
from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import CuWavelet
from subshader.viz.plotter import ShaderPlot
from subshader.utils.logging import logger_init, get_logger
```

**No Relative Imports:** Codebase uses absolute imports from package root (no `from ..config import` patterns)

