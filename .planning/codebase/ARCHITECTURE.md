# Architecture

**Analysis Date:** 2026-03-21

## Pattern Overview

**Overall:** Three-stage pipeline architecture with layered abstraction

**Key Characteristics:**
- Linear data flow: Audio Source → DSP Block → Renderer
- Modular separation of concerns with clear interfaces between stages
- Configuration-driven parameter management across all components
- GPU acceleration for computationally intensive signal processing
- Graceful exception handling with custom exception hierarchy

## Layers

**Audio Input Layer:**
- Purpose: Load audio files and deliver overlapping window frames to the DSP stage
- Location: `src/subshader/audio/audio_input.py`
- Contains: AudioInput class for file reading, chunk extraction with configurable overlap
- Depends on: soundfile, numpy, logging
- Used by: SubShader main orchestrator

**DSP Stage (Signal Processing Layer):**
- Purpose: Perform Continuous Wavelet Transform (CWT) on raw audio samples across chromatic scale
- Location: `src/subshader/dsp/`
- Contains:
  - Wavelet base class and implementations (PyWavelet, NumPyWavelet, CuPyWavelet)
  - WaveletKernel for Morlet wavelet generation
  - Gaussian envelope shaping
  - Post-processing: scale normalization, edge artifact removal, downsampling
- Depends on: numpy, cupy, pywavelets, scipy, matplotlib
- Used by: SubShader main orchestrator

**Visualization Layer (Rendering):**
- Purpose: Store processed DSP results in circular buffer and render via GPU shader
- Location: `src/subshader/viz/`
- Contains:
  - ShaderPlot: main plotter interface with ModernGL renderer
  - GLContext: GLFW window and OpenGL context management
  - Renderer: shader compilation, texture management, gamma correction
  - CircularFrameBuffer: chronological storage with intensity tracking
  - IntensityTracker: global max across frames for consistent colormap scaling
  - Comparison Navigator: legacy PyQtGraph implementation
- Depends on: glfw, moderngl, pyqtgraph, pyqt5, matplotlib, numpy
- Used by: SubShader main orchestrator

**Configuration Layer:**
- Purpose: Centralized, validated parameter management
- Location: `src/subshader/config.py`
- Contains: Dataclass-based config objects (AudioConfig, WaveletConfig, VisualizationConfig, ColorNormalizationConfig, ProcessingConfig)
- Validation includes: file existence, GPU memory, CPU memory, OpenGL texture limits, performance targets
- Used by: All components during initialization

**Utilities & Cross-Cutting Concerns:**
- Location: `src/subshader/utils/`
- Contains: Logging, environment setup, loop timing, frame counting, GL diagnostics
- Used by: All layers

**Exception Handling Layer:**
- Location: `src/subshader/exceptions.py`
- Contains: Custom exception hierarchy (SubShaderException, EndOfAudioException, WindowCloseException, AudioFileNotFoundError)
- Pattern: log_level attribute on exceptions, singleton ExceptionReporter for unified handling

## Data Flow

**Main Loop Data Pipeline:**

1. **Audio Chunk Retrieval** → AudioInput.get_chunk()
   - Reads next chunk with overlap from file
   - Returns float64 numpy array shape (chunk_size,)
   - Returns None when EOF reached

2. **Continuous Wavelet Transform** → Wavelet.cwt(audio_data)
   - Input: shape (chunk_size,) float64 audio samples
   - Class-specific implementation (CuWavelet uses GPU via CuPy)
   - Step 2a: class_specific_cwt() → complex coefficients shape (num_freqs, chunk_size)
   - Step 2b: normalize_by_scale() → scale-dependent energy normalization
   - Step 2c: compute_mag() → convert to real magnitudes via absolute value
   - Step 2d: discard_unreliable_coefs() → slice to reliable center region
   - Step 2e: downsample() → reduce time dimension to target_width
   - Output: shape (num_freqs, target_width) float32/float64 normalized coefficients

3. **Circular Buffer Update** → CircularFrameBuffer.push_frame()
   - Stores frame chronologically in ring buffer (FIFO)
   - Updates IntensityTracker with new frame's percentile max
   - Maintains global_max across all stored frames

4. **GPU Texture Upload & Render** → Renderer.update_texture() + render_graphic()
   - Flattens entire buffer: shape (num_frames * num_freqs, target_width)
   - Uploads as 2D GPU texture
   - Fragment shader applies colormap and gamma correction
   - Displays as scale (Y) vs time (X) heatmap

**State Management:**
- Audio position: maintained in AudioInput.file_pos (hop-based seeking)
- Frame history: maintained in CircularFrameBuffer with circular indexing
- Global intensity: maintained in IntensityTracker with exponential decay
- GPU state: maintained in Renderer and GLContext (shader, texture, context)

## Key Abstractions

**Wavelet Base Class:**
- Purpose: Define interface and common logic for CWT implementations
- Examples: `src/subshader/dsp/wavelet.py` - Wavelet, PyWavelet, AntsWavelet, NumPyWavelet, CuPyWavelet, CuWavelet
- Pattern: Template method pattern for cwt() with abstract subclass hooks (class_specific_cwt, normalize_by_scale, discard_unreliable_coefs, cleanup)
- Implementations vary by computation location (CPU vs GPU) and algorithm source

**Plotter Base Class:**
- Purpose: Define interface for visualization backends
- Examples: `src/subshader/viz/plotter.py` - Plotter, ShaderPlot
- Pattern: Abstract base with update_plot() and should_window_close() methods
- Allows swapping renderer implementations (currently ShaderPlot with ModernGL)

**Configuration Dataclasses:**
- Purpose: Centralized, validated parameter containers
- Examples: AudioConfig, WaveletConfig, VisualizationConfig, ColorNormalizationConfig
- Pattern: Dataclass with validate() method returning error list
- Supports composition: ProcessingConfig contains sub-configs

**SubShader Main Orchestrator:**
- Purpose: Coordinate pipeline execution and resource lifecycle
- Location: `src/subshader/__main__.py`
- Pattern: Initialization in __init__, main loop in loop(), cleanup in cleanup()
- Manages dependencies: creates AudioInput, CuWavelet, ShaderPlot in order

## Entry Points

**Module Entry Point:**
- Location: `src/subshader/__main__.py`
- Triggers: `python -m subshader` or direct import
- Responsibilities:
  - Logging initialization via logger_init()
  - Configuration loading via get_default_config()
  - SubShader orchestrator creation and loop execution
  - Exception handling and graceful shutdown
  - Resource cleanup

**Configuration Override:**
- Before loop starts, config values can be modified programmatically
- Example in __main__.py: `config.audio.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav"`
- Validation happens in get_default_config() before loop

## Error Handling

**Strategy:** Layered exception hierarchy with log-level preservation

**Patterns:**

**Graceful Exit Exceptions:**
```python
# End of audio file - logged as INFO
EndOfAudioException("Audio file processing complete - reached end of file.")

# Window closed by user - logged as INFO
WindowCloseException("Window closed by user")

# Keyboard interrupt - logged as WARNING
KeyboardInterrupt
```

**Error Exceptions:**
```python
# File not found - logged as ERROR
AudioFileNotFoundError("Audio file not found: {path}")

# Configuration validation failures - logged as ERROR, raised with full message list
ValueError("Configuration validation failed:\n" + validation_errors)

# Runtime errors from GPU/OpenGL - logged as ERROR
RuntimeError
```

**Handler Pattern:**
- ExceptionReporter.report() inspects exception type and logs at appropriate level
- SubShader.loop() catches GRACEFUL_EXCEPTIONS tuple
- All exceptions propagate to main() finally block for cleanup
- cleanup() is idempotent (safe to call multiple times, handles None attributes)

## Cross-Cutting Concerns

**Logging:**
- Centralized via `src/subshader/utils/logging.py`
- logger_init(log_level, console_output, file_output) configures root logger
- All modules import and use get_logger(__name__)
- Formatted with timestamp, module name, level, message
- File output to `logs/subshader.log`

**Validation:**
- Config validation at startup via ProcessingConfig.validate()
- Checks: file existence, GPU memory, CPU memory, OpenGL limits, performance targets
- Input validation in each component: AudioInput checks chunk_size > 0, Wavelet checks input_n matches expected shape
- Shape validation throughout pipeline

**Authentication & Authorization:**
- Not applicable - local audio processing only

