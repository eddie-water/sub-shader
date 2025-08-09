# Sub Shader

Sub Shader is a real-time audio analysis tool that uses GPU-accelerated time-frequency analysis to visualize audio data. It takes an audio file, performs Continuous Wavelet Transform (CWT) analysis, and renders the results using OpenGL shaders for high-performance visualization.

## SubShader Performance Update

### Performance
- **Current FPS**: ~40 FPS
- **Optimization**: Moved downsampling to wavelet class, reducing CPU→GPU data transfer

### Recent Changes

#### **App Structure**
- **SubShader Module**: Now handles main loop, timing, central logging, and graceful shutdown
- **Modular Design**: Separated audio processing, rendering, and display components

#### **Shader Implementation** 
- **Graphics Context**: ModernGL/GLFW setup with fullscreen/windowed modes
- **Geometry**: VBO/VAO fullscreen quad for rendering
- **Texture Pipeline**: Direct texture upload with shader rendering

#### **Graphics Pipeline**
```
Audio Data → update_plot() → Circular Frame Buffer → Flattened Texture → 
Clear Back Buffer → Render → Swap Buffers → Display
```

#### **Logging System**
- **Replaced print statements**: Now using structured logging (info, debug, warning, error)
- **GPU Transfer Tracking**: Logs all CPU↔GPU transfers and error paths
- **Better debugging**: Clear visibility into performance and errors

#### **Code Improvements**
- **Naming**: Better function and class names (CuWavelet, LoopTimer)
- **Error Handling**: Division-by-zero and EOF handling
- **Documentation**: Improved docstrings and code organization

## Current Results

### Shader Plot Output
![SubShader Visualization](https://github.com/user-attachments/assets/19f9c2a9-9964-4477-aa27-08e7447f6437)

**Source**: [Beltran Coachella Set](https://soundcloud.com/listenbeltran/beltran-coachella-yuma-weekend-1-2025) (10:20 - 10:27)

### Performance Display
![FPS Metrics](https://github.com/user-attachments/assets/51bf7c04-2cc2-45dd-808f-6ba320e7a0b2)

## Installation

### Create Virtual Environment
Create a virtual environment to avoid cluttering your system:
```bash
python3 -m venv venv
```

### Activate Virtual Environment
On Linux/WSL:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Install Dependencies
Install the package and all dependencies in editable mode:
```bash
pip install -e .
```

This installs the project as an importable package, so you don't need to reinstall after code changes.

## Usage

### Run Main Application
```bash
python -m subshader
```

### Deactivate Virtual Environment
When finished:
```bash
deactivate
```

## Development

### Performance Benchmark
**Note**: The benchmark script may not work with the current codebase due to recent refactoring. It was designed for earlier versions of the project.

```bash
# WARNING: This script may be outdated and not work with current implementation
python research/benchmark.py
```

### Development Progress

#### **Completed**
- ✅ Audio input with EOF handling
- ✅ Dynamic scrolling plot
- ✅ Modular architecture
- ✅ Debug utilities (QuickPlot/MultiQuickPlot)
- ✅ Code cleanup (removed unused files)

## Technical Details

### Continuous Wavelet Transform
The project implements GPU-accelerated Continuous Wavelet Transform (CWT) using Complex Morlet Wavelets. This provides better time-frequency localization compared to traditional Fourier Transform methods.

### GPU Acceleration
- Uses CuPy for GPU-accelerated wavelet computations
- Minimizes CPU↔GPU data transfers through strategic downsampling
- Direct texture upload to OpenGL for rendering

### Real-time Visualization
- OpenGL shader-based rendering for high performance
- Circular buffer for scrolling visualization
- Fullscreen and windowed display modes

## Requirements

- Python 3.8+
- CUDA-capable GPU (I have a 4060 ti)
- OpenGL 3.3+ support

See `pyproject.toml` for complete dependency list.