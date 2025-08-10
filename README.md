# SubShader

SubShader is a real-time audio visualizer. It reads in an audio file, performs the Continuous Wavelet Transform using CuPy, and plots the results in real-time with a 2D shader. Currently getting around 40 FPS.

## Overview

**Audio File → CWT Analysis → GPU Texture → Shader Rendering → Real-time Visualization**

### Current Status

![SubShader Visualization](assets/images/beltran_souncloud_wav_0m_8s_to_0m_25s.png)

**Source**: [Beltran Coachella Soundcloud Rip](https://soundcloud.com/listenbeltran/beltran-coachella-yuma-weekend-1-2025) ~(8:22 - 8:31)

![SubShader Visualization](https://github.com/user-attachments/assets/19f9c2a9-9964-4477-aa27-08e7447f6437)

**Source**: [Beltran Coachella Soundclou Rip](https://soundcloud.com/listenbeltran/beltran-coachella-yuma-weekend-1-2025) ~(10:19 - 10:27)

### Software Flow
- **Audio Input**: Retrieves chunks of audio from file.
- **CWT**: Performs time-frequency analysis on the audio, accelerated with CuPy.
- **Real-Time Plotting**: Using a 2D shader to visualize the CWT.

### What is the CWT and why use it?

The Continuous Wavelet Transform (CWT) is really good for analyzing audio for musical content. The Fast Fourier Transform (FFT), which is typically the go-to for DSP / time-frequency analysis, unfortunately has a fixed time-frequency resolution. You can only configure them to have A: good frequency resolution (accurate low frequencies, but blurry timing) or B: good time resolution (accurate timing, but blurry frequencies). 

The CWT overcomes this limitation by adapting the number of samples per transform: 
- Uses more time samples for lower frequencies, since they tend to last longer in time like basslines and sustained melodic notes, which also gives us fine frequency resolution, which is advantageous because low frequencies are easily differentiable to the ear
- Uses fewer time samples for higher frequencies, since short transient events like percussion don't last very long in time, which gives us precise timing for quick events, and is advantageous because the ear is bad at differentiating high frequencies

**Note**: After auditing the code, I realized the current implementation doesn't fully implement this adaptive behavior. All wavelets are generated using the same number of time samples, which means the time resolution is fixed and the frequency resolution is also fixed. This means we're not getting the full benefits of true CWT. I've created [this issue](https://github.com/users/eddie-water/projects/1/views/1?pane=issue&itemId=113509598&issue=eddie-water%7Csub-shader%7C36) to track fixing this.

## Performance

Currently achieving around 40 FPS.

## Installation

### Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/WSL)
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Run
```bash
python -m subshader
```

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenGL 3.3+ support
