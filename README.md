# Sub Shader

Sub Shader is a real-time audio visualizer written in Python. It's an audio-graphics pipeline that analyzes and converts audio into a visual representation using modern techniques in digital signal processing and parallel computing.

This project is a showcase of my skills and interests in DSP and GPU acceleration. It's given me a deeper understanding of the foundations of signal processing and how they can be generalized and used in other contexts beyond monitoring audio. It also serves as an exercise on offloading parallel operations onto a GPU. The top-level design and performance of Sub Shader are detailed in this document. 

*[Insert Demo Clip]*

---

## Design

```
Audio Source → DSP Block → Renderer
```

Sub Shader splits the pipeline into three modules:

### Audio Source

Loads audio from file and delivers overlapping window frames of audio samples to the DSP Stage. The overlap reduces edge artifacts at window boundaries.

For more details → [Audio README](AUDIO_README.md)

### DSP Stage

Performs the Continuous Wavelet Transform using CUDA on the raw audio samples across the chromatic scale. Post-procesing includes scale normalization, discarding of edge-contaminated results, and downsampling.

For a comprehensive and intuitive explanation → [DSP README](DSP_README.md)

### Renderer

Stores each result chronologically in a circular buffer, and uploads the entire buffer as a single GPU texture. The renderer feeds the texture data to a fragment shader that colormaps the results into a scale vs time plot.

For specific details → [Renderer README](RENDERER_README.md)

<!-- Placeholder: init flowchart -->

<!-- Placeholder: runtime flowchart -->

---
<!-- TODO Pick ip from here -->
## Performance

The CWT trades compute cost for better time-frequency resolution. Below is a comparison of Fast Fourier Transform (FFT), Short-Time Fourier Transform (STFT), and the Continous Wavelet Transform (CWT) on a few example audio signals.

### Chirp (frequency sweep)

<!-- Placeholder: chirp FFT vs STFT vs CWT comparison figure -->

A linearly swept frequency is the clearest demonstration of the resolution tradeoff. FFT collapses the whole sweep into a smeared spectrum. The STFT gets closer but its fixed window size gives it poor low-end resolution. CWT tracks the sweep continuously because its window width adapts to the frequency being analyzed.

### MIDI Synth

<!-- Placeholder: MIDI FFT vs STFT vs CWT comparison figure -->

Harmonic structure from a synthesized source is where CWT's logarithmic frequency spacing shows its value. Each overtone resolves cleanly at its correct frequency and time.

### Music (percussion + sustained bass)

*[House Music Beat]*

Here we can see the classic four-on-the-floor house rhythm come in and come out. From the *[link source]*. Compare the shitty fft vs stft vs pywt vs good cwt

---

## Benchmark

*[Placeholder: full timing comparison STFT, PYWT, CWT from research/benchmark.py]*

*[Placeholder: full timing breakdown of Sub Shader using fastest DSP block from research/benchmark.py — runtime per block and end-to-end]*

*[Placeholder: compare numpy vs cupy from research/benchmark.py — runtime per block and end-to-end]*

---

## Installation

*[Insert instructions on how to install using venv and link to requirements]*

```

### Requirements

- Python 3.8+
- CUDA-capable GPU
- OpenGL 3.3+

## Future Improvements
This project has given me a deeper understanding of the foundational concepts in real-time signal processing. There are DSP fundamentals that can be generalized, allowing for higher level pattern detection and feature extraction beyond audio contexts. Higher precision results require higher compute, so we explore why off-loading some of the work  This project goes into the details of the why I made these design decisions 

*[List future improvements]*