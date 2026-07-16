# SubShader Timing

## Overview

✅ **Real-Time Performance**

- **Processing Speed** — ~612,946 samples per second
- **FPS** — ~75 frames per second
- **Deadline** — 14× under the 186 ms deadline

## Pipeline Profiling

**Test parameters**

| Number of Samples | Sampling Rate | Lowest Frequency | Highest Frequency | Frequency Resolution | Number of Frequencies |
| --- | --- | --- | --- | --- | --- |
| 16,384 | 44.1 kHz | 27.5 Hz | 21.1 kHz | 12 per octave | 116 |

### Pipeline Structure

![Start up — one-time construction, CPU and GPU lanes](pipeline_init.drawio.png)

![Runtime loop — the per-frame pipeline, CPU and GPU lanes](pipeline_runtime.drawio.png)

### Start Up vs Runtime — Measured Per-Stage Timing

![Pipeline timing — startup construction and runtime loop, each to scale](timing_pipeline.png)

## Fourier vs Wavelet Implementations

![Fourier vs Wavelet — time per frame (log scale) and frequency resolution](timing_methods.png)

## Performance Across Settings

![Compute per frame for each backend, chunk size, and resolution](timing_config.png)
