# Intensity Normalization

Raw CWT coefficient magnitudes vary wildly across signals (quiet recording vs. loud one), so a fixed colormap range would look wildly different frame to frame without a consistent reference. SubShader pre-scans the entire audio file before playback, computes a percentile-based magnitude, and holds that value fixed for every frame — the same coefficient magnitude always maps to the same brightness.

**Key facts**

- Reference: the **99th percentile** of coefficient magnitude across the whole file (`global_intensity_percentile` in [`config.py`](../../src/subshader/config.py)) — a percentile rather than the max, so one spike can't dim the entire visualization
- Held constant: `global_max = max(fixed_max, floor_value)` is set once in [`IntensityTracker`](../../src/subshader/renderer/intensity.py) and never updated during playback
- Safety floor: the shader divides by `max(intensity_max, 1e-8)` so silent or warmup frames can't divide by zero
- **Open question** (flagged in [RENDERER.md](../../src/subshader/renderer/RENDERER.md#color-mapping)): a fixed pre-scan gives frame-to-frame consistency, but whether an absolute reference is what the visualization actually *needs* — vs. something adaptive — is an unresolved polish item. It also can't work for live input, where there's no file to pre-scan.

## In SubShader

Computed by a pipeline pre-scan before the render loop starts, consumed per-frame as the `intensity_max` uniform in the fragment shader (see [Colormap and Gamma Correction](colormap-and-gamma-correction.md)). Documented in [RENDERER.md](../../src/subshader/renderer/RENDERER.md#color-mapping).

---

**Related:** [Wavelet Magnitude Normalization](wavelet-magnitude-normalization.md) · [Colormap and Gamma Correction](colormap-and-gamma-correction.md) · [Circular Frame Buffer](circular-frame-buffer.md) · [RENDERER](../../src/subshader/renderer/RENDERER.md) · [MAP](../../MAP.md)
