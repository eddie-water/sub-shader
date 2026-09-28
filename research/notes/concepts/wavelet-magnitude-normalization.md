# Wavelet Magnitude Normalization

A Morlet wavelet's Gaussian width scales with `1/frequency`, so a low-frequency kernel integrates far more raw energy than a high-frequency one — without correction, low bands would appear disproportionately bright regardless of actual signal content. SubShader normalizes each wavelet kernel to unit L1 area at construction time so every frequency band's CWT response is on a comparable scale before magnitude conversion.

**Key facts**

- The bias is big: a 100 Hz kernel integrates **~100× more energy** than a 10 kHz one (from the L1 ∝ 1/f relationship noted in [`wavelet_kernel.py`](../../../src/subshader/dsp/wavelet_kernel.py))
- The fix is one line at construction: `kernel_t /= np.sum(np.abs(kernel_t))` — paid once at startup, free at runtime
- Because it happens at kernel construction, the per-frame `CWT._normalize_by_scale()` in [`cwt.py`](../../../src/subshader/dsp/cwt.py) is a documented no-op kept for API symmetry

This corrects for the *analysis kernels'* energy bias; the separate question of mapping the resulting magnitudes onto a consistent color range is [Intensity Normalization](intensity-normalization.md).

## In SubShader

Applied in [`WaveletKernel.__init__`](../../../src/subshader/dsp/wavelet_kernel.py). Complex coefficients are then converted to real magnitude via `_compute_mag()` (`np.abs`) in [`cwt.py`](../../../src/subshader/dsp/cwt.py) as the first post-processing step after the transform.

---

**Related:** [Morlet Wavelet](morlet-wavelet.md) · [Continuous Wavelet Transform](continuous-wavelet-transform.md) · [Intensity Normalization](intensity-normalization.md) · [DSP](../../../src/subshader/dsp/DSP.md) · [MAP](../MAP.md)
