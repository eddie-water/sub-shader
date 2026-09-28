# Time-Frequency Resolution Tradeoff

A fixed analysis window can't be optimal at both ends: a short window gives good time resolution but poor low-frequency resolution, a long window gives good frequency resolution but poor time resolution. This tradeoff is fundamental to any fixed-window method (like the STFT) and is exactly what motivates wavelet analysis, which varies window length with frequency instead of fixing it.

![Fourier vs Wavelet on the same chirp-and-clicks signal](../../assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v43_equal_bands.png)

*The same signal analyzed both ways. The STFT smears the low-frequency sweep and fragments the highs; the CWT traces the contour cleanly at every frequency and resolves each click as a distinct event in time.*

**Why the tradeoff can be sidestepped:** low frequencies change pitch meaningfully with tiny Hz shifts but evolve slowly in time — they need fine frequency resolution, coarse time resolution. High frequencies are the opposite. A fixed window forces one compromise on both; a per-frequency window gives each end what it actually needs.

## In SubShader

Framed in [DSP §3.4](../../src/subshader/dsp/DSP.md#34-stft-and-the-resolution-tradeoff) and resolved in §4 ("What if the basis function's width varied with frequency?"). Quantified in [TIMING.md](../../assets/timing/TIMING.md#why-this-method), which compares STFT, PyWavelets CWT, and SubShader's GPU CWT on speed vs. frequency resolution.

---

**Related:** [Short-Time Fourier Transform](short-time-fourier-transform.md) · [Continuous Wavelet Transform](continuous-wavelet-transform.md) · [Mother Wavelet and Scaling](mother-wavelet-and-scaling.md) · [TIMING](../../assets/timing/TIMING.md) · [MAP](../../MAP.md)
