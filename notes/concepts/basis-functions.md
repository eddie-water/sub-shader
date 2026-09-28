# Basis Functions

A basis is a set of reference patterns used to fully decompose a signal into independent components with no information lost and none double-counted — N independent samples in time become N independent measurements in frequency. Fourier analysis uses sine waves at integer multiples of a fundamental frequency as its basis; wavelet analysis uses scaled copies of a single mother wavelet instead, trading strict orthogonality for adaptive time-frequency resolution.

![Measuring a signal against pure sine references at 2 Hz and 10 Hz](../../assets/images/dsp/figures/by_figure/fig_2_6_sine_basis/fig_2_6_sine_basis_2hz_10hz_v21.png)

*A signal measured against pure sine references — the dot product against each basis function reveals how much of that frequency is present.*

**Why integer multiples:** sine waves at integer multiples of the fundamental are mutually orthogonal — over the window, their sign agreement cancels to exactly zero — so each measures information no other one can. The CWT deliberately gives this up: its scaled wavelets overlap in what they measure, which is redundant but yields smoother coverage and better edge behavior.

## In SubShader

Introduced in [DSP §2.6](../../src/subshader/dsp/DSP.md#26-basis-functions---full-signal-decomposition) (sinusoidal basis) and extended in §4 (wavelet basis functions). The wavelet basis is built once at startup as a bank of 116 [`WaveletKernel`](../../src/subshader/dsp/wavelet_kernel.py) instances, one per frequency in the [chromatic scale](chromatic-frequency-scale.md).

---

**Related:** [Inner Product and Correlation](inner-product-and-correlation.md) · [Mother Wavelet and Scaling](mother-wavelet-and-scaling.md) · [Sampling and Nyquist Limit](sampling-and-nyquist-limit.md) · [DSP](../../src/subshader/dsp/DSP.md) · [MAP](../MAP.md)
