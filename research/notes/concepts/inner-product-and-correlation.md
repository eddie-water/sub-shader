# Inner Product and Correlation

The inner product measures similarity between a signal and a reference pattern by multiplying corresponding values and summing the result — a "sum of products." For discrete signals this is the dot product; the result is large when the two patterns share a sign pattern (correlated), near zero when unrelated, and negative when opposed. Every transform in the pipeline (Fourier, STFT, CWT) is this same operation applied against a different family of reference patterns.

![Sign accumulation — each pairwise product votes agreement or disagreement](../../../assets/images/dsp/figures/by_figure/fig_2_5_sign_accumulation/fig_2_5_sign_accumulation_composite_v49_border_flush.png)

*Each pairwise product votes: agree in sign → push the sum up, disagree → pull it down, zero → abstain. The accumulated total is the correlation score.*

**Why it works without trigonometry:** the dot product equals `|a||b|cos(θ)` — the relative angle is already baked into the components before you multiply them. Projection reveals the same relationship geometrically; the dot product computes it with nothing but multiply-and-add.

## In SubShader

Foundational math in [DSP §2](../../../src/subshader/dsp/DSP.md#2-foundations) (Inner Product → Dot Product → Vector Projection → Sign Accumulation). Implemented implicitly wherever a signal is multiplied against a basis/kernel and summed — most concretely in [`WaveletKernel`](../../../src/subshader/dsp/wavelet_kernel.py) construction and the FFT-domain multiply in [`CWT.transform()`](../../../src/subshader/dsp/cwt.py). One frame of SubShader output is ~1.9 million of these comparisons (16,384 samples × 116 wavelets), which is why the [convolution shortcut](convolution-and-kernel.md) and [GPU offload](gpu-accelerated-fft-convolution.md) matter.

---

**Related:** [Basis Functions](basis-functions.md) · [Convolution and Kernel](convolution-and-kernel.md) · [DSP](../../../src/subshader/dsp/DSP.md) · [MAP](../MAP.md)
