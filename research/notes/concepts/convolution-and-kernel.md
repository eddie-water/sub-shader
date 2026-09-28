# Convolution and Kernel

Convolution is correlation (inner product) computed at every position — sliding a kernel across the signal to produce one coefficient per position rather than one for the whole window. The kernel is the pattern used in that sliding comparison; in the CWT the kernel is the wavelet itself. The Convolution Theorem lets this be computed as a multiply in the frequency domain (FFT the signal, FFT the kernel, multiply, inverse FFT) — `O(N log N)` instead of the `O(N²)` direct sliding sum, with identical results.

**Key numbers**

- Convolution length: `conv_n = input_n + time_support_n − 1` → up to **26,006 samples** for the widest (A0) kernel against a 16,384-sample chunk ([`wavelet_kernel.py`](../../../src/subshader/dsp/wavelet_kernel.py))
- Precomputed at startup: the frequency-domain kernel bank `kernel_f_bank`, 116 × 26,006 complex64 ≈ **24 MB** — so runtime cost per frame is one forward FFT, one broadcast multiply, and one inverse FFT
- Result: 116 full-resolution correlation tracks per chunk, one per analysis frequency

## In SubShader

`CpuCWT.transform()` / `GpuCWT.transform()` in [`cwt.py`](../../../src/subshader/dsp/cwt.py) implement exactly this: `fft(data) * kernel_f_bank`, then `ifft`, then trim back to `input_n`. Conceptual background: [DSP §3.5](../../../src/subshader/dsp/DSP.md#35-convolution-theorem) and §4.4 — the same trick the FFT uses to speed up Fourier analysis carries straight over to wavelets.

---

**Related:** [Inner Product and Correlation](inner-product-and-correlation.md) · [Morlet Wavelet](morlet-wavelet.md) · [GPU-Accelerated FFT Convolution](gpu-accelerated-fft-convolution.md) · [DSP](../../../src/subshader/dsp/DSP.md) · [MAP](../MAP.md)
