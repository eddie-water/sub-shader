# Cone of Influence

Wavelets have finite but nonzero width, so results near the edges of an analysis window are contaminated by the wavelet extending past the available signal — wider wavelets (lower frequencies) are affected over a larger region. The Cone of Influence marks which coefficients are trustworthy; SubShader discards the unreliable edge region rather than displaying it.

**How the keep-region is sized** (defaults, from [`cwt.py`](../../src/subshader/dsp/cwt.py))

- The widest wavelet — A0 at 27.5 Hz — spans **9,623 samples** (218 ms), so its edge contamination reaches deepest
- The keep width is rounded down to the nearest power of two: **8,192 center samples** of the 16,384-sample chunk survive; the outer 4,096 on each side are discarded
- With 50% overlap, that reliable center exactly equals the hop size — so consecutive frames tile the timeline with **no gaps and no double-counting**. The chunk/overlap defaults were chosen to make these numbers line up.

SubShader uses a uniform (rectangular) keep-region sized by the *widest* wavelet rather than a per-frequency cone — high frequencies could keep more, but a rectangular output is what the renderer needs.

## In SubShader

`CWT._create_reliable_slice()` computes the slice once at startup; `discard_unreliable_coefs()` applies it every frame, and `extract_hop_center()` then takes the trailing hop-width portion. Planned as its own subsection in [DSP §5.2.2](../../src/subshader/dsp/DSP.md#4-wavelet-transform-adaptive-resolution) (edge effects / COI).

---

**Related:** [Mother Wavelet and Scaling](mother-wavelet-and-scaling.md) · [Continuous Wavelet Transform](continuous-wavelet-transform.md) · [Audio Overlap and Hop Size](audio-overlap-and-hop-size.md) · [Real-Time Frame Budget](real-time-frame-budget.md) · [DSP](../../src/subshader/dsp/DSP.md) · [MAP](../MAP.md)
