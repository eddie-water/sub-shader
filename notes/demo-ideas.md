# Demo & Figure Ideas

A ranked backlog of demos, showcase pieces, and figures worth making — easiest first. Five concept
cards already shipped via `research/dsplot/figures/concept_cards.py` (Morlet construction, wavelet
scaling, chromatic scale, colormap+gamma, overlap/hop); these are what's left on the table.
Sources: the planned-figure tables in [DSP.md](../src/subshader/dsp/DSP.md), the `[PLACEHOLDER]`
markers in [RENDERER.md](../src/subshader/renderer/RENDERER.md), and gaps noticed while building the
concept notes.

## Quick wins (an hour or two each, dsplot single-panel)

- **Two close sinusoids diverging** (DSP fig 3.1.a) — 10 Hz vs 11 Hz overlaid: identical over a
  short window, clearly different over a long one. The cleanest possible "duration = frequency
  resolution" visual. Serves [Sampling and Nyquist Limit](concepts/sampling-and-nyquist-limit.md).
- **Only the matching bins light up** (DSP fig 3.1.b) — a 5 Hz + faint 10 Hz signal dotted against
  every basis frequency; bar chart of results. The orthogonality payoff in one image. Serves
  [Basis Functions](concepts/basis-functions.md).
- **Kernel energy bias, before/after** — L1 norms of three raw Morlet kernels (55 Hz / 880 Hz /
  10 kHz) next to their normalized versions. Makes the ~100× bias visible. Serves
  [Wavelet Magnitude Normalization](concepts/wavelet-magnitude-normalization.md).

## Medium (half a day each)

- **Two signals, same spectrum** (DSP fig 3.2.a) — four tones simultaneous vs. one-per-quarter,
  with their identical DFT magnitude spectra side by side. The stationarity argument in one shot.
  Serves [Stationarity Assumption](concepts/stationarity-assumption.md).
- **STFT window sweep** (DSP fig 3.3.a) — the same chirp at 23 ms / 100 ms / 300 ms windows,
  showing the tradeoff slide from time-sharp to frequency-sharp. Prototypes already exist in
  `assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/archive/stft_window_sweep_*.png`.
  Serves [Time-Frequency Resolution Tradeoff](concepts/time-frequency-resolution-tradeoff.md).
- **STFT tiling vs CWT tiling** (DSP fig 4.4) — the classic rectangles diagram: uniform grid vs.
  constant-Q tiles. Serves the tradeoff and [CWT](concepts/continuous-wavelet-transform.md) notes.
- **Cone of influence mask** (DSP fig 4.5) — a CWT frame with the discarded edge regions shaded
  and the 8,192-sample keep window annotated. Serves
  [Cone of Influence](concepts/cone-of-influence.md).
- **Circular buffer ring diagram** (RENDERER placeholder) — ring of 32 slots with the write
  pointer, unrolling into the flattened texture. Serves
  [Circular Frame Buffer](concepts/circular-frame-buffer.md).

## Ambitious but worth it

- **Sliding-wavelet animation** (DSP fig 4.3, GIF) — a Morlet sliding across the chirp with the
  correlation trace drawing in below; the moment convolution "clicks." dsplot has a DynamicPanel
  layer for exactly this. Serves [Convolution and Kernel](concepts/convolution-and-kernel.md).
- **Screen-capture showcase reel** — SubShader running live on a real track next to a DAW
  spectrogram, 30–60 s. The single most convincing demo artifact for the README; still-image
  precedents exist in `assets/images/claude/diagnostics/` (SubShader vs FL Studio comparisons).
- **Intensity normalization before/after** (RENDERER placeholder) — the same frame rendered at
  three loudness levels with and without the fixed reference. Serves
  [Intensity Normalization](concepts/intensity-normalization.md).
- **3D spectrogram build-up GIF** (DSP "goal visual") — the CWT surface of the chirp assembling
  column by column in 3D. dsplot's `StaticPanel3D`/`DynamicPanel3D` exist; this is the flagship
  version of the fig-1 story.

---

**Related:** [MAP](../MAP.md) · [DSP](../src/subshader/dsp/DSP.md) · [RENDERER](../src/subshader/renderer/RENDERER.md)
