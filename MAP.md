# MAP — SubShader Knowledge Base

The topics this project explores, and the docs that go deep on them. Concept notes live in
`notes/concepts/` and link to each other and to the docs/source that implement them. All links
are GitHub-safe relative markdown links (no wikilinks), so the web renders identically on
GitHub and in Obsidian's graph view.

## Concepts

**Signal Theory**
- [Inner Product and Correlation](notes/concepts/inner-product-and-correlation.md) — similarity measurement via multiply-and-sum; the operation underlying every transform.
- [Basis Functions](notes/concepts/basis-functions.md) — reference pattern sets used for full, non-redundant signal decomposition.
- [Sampling and Nyquist Limit](notes/concepts/sampling-and-nyquist-limit.md) — sample rate vs. detectable frequency ceiling; window duration vs. frequency resolution.
- [Stationarity Assumption](notes/concepts/stationarity-assumption.md) — why plain Fourier analysis loses *when* a frequency occurred.
- [Short-Time Fourier Transform](notes/concepts/short-time-fourier-transform.md) — windowed Fourier analysis to recover time localization.
- [Time-Frequency Resolution Tradeoff](notes/concepts/time-frequency-resolution-tradeoff.md) — the fixed-window tradeoff that motivates wavelet analysis.

**Wavelet Analysis**
- [Continuous Wavelet Transform (CWT)](notes/concepts/continuous-wavelet-transform.md) — adaptive-resolution time-frequency decomposition via scaled wavelets.
- [Mother Wavelet and Scaling](notes/concepts/mother-wavelet-and-scaling.md) — the prototype wavelet shape and its stretched/compressed daughters.
- [Morlet Wavelet](notes/concepts/morlet-wavelet.md) — a Gaussian-enveloped sinusoid; SubShader's specific wavelet shape.
- [Convolution and Kernel](notes/concepts/convolution-and-kernel.md) — sliding correlation, and the frequency-domain shortcut via FFT.
- [Chromatic Frequency Scale](notes/concepts/chromatic-frequency-scale.md) — exponentially-spaced analysis frequencies matching musical semitones.
- [Cone of Influence](notes/concepts/cone-of-influence.md) — the edge region where wavelet coefficients are unreliable.
- [Wavelet Magnitude Normalization](notes/concepts/wavelet-magnitude-normalization.md) — correcting for the 1/f energy bias across wavelet kernels.

**GPU & Real-Time**
- [GPU-Accelerated FFT Convolution](notes/concepts/gpu-accelerated-fft-convolution.md) — offloading the wavelet bank convolution to CuPy.
- [Real-Time Frame Budget](notes/concepts/real-time-frame-budget.md) — the per-chunk deadline the whole pipeline must beat.
- [Circular Frame Buffer](notes/concepts/circular-frame-buffer.md) — allocation-free scrolling history of CWT frames.
- [Audio Overlap and Hop Size](notes/concepts/audio-overlap-and-hop-size.md) — overlapping analysis windows and the non-redundant hop between them.

**Visualization**
- [Intensity Normalization](notes/concepts/intensity-normalization.md) — fixed percentile reference for consistent colormap brightness across frames.
- [Colormap and Gamma Correction](notes/concepts/colormap-and-gamma-correction.md) — per-pixel GPU color mapping with a perceptual brightness curve.

## Deep-Dive Docs

- [README](README.md) — Public-facing overview: what SubShader is, how it works, how to run it.
- [AUDIO](src/subshader/audio/AUDIO.md) — Audio input/playback module: chunk delivery and render-loop clock.
- [DSP](src/subshader/dsp/DSP.md) — Continuous Wavelet Transform module: motivations, math, GPU implementation.
- [RENDERER](src/subshader/renderer/RENDERER.md) — Circular buffer + shader rendering: CWT frames to screen pixels.
- [TIMING](assets/timing/TIMING.md) — Measured per-stage pipeline timing results and real-time budget analysis.
- [Audio Assets README](assets/audio/README.md) — Test audio files used for accuracy and performance validation.

## Backlog

- [Demo & Figure Ideas](notes/demo-ideas.md) — Ranked list of demos and figures worth making, easiest first.
