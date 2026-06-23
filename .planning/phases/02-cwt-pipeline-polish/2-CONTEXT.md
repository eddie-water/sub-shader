# Phase 2: CWT Pipeline Polish - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

CWT output looks visually correct across all frequency bands and the fix is covered by a test. Requirements: PIPE-01 (brightness bias), QUAL-02 (incremental test suite).

</domain>

<decisions>
## Implementation Decisions

### Normalization fix
- **D-01:** Normalize wavelet kernels to unit area in `WaveletKernel.__init__` — the kernel itself produces balanced output, no post-hoc correction needed
- **D-02:** The normalization code must be explicitly commented explaining the energy-bias reasoning (wider wavelets collect more energy because longer time support integrates more signal) — this scaffolds a future DSP README section on normalization design intuition
- **D-03:** Current `normalize_by_scale` multiplying by `sqrt(freq)` is insufficient — CWT output amplitude scales as `1/f` (proportional to Gaussian width = `num_fwhm_cycles/f`), so `sqrt(f)` only half-corrects. Kernel unit-area normalization addresses this at the source
- **D-04:** `cwt_out_type` config field is dead code (defined but never used) — clean up during this phase

### PyWavelet handling
- **D-05:** Use PyWavelet as-is with any built-in normalization options it provides — do not hand-write normalization for it
- **D-06:** If SubShader's normalized output looks better than PyWavelet's, that's a feature worth showing in benchmark figures

### Intensity tracker
- **D-07:** Intensity tracker position is correct — it already receives post-processed CWT output (after normalize, magnitude, discard, downsample)
- **D-08:** Tracker's job is frame-to-frame color consistency: `max(decayed_global_max, frame_99th_percentile)` so the shader's color range doesn't jump between frames
- **D-09:** Evaluate whether tracker parameters (decay_rate=0.001, percentile=99, warmup=10) need tuning after the kernel normalization fix, but don't over-engineer — current design is sound

### Test strategy
- **D-10:** Tests written at each CWT processing stage to guide the fix — not just end-to-end validation
- **D-11:** Test signals: synthetic multi-tone (equal-amplitude sinusoids at spread frequencies, e.g. 100 Hz, 1 kHz, 5 kHz, 10 kHz) + existing chirp signal from project assets
- **D-12:** Tests persist as the incremental test suite (QUAL-02) for future milestones
- **D-13:** "Visually correct" = equal-amplitude input tones produce comparable CWT magnitudes. DAW spectrogram and STFT are the correctness reference

### Claude's Discretion
- Which frequencies to use in multi-tone test signal
- Tolerance bounds for "comparable magnitude" assertion
- Whether `normalize_by_scale` method is removed entirely or kept as a no-op after kernel normalization
- Downsampling method assessment (current fractional-hop approach)
- Benchmark figure regeneration after fix

</decisions>

<specifics>
## Specific Ideas

- "SubShader should be better than the most popular library out there" — PyWavelet comparison in figures should demonstrate this
- The chirp signal comparison figure is the visual proof: STFT shows uniform brightness across the sweep, CWT should match
- Polyphonic MIDI composition (pure sine oscillator) is the other key reference — DAW spectrogram shows comparable intensity across tones

</specifics>

<canonical_refs>
## Canonical References

### Phase requirements
- `.planning/REQUIREMENTS.md` — PIPE-01 (brightness bias), QUAL-02 (incremental tests)
- `.planning/ROADMAP.md` §Phase 2 — Success criteria (3 items) that must be TRUE after this phase

### Prior phase context
- `.planning/phases/01-codebase-hardening/1-CONTEXT.md` — GPU fallback decisions, exception hierarchy cleanup (Phase 2 builds on this)

### Benchmark figures (visual references for "correct")
- `assets/images/benchmarks/chirp_signal_comparison.png` — Shows brightness decay at high frequencies in both CWT implementations
- `assets/images/benchmarks/polyphonic_signal_comparison.png` — Shows brightness decay vs DAW/STFT reference

</canonical_refs>

<code_context>
## Existing Code Insights

### Root Cause Analysis
- `src/subshader/dsp/wavelet_kernel.py:59` — Kernel is `sinusoid * gaussian` with no energy normalization. Lower-frequency kernels have longer time support (more samples), so more total energy
- `src/subshader/dsp/gaussian.py:30` — Gaussian FWHM = `num_fwhm_cycles / f`, so width ∝ 1/f. The integral of the Gaussian (which determines CWT output magnitude) scales as 1/f
- `src/subshader/dsp/wavelet.py:440` — `normalize_by_scale` multiplies by `sqrt(freq)`, but output scales as 1/f, so correction is half-strength. Result: persistent high-frequency brightness decay
- `src/subshader/dsp/wavelet.py:325` — PyWavelet's `normalize_by_scale` is a pass-through (returns unchanged). pywt.cwt() internally applies `sqrt(scale)` = `1/sqrt(freq)` which worsens the bias
- `src/subshader/config.py:106` — `cwt_out_type = "pow"` defined but never used anywhere in codebase (dead config)

### Pipeline Flow (where normalization happens)
1. `wavelet.py:150` — `class_specific_cwt()` → raw complex coefficients
2. `wavelet.py:154` — `normalize_by_scale()` → sqrt(freq) correction (insufficient)
3. `wavelet.py:157` — `compute_mag()` → absolute value
4. `wavelet.py:160` — `discard_unreliable_coefs()` → center slice to widest kernel's valid region
5. `wavelet.py:163` — `downsample()` → fractional-hop to target_width
6. `plotter.py:108` — `push_frame()` → intensity tracker updates global max
7. `fragment_shader.glsl:70` — `value / intensity_max` → [0,1] mapping with gamma

### Reusable Assets
- `tests/conftest.py` — Existing fixtures (`project_root`, `valid_audio_path`) for test infrastructure
- `research/benchmark.py` — Figure generation pipeline (`ReadmeFigures` class) for regenerating comparison figures
- `assets/audio/` — Existing chirp and polyphonic audio files for test signals

### Integration Points
- `WaveletKernel.__init__()` in `wavelet_kernel.py` — where kernel normalization must be added
- `AntsWavelet.normalize_by_scale()` in `wavelet.py` — may become no-op or be removed after kernel fix
- `research/benchmark.py` — regenerate figures after fix to show improvement

</code_context>

<deferred>
## Deferred Ideas

- DSP README section on normalization design intuition — Phase 5 (Documentation)
- Per-band intensity tracking (normalize each frequency row independently) — not needed if kernel normalization fixes the root cause; revisit if needed
- EGL headless rendering — v2 milestone

</deferred>

---

*Phase: 02-cwt-pipeline-polish*
*Context gathered: 2026-03-21*
