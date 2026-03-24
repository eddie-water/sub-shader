# Phase 6: Finalize Example Audio and Comparison Figures for README - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning
**Source:** discuss-phase conversation

<domain>
## Phase Boundary

Curate final audio examples and generate the polished comparison grid figure for the top-level README. The README should have one impactful visual comparison (the grid) that highlights SubShader's advantages over common DSP methods. Detailed per-signal analysis lives in DSP.md, not here.

**Deliverables:**
- Bouncing chirp WAV file (synthesized programmatically, ideated with user)
- Finalized comparison grid PNG (3 signals × 3 representations)
- README.md updated with grid figure + scaffold captions for per-signal sections
- Timing bar chart moved to DSP.md and regenerated with updated styling

</domain>

<decisions>
## Implementation Decisions

### Audio Signal Selection
- **D-01:** Chirp column uses a NEW bouncing chirp signal — frequency contour that rises overall with periodic parabolic dips (like a ball bouncing upward). Logarithmic bounce pattern: ~20Hz → 200Hz → 2kHz → 20kHz (decade per bounce). Exact parameters confirmed visually by user using stub figure generator before final grid.
- **D-02:** Polyphonic column uses `polyphonic_audio_example.wav` (existing file)
- **D-03:** Musical column uses `beltran_sc_rip_8_bar.wav` (existing 8-bar excerpt)

### Figure Polish Level
- **D-04:** Colormap stays as current benchmark.py default (no change)
- **D-05:** Axis labels/titles stay as-is for now — deferred until audio examples are finalized and user can evaluate visually
- **D-06:** DPI not locked — generate at 150, 200, 250, 300 DPI for user comparison. User picks final DPI based on visual quality vs file size.

### README Visual Flow
- **D-07:** Top-level README is a landing page — one impactful comparison grid figure, not individual per-signal figures. Deep analysis lives in DSP.md.
- **D-08:** Comparison grid centered at ~80% width in README. Figure itself generated wide so it's detailed when opened in a new tab / zoomed.
- **D-09:** Replace REWRITE markers in per-signal sections (Chirp, Polyphonic, Musical) with scaffold captions. User rewrites later.
- **D-10:** Timing bar chart moved from README to DSP.md. Regenerated with updated data and styling to match comparison grid polish level.

### Scope Boundary
- **D-11:** Phase 5 plan 05-04 (DSP.md foundation figures) is NOT in Phase 6 scope — will be executed as Phase 5 work after Phase 6.
- **D-12:** Bouncing chirp audio generation IS in Phase 6 scope — ideated with user, not a pre-existing file.
- **D-13:** Decorator utility functions (grid lines, vertical markers) are NOT in Phase 6 scope — just the figures.

### Claude's Discretion
- Exact bouncing chirp synthesis implementation (waveform math, sample rate, duration)
- Scaffold caption wording for per-signal sections
- How to restructure the timing bar chart for DSP.md placement
- Subplot spacing and layout details for comparison grid

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Benchmark & Figures
- `research/benchmark.py` — figure generation pipeline; `--comparison-grid` flag already implemented; all figure generation goes through this file
- `assets/images/benchmarks/comparison_grid.png` — current comparison grid output (to be regenerated)
- `assets/images/benchmarks/timing_bar_chart.png` — timing chart to move to DSP.md and regenerate

### Audio Files
- `assets/audio/daw/polyphonic_audio_example.wav` — polyphonic column source
- `assets/audio/daw/beltran_sc_rip_8_bar.wav` — musical column source
- `fm_sine.py` — existing FM synthesis script in repo root, may inform bouncing chirp generation

### Documentation
- `README.md` — current top-level README with PLACEHOLDER/REWRITE markers
- `DSP.md` — destination for timing bar chart (if exists; may need creation reference from Phase 5 plans)

### Phase 5 Context
- `.planning/phases/05-documentation/05-CONTEXT.md` — prior decisions on comparison grid layout, figure generation strategy

### Reference Images
- `assets/images/claude/bouncing_chirp.png` — user's hand-drawn frequency contour sketch (reference for bouncing chirp shape)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `research/benchmark.py` — already has `--comparison-grid` flag generating 3×3 grid with per-row vmax normalization
- `research/utilities/dsp_helpers.py` — DSP helper functions used by benchmark
- `research/utilities/constants.py` — audio file path constants
- `fm_sine.py` — FM synthesis script, potential starting point for bouncing chirp generation

### Established Patterns
- All figure generation goes through `benchmark.py` flags — no standalone scripts
- Audio files stored in `assets/audio/daw/` (generated) or `assets/audio/songs/` (sourced)
- Benchmark images stored in `assets/images/benchmarks/`
- Stub images (PyWavelet-skipped) in `assets/images/benchmarks/stubs/`

### Integration Points
- `benchmark.py --comparison-grid` generates `comparison_grid.png`
- README.md references images via relative paths: `assets/images/benchmarks/`
- `research/utilities/constants.py` defines audio file path constants

</code_context>

<specifics>
## Specific Ideas

- Bouncing chirp frequency contour: ascending overall with periodic parabolic dips — like a ball bouncing upward with increasing height. Each bounce covers a frequency decade (20→200→2k→20k). Reference sketch: `assets/images/claude/bouncing_chirp.png`
- User will use stub figure generator (skip PyWavelet) to rapidly iterate on bouncing chirp parameters before committing to final version
- DPI comparison: generate grid at 4 DPI levels (150/200/250/300) so user can visually pick the sweet spot
- Comparison grid should be the single hero image in the top-level README Performance section

</specifics>

<deferred>
## Deferred Ideas

- Decorator utility functions (vertical lines, grid overlays for accuracy verification) — future phase or ad-hoc
- Axis label/title styling for comparison grid — deferred until audio examples are confirmed
- Phase 5 plan 05-04 (DSP.md foundation figures) — executed as Phase 5 work after Phase 6
- Cleaning up bouncing chirp generation into a permanent benchmark.py flag — later phase

</deferred>

---

*Phase: 06-finalize-example-audio-and-comparison-figures-for-readme*
*Context gathered: 2026-03-24 via discuss-phase conversation*
