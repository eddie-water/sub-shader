# Phase 5: Documentation - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning
**Source:** discuss-phase conversation

<domain>
## Phase Boundary

Each module has a README that explains what it does, why it does it that way, and how to use it — written in the user's voice. Claude scaffolds structure, chooses meaningful examples, helps with technical accuracy. User authors all final prose.

**Deliverables:**
- `README.md` — top-level landing page (exists, ~50% draft)
- `DSP.md` — wavelet foundations (sections 1-6), wavelet implementation, post-processing, future aspirations
- `AUDIO.md` — audio input, audio player, the overlap pattern
- `RENDERER.md` — plot buffer, shader pipeline

</domain>

<decisions>
## Implementation Decisions

### Scaffold Format
- Headers, subheaders, bullet-point placeholders with specific guidance
- Placeholder examples: "explain the connection to X here", "make an analogy here", "example that highlights non-stationary advantages in CWT"
- NOT draft prose — user writes final text
- Where ~50% already exists (foundations outline, top-level draft), scaffold fills gaps and marks what's done vs what's left
- Suggest candidate analogies user can accept/reject/rewrite

### Top-level README (README.md)
- Keep existing personal "technical showcase" voice and framing
- Keep all three signal comparisons (chirp, polyphonic, musical)
- Flag stream-of-consciousness notes as "rewrite needed" with intent and placement guidance
- numpy_vs_cupy_diff.png belongs in DSP section, not top-level
- Timing bar chart needs cleanup and belongs in STFT vs PyWavelet vs NumPy vs CuPy comparison section

### Comparison Grid Figure
- One comparison grid: columns = audio signals, rows = representations (STFT, PyWavelet CWT, SubShader CWT)
- Chirp: reduce to 10 seconds; consider bouncing chirp (from overlap diagnostic) instead of linear
- Polyphonic: keep at ~16 seconds as-is
- Musical: reduce to ~8 bars (some bass, some none)
- Minimal/no axis labels, squeeze subplots close together
- Add decorators and grid lines
- Workshop exact layout during implementation — not locked yet

### DSP README (DSP.md)
- Cover sections 1-6 of wavelet foundations outline (inner product → CWT implementation)
- Sections 7-10 (feature hierarchy, ML, applications) consolidated into single "Future" section preserving their info
- Depth: accurate, practical explanations — match the foundations notebook voice
- All figure generation via benchmark flags (not notebook cells) — use image placeholders in scaffold
- Streamline: pull notebook figure-generation code into benchmark.py so all images come from one place

### Module READMEs
- Standalone .md files (not notebooks)
- AUDIO.md: audio input, audio player, overlap pattern
- RENDERER.md: plot buffer, shader pipeline
- No submodule READMEs under src/ — these live at project root level

### Claude's Discretion
- Exact section ordering within each README
- Which existing research/docs/demo/readmes/ content to incorporate vs discard
- Specific placeholder wording for scaffolds
- How to structure the "Future" consolidated section in DSP.md

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Documentation Drafts
- `README.md` — current top-level README draft (~130 lines, 50% complete)
- `research/docs/demo/readmes/subshader_toplevel_readme_claude.md` — top-level README spec (<250 lines, landing page)
- `research/docs/demo/readmes/dsp_submodule_readme_claude.md` — DSP README workflow instructions
- `research/docs/demo/readmes/audio_submodule_readme_claude.md` — audio module outline
- `research/docs/demo/readmes/visualizer_submodule_readme_claude.md` — visualizer module outline
- `research/docs/demo/readmes/discussion_summary.md` — pedagogical decisions, voice guidelines, terminology ladder

### DSP Foundations
- `research/docs/demo/wavelet_foundations_outline.md` — 10-section outline (inner product → applications)
- `research/docs/demo/wavelet_foundations.ipynb` — notebook through section 2.4 with working figures

### Benchmark & Figures
- `research/benchmark.py` — figure generation and timing; all future figure generation goes here
- `assets/images/benchmarks/` — existing comparison figures (chirp, polyphonic, musical, numpy_vs_cupy_diff)
- `assets/images/diagnostics/overlap_redundancy_diagnostic.png` — bouncing chirp example

### Source Code (for accurate documentation)
- `src/subshader/audio/audio_input.py` — audio input pipeline
- `src/subshader/dsp/wavelet.py` — CWT implementation hierarchy
- `src/subshader/viz/` — rendering pipeline (plotter, shader, frame buffer)
- `src/subshader/config.py` — configuration dataclasses

</canonical_refs>

<specifics>
## Specific Ideas

- Bouncing chirp from overlap_redundancy_diagnostic.png could replace one of the three comparison signals (possibly musical) — more dynamic than linear chirp
- Paint mixing analogy for signal decomposition (already in notebook) — good template for analogy style
- Terminology ladder from discussion_summary.md: build vocabulary incrementally, don't front-load jargon
- Grid figure: 3 columns (signals) × 3-4 rows (representations) with cropped time windows for readability

</specifics>

<deferred>
## Deferred Ideas

- Wavelet foundations sections 7-10 (feature hierarchy, ML integration, applications) — consolidated into "Future" section, full treatment deferred
- Interactive notebook-based documentation — decided against for now, using static .md with image placeholders
- Module READMEs under src/ directories — using project-root .md files instead

</deferred>

---

*Phase: 05-documentation*
*Context gathered: 2026-03-23 via discuss-phase conversation*
