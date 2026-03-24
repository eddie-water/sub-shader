# Phase 5: Documentation - Research (Updated)

**Researched:** 2026-03-23 (updated 2026-03-23 post-execution)
**Domain:** Technical documentation scaffolding — README structure, pedagogical writing, benchmark figure pipeline
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scaffold Format**
- Headers, subheaders, bullet-point placeholders with specific guidance
- Placeholder examples: "explain the connection to X here", "make an analogy here"
- NOT draft prose — user writes final text
- Where ~50% already exists (foundations outline, top-level draft), scaffold fills gaps and marks what's done vs what's left
- Suggest candidate analogies user can accept/reject/rewrite

**Top-level README (README.md)**
- Keep existing personal "technical showcase" voice and framing
- Keep all three signal comparisons (chirp, polyphonic, musical)
- Flag stream-of-consciousness notes as "rewrite needed" with intent and placement guidance
- numpy_vs_cupy_diff.png belongs in DSP section, not top-level
- Timing bar chart needs cleanup and belongs in STFT vs PyWavelet vs NumPy vs CuPy comparison section

**Comparison Grid Figure**
- One comparison grid: columns = audio signals, rows = representations (STFT, PyWavelet CWT, SubShader CWT)
- Chirp: reduce to 10 seconds; consider bouncing chirp (from overlap diagnostic) instead of linear
- Polyphonic: keep at ~16 seconds as-is
- Musical: reduce to ~8 bars (some bass, some none)
- Minimal/no axis labels, squeeze subplots close together
- Add decorators and grid lines
- Workshop exact layout during implementation — not locked yet

**DSP README (DSP.md)**
- Cover sections 1-6 of wavelet foundations outline (inner product → CWT implementation)
- Sections 7-10 (feature hierarchy, ML, applications) consolidated into single "Future" section
- Depth: accurate, practical explanations — match the foundations notebook voice
- All figure generation via benchmark flags (not notebook cells) — use image placeholders in scaffold
- Streamline: pull notebook figure-generation code into benchmark.py so all images come from one place

**Module READMEs**
- Standalone .md files (not notebooks) at project root level
- AUDIO.md: audio input, audio player, overlap pattern
- RENDERER.md: plot buffer, shader pipeline
- No submodule READMEs under src/

### Claude's Discretion
- Exact section ordering within each README
- Which existing research/docs/demo/readmes/ content to incorporate vs discard
- Specific placeholder wording for scaffolds
- How to structure the "Future" consolidated section in DSP.md

### Deferred Ideas (OUT OF SCOPE)
- Wavelet foundations sections 7-10 full treatment — consolidated into "Future" section only
- Interactive notebook-based documentation
- Module READMEs under src/ directories
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOCS-01 | Top-level README — project overview, benchmark figures, visual comparisons, install/usage instructions | README.md scaffold complete; comparison grid PLACEHOLDER tag in place; install section flagged for Phase 4 output |
| DOCS-02 | DSP module README — pedagogical explanation of CWT pipeline, wavelet choices, normalization, with visuals | DSP.md scaffold complete with 68 WRITE placeholders, 8 image placeholders, 6 candidate analogies, all code verified |
| DOCS-03 | Rendering module README — OpenGL/shader pipeline, frame buffer, intensity mapping | RENDERER.md scaffold complete with 9 sections, 22 WRITE placeholders, all code verified |
| DOCS-04 | Audio module README — audio capture, chunking, overlap strategy | AUDIO.md scaffold complete with 6 sections, 12 WRITE placeholders, all code verified |
| DOCS-05 | Meaningful examples — no filler, no superfluous content | comparison_grid.png generated; all code examples extracted from actual source |
| DOCS-06 | Documentation scaffolded by Claude, authored by user in their own voice | All four scaffold documents complete — zero Claude-written prose paragraphs |
</phase_requirements>

---

## Summary

**Plans 01, 02, and 03 are complete as of 2026-03-23.** All four scaffold documents exist at project root: README.md (updated), DSP.md, AUDIO.md, RENDERER.md. The comparison grid figure (3x3, 3600x2000 px, 200 DPI) has been generated at `assets/images/benchmarks/comparison_grid.png`.

This RESEARCH.md update serves as the accurate post-execution state record for any new plans in this phase. The remaining work is split into two categories:

1. **Figure generation for DSP.md** — DSP.md contains 8 image `[PLACEHOLDER:]` markers. Two of these reference figures that require a new `--foundations-figures` flag in benchmark.py (inner product / vector geometry figures currently only in the notebook). The `numpy_vs_cupy_diff.png` already exists and is already embedded in DSP.md using live image syntax. The STFT windowing, wavelet scaling, and Fourier basis figures also need generating.

2. **User prose authoring** — the scaffolds are complete; the user writes all final prose. This is out of scope for Claude's execution tasks. The scaffold markers guide this work.

**Primary recommendation:** If a new plan is needed, it should focus on migrating notebook figure-generation code into benchmark.py via a `--foundations-figures` flag, and then generating the DSP.md figure set into `assets/images/dsp/`.

---

## Current Execution State

### What Plans 01-03 Delivered

| Plan | Deliverable | Status | Key Files |
|------|-------------|--------|-----------|
| 05-01 | Comparison grid figure + benchmark.py flag | Complete | `assets/images/benchmarks/comparison_grid.png`, `research/benchmark.py` |
| 05-02 | DSP.md scaffold (sections 1-7, verified code) | Complete | `DSP.md` |
| 05-03 | README.md scaffold + AUDIO.md + RENDERER.md | Complete | `README.md`, `AUDIO.md`, `RENDERER.md` |

### What Each Scaffold Document Contains Now

**README.md** (updated — was ~50% draft)
- 8 `[REWRITE:]` tags on stream-of-consciousness passages (all with intent + placement guidance)
- 10 `[WRITE:]` tags on gaps (install section, benchmark timing breakdown)
- 2 `[PLACEHOLDER:]` tags (comparison grid figure, demo video clip)
- `[MOVED:]` marker for `numpy_vs_cupy_diff.png` (now in DSP.md Section 6.4)
- Cross-links fixed: all three point to `AUDIO.md`, `DSP.md`, `RENDERER.md`
- Python requirement updated from `3.8+` to `3.9+`

**DSP.md** (new)
- 7 sections (Motivation through Future) matching wavelet_foundations_outline.md
- 68 `[WRITE:]` placeholders
- 8 `[PLACEHOLDER:]` image markers
- 6 `candidate analogy:` labeled suggestions
- CWT pipeline code block verified against wavelet.py (all 6 method calls)
- WaveletConfig parameter table with actual defaults (7 fields verified against config.py)
- `numpy_vs_cupy_diff.png` embedded with live image syntax (file exists)
- Appendix: Concept Ladder table from foundations outline

**AUDIO.md** (new)
- 6 sections: Role in Pipeline, The Overlap Strategy, AudioInput, AudioPlayer, Configuration, Usage Example
- 12 `[WRITE:]` placeholders
- `hop_size = int(chunk_size * (1.0 - overlap_factor))` formula with concrete example
- Code examples extracted from `audio_input.py`, `audio_player.py`, `__main__.py`
- AudioConfig fields table with actual defaults (3 fields)
- All design decisions documented: float32 storage, blocksize=0, threading.Lock, seamless looping

**RENDERER.md** (new)
- 9 sections: Role in Pipeline, Why Shaders, Circular Frame Buffer, Intensity Normalization, Init CPU-GPU Transfers, Runtime Render Loop, Shader Pipeline, Configuration, Diagram
- 22 `[WRITE:]` placeholders
- `frame_order` code extracted from `plotter.py CircularFrameBuffer.push_frame`
- IntensityTracker decay code extracted from `plot_normalizer.py` (3 lines)
- `flattened_buffer` pre-allocation documented (5 mentions)
- `intensity_max, 1e-8` floor documented (2 locations)
- VisualizationConfig and ColorNormalizationConfig fields tables with actual defaults

---

## Assets State

### Benchmark Figures (all in `assets/images/benchmarks/`)

| File | Exists | Used In | Notes |
|------|--------|---------|-------|
| `comparison_grid.png` | YES | README.md placeholder | 3x3, 3600x2000 px, 200 DPI — user has not yet reviewed final layout |
| `chirp_signal_comparison.png` | YES | Superseded by grid | Individual signal comparisons; no longer referenced in README |
| `polyphonic_signal_comparison.png` | YES | Superseded by grid | Same |
| `musical_signal_comparison.png` | YES | Superseded by grid | Same |
| `numpy_vs_cupy_diff.png` | YES | DSP.md Section 6.4 | Live image reference (not a placeholder) |
| `timing_bar_chart.png` | YES | README.md Benchmark section | Embedded directly (live path, not placeholder) |
| `stubs/chirp_signal_comparison_STUB_PYWT.png` | YES | Stubs only | Stub for pywt row — real renders from `--comparison-grid` |
| `stubs/polyphonic_signal_comparison_STUB_PYWT.png` | YES | Stubs only | Same |
| `stubs/musical_signal_comparison_STUB_PYWT.png` | YES | Stubs only | Same |

### Diagnostic Figures

| File | Exists | Used In | Notes |
|------|--------|---------|-------|
| `assets/images/diagnostics/overlap_redundancy_diagnostic.png` | YES | Not yet referenced | Bouncing chirp candidate for Column 1 of comparison grid — deferred to user review |

### DSP Foundations Figures (all MISSING — need generation)

DSP.md contains 8 `[PLACEHOLDER:]` markers. Two are already resolved (`numpy_vs_cupy_diff.png` is live). The remaining 6 require generating new figures:

| Placeholder | Where in DSP.md | Generator | Output Path | Status |
|-------------|-----------------|-----------|-------------|--------|
| 2D vector dot product geometric interpretation | Section 2.3 | benchmark.py `--foundations-figures` (TBD) | `assets/images/dsp/dot_product_geometry.png` | MISSING |
| Parallel/perpendicular vector pairs | Section 2.4.1 | Same | `assets/images/dsp/vector_similarity.png` | MISSING |
| Basis decomposition 2D vector | Section 2.4.3 | Same | `assets/images/dsp/basis_decomposition.png` | MISSING |
| Sign accumulation color-coded | Section 2.5 | Same | `assets/images/dsp/sign_accumulation.png` | MISSING |
| Fourier basis functions | Section 3.3 | Same | `assets/images/dsp/fourier_basis.png` | MISSING |
| STFT windowing illustration | Section 4.1 | Same | `assets/images/dsp/stft_windowing.png` | MISSING |
| Wavelet scaling at different frequencies | Section 5.4 | Same | `assets/images/dsp/wavelet_scaling.png` | MISSING |
| numpy_vs_cupy_diff.png | Section 6.4 | EXISTS — already in benchmark.py `--figures` | `assets/images/benchmarks/numpy_vs_cupy_diff.png` | DONE |

Source notebook for figures 1-5: `research/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` (through Section 2.4)
Figures 6-7 (STFT windowing, wavelet scaling) require new implementation in benchmark.py.

---

## What Remains for Phase 5 Completion

### Remaining Engineering Work (potential Plan 04)

**DSP.md Figure Generation** — blocked by missing `--foundations-figures` flag in benchmark.py:

1. Audit `wavelet_foundations.ipynb` for existing figure-generation cells (Sections 2.3-2.4 have working matplotlib figures)
2. Add `--foundations-figures` flag to `benchmark.py` that generates the 6 missing DSP.md figures
3. Output to `assets/images/dsp/` (new directory)
4. Update DSP.md `[PLACEHOLDER:]` markers to live image paths after generation

This is a legitimate Plan 04 scope. It is self-contained, has clear verification (6 PNG files exist), and unblocks the user from seeing their DSP.md visually complete.

### User Authoring Work (not Claude's task)

The scaffold format with `[WRITE:]`, `[REWRITE:]`, and `[PLACEHOLDER:]` markers is the contract: the user fills in all prose. Per DOCS-06 and CONTEXT.md, Claude does not write draft paragraphs. The following are the author's surface areas:

| Document | Marker Count | Biggest Sections |
|----------|-------------|-----------------|
| README.md | 8 REWRITE + 10 WRITE | Performance section prose, Future Improvements, Install |
| DSP.md | 68 WRITE | All of Sections 1-7 prose content |
| AUDIO.md | 12 WRITE | All section prose |
| RENDERER.md | 22 WRITE | All section prose |

---

## Architecture Patterns

### Scaffold Format (established and in use)

All four documents follow the same marker system:
- `[DONE: keep this text]` — existing content ready as-is
- `[REWRITE: intent="..." placement="..."]` — rough draft passage needing authoring
- `[PLACEHOLDER: figure — "description"]` — image not yet generated
- `[WRITE: "specific topic"]` — blank section needing prose
- `candidate analogy:` — one-liner suggestion, labeled, never embedded in prose

This format is consistent across all four documents. Any new plan must not introduce a different marker style.

### Benchmark Figure Generation Pattern

All figure generation lives in `research/benchmark.py` via argparse flags. The established pattern:
1. Add `--flag-name` to argparse section
2. Write a standalone `def generate_flag_name()` function
3. Append to run_modes dispatch in `if __name__ == "__main__"` block
4. Save output to `assets/images/[subdirectory]/filename.png`

The `generate_comparison_grid()` function established in Plan 05-01 is the canonical model. Any `--foundations-figures` implementation must follow the same pattern.

### DSP Foundations Figure Style

Figures from `wavelet_foundations.ipynb` are matplotlib-based. The notebook Sections 2.3-2.4 already have working figure cells. Migration to benchmark.py means:
- Extract the matplotlib figure code from notebook cells
- Wrap in a function with consistent style parameters
- Save to `assets/images/dsp/` with `plt.savefig(path, dpi=150, bbox_inches='tight')`

The notebook is the authoritative source for these figures. Do not regenerate from scratch — port existing working code.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Comparison grid figure | Custom layout from scratch | Existing `generate_comparison_grid()` in benchmark.py | Already built in Plan 05-01 |
| Runnable usage examples | Illustrative pseudo-code | Extract from `__main__.py` | DOCS-05 requires accurate examples |
| DSP foundations figures | New matplotlib code | Port from `wavelet_foundations.ipynb` cells | Notebook already has working vetted code |
| Placeholder image references | Made-up paths | Reference paths that actually exist | Broken image links make README unusable |
| DSP terminology | Write from scratch | Copy from foundations outline Section 10 concept ladder (already in DSP.md Appendix) | Already written, format established |

---

## Common Pitfalls

### Pitfall 1: Writing Prose Instead of Scaffold
**What goes wrong:** Any plan task that outputs paragraph-length content violates DOCS-06 and USER CONSTRAINTS.
**Prevention:** Every content section is a `[WRITE:]` placeholder. Candidate analogies are one-liners. No exceptions.
**Warning signs:** Any scaffold section readable as final prose.

### Pitfall 2: Stale Image References in DSP.md
**What goes wrong:** DSP.md has 6 `[PLACEHOLDER:]` markers for figures not yet generated. Replacing them with paths that don't exist breaks the document.
**Prevention:** Only replace a `[PLACEHOLDER:]` with a live path after the file exists at that path. The generation task and the link-update task must be in the same plan or sequenced correctly.
**Warning signs:** `![...]` image syntax in DSP.md pointing to a path under `assets/images/dsp/` that doesn't exist.

### Pitfall 3: New Figures Not Following benchmark.py Pattern
**What goes wrong:** Figure generation code scattered across notebook cells, standalone scripts, or inline in a plan task — breaking the single-source-of-truth invariant.
**Prevention:** All figure generation MUST be in `research/benchmark.py` behind an argparse flag.
**Warning signs:** Any plan task that runs matplotlib inline without adding a flag to benchmark.py.

### Pitfall 4: Comparison Grid Layout Not Yet Locked
**What goes wrong:** Treating the comparison_grid.png as final before user review. CONTEXT.md explicitly says "workshop exact layout during implementation — not locked yet."
**Prevention:** Any plan touching the comparison grid should present it to the user as a checkpoint before embedding in README.md. The `--comparison-grid` flag is the iteration mechanism.
**Warning signs:** README.md Performance section replacing `[PLACEHOLDER:]` with a live path before user approval.

### Pitfall 5: Inaccurate Code Examples
**What goes wrong:** Method names or parameter names in scaffold documents drift from actual source.
**Prevention:** Any new code example must be cross-checked against the actual source file using the venv Python environment.
**Warning signs:** Comments like `# simplified` or `# illustrative` in any code block.

---

## Code Examples

All code examples in the scaffold documents have been verified against actual source. They are not repeated here — see the scaffold documents themselves.

For reference, the verification commands used in Plan 05-02 and 05-03:

```bash
# Verify Wavelet methods
source venv/bin/activate && python -c "
from subshader.dsp.wavelet import Wavelet
methods = ['cwt', 'class_specific_cwt', 'normalize_by_scale', 'compute_mag',
           'discard_unreliable_coefs', 'extract_hop_center', 'downsample']
for m in methods:
    assert hasattr(Wavelet, m), f'Missing method: {m}'
print('All verified')
"

# Verify AudioInput/AudioPlayer methods
source venv/bin/activate && python -c "
from subshader.audio.audio_input import AudioInput
from subshader.audio.audio_player import AudioPlayer
from subshader.viz.plotter import CircularFrameBuffer, ShaderPlot
from subshader.viz.plot_normalizer import IntensityTracker
print('All classes importable')
"

# Verify WaveletConfig fields
source venv/bin/activate && python -c "
from subshader.config import WaveletConfig
print([f for f in WaveletConfig.__dataclass_fields__.keys()])
# Expected: ['typical_sampling_freq', 'notes_per_octave', 'num_octaves',
#            'root_note_a0_hz', 'num_cycles', 'num_fwhm_cycles', 'target_width']
"
```

---

## Benchmark Figure Pipeline

### Current State of benchmark.py Flags

| Flag | Function | Output | Status |
|------|----------|--------|--------|
| `--figures` | `_generate_comparison_figure()` | 3 individual signal comparison PNGs | Working |
| `--figures-chirp` | Same, chirp only | `chirp_signal_comparison.png` | Working |
| `--figures-polyphonic` | Same, polyphonic only | `polyphonic_signal_comparison.png` | Working |
| `--figures-musical` | Same, musical only | `musical_signal_comparison.png` | Working |
| `--comparison-grid` | `generate_comparison_grid()` | `comparison_grid.png` (3x3, 200 DPI) | Working (Plan 05-01) |
| `--foundations-figures` | Not yet implemented | 6 DSP pedagogical figures | MISSING — Plan 04 scope |

### DSP Foundations Figure Source

The notebook `research/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` contains working matplotlib figure cells through Section 2.4. The Plan 04 implementer must:
1. Read the notebook to extract existing figure cells
2. Port them to a `generate_foundations_figures()` function in benchmark.py
3. Implement the remaining figures (STFT windowing, wavelet scaling) not yet in the notebook
4. Save to `assets/images/dsp/` at 150 DPI

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + structural file checks |
| Config file | none — documentation phase |
| Quick run command | `python -c "import subshader"` |
| Full suite command | `python -m pytest tests/ -x -q` |
| Estimated runtime | ~10 seconds |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| DOCS-01 | README.md exists with required sections | structural | `grep -c "##" README.md` | Manual visual review for voice quality |
| DOCS-02 | DSP.md exists and covers sections 1-6 | structural | `test -f DSP.md && grep -c "##" DSP.md` | Section completeness is human judgment |
| DOCS-03 | RENDERER.md exists and covers pipeline | structural | `test -f RENDERER.md` | Same |
| DOCS-04 | AUDIO.md exists and covers overlap | structural | `test -f AUDIO.md` | Same |
| DOCS-05 | Code examples are runnable | smoke | See verification commands above | All examples verified in Plans 02-03 |
| DOCS-06 | Scaffold, not prose | manual | n/a | Human review only — no Claude prose paragraphs |

### Sampling Rate
- **Per task commit:** Verify files exist and contain expected markers via grep
- **Per wave merge:** Run full pytest suite + verify image paths reference existing files
- **Phase gate:** All scaffold files exist, no broken image links, no `*_README.md` cross-links

### Wave 0 Gaps
None — no new test infrastructure is needed. The phase deliverables are .md scaffold files and generated images.

---

## Open Questions

1. **Comparison grid layout approval**
   - What we know: `comparison_grid.png` exists (3x3, 3600x2000 px, Plan 05-01 output). The plan had a checkpoint task that auto-advanced without explicit user review.
   - What is unclear: Whether the user has reviewed and approved the current layout, or whether layout iteration is still pending.
   - Recommendation: If a new plan is added, include a checkpoint task presenting `comparison_grid.png` for explicit user layout feedback before the README.md `[PLACEHOLDER:]` is replaced with a live path.

2. **Comparison grid PLACEHOLDER vs live path in README.md**
   - What we know: README.md currently has `[PLACEHOLDER: figure — "3x3 comparison grid..."]` — not a live image reference. The file exists at `assets/images/benchmarks/comparison_grid.png`.
   - What is unclear: The CONTEXT.md decision to workshop layout before locking means the placeholder should remain until user approves the layout.
   - Recommendation: The placeholder-to-live-path replacement should be a distinct task gated on user approval, not automated.

3. **Installation section in README.md**
   - What we know: Phase 4 (install experience, INST-01/02) is pending. README.md Installation section is currently `[WRITE: "Installation instructions — fill from Phase 4 output (INST-01/INST-02)"]`.
   - What is unclear: Whether Phase 4 is planned or deferred. REQUIREMENTS.md shows INST-01/02/03 as `[ ]` pending.
   - Recommendation: Leave installation placeholder as-is until Phase 4 completes.

4. **DSP.md image directory naming**
   - What we know: Existing benchmark figures live under `assets/images/benchmarks/`. DSP foundations figures are pedagogical, not benchmark performance figures.
   - Recommendation: Use `assets/images/dsp/` for foundations figures — keeps pedagogical visuals separate from benchmark comparison figures. This is consistent with the RESEARCH.md plan written before execution.

---

## Sources

### Primary (HIGH confidence)
- `.planning/phases/05-documentation/05-01-SUMMARY.md` — Plan 01 execution record
- `.planning/phases/05-documentation/05-02-SUMMARY.md` — Plan 02 execution record
- `.planning/phases/05-documentation/05-03-SUMMARY.md` — Plan 03 execution record
- `README.md` — current scaffold state (directly read)
- `DSP.md` — current scaffold state (directly read)
- `AUDIO.md` — current scaffold state (directly read)
- `RENDERER.md` — current scaffold state (directly read)
- `research/benchmark.py` — confirmed flags via grep (generate_comparison_grid exists, --foundations-figures does not exist)
- `assets/images/benchmarks/` — all files listed via ls (comparison_grid.png confirmed)
- `.planning/phases/05-documentation/05-CONTEXT.md` — all locked decisions and constraints

### Secondary (MEDIUM confidence)
- `research/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` — cited in SUMMARY.md as source for foundations figures; not directly read in this update pass
- `research/docs/demo/discussion_summary.md` — terminology and voice constraints; confirmed applied per SUMMARY.md records

---

## Metadata

**Confidence breakdown:**
- Execution state (what's done): HIGH — all three SUMMARY.md files read, files confirmed to exist
- Scaffold quality: HIGH — all four documents directly read; marker counts verified
- Figure pipeline gaps: HIGH — confirmed via grep that `--foundations-figures` does not exist in benchmark.py
- Open questions: MEDIUM — comparison grid approval status depends on user interaction not captured in research

**Research date:** 2026-03-23 (initial) / 2026-03-23 (updated post-execution)
**Valid until:** Stable for planning purposes — all dependencies are internal files, not external APIs
