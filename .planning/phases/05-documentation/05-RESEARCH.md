# Phase 5: Documentation - Research

**Researched:** 2026-03-23
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
| DOCS-01 | Top-level README — project overview, benchmark figures, visual comparisons, install/usage instructions | README.md exists at ~50% draft; scaffold must preserve voice, add performance section with comparison grid figure |
| DOCS-02 | DSP module README — pedagogical explanation of CWT pipeline, wavelet choices, normalization, with visuals | wavelet_foundations_outline.md sections 1-6 cover exactly this content; notebook through 2.4 already written |
| DOCS-03 | Rendering module README — OpenGL/shader pipeline, frame buffer, intensity mapping | plotter.py is fully readable; visualizer_submodule_readme_claude.md gives the topic list |
| DOCS-04 | Audio module README — audio capture, chunking, overlap strategy | audio_input.py + audio_player.py are fully readable; audio_submodule_readme_claude.md gives the topic list |
| DOCS-05 | Meaningful examples — no filler, no superfluous content | All code examples must be extracted from actual source; placeholder images must reference real assets |
| DOCS-06 | Documentation scaffolded by Claude, authored by user in their own voice | Scaffold = structure + placeholders + candidate analogies, NOT draft prose |
</phase_requirements>

---

## Summary

Phase 5 is a scaffolding phase, not a writing phase. The deliverable is four scaffold files (README.md, DSP.md, AUDIO.md, RENDERER.md) that give the user exactly enough structure to sit down and fill in prose — without Claude writing the prose for them. The existing content inventory is rich: a 10-section wavelet foundations outline (sections 1-6 already detailed), a working top-level README draft at ~50%, and detailed topic lists for the audio and renderer modules.

The primary technical work in this phase is the comparison grid figure — a 3x3 (or 3x4) matplotlib grid with columns = signals (chirp, polyphonic, musical) and rows = representations (STFT, PyWavelet CWT, SubShader CWT). This figure is the visual centerpiece of the README performance section, and the benchmark.py file is the designated home for all figure generation. Notebook figure-generation code should be migrated there.

The scaffold methodology is consistent across all four documents: section headings, sub-headings, bullet-point guidance ("explain X here", "analogy opportunity"), image placeholder markers, and a few candidate analogies the user can accept or rewrite. The only code in the scaffolds is accurate, runnable examples extracted from actual source files — no illustrative filler that silently fails.

**Primary recommendation:** Build scaffolds from canonical sources (foundations outline, source code, existing readme drafts), never invent content — then flag what is already written vs what needs authoring.

---

## Content Inventory

### What Exists (can be incorporated directly)

| Item | Location | Status | Notes |
|------|----------|--------|-------|
| Top-level README draft | `README.md` | ~50% done | Voice is right; several sections are stream-of-consciousness placeholders |
| Wavelet foundations outline | `research/docs/demo/readmes/wavelet/wavelet_foundations_outline.md` | Complete | Sections 1-6 are fully detailed; 7-10 thorough but deferred |
| Foundations notebook | `research/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` | Through 2.4 | Has working figures (basic vectors, projections); figure gen code to migrate to benchmark.py |
| DSP README spec | `research/docs/demo/readmes/wavelet/dsp_submodule_readme_claude.md` | Outline only | Confirms: use foundations outline, figure gen via benchmark flags |
| Audio README spec | `research/docs/demo/readmes/audio/audio_submodule_readme_claude.md` | Outline only | 4-row overlap viz described; style guidelines explicit |
| Renderer README spec | `research/docs/demo/readmes/plotter/visualizer_submodule_readme_claude.md` | Outline only | Topics listed; diagram placeholder noted |
| Discussion summary | `research/docs/demo/discussion_summary.md` | Done | Terminology ladder, voice guidelines, pedagogy decisions — critical for DSP scaffold |
| Chirp comparison | `assets/images/benchmarks/chirp_signal_comparison.png` | Exists | STFT + pywt + SubShader CWT, but no grid format yet |
| Polyphonic comparison | `assets/images/benchmarks/polyphonic_signal_comparison.png` | Exists | Same format |
| Musical comparison | `assets/images/benchmarks/musical_signal_comparison.png` | Exists | Same format |
| numpy_vs_cupy_diff | `assets/images/benchmarks/numpy_vs_cupy_diff.png` | Exists | Moves to DSP.md per decision |
| Bouncing chirp | `assets/images/diagnostics/overlap_redundancy_diagnostic.png` | Exists | Candidate replacement for linear chirp in grid |
| Timing bar chart | `assets/images/benchmarks/timing_bar_chart.png` | Exists | Goes in STFT vs pywt vs numpy vs cupy timing section |
| PyWt stub images | `assets/images/benchmarks/stubs/` | Stubs | chirp/polyphonic/musical _STUB_PYWT.png — needs real PyWavelet rows |

### What Does Not Exist (needs to be created during implementation)

| Item | Blocker | Owner |
|------|---------|-------|
| Comparison grid figure (3×3 or 3×4) | benchmark.py needs new grid-layout flag | Implementer |
| Inner product / vector figures in DSP.md | Already in notebook; need migration to benchmark.py | Implementer |
| `DSP.md` file | Does not exist yet | Implementer (scaffold) |
| `AUDIO.md` file | Does not exist yet | Implementer (scaffold) |
| `RENDERER.md` file | Does not exist yet | Implementer (scaffold) |
| PyWavelet rows in comparison images | Stubs exist; real renders needed | Implementer |

---

## Architecture Patterns

### Scaffold Structure (all four documents)

Each scaffold follows the same anatomy:

```
# [Title]

## Section Heading

[One sentence of what goes here]

- **Sub-point:** [Specific guidance, e.g. "Explain why overlap_factor=0.5 is the default"]
- **Analogy opportunity:** [Candidate phrasing user can accept/rewrite]
- [IMAGE PLACEHOLDER: description of what figure goes here, path when generated]

### Sub-section

[Guidance continues...]
```

Placeholders MUST be formatted so a human scanning the file instantly knows what is done vs what needs writing:
- `[DONE: keep this text]` — existing content that is ready
- `[REWRITE: intent="..." placement="..."]` — existing stream-of-consciousness that needs authoring
- `[PLACEHOLDER: figure — "description of what this shows"]` — image that needs generating
- `[WRITE: "specific topic to cover here"]` — blank section needing prose

### README.md Scaffold Plan

**Sections confirmed from existing draft and spec:**

1. **Project Summary** — keep existing paragraph; flag two sentences as REWRITE
2. **Design** — keep architecture diagram, links to AUDIO.md / DSP.md / RENDERER.md (update file names from current *_README.md to match decided names)
3. **Performance** — replace three separate images with single comparison grid; add brief STFT vs PyWt vs SubShader prose guidance per signal
4. **Benchmark** — timing bar chart + numpy_vs_cupy_diff.png moves to DSP.md; this section becomes SubShader-only timing
5. **Installation** — placeholder for Phase 4 output (install instructions come from INST-01/02)
6. **Future Improvements** — keep; flag for user authoring

**Stream-of-consciousness passages in current README.md that need REWRITE flags:**
- Lines 51-55: commented-out STFT/PyWavelet explanation (good intent, rough execution)
- Lines 76-104: musical signal section with "TODO", "shitty fft", "link source" — preserve intent, flag execution
- Lines 102-104: "Comparing the Fourier Transform to the CWT is like comparing apples to oranges..." — user voice starting to emerge; flag as "keep tone, rewrite paragraph"
- Lines 129-131: Future Improvements — stream-of-consciousness; flag and preserve

### DSP.md Scaffold Plan

**Source: wavelet_foundations_outline.md sections 1-6**

The outline is already at scaffold depth — headers, sub-headers, and bullet points exist. The task is:
1. Convert outline to README scaffold format (strip notebook-style directives)
2. Insert image placeholder markers at each `*[Figure/Visual/Example: ...]`
3. Map each placeholder to a benchmark.py flag
4. Add "Future" section collapsing sections 7-10 into a compact forward pointer
5. Keep terminology ladder from discussion_summary.md

**Sections to cover:**
- 1. Motivation
- 2. Foundations: Inner Product (2.1-2.7)
- 3. Fourier Transform
- 4. STFT
- 5. Wavelet Transform
- 6. Implementation Deep Dive (wavelet construction, post-processing pipeline, GPU acceleration)
- 7. Future (collapse 7-10)

**Key pedagogical constraints from discussion_summary.md:**
- Do not use "features" or "patterns" before section 7
- Use "properties" in section 2, "components" as bridge, "features" only when ML appears
- Concrete examples before abstractions in every section
- Candidate analogies: paint mixing (already in outline), "casting a shadow" (projection), "measuring with different rulers at different frequencies" (adaptive resolution)

### AUDIO.md Scaffold Plan

**Source: audio_input.py, audio_player.py, audio_submodule_readme_claude.md**

Sections (Claude's discretion on ordering):

1. **Role in the Pipeline** — AudioInput delivers overlapping chunks to DSP; AudioPlayer drives the render clock
2. **The Overlap Strategy** — why overlap_factor exists; edge discontinuity and aliasing explanation; 4-row visualization placeholder
3. **AudioInput** — how get_chunk() works; hop_size = chunk_size * (1 - overlap_factor); stereo-to-mono; file_pos tracking
4. **AudioPlayer** — audio-clock-driven sync design; callback thread; get_playback_sample() as timing reference; seamless loop
5. **Configuration** — AudioConfig dataclass parameters with real defaults from config.py
6. **Usage Example** — runnable code extracted from __main__.py pattern

**Key implementation details to expose in scaffold (from source code):**
- `hop_size = int(chunk_size * (1.0 - overlap_factor))` — the relationship that drives everything
- `file_pos += hop_size` (not chunk_size) — this is what makes overlap work
- AudioPlayer stores `_data` as float32 because PortAudio callback expects float32
- `blocksize=0` lets PortAudio choose optimal hardware buffer (not a magic number)
- `threading.Lock()` on `_current_frame` — single int read/write, low contention

### RENDERER.md Scaffold Plan

**Source: plotter.py (ShaderPlot, GLContext, Renderer, CircularFrameBuffer, IntensityTracker), viz/shaders/, plot_normalizer.py, visualizer_submodule_readme_claude.md**

Sections:

1. **Role in the Pipeline** — receives CWT frames; stores history; renders as texture
2. **Why Shaders** — Python plotting libraries (matplotlib, pyqtgraph) can't sustain real-time frame rates with the data volume; GPU renders the entire history as a texture in one draw call
3. **The Circular Frame Buffer** — how CircularFrameBuffer stores frames, maintains chronological order, pre-allocates flattened_buffer; frame_index and ordering logic
4. **Intensity Normalization** — IntensityTracker; percentile-based global_max; exponential decay; warmup_frames; why this instead of per-frame min/max
5. **Init: CPU-GPU Transfers** — what goes to GPU at init (quad geometry VBO, texture allocation); what goes every frame (texture data write only); design rationale for minimizing per-frame transfers
6. **Runtime: The Render Loop** — push_frame → update_texture → set_intensity_max → clear → render → display; the sequence and why it's ordered this way
7. **Shader Pipeline** — vertex shader (full-screen quad geometry), fragment shader (colormap + gamma correction); `intensity_max` uniform; `gamma` uniform
8. **Configuration** — VisualizationConfig parameters (num_frames, gamma, color_norm)
9. **Diagram placeholder** — user has diagram; placeholder goes here

**Key implementation details to expose:**
- `flattened_buffer` is pre-allocated, not rebuilt each frame — this is the performance-critical design
- `frame_order = [(frame_index + i) % num_frames for i in range(num_frames)]` — this is how chronological order is maintained in a circular buffer
- IntensityTracker uses `(1 - decay_rate) * global_max` — slow decay so max doesn't collapse on quiet passages
- `shader['intensity_max'] = max(intensity_max, 1e-8)` — floor prevents division-by-zero in shader

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Comparison grid figure | Custom multi-subplot layout from scratch | Extend existing benchmark.py pattern | Existing signals and rendering code already there; consistency with all other figures |
| Runnable usage examples | Illustrative pseudo-code | Extract from `__main__.py` and tests | DOCS-05 requires accurate, runnable examples |
| Placeholder image references | Made-up paths | Reference actual existing paths under `assets/images/` | Broken image links make README unusable |
| DSP terminology definitions | Write from scratch | Copy from foundations_outline.md section 10 terminology ladder | Already written, tested for consistency |

---

## Common Pitfalls

### Pitfall 1: Writing Prose Instead of Scaffold
**What goes wrong:** Claude writes draft prose, user reads it, dislikes the voice, rewrites from scratch. The scaffold becomes wasted work and the user loses the structure.
**Why it happens:** Default behavior is to produce complete-looking output.
**How to avoid:** Every paragraph-length gap in the scaffold must be a `[WRITE: ...]` placeholder, never a prose paragraph. Candidate analogies are one-liners labeled "candidate analogy:", not embedded in prose.
**Warning signs:** Any scaffold section that could be published as-is without user edits.

### Pitfall 2: Stale Image References
**What goes wrong:** Scaffold references image paths that don't exist yet (e.g., `comparison_grid.png`), breaking the README before images are generated.
**Why it happens:** Placeholder images are planned but not yet generated.
**How to avoid:** Use two-tier placeholder format: `[IMAGE PLACEHOLDER: description]` in scaffold text, with a separate task to generate and replace. Do not embed paths for images that don't yet exist.
**Warning signs:** Any README that links to a file not present in `assets/images/`.

### Pitfall 3: Inaccurate Code Examples
**What goes wrong:** Code examples are illustrative-only and contain wrong method names, wrong parameter names, or wrong invocation patterns. Users copy them and they fail silently.
**Why it happens:** Code written from memory or documentation, not from the actual source.
**How to avoid:** Every code example must be extracted from `src/subshader/` source or from a file that is actively run (e.g., `__main__.py`, `research/benchmark.py`). Cross-check parameter names against actual signatures.
**Warning signs:** Code with comments like "# simplified" or "# illustrative" or any TODO.

### Pitfall 4: Forgetting the README.md Link Updates
**What goes wrong:** The existing `README.md` links to `AUDIO_README.md`, `DSP_README.md`, `RENDERER_README.md` — the decided file names are `AUDIO.md`, `DSP.md`, `RENDERER.md`. If links are not updated, navigation breaks.
**Why it happens:** The existing draft predates the naming decision in CONTEXT.md.
**How to avoid:** README.md scaffold task must explicitly update all three cross-links.
**Warning signs:** Any occurrence of `*_README.md` in the final scaffold.

### Pitfall 5: Missing benchmark.py Figure Migration Task
**What goes wrong:** DSP.md scaffold references figures from the notebook but no task migrates the notebook's figure generation code to benchmark.py. Implementer has no way to generate the referenced images.
**Why it happens:** Easy to scaffold references without tracking the generation dependency.
**How to avoid:** Plan must include an explicit task: "Migrate notebook figure generation code from `wavelet_foundations.ipynb` to `benchmark.py` with a CLI flag." DSP.md image placeholders are not live until this task is complete.
**Warning signs:** Any DSP.md image placeholder that does not have a corresponding benchmark.py flag.

### Pitfall 6: Over-Scaffolding the Comparison Grid Before Layout Is Locked
**What goes wrong:** CONTEXT.md says "workshop exact layout during implementation — not locked yet." A plan that pre-commits to a specific grid dimensions, axis label decisions, or color scheme wastes implementation time when those decisions change.
**How to avoid:** Comparison grid task should be: implement a working draft, present to user, iterate. Do not scaffold the grid as if all layout decisions are final.

---

## Code Examples

Accurate patterns from actual source (for use in README scaffolds):

### AudioInput — Core chunk retrieval pattern
```python
# Source: src/subshader/audio/audio_input.py AudioInput.__init__ + get_chunk
audio = AudioInput(
    path="assets/audio/daw/a2a3_a4_minor_scale.wav",
    chunk_size=4096,
    overlap_factor=0.5
)
# hop_size = 4096 * (1 - 0.5) = 2048 samples
# get_chunk() returns 4096 samples, advances file_pos by 2048

chunk = audio.get_chunk()  # Returns None at end of file
```

### AudioPlayer — timing reference pattern
```python
# Source: src/subshader/audio/audio_player.py
player = AudioPlayer(audio_data=audio.get_entire_audio(), sample_rate=audio.get_sample_rate())
player.start()

# In render loop — audio clock drives the position
current_sample = player.get_playback_sample()
```

### ShaderPlot — single render frame
```python
# Source: src/subshader/viz/plotter.py ShaderPlot.update_plot
plotter.update_plot(cwt_frame)  # push_frame → update_texture → render
```

### CircularFrameBuffer — chronological ordering
```python
# Source: src/subshader/viz/plotter.py CircularFrameBuffer.push_frame
# frame_index points to NEXT write slot; ordering computed from there
frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]
```

### IntensityTracker — decay with floor
```python
# Source: src/subshader/viz/plot_normalizer.py IntensityTracker.update
self.global_max = (1.0 - self.decay_rate) * self.global_max
self.global_max = max(self.global_max, self.floor_value)
self.global_max = max(self.global_max, frame_max)
```

### Wavelet CWT pipeline steps
```python
# Source: src/subshader/dsp/wavelet.py Wavelet.cwt
cwt_coefs = self.class_specific_cwt(input_data)     # GPU or NumPy implementation
cwt_coefs = self.normalize_by_scale(cwt_coefs)       # L1 kernel norm at construction (no-op here)
mag_coefs = self.compute_mag(cwt_coefs)              # |complex| → magnitude
reliable_coefs = self.discard_unreliable_coefs(mag_coefs)  # trim cone of influence
hop_center_coefs = self.extract_hop_center(reliable_coefs) # trim overlapping wings
downsampled_coefs = self.downsample(hop_center_coefs, self.output_n)  # to target_width
```

---

## Benchmark Figure Pipeline

### Existing Figures
| File | What It Shows | Used In |
|------|--------------|---------|
| `chirp_signal_comparison.png` | Chirp: STFT + PyWt + SubShader CWT, side by side | To be replaced by comparison grid |
| `polyphonic_signal_comparison.png` | Polyphonic: same | To be replaced |
| `musical_signal_comparison.png` | Musical: same | To be replaced |
| `numpy_vs_cupy_diff.png` | NumPy vs CuPy coefficient diff | Moves from README.md to DSP.md |
| `timing_bar_chart.png` | Timing bars (needs cleanup) | README.md Benchmark section |
| `overlap_redundancy_diagnostic.png` | Bouncing chirp sweep | Candidate for grid Column 1 |
| `stubs/chirp_signal_comparison_STUB_PYWT.png` | PyWt row stub | Stubs — real renders needed |

### New Figure Needed: Comparison Grid
- Columns: chirp (10s or bouncing), polyphonic (~16s), musical (~8 bars)
- Rows: STFT, PyWavelet CWT, SubShader CWT
- Style: minimal axis labels, squeezed subplots, grid lines, decorators
- Location: `assets/images/benchmarks/comparison_grid.png`
- Generator: `research/benchmark.py` — new flag `--comparison-grid`
- Layout is NOT locked — workshop during implementation task

### New Figures Needed: DSP.md Inner Product / Vector Illustrations
- Source: `research/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` (Sections 2.3-2.4)
- Migration target: `research/benchmark.py` — new flag `--foundations-figures`
- Output path: `assets/images/dsp/` (new directory)

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none detected — tests are inline per-module |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| DOCS-01 | README.md exists and contains required sections | manual | n/a | Visual review — no automated check for prose quality |
| DOCS-02 | DSP.md exists and covers sections 1-6 | manual | n/a | Section completeness is a human judgment |
| DOCS-03 | RENDERER.md exists and covers shader/buffer pipeline | manual | n/a | Same |
| DOCS-04 | AUDIO.md exists and covers overlap strategy | manual | n/a | Same |
| DOCS-05 | All code examples are runnable | smoke | `python -c "$(cat code_example)"` or extract and run | Implementer should test each snippet before placing in scaffold |
| DOCS-06 | Scaffold, not prose | manual | n/a | Verified by human review that no paragraph-length prose is present |

Documentation phase is manual-only for quality gates. The only automatable check is DOCS-05 (code correctness), which is a pre-condition for placing examples in scaffolds, not a post-condition test.

### Wave 0 Gaps
None — no new test infrastructure is needed for this phase. The phase deliverables are .md scaffold files, not code.

---

## Open Questions

1. **Comparison grid: bouncing chirp vs linear chirp as Column 1**
   - What we know: `overlap_redundancy_diagnostic.png` exists (bouncing chirp); linear chirp is the existing standard
   - What's unclear: Which tells the clearer story for non-stationary signal advantage — the bouncing chirp is more dynamic but the linear sweep is simpler to explain
   - Recommendation: Present both options to user during comparison grid implementation; let them choose after seeing the renders

2. **PyWavelet comparison rows: stub images vs real renders**
   - What we know: Stubs exist under `assets/images/benchmarks/stubs/`; real PyWavelet renders require benchmark.py to run PyWavelet CWT
   - What's unclear: Whether the existing benchmark.py already generates these or whether the stubs are pure placeholders
   - Recommendation: Read `research/benchmark.py` at implementation time to confirm what flags exist; if PyWt renders are missing, that is a prerequisite for the comparison grid task

3. **DSP.md figure directory**
   - What we know: All existing figures live under `assets/images/benchmarks/` or `assets/images/diagnostics/`
   - What's unclear: Should inner-product/vector figures from the DSP foundations live under `assets/images/dsp/` or `assets/images/benchmarks/`
   - Recommendation: Use `assets/images/dsp/` — keeps DSP pedagogical figures separate from benchmark performance figures

4. **Installation section in README.md**
   - What we know: Phase 4 (install experience) is pending; INST-01/02 requirements are not yet complete
   - What's unclear: Whether Phase 4 will be done before or after Phase 5
   - Recommendation: Scaffold the Installation section with a placeholder; flag it as "fill from Phase 4 output"

---

## Sources

### Primary (HIGH confidence)
- `research/docs/demo/readmes/wavelet/wavelet_foundations_outline.md` — all 10 sections; used directly for DSP.md scaffold structure
- `research/docs/demo/discussion_summary.md` — voice guidelines, terminology ladder, pedagogy decisions
- `src/subshader/audio/audio_input.py` — AudioInput class; all code examples verified against source
- `src/subshader/audio/audio_player.py` — AudioPlayer class; timing reference design
- `src/subshader/dsp/wavelet.py` — Wavelet hierarchy; CWT pipeline steps
- `src/subshader/viz/plotter.py` — ShaderPlot, CircularFrameBuffer, Renderer, GLContext
- `src/subshader/viz/plot_normalizer.py` — IntensityTracker decay logic
- `README.md` — existing top-level draft; confirmed ~50% status, identified REWRITE passages
- `research/docs/demo/readmes/audio/audio_submodule_readme_claude.md` — AUDIO.md topic list
- `research/docs/demo/readmes/plotter/visualizer_submodule_readme_claude.md` — RENDERER.md topic list
- `.planning/phases/05-documentation/05-CONTEXT.md` — all locked decisions and constraints

### Secondary (MEDIUM confidence)
- `research/docs/demo/readmes/wavelet/dsp_submodule_readme_claude.md` — DSP README workflow instructions; used to confirm benchmark.py as figure generation home
- `research/docs/demo/readmes/subshader_toplevel_readme_claude.md` — top-level README spec; confirmed scope

---

## Metadata

**Confidence breakdown:**
- Scaffold structure: HIGH — CONTEXT.md decisions are explicit; source files are all readable
- Content inventory: HIGH — all files read directly; no guesswork on what exists
- Figure pipeline: MEDIUM — benchmark.py not deeply read; PyWavelet row status unclear
- Code examples: HIGH — all extracted from actual source, not from memory

**Research date:** 2026-03-23
**Valid until:** Stable — documentation scaffolding does not depend on rapidly-changing APIs
