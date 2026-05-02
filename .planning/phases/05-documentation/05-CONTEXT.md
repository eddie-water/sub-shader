# Phase 5: Documentation - Context

**Gathered:** 2026-03-23 (original) · **Updated:** 2026-05-01 (post-Phases 5.1/5.2/6/7/8)
**Status:** Ready for planning (replan)
**Source:** discuss-phase conversation (original) + discuss-phase update conversation (refresh)

<domain>
## Phase Boundary

Each module has a README that explains what it does, why it does it that way, and how to use it — written in the user's voice. Claude scaffolds structure, chooses meaningful examples, generates figures inline against shared helpers, helps with technical accuracy. User authors all final prose.

**Deliverables:**
- `README.md` — top-level landing page (exists, ~143 lines, has REWRITE markers from Phase 6/7 scaffold work; needs final pass)
- `src/subshader/dsp/DSP.md` — wavelet foundations + CWT implementation + post-processing + future applications (exists as 359-line scaffold with WRITE markers + 7 PLACEHOLDER figures)
- `src/subshader/audio/AUDIO.md` — AudioStream facade + reader.py + player.py (exists as 127-line scaffold; references stale `audio_input.py` paths to refresh)
- `src/subshader/renderer/RENDERER.md` — frame buffer + shader pipeline + color mapping + config (exists as 128-line scaffold rewritten 2026-04-28 with current paths; concern-based structure)

</domain>

<decisions>
## Implementation Decisions

### Cross-Doc Authoring Rules
- **D-01:** Each README is self-contained — no "see DSP.md §4" cross-references that force the reader to hop docs to understand a concept.
- **D-02:** When a concept touches multiple modules, each affected doc owns its own explanation from its audience's angle — same fact, reframed per lens. Duplication of facts is OK; duplication of identical prose is over-coupling.
- **D-03:** No enforced uniformity in organizing principle across docs — each picks the structural shape that reads best for its content (file-mirrored vs concern-based vs narrative-flow). Different shapes per doc are allowed and expected.

### Authoring Rhythm
- **D-04:** Workshop the long-term storyline (prose arc + figure inventory with justifications) BEFORE section-by-section authoring begins. Workshop output is an edit to the doc scaffold itself — section structure locked + each `[PLACEHOLDER:]` replaced with a one-line pedagogical justification. No separate blueprint doc.
- **D-05:** Per-section loop is prose-first → figure-to-match. User authors prose against `[WRITE:]` markers; when a section needs a figure, user describes what visual would land the point; Claude generates it; user approves/iterates; user continues. Figure follows prose's actual emphasis.
- **D-06:** Every figure must justify its inclusion against the learning goal. No decorative figures. Cut figures that can't justify themselves during the workshop pass.
- **D-07:** Bridge sections may emerge mid-authoring when gaps surface. Expected discovery, not constant restructure — minimize but don't fight it.
- **D-08:** Figures must share visual language and reference prior figures' primitives. No one-off styling per figure. Foundation figure generation must use the same helper functions as `research/figures.py` (ReadmeFigures) and `research/comparison.py` (comparison grid) — no parallel helper system. Reuse `research/utilities/style.py` constants where applicable.

### Scaffold Format (preserved from original)
- **D-09:** Headers, subheaders, bullet-point placeholders with specific guidance. NOT draft prose — user writes final text.
- **D-10:** Where ~50% already exists (DSP foundations outline, top-level README draft, RENDERER.md rewrite), scaffold fills gaps and marks what's done vs what's left.
- **D-11:** Suggest candidate analogies user can accept/reject/rewrite (e.g., "unmixing paint" for signal decomposition; "casting a shadow" for vector projection).

### Top-level README (README.md)
- **D-12:** Keep existing personal "technical showcase" voice and framing.
- **D-13:** Hero comparison grid (per-signal columns × representation rows) is shipped via Phases 6/7 — README polish does NOT regenerate; it polishes prose and captions only.
- **D-14:** REWRITE-flagged passages get user prose pass with intent + placement guidance preserved.
- **D-15:** numpy_vs_cupy_diff.png belongs in DSP section, not top-level (preserved from original).

### DSP.md (`src/subshader/dsp/DSP.md`)
- **D-16:** Cover sections 1-6 of wavelet foundations outline (inner product → CWT implementation) per existing scaffold.
- **D-17:** Sections 7-10 (feature hierarchy, ML, applications) consolidated into single "Future" section preserving their info. Plan 05-06 carves out the "Future Applications" blurb specifically (financial time series, heartbeat, brain signals) — folds into the consolidated §7+ near the end of DSP.md authoring.
- **D-18:** Depth: accurate, practical explanations — match the foundations notebook voice (now archived at `research/archive/docs/demo/readmes/wavelet/wavelet_foundations.ipynb`).
- **D-19:** Foundation figure code lives in / alongside `research/figures.py`, leveraging shared helpers and `research/utilities/style.py` constants. Wired into `research/test_suite.py` CLI as a flag (e.g., `--foundations-figures` or `--figures` extension). Replaces the stale 05-04-PLAN's reference to `research/benchmark.py`.
- **D-20:** Code examples extracted from current source files: `src/subshader/dsp/cwt.py`, `src/subshader/dsp/pywavelet.py`, `src/subshader/dsp/stft.py`, `src/subshader/config.py` — NOT old `wavelet.py` paths. No illustrative stubs.
- **D-21:** DSP scaffold uses 'properties' not 'features/patterns' before §7 — terminology ladder preserved (per archived `discussion_summary.md`).

### AUDIO.md (`src/subshader/audio/AUDIO.md`)
- **D-22:** Structure mirrors module file layout: §1 AudioStream facade (the API consumers actually call), §2 `reader.py` (file I/O + chunking + overlap mechanics), §3 `player.py` (callback thread + audio-clock source of truth). 1:1 code↔doc correspondence.
- **D-23:** Each section opens with "this file does X" and shows the API. Reader navigating the code finds direct mapping.
- **D-24:** Refresh stale `audio_input.py` references throughout the scaffold to current `reader.py` / `audio_stream.py` paths during the workshop pass.
- **D-25:** Overlap strategy explained from the audio-I/O angle here (mechanics of `hop_size`, `file_pos` advancement) — reframed from DSP angle in DSP.md (edge effects, reliable interior coefficients) per D-02.
- **D-26:** [DEFERRED] Audio-clock-driven sync mechanism's home (AUDIO.md §3 player.py vs `pipeline.py` docstring vs separate SYNC.md) — decide during authoring.

### RENDERER.md (`src/subshader/renderer/RENDERER.md`)
- **D-27:** Keep current concern-based structure (Role → Frame Buffer → Why a Shader → Color Mapping → Config → Diagram). Frame buffer + intensity + renderer interlock too tightly to discuss in isolation per file. Do NOT restructure to file-mirrored.
- **D-28:** Author against current `IntensityTracker` implementation (fixed pre-scan reference). Polish backlog item — IntensityTracker normalization rethink — does NOT block Phase 5. If rework happens later, the Color Mapping section gets a localized revision then.
- **D-29:** Two PLACEHOLDER figures to address: (a) intensity normalization before/after at three magnitudes; (b) full renderer pipeline diagram. Apply D-08 shared-helper rule.

### Plan Sequencing
- **D-30:** One plan per doc — each independently shippable. Plans are: 05-04 (DSP.md authoring; existing PLAN replaced wholesale due to stale paths/tooling), 05-NEW (AUDIO.md authoring), 05-06 (DSP.md §7 Future Applications blurb — stays standalone, folds in near end of DSP authoring), 05-NEW (RENDERER.md authoring), 05-NEW (README.md final polish).
- **D-31:** Plan order: DSP.md first (largest, most figures, most blocking) → AUDIO.md → RENDERER.md → README polish. 05-06 Future Applications inserts during DSP authoring tail.
- **D-32:** IntensityTracker rework deferred entirely from Phase 5 — handled in v1.1 / next milestone. Polish backlog stays parked.

### Claude's Discretion
- Exact section ordering within each README (within the locked structures above)
- Which existing archived `research/archive/docs/demo/readmes/` content to incorporate vs discard
- Specific placeholder wording for scaffold updates
- How to structure the consolidated `Future` section in DSP.md (D-17 / 05-06)
- Workshop edit specifics — which figures to cut vs keep based on D-06 justification test
- Audio-clock sync mechanism's home (D-26 deferred) — decide during authoring with reference to D-02 cross-doc rule

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Active Documentation Targets (in-flight)
- `README.md` — current top-level README (~143 lines, REWRITE markers present)
- `src/subshader/dsp/DSP.md` — DSP scaffold (359 lines, WRITE markers + 7 PLACEHOLDER figures)
- `src/subshader/audio/AUDIO.md` — AUDIO scaffold (127 lines, references stale `audio_input.py` paths to refresh)
- `src/subshader/renderer/RENDERER.md` — RENDERER scaffold (128 lines, current paths, concern-based shape, 2 PLACEHOLDER figures)

### Source Code (for accurate documentation)
- `src/subshader/audio/audio_stream.py` — AudioStream facade exposing `start() / get_chunk() / next_chunk() / get_playback_sample() / has_looped() / cleanup()`
- `src/subshader/audio/reader.py` — file I/O, chunking, overlap mechanics (was `audio_input.py`)
- `src/subshader/audio/player.py` — callback-driven playback, audio-clock source of truth
- `src/subshader/dsp/dsp.py` — DSP ABC with `pre() / transform() / post()` interface
- `src/subshader/dsp/cwt.py` — CWT base + CpuCWT + GpuCWT (was part of `wavelet.py`)
- `src/subshader/dsp/pywavelet.py` — PyWavelet reference backend
- `src/subshader/dsp/stft.py` — STFT comparison backend
- `src/subshader/dsp/gaussian.py`, `src/subshader/dsp/wavelet_kernel.py` — kernel construction + L1 normalization
- `src/subshader/renderer/renderer.py` — Renderer + GPURenderer + GLContext orchestration
- `src/subshader/renderer/frame_buffer.py` — CircularFrameBuffer (pre-allocated `flattened_buffer`, in-place updates)
- `src/subshader/renderer/intensity.py` — IntensityTracker (fixed pre-scan reference, current implementation)
- `src/subshader/renderer/shaders/{vertex.glsl, fragment.glsl}` — GPU shaders
- `src/subshader/pipeline.py` — `SubShader` orchestrator class (audio-clock-driven render loop)
- `src/subshader/__main__.py` — thin CLI entry point
- `src/subshader/config.py` — `PipelineConfig`, `CWTConfig`, `RendererConfig`, `ColorNormalizationConfig` dataclasses

### Figure & Helper Infrastructure
- `research/figures.py` — `ReadmeFigures` per-signal figure generation (foundation figures live here or alongside, sharing helpers)
- `research/comparison.py` — `generate_comparison_grid()` + `generate_timing_bar_chart()` + `COMPARISON_METHODS` extensible config
- `research/timing.py` — pipeline timing reads from `@timed` instance attributes
- `research/test_suite.py` — single CLI dispatcher (4 modes: `--timing`, `--test`, `--compare-methods`, `--figures`); foundation-figures flag wires in here
- `research/utilities/style.py` — single source of truth for visual constants (dark theme, fonts, colors, DPI). All figures use these constants.
- `research/utilities/signals.py` — `SIGNALS` registry for comparison grid inputs
- `research/utilities/wav_export.py` — audio export helper

### Existing Asset Locations
- `assets/images/figures/` — per-signal README figures (bouncing chirp, polyphonic, musical, beltran)
- `assets/images/generated/` — test_suite.py outputs
- `assets/images/reference/` — committed reference inputs
- `assets/images/diagnostics/` — overlap/redundancy diagnostics
- `assets/images/dsp/` — DSP.md foundation figures land here (per stale 05-04-PLAN; path remains valid)
- `assets/audio/reference/` — committed audio inputs (default: `beltran_sc_rip.wav`)
- `assets/audio/generated/` — synthesized signals (bouncing chirp, polyphonic MIDI)
- `assets/timing/` — timestamped `--timing` output

### Archived Reference Material (still consultable)
- `research/archive/docs/demo/readmes/wavelet/wavelet_foundations.ipynb` — notebook through §2.4 with working figures (was active reference for DSP scaffold)
- `research/archive/docs/demo/readmes/wavelet/wavelet_foundations_outline.md` — original 10-section outline
- `research/archive/docs/demo/readmes/wavelet/plot_vectors.py` — quiver-arrow vector plotting helper (port style into shared figures.py helpers)
- `research/archive/docs/demo/discussion_summary.md` — pedagogical decisions, voice guidelines, terminology ladder (`properties` not `features` before §7)
- `research/archive/docs/demo/readmes/{audio,plotter,wavelet}/*_submodule_readme_claude.md` — original module README outlines (cherry-pick what's still relevant)

### Project-Level Context
- `.planning/PROJECT.md` — Demo Ready milestone scope, documentation philosophy
- `.planning/REQUIREMENTS.md` — DOCS-01..DOCS-06 requirements (currently marked Complete in traceability table — this update will reopen the Complete status if any DOCS-* requirement is materially changed; verify after replan)
- `.planning/ROADMAP.md` — Phase 5 active, plans 05-04 / 05-06 outstanding
- `.planning/phases/08-codebase-refactoring-and-module-cleanup/08-CONTEXT.md` — D-01..D-38 from Phase 8 establish the current module layout this CONTEXT depends on

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`research/figures.py` ReadmeFigures**: per-signal figure generation pattern — Claude helpers, style.py-driven aesthetics. Foundation figures extend this rather than reinvent.
- **`research/utilities/style.py` constants**: dark theme, font sizes, colors, DPI — single source of truth. All Phase 5 figures consume these.
- **`research/comparison.py` helpers**: `create_figure_scaffold()`, `render_top_row()`, `render_spectrogram_row()` — shared primitives for grid layout.
- **`research/archive/docs/demo/readmes/wavelet/plot_vectors.py`**: existing quiver-arrow plotting code — port the styling pattern into shared helpers; do NOT call from archive directly.
- **`research/utilities/signals.py` SIGNALS registry**: extensible signal list for comparison work.
- **Existing scaffolds**: DSP.md (359 lines), AUDIO.md (127 lines), RENDERER.md (128 lines), README.md (143 lines) — all have committed structure to extend, not rewrite from zero.

### Established Patterns
- **`@timed` decorator** (`src/subshader/utils/timing.py`) — pipeline stages already instrumented. DSP.md §6 computational cost section uses `_timing_*_ms` instance attributes via `research/timing.py`.
- **`PipelineConfig` inheritance** — `CWTConfig` and `RendererConfig` extend the base. Config docs in module READMEs reference the right subclass.
- **DSP ABC** — `dsp/dsp.py` defines `pre() / transform() / post()`; backends inherit. DSP.md implementation section frames the abstraction.
- **AudioStream facade** — wraps `reader.py` + `player.py`; consumers see one API. AUDIO.md §1 frames it this way.
- **`global_intensity_percentile`** — pipeline pre-scan computes fixed `intensity_max` reference from this percentile (default 99.0). RENDERER.md Color Mapping section documents current behavior.

### Integration Points
- **DSP.md ↔ figure generation**: `--foundations-figures` (or equivalent) flag in `research/test_suite.py` triggers helpers in/alongside `research/figures.py` → PNGs land in `assets/images/dsp/` → markdown image refs in DSP.md.
- **All docs ↔ source code**: examples extracted directly from current source paths (D-20). Stale paths in scaffolds (e.g., AUDIO.md's `audio_input.py` refs) refresh during the workshop pass.
- **Module READMEs ↔ project root**: top-level README does NOT cross-link to module READMEs in a way that requires the reader to hop docs (D-01). Module READMEs are discoverable by reader who navigates the source tree.
- **Cross-doc concept ownership**: overlap strategy → AUDIO.md (audio I/O angle) + DSP.md (CWT correctness angle), each self-contained per D-02. CWT output shape → DSP.md (post-processing angle) + RENDERER.md (input contract angle), each self-contained.

</code_context>

<specifics>
## Specific Ideas

- Bouncing chirp from overlap_redundancy_diagnostic.png shipped via Phase 6 — used in comparison grid (preserved from original)
- Paint mixing analogy for signal decomposition (already in archived notebook) — good template for analogy style (preserved)
- "Casting a shadow" candidate analogy for vector projection (in DSP.md scaffold §2.4.2)
- Terminology ladder from archived `discussion_summary.md`: build vocabulary incrementally, don't front-load jargon ('properties' before §7, then 'features/patterns')
- Comparison grid: 3 columns (signals) × N rows (representations) — shipped (preserved)
- IntensityTracker design tension noted in RENDERER.md scaffold's polish-backlog callout: fixed pre-scan reference may not match the original "frame-to-frame consistency" goal — explicitly deferred per D-32

</specifics>

<deferred>
## Deferred Ideas

- **IntensityTracker normalization rework** — polish-backlog item; deferred to v1.1 / next milestone per D-32. Phase 5 ships docs against current behavior.
- **Audio-clock-driven sync mechanism's documentation home** — AUDIO.md §3 vs `pipeline.py` docstring vs separate SYNC.md. Decided during authoring per D-26.
- **Wavelet foundations sections 7-10 full treatment** — feature hierarchy, ML integration, applications. Consolidated into single `Future` section per D-17. Full per-section treatment deferred beyond v1.
- **Interactive notebook-based documentation** — decided against; using static .md with image placeholders (preserved).
- **Module READMEs at project root** — superseded by Phase 8 reorg: docs now live at `src/subshader/{audio,dsp,renderer}/{AUDIO,DSP,RENDERER}.md`. Original CONTEXT decision overridden.
- **Source-code drift not yet committed** (per ROADMAP Polish Backlog): `research/comparison.py`, `research/figures.py`, `research/test_suite.py`, `research/timing.py`, `research/utilities/dsp_helpers.py`, `research/utilities/style.py`, `research/tests/audio/test_audio_overlap.py`, `research/tests/dsp/test_wavelet_kernel.py` — verify intent and commit before milestone close (out of Phase 5 scope; flagged for milestone audit).

</deferred>

---

*Phase: 05-documentation*
*Context originally gathered: 2026-03-23 via discuss-phase conversation*
*Context updated: 2026-05-01 via discuss-phase update — refreshed canonical_refs and code_context for post-Phase-8 module layout, added cross-doc rules, locked authoring rhythm, sequenced remaining plans*
