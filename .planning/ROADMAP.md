# Roadmap: SubShader — Demo Ready

## Overview

The pipeline already works. This milestone gets it from "works on my machine with hardcoded paths" to "anyone can clone it, install it, run it, and understand what they're looking at." Five phases in dependency order: harden the codebase, polish the CWT output, sync audio to the visualization, make install frictionless, then document everything.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Codebase Hardening** - Fix silent failure modes and relocate GPU fallback to the right place
- [x] **Phase 2: CWT Pipeline Polish** - Resolve frequency-band brightness bias and establish incremental test pattern (completed 2026-03-21)
- [ ] **Phase 3: Audio-Visual Sync** - Wire file-based audio playback to CWT rendering with sub-100ms perceived lag
- [ ] **Phase 4: Install Experience** - Make clone-install-run work without manual configuration or surprises
- [ ] **Phase 5: Documentation** - Four READMEs scaffolded by Claude, authored by user in their own voice

## Phase Details

### Phase 1: Codebase Hardening
**Goal**: The pipeline fails loudly and correctly — no silent blank frames, no swallowed GPU errors, no hardcoded paths
**Depends on**: Nothing (first phase)
**Requirements**: PIPE-02, PIPE-03, QUAL-01, QUAL-03
**Success Criteria** (what must be TRUE):
  1. Running the tool with a missing or wrong audio path raises a clear error immediately — not a silent blank visualization
  2. GPU unavailability at startup is detected, logged explicitly, and the session continues on NumPy — not silently degraded
  3. GPU fallback lives in DSP block instantiation, not in benchmark code — you can delete the benchmark without breaking fallback
  4. Code in the changed files uses descriptive function names and well-factored helpers — no new comment litter or spaghetti added
**Plans:** 1/2 plans executed
Plans:
- [x] 01-01-PLAN.md — Consolidate exceptions, fix config, create gpu utility, set up tests
- [x] 01-02-PLAN.md — Wire GPU fallback, guard CuPy imports, fix silent failures

### Phase 2: CWT Pipeline Polish
**Goal**: CWT output looks visually correct across all frequency bands and the fix is covered by a test
**Depends on**: Phase 1
**Requirements**: PIPE-01, QUAL-02
**Success Criteria** (what must be TRUE):
  1. Low-frequency bands no longer appear disproportionately brighter than high-frequency bands on the same audio input
  2. A pytest test exists that asserts normalized CWT output stays within expected brightness bounds — the first test in the incremental suite
  3. The test can be run with `pytest` from the project root without manual setup
**Plans:** 2/2 plans complete
Plans:
- [x] 02-01-PLAN.md — TDD: kernel L1 normalization fix + brightness bias tests + dead code cleanup
- [x] 02-02-PLAN.md — Regenerate benchmark figures + visual verification checkpoint

### Phase 3: Audio-Visual Sync
**Goal**: Users can play an audio file and watch the CWT visualization track it in real time with no perceptible drift
**Depends on**: Phase 2
**Requirements**: AUDIO-01, AUDIO-02
**Success Criteria** (what must be TRUE):
  1. Running the tool with an audio file argument plays the audio and renders CWT frames simultaneously — not sequentially
  2. Transient events in the audio (a drum hit, a sharp consonant) appear in the visualization within ~100ms of being heard
  3. The visualization does not drift ahead or behind the audio over a 60-second playback — sync holds for the duration
**Plans:** 2 plans
Plans:
- [x] 03-01-PLAN.md — Create AudioPlayer class with sounddevice, add dependency, unit tests
- [x] 03-02-PLAN.md — Wire AudioPlayer into orchestrator with CLI arg, sync loop, and human verification

### Phase 4: Install Experience
**Goal**: A developer with Python and a compatible GPU can clone the repo, install, and run the visualization without reading source code
**Depends on**: Phase 3
**Requirements**: INST-01, INST-02, INST-03
**Success Criteria** (what must be TRUE):
  1. `git clone && pip install -e . && subshader demo.wav` works on a fresh environment without any manual configuration step
  2. `pip install` completes without version conflicts or build errors for the listed dependencies
  3. Running without a GPU (or with GPU unavailable) prints a clear message stating CPU fallback is active — it does not crash or silently degrade
**Plans**: TBD

### Phase 5: Documentation
**Goal**: Each module has a README that explains what it does, why it does it that way, and how to use it — written in the user's voice
**Depends on**: Phase 4
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05, DOCS-06
**Success Criteria** (what must be TRUE):
  1. The top-level README lets a reader with no prior context understand what SubShader is, see benchmark figures, and get it running — without reading source code
  2. The DSP README explains the CWT pipeline, wavelet choices, and normalization in terms a developer without a DSP background can follow, supported by visuals
  3. The rendering and audio module READMEs exist and explain their respective pipelines at the same depth — not placeholder stubs
  4. Every code example in every README is accurate and runnable — no illustrative filler that silently fails
  5. The prose reads in the user's voice — Claude's scaffold is not detectable as generated text
**Plans:** 4 plans
Plans:
- [x] 05-01-PLAN.md — Comparison grid figure + benchmark.py figure pipeline
- [x] 05-02-PLAN.md — DSP.md scaffold from wavelet foundations outline
- [x] 05-03-PLAN.md — README.md update + AUDIO.md + RENDERER.md scaffolds
- [ ] 05-04-PLAN.md — DSP.md foundation figures via benchmark.py --foundations-figures

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Codebase Hardening | 1/2 | In Progress|  |
| 2. CWT Pipeline Polish | 2/2 | Complete   | 2026-03-21 |
| 3. Audio-Visual Sync | 0/2 | Not started | - |
| 4. Install Experience | 0/TBD | Not started | - |
| 5. Documentation | 3/4 | In Progress | - |
| 5.1 Research Toolkit | 2/2 | Complete | - |
| 5.2 Timing & Grid Polish | 0/2 | Not started | - |
| 6. Finalize Audio & Figures | 2/3 | In Progress|  |

### Phase 05.2: Benchmark timing profiling and comparison grid polish (INSERTED)

**Goal:** Add cwt_timed() sub-stage profiling to --timing output, add NumPy timing to --comparison table, and commit existing grid visual improvements
**Requirements**: TBD-01, TBD-02, TBD-03, TBD-04
**Depends on:** Phase 5.1
**Plans:** 2/2 plans complete

Plans:
- [x] 05.2-01-PLAN.md — cwt_timed() method + TimedSubShader sub-stage breakdown + render timing + tests
- [x] 05.2-02-PLAN.md — NumPy (SubShader CPU) timing row in --comparison table

### Phase 05.1: Research toolkit restructure (INSERTED)

**Goal:** Restructure monolithic benchmark.py into modular research toolkit and migrate unit tests to colocated pytest files alongside source modules
**Requirements**: RTK-01, RTK-02, RTK-03, RTK-04, RTK-05, RTK-06
**Depends on:** Phase 5
**Plans:** 2/2 plans complete

Plans:
- [x] 05.1-01-PLAN.md — Split benchmark.py into figures.py, timing.py, wav_export.py + thin CLI dispatcher
- [x] 05.1-02-PLAN.md — Migrate unit_tests.py to colocated pytest files + relocate standalone scripts

### Phase 6: Finalize example audio and comparison figures for README

**Goal:** Curate final audio examples (bouncing chirp synthesis) and generate the polished comparison grid figure for the top-level README, with timing analysis relocated to DSP.md
**Requirements**: FIG-01, FIG-02, FIG-03, FIG-04, FIG-05, FIG-06
**Depends on:** Phase 5
**Plans:** 2/3 plans executed

Plans:
- [x] 06-01-PLAN.md — Bouncing chirp synthesis + comparison grid at multiple DPIs + user DPI selection
- [x] 06-02-PLAN.md — Final grid generation (full PyWavelet) + README/DSP.md updates
- [x] 06-03-PLAN.md — Fix duration_s overlap bug + add --comparison flag with per-method timing stats (gap closure)
