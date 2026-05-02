# Phase 5: Documentation - Discussion Log (Update)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the conversation that produced them.

**Date:** 2026-05-01 (update; original 2026-03-23)
**Phase:** 05-documentation
**Mode:** discuss (update of existing CONTEXT.md)
**Trigger:** ROADMAP shifted significantly since original CONTEXT (Phases 5.1, 5.2, 6, 7, 8 shipped; module layout reorganized; READMEs moved into modules; research toolkit split). Stale paths in canonical_refs forced refresh.
**Areas selected:** DSP.md authoring approach, AUDIO.md authoring scope, RENDERER.md scope (post-IntensityTracker rework), Plan sequencing for remaining work

---

## Pre-Discussion: Remote Pairing Detour

User asked about pairing Claude Code session to phone via QR code (`/remote-control`). Verified feature exists; verified voice input is NOT available in Remote Control sessions on mobile (open GitHub issues #32597, #29399). User opted to continue at the keyboard rather than type on phone.

---

## Area 1: DSP.md Authoring Approach

### Q1.1 — Foundation figure rhythm
**Options:**
- (1) Inline per section (Recommended — matches ROADMAP)
- (2) Batch all 7 first, then author
- (3) Hybrid: batch 4 vector-math, defer 3

**User selected:** (1) Inline per section, with addition: "I'd like to keep the horizon long term story and figures to be kept in mind while figure generation - consistent aesthetic style and in a way that builds upon previous figures and content - common views and consistent visual tone"

**Decisions captured:** D-04, D-05, D-06, D-07, D-08

### Q1.2 — Where foundation-figure code lives
**Options:**
- (1) New module `research/foundations.py` + flag in `test_suite.py`
- (2) Extend `research/figures.py` with foundations alongside ReadmeFigures
- (3) Inline next to DSP.md (`src/subshader/dsp/figures.py` or `research/dsp_figures.py`)

**User selected:** "It should live in the figure generation set of helper functions used in a common way - there are already some from the top level figure generation - lets leverage common helper functions"

**Interpretation:** lean toward (2) — extend `research/figures.py` and reuse shared helpers. Hard constraint: no parallel helper system.

**Decisions captured:** D-08, D-19

### Q1.3 — Per-section authoring loop
**Options:**
- (1) Prose first → I generate figure to match (Recommended)
- (2) Figure first → user authors prose around it
- (3) Side-by-side draft of both, user rewrites prose, I redo figure

**User selected:** (1), with addition: "we need to make sure prose and figures are at least somewhat or generally planned in the long term - leaving room for flexibility if we start writing and realize we need to add sections to bridge gaps - ideally we're not doing this all the time, i don't want superfluous visualizations, everything should be justified for the overall better learning experience for the reader"

**Decisions captured:** D-05, D-06, D-07

### Q1.4 — Workshop output
**Options:**
- (1) Edit DSP.md scaffold itself — lock structure + justify figures inline (Recommended)
- (2) Separate blueprint doc `DSP-BLUEPRINT.md`
- (3) Conversational only — capture in CONTEXT.md

**User selected:** (1)

**Decisions captured:** D-04

### Q1.5 — Continue Area 1 or move on
**Options:**
- (1) Move to Area 2 (Recommended)
- (2) One more on DSP.md — sections 7-10 scope
- (3) One more on DSP.md — code example sourcing strategy

**User selected:** Custom: "Only in the context of defining the scope of each document so the explanation found in one document aren't coupled in another"

**Interpretation:** pivot to a meta-question about cross-doc scope partitioning.

**Decisions captured:** D-01 (cross-doc rule established as direct response)

### Q1.6 — Cross-doc concept handling (meta)
**Options:**
- (1) Each doc owns its full explanation, no cross-refs (Recommended)
- (2) Single owner per concept; other docs treat it as black-box context
- (3) Strict module boundary — cross-module concerns omitted or in README only

**User selected:** (1), with qualifier: "no need to double explain"

**Interpretation:** each doc owns its angle on shared concepts; reframe per audience; identical paragraphs are over-coupling but reframed explanations are not duplication.

**Decisions captured:** D-02

---

## Area 2: AUDIO.md Authoring Scope

### Q2.1 — Structural shape
**Options:**
- (1) 3-section by file: AudioStream facade → reader.py → player.py (Recommended)
- (2) 2-section by concern: data delivery → timing reference
- (3) Pipeline-flow narrative: file → chunk → callback → hardware

**User selected:** (1)

**Decisions captured:** D-22, D-23, D-24

### Q2.2 — Audio-clock sync mechanism's home
**Options:**
- (1) AUDIO.md §3 player.py owns the full design (Recommended)
- (2) AUDIO.md gives the API; pipeline.py docstring explains the loop
- (3) Defer to system-level SYNC.md or README architecture section

**User selected:** "figure this out later"

**Decisions captured:** D-26 (deferred to authoring time)

---

## Area 3: RENDERER.md Scope (Post-IntensityTracker Rework)

### Q3.1 — Author against current intensity strategy or wait for rework
**Options:**
- (1) Author now against current implementation; revise if rework happens (Recommended)
- (2) Author everything except the intensity section; leave it stubbed
- (3) Block authoring until rework resolves

**User selected:** (1)

**Decisions captured:** D-28, D-32

### Q3.2 — Restructure RENDERER.md to match AUDIO.md, or keep concern-based
**Options:**
- (1) Keep RENDERER.md concern-based; AUDIO.md file-mirrored (let each doc pick)
- (2) Restructure RENDERER.md to file-mirrored for consistency
- (3) Re-align AUDIO.md to concern-based instead

**User selected:** (1)

**Decisions captured:** D-03, D-27

---

## Area 4: Plan Sequencing for Remaining Work

### Q4.1 — Plan structure
**Options:**
- (1) One plan per doc (Recommended)
- (2) One omnibus plan covering all docs
- (3) Two plans: shared infrastructure first, then per-doc authoring

**User selected:** (1)

**Decisions captured:** D-30

### Q4.2 — Plan ordering
**Options:**
- (1) DSP.md first (largest, most figures, most blocking)
- (2) Smallest first — RENDERER → AUDIO → DSP
- (3) README polish first to set voice anchor
- (4) Workshop all 4 scaffolds first as setup plan, then author

**User selected:** (1)

**Decisions captured:** D-31

### Q4.3 — IntensityTracker rework timing
**Options:**
- (1) Defer entirely — ship Phase 5 against current behavior, address in v1.1
- (2) Address right before RENDERER.md (insert quick task between AUDIO and RENDERER plans)
- (3) Address right after RENDERER.md ships, then revise the section

**User selected:** (1)

**Decisions captured:** D-32

---

## Deferred Ideas

- IntensityTracker normalization rework (polish-backlog → v1.1)
- Audio-clock sync mechanism's documentation home (deferred to authoring)
- DSP §7-10 full per-section treatment (consolidated into single Future section)

## Claude's Discretion (per CONTEXT.md `<decisions>`)

- Section ordering within each README (within locked structures)
- Which archived `research/archive/docs/demo/readmes/` content to incorporate vs discard
- Specific placeholder wording during workshop edits
- Structure of the consolidated `Future` section in DSP.md
- Workshop-edit specifics — which figures to cut vs keep based on D-06 justification test
- Audio-clock sync mechanism's home (during authoring, with reference to D-02)

---

## Conversation Notes (out-of-band)

- User checkpoint-paused mid-discussion to investigate `/remote-control` for mobile pairing. Verified feature exists; verified voice input not yet supported in Remote Control sessions. User opted to continue at the keyboard. Checkpoint file `05-DISCUSS-CHECKPOINT.json` cleaned up after CONTEXT.md was written.
- User's "no superfluous visualizations" + "long-term story" + "common views and consistent visual tone" forced D-06 and D-08 as hard constraints, not nice-to-haves.
- User's "no need to double explain" qualifier on D-02 is the operative rule for resolving cross-doc coupling: same fact reframed per audience is fine; copy-paste prose is the failure mode to avoid.
- User cancelled an in-flight Windows/WSL2 power-state research subagent invocation when deciding to skip Remote Control entirely — context preserved in conversation but no decision required.
