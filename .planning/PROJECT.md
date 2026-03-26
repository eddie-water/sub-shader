# SubShader

## What This Is

SubShader is a real-time audio visualization tool that uses continuous wavelet transforms (CWT) to render frequency content from audio. It runs a GPU-accelerated pipeline (CuPy + OpenGL) on a three-stage architecture: audio input, DSP processing, and shader-based rendering. The project is targeting a "Demo Ready" milestone where anyone can clone the repo, install, and see real-time audio visualization with well-documented, pedagogical READMEs.

## Core Value

The visualization accurately tracks audio input in real time with minimal latency — if the visual doesn't feel synced to the audio, nothing else matters.

## Requirements

### Validated

- ✓ CWT pipeline with GPU acceleration (CuPy) — existing
- ✓ CWT pipeline with CPU fallback (NumPy) — existing
- ✓ OpenGL shader-based rendering via ModernGL — existing
- ✓ Circular frame buffer with intensity tracking — existing
- ✓ Configuration system with dataclass validation — existing
- ✓ Custom exception hierarchy with graceful shutdown — existing
- ✓ Wavelet abstraction with multiple backend implementations — existing
- ✓ Audio file input with chunked reading and overlap — existing
- ✓ Scale normalization, edge artifact removal, downsampling — existing
- ✓ Codebase hardened — no silent failures, GPU fallback in DSP init — Phase 1
- ✓ GPU availability checked at init — falls back to NumPy with clear message — Phase 1
- ✓ CWT brightness bias fixed — L1 kernel normalization equalizes frequency bands — Phase 2
- ✓ CWT normalization test suite — 6 pytest tests covering kernel norm, magnitude ratio, no-op — Phase 2
- ✓ File-based audio playback synced to CWT visualization with <100ms perceived lag — Phase 3 (human verification pending)
- ✓ Top-level README with embedded comparison grid hero figure and scaffold captions — Phase 6
- ✓ Research toolkit restructured — benchmark.py split into modular modules, colocated pytest tests — Phase 5.1
- ✓ CWT sub-stage profiling via cwt_timed() + 4-method comparison timing (STFT, PyWavelet, CPU, GPU) — Phase 5.2
- [ ] Module-level READMEs for DSP, rendering, and audio capture modules
- [ ] DSP documentation that is pedagogical, visual, and accessible to non-specialists
- [ ] Unit tests (pytest) built incrementally as issues are discovered
- [ ] Clean, readable code — descriptive function names, helpers over spaghetti, minimal comments
- [ ] Clone → install → run works without manual configuration

### Out of Scope

- Hosted demo — future milestone after demo-ready locally
- Live audio capture — future milestone (before/with hosting)
- Cross-platform desktop distribution — future consideration
- Client-side WebGL pipeline port — server-rendered approach keeps existing Python pipeline
- Comprehensive upfront test suite — tests built pragmatically as issues surface
- Mobile support — desktop browser sufficient
- Adaptive resolution — future enhancement

## Context

- **Existing codebase:** ~mature three-stage pipeline (audio → DSP → render) with working GPU and CPU paths
- **Dev environment:** AMD Ryzen 9700, NVIDIA 4060 Ti 16GB VRAM, Windows/WSL2
- **Milestone scope:** Demo-ready locally first. Hosted demo is a separate future milestone.
- **Documentation philosophy:** User will author docs in their own voice. Claude scaffolds structure, chooses meaningful examples, and helps with technical accuracy. No filler, no superfluous content.
- **Code quality:** Readable code is the priority. Descriptive function names, well-factored helpers, minimal comments. Good code explains itself.
- **Audio sync:** File-based playback synced to visualization. Live capture deferred to v2.
- **Workflow:** User works on documentation while Claude handles engineering tasks in parallel worktrees. Nothing merges to develop without user confirmation.

## Constraints

- **GPU:** NVIDIA 4060 Ti 16GB VRAM — local dev machine
- **Tech stack:** Python, CuPy, ModernGL, existing pipeline — no rewrites
- **Documentation voice:** User authors final prose. Claude scaffolds and suggests, doesn't write final copy.
- **Test approach:** Pytest, incremental, not a dedicated phase. Shouldn't require constant user attention.
- **Code style:** Descriptive names, helpers, no comment litter. Structure over documentation.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Demo-ready locally before hosted demo | Get it working and documented first; hosting is a separate milestone | — Pending |
| GPU check at init only, no mid-session switching | Simpler; just pick CPU or GPU path and run | — Pending |
| File-based audio for v1, live capture for v2 | Sync is the priority, not input source | — Validated Phase 3 |
| Documentation scaffolded by Claude, authored by user | User wants their voice in docs; Claude helps structure and pick examples | — Pending |
| Tests built incrementally, not as dedicated phase | User wants to focus on READMEs, not babysit test infrastructure | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-25 after Phase 06 completion*
