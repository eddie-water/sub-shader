# SubShader

## What This Is

SubShader is a real-time audio visualization tool that uses continuous wavelet transforms (CWT) to render frequency content from live audio. It runs a GPU-accelerated pipeline (CuPy + OpenGL) on a three-stage architecture: audio input, DSP processing, and shader-based rendering. The project is targeting a "Demo Ready" milestone where anyone can visit a hosted URL and see real-time audio visualization without installing anything.

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

### Active

- [ ] Server-rendered hosted demo (GPU runs CWT server-side, browser displays via WebSocket + canvas)
- [ ] Real-time audio sync with minimal perceptible latency
- [ ] Live audio capture from system audio input (not just file playback)
- [ ] Top-level README with embedded benchmark figures and visual comparisons
- [ ] Module-level READMEs for DSP, rendering, and audio capture modules
- [ ] DSP documentation that is pedagogical, visual, and accessible to non-specialists
- [ ] NumPy fallback as local dev safety net (auto-detect GPU disconnection/failure)
- [ ] Unit tests (pytest) built incrementally as issues are discovered
- [ ] Clean, readable code — descriptive function names, helpers over spaghetti, minimal comments
- [ ] Secure hosted deployment (rate limiting, controlled audio input, read-only streaming)

### Out of Scope

- Cross-platform desktop distribution — pivoted to hosted demo instead
- "Works on any computer" install experience — hosting eliminates this
- Client-side WebGL pipeline port — server-rendered approach keeps existing Python pipeline
- Comprehensive upfront test suite — tests built pragmatically as issues surface
- Mobile support — desktop browser is sufficient for demo

## Context

- **Existing codebase:** ~mature three-stage pipeline (audio → DSP → render) with working GPU and CPU paths
- **Dev environment:** AMD Ryzen 9700, NVIDIA 4060 Ti 16GB VRAM, Windows/WSL2
- **Architecture pivot:** Originally planned cross-platform desktop app, pivoted to server-rendered hosted demo to avoid cross-platform compatibility rabbit hole
- **Documentation philosophy:** User will author docs in their own voice. Claude scaffolds structure, chooses meaningful examples, and helps with technical accuracy. No filler, no superfluous content.
- **Code quality:** Readable code is the priority. Descriptive function names, well-factored helpers, minimal comments. Good code explains itself.
- **Audio sync:** Worth pursuing for real-time feel but scoped pragmatically. May need resolution reduction for acceptable latency. Not a rabbit hole.
- **Server architecture:** Python/CuPy pipeline runs on user's GPU (4060 Ti), streams rendered frames to browser clients. Read-only visual streaming — no user-executed code on server.

## Constraints

- **GPU:** NVIDIA 4060 Ti 16GB VRAM — server compute budget for hosted demo
- **Tech stack:** Python, CuPy, ModernGL, existing pipeline — no rewrites
- **Documentation voice:** User authors final prose. Claude scaffolds and suggests, doesn't write final copy.
- **Test approach:** Pytest, incremental, not a dedicated phase. Shouldn't require constant user attention.
- **Code style:** Descriptive names, helpers, no comment litter. Structure over documentation.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Server-rendered hosted demo over cross-platform desktop | Cross-platform compatibility is high effort; hosting leverages existing pipeline | — Pending |
| NumPy fallback kept for local dev only | GPU sometimes disconnects; not a user-facing feature | — Pending |
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
*Last updated: 2026-03-21 after initialization*
