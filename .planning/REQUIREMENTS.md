# Requirements: SubShader — Demo Ready

**Defined:** 2026-03-21
**Core Value:** The visualization accurately tracks audio input in real time with minimal latency

## v1 Requirements

Requirements for "Demo Ready" — locally installable, documented, real-time audio visualization.

### Pipeline Fixes

- [ ] **PIPE-01**: CWT normalization produces consistent brightness across frequency bands (investigate low-frequency brightness bias)
- [x] **PIPE-02**: GPU fallback lives in DSP block instantiation, not benchmark code — auto-detects GPU failure and falls back to NumPy
- [x] **PIPE-03**: GPU availability checked at init — if unavailable, run on NumPy path for the session

### Audio Sync

- [ ] **AUDIO-01**: Audio playback and visualization are synced — file-based audio with real-time CWT rendering
- [ ] **AUDIO-02**: Audio-visual sync with minimal perceptible latency (<100ms perceived lag)

### Documentation

- [ ] **DOCS-01**: Top-level README — project overview, benchmark figures, visual comparisons, install/usage instructions
- [ ] **DOCS-02**: DSP module README — pedagogical explanation of CWT pipeline, wavelet choices, normalization, with visuals
- [ ] **DOCS-03**: Rendering module README — OpenGL/shader pipeline, frame buffer, intensity mapping
- [ ] **DOCS-04**: Audio module README — audio capture, chunking, overlap strategy
- [ ] **DOCS-05**: Meaningful examples chosen for each README (no filler, no superfluous content)
- [ ] **DOCS-06**: Documentation scaffolded by Claude, authored by user in their own voice

### Code Quality

- [x] **QUAL-01**: Clean, readable code — descriptive function names, well-factored helpers, minimal comments
- [ ] **QUAL-02**: Pytest unit tests built incrementally as issues surface (not comprehensive upfront suite)
- [x] **QUAL-03**: Existing readability maintained — no unnecessary refactoring

### Install Experience

- [ ] **INST-01**: Clone → install → run works without manual configuration
- [ ] **INST-02**: Dependencies install cleanly via pip/setup
- [ ] **INST-03**: Clear error messages if GPU not available (with automatic CPU fallback)

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Hosted Demo

- **HOST-01**: Server-rendered WebSocket frame streaming (GPU runs CWT, browser displays)
- **HOST-02**: Pre-selected audio loop plays server-side, demo is live on page load
- **HOST-03**: Auto-reconnect, rate limiting, connection status indicator
- **HOST-04**: HTTPS deployment with connection limits and read-only streaming
- **HOST-05**: Headless GPU rendering via EGL for server deployment

### Live Audio Capture

- **LIVE-01**: Live audio capture from system audio input (before or alongside hosted demo)

### Enhancements

- **ENH-01**: GPU benchmark panel visible in UI (CuPy vs NumPy timing)
- **ENH-02**: Color palette controls for shader
- **ENH-03**: Resolution / quality slider
- **ENH-04**: Curated track selection (3-5 demo tracks)
- **ENH-05**: Browser microphone input
- **ENH-06**: Adaptive resolution reduction to maintain frame rate under load

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cross-platform desktop distribution | Pivoted to "demo ready locally" then hosted later |
| Client-side WebGL/WASM pipeline port | Server-rendered approach keeps existing Python pipeline |
| Mobile support | Desktop browser sufficient |
| User accounts / auth | Not a demo-stage concern |
| Multi-user concurrent GPU pipelines | Single 4060 Ti; one pipeline is the design |
| User-uploaded arbitrary audio files | Abuse surface; defer until rate limiting hardened |
| Comprehensive upfront test suite | Tests built pragmatically as issues surface |
| Real-time MIDI/OSC control | Out of scope for this milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 2 | Pending |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| AUDIO-01 | Phase 3 | Pending |
| AUDIO-02 | Phase 3 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |
| DOCS-04 | Phase 5 | Pending |
| DOCS-05 | Phase 5 | Pending |
| DOCS-06 | Phase 5 | Pending |
| QUAL-01 | Phase 1 | Complete |
| QUAL-02 | Phase 2 | Pending |
| QUAL-03 | Phase 1 | Complete |
| INST-01 | Phase 4 | Pending |
| INST-02 | Phase 4 | Pending |
| INST-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 17 total (note: REQUIREMENTS.md header previously stated 16 — DOCS-06 is the 17th)
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-03-21 after roadmap creation — traceability complete*
