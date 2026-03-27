# Requirements: SubShader — Demo Ready

**Defined:** 2026-03-21
**Core Value:** The visualization accurately tracks audio input in real time with minimal latency

## v1 Requirements

Requirements for "Demo Ready" — locally installable, documented, real-time audio visualization.

### Pipeline Fixes

- [x] **PIPE-01**: CWT normalization produces consistent brightness across frequency bands (investigate low-frequency brightness bias)
- [x] **PIPE-02**: GPU fallback lives in DSP block instantiation, not benchmark code — auto-detects GPU failure and falls back to NumPy
- [x] **PIPE-03**: GPU availability checked at init — if unavailable, run on NumPy path for the session

### Audio Sync

- [x] **AUDIO-01**: Audio playback and visualization are synced — file-based audio with real-time CWT rendering
- [x] **AUDIO-02**: Audio-visual sync with minimal perceptible latency (<100ms perceived lag)

### Documentation

- [x] **DOCS-01**: Top-level README — project overview, benchmark figures, visual comparisons, install/usage instructions
- [x] **DOCS-02**: DSP module README — pedagogical explanation of CWT pipeline, wavelet choices, normalization, with visuals
- [x] **DOCS-03**: Rendering module README — OpenGL/shader pipeline, frame buffer, intensity mapping
- [x] **DOCS-04**: Audio module README — audio capture, chunking, overlap strategy
- [x] **DOCS-05**: Meaningful examples chosen for each README (no filler, no superfluous content)
- [x] **DOCS-06**: Documentation scaffolded by Claude, authored by user in their own voice

### Code Quality

- [x] **QUAL-01**: Clean, readable code — descriptive function names, well-factored helpers, minimal comments
- [x] **QUAL-02**: Pytest unit tests built incrementally as issues surface (not comprehensive upfront suite)
- [x] **QUAL-03**: Existing readability maintained — no unnecessary refactoring

### Install Experience

- [ ] **INST-01**: Clone → install → run works without manual configuration
- [ ] **INST-02**: Dependencies install cleanly via pip/setup
- [ ] **INST-03**: Clear error messages if GPU not available (with automatic CPU fallback)

### Figures & Audio Examples

- [x] **FIG-01**: Bouncing chirp audio signal synthesized programmatically — ascending frequency contour with parabolic dips across three decades
- [x] **FIG-02**: Comparison grid uses curated audio examples: bouncing chirp, polyphonic MIDI, musical excerpt
- [x] **FIG-03**: Comparison grid generated at multiple DPI levels for user quality/filesize selection
- [x] **FIG-04**: README Performance section has single hero comparison grid figure centered at ~80% width
- [x] **FIG-05**: README per-signal sections have scaffold captions replacing REWRITE markers
- [x] **FIG-06**: Timing bar chart relocated from README to DSP.md computational cost section

### Research Toolkit

- [x] **RTK-01**: benchmark.py split into figures.py (~800 lines), timing.py (~100 lines), wav_export.py (~15 lines) by concern
- [x] **RTK-02**: benchmark.py reduced to thin CLI dispatcher (~60-70 lines) preserving all existing flags
- [x] **RTK-03**: All existing CLI commands (`--comparison-grid`, `--timing`, `--figures`, etc.) produce identical output after restructure
- [x] **RTK-04**: unit_tests.py (10 test categories) migrated to colocated pytest test files alongside source modules
- [x] **RTK-05**: `pytest src/` discovers and runs all migrated tests with zero configuration
- [x] **RTK-06**: Standalone scripts relocated (font_showcase to utilities/, overlap_diagnostic deleted after test logic folded in)

### Visual Style System

- [x] **STY-01**: All visual constants centralized in `research/utilities/style.py` as module-level names — single source of truth
- [x] **STY-02**: Backend toggle (set_backend/get_backend/get_active_style) and style dict pattern removed from plotting.py
- [x] **STY-03**: Seaborn import and SEABORN_STYLE removed — one canonical dark style only
- [x] **STY-04**: plotting.py primitives (create_figure_scaffold, render_top_row, render_spectrogram_row) use style.py constants directly
- [x] **STY-05**: Style system designed for reusability — works for comparison grid, per-signal figures, future documentation figures
- [x] **STY-06**: Comparison grid column titles have visible top margin (increased pad from 8 to 20+)
- [x] **STY-07**: Comparison grid column titles centered over spectrogram columns

### Pipeline Timing

- [x] **TIM-01**: @timed decorator in src/subshader/utils/timing.py wraps pipeline methods with perf_counter
- [x] **TIM-02**: All wavelet pipeline stages (class_specific_cwt, normalize_by_scale, compute_mag, discard_unreliable_coefs, extract_hop_center, downsample) decorated with @timed
- [x] **TIM-03**: cwt_timed() duplicate code path removed from wavelet.py — timing via instance attributes only
- [x] **TIM-04**: research/timing.py reads timing from @timed instance attributes, not from parallel pipeline reimplementation

### Research Toolkit v2

- [x] **RTK2-01**: benchmark.py renamed to test_suite.py as single CLI entry point
- [x] **RTK2-02**: --seaborn flag removed from CLI (seaborn backend killed with style consolidation)
- [x] **RTK2-03**: --test flag runs pytest on research/tests/ (replaces --unit-tests running on src/)
- [x] **RTK2-04**: wav_export.py moved from research/ root to research/utilities/wav_export.py
- [x] **RTK2-05**: Historical directories (ants, docs, gpu_basics, misc, python) archived to research/archive/
- [x] **RTK2-06**: All test files migrated from src/ to research/tests/ mirroring src/ structure
- [x] **RTK2-07**: comparison.py extracted from figures.py with generate_comparison_grid() and generate_timing_bar_chart()
- [x] **RTK2-08**: COMPARISON_METHODS extensible config list in comparison.py — adding a method is one list append
- [x] **RTK2-09**: figures.py (ReadmeFigures) uses style.py constants — no hardcoded visual values

### Frequency Range Configuration

- [x] **FREQ-01**: WaveletConfig root_note_a0_hz (27.5Hz) and num_octaves (10) confirmed as configurable parameters with existing Nyquist trimming — no new code needed

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
| PIPE-01 | Phase 2 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| AUDIO-01 | Phase 3 | Complete |
| AUDIO-02 | Phase 3 | Complete |
| DOCS-01 | Phase 5 | Complete |
| DOCS-02 | Phase 5 | Complete |
| DOCS-03 | Phase 5 | Complete |
| DOCS-04 | Phase 5 | Complete |
| DOCS-05 | Phase 5 | Complete |
| DOCS-06 | Phase 5 | Complete |
| QUAL-01 | Phase 1 | Complete |
| QUAL-02 | Phase 2 | Complete |
| QUAL-03 | Phase 1 | Complete |
| INST-01 | Phase 4 | Pending |
| INST-02 | Phase 4 | Pending |
| INST-03 | Phase 4 | Pending |
| FIG-01 | Phase 6 | Complete |
| FIG-02 | Phase 6 | Complete |
| FIG-03 | Phase 6 | Complete |
| FIG-04 | Phase 6 | Complete |
| FIG-05 | Phase 6 | Complete |
| FIG-06 | Phase 6 | Complete |
| RTK-01 | Phase 5.1 | Complete |
| RTK-02 | Phase 5.1 | Complete |
| RTK-03 | Phase 5.1 | Complete |
| RTK-04 | Phase 5.1 | Complete |
| RTK-05 | Phase 5.1 | Complete |
| RTK-06 | Phase 5.1 | Complete |
| STY-01 | Phase 7 | Complete |
| STY-02 | Phase 7 | Complete |
| STY-03 | Phase 7 | Complete |
| STY-04 | Phase 7 | Complete |
| STY-05 | Phase 7 | Complete |
| STY-06 | Phase 7 | Complete |
| STY-07 | Phase 7 | Complete |
| TIM-01 | Phase 7 | Complete |
| TIM-02 | Phase 7 | Complete |
| TIM-03 | Phase 7 | Complete |
| TIM-04 | Phase 7 | Complete |
| RTK2-01 | Phase 7 | Complete |
| RTK2-02 | Phase 7 | Complete |
| RTK2-03 | Phase 7 | Complete |
| RTK2-04 | Phase 7 | Complete |
| RTK2-05 | Phase 7 | Complete |
| RTK2-06 | Phase 7 | Complete |
| RTK2-07 | Phase 7 | Complete |
| RTK2-08 | Phase 7 | Complete |
| RTK2-09 | Phase 7 | Complete |
| FREQ-01 | Phase 7 | Complete |

**Coverage:**
- v1 requirements: 47 total
- Mapped to phases: 47
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-03-27 — Phase 7 visual style, timing, and research toolkit v2 requirements added*
