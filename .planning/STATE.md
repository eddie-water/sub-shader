---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Ready to execute
stopped_at: Completed 08-02-PLAN.md
last_updated: "2026-04-06T23:15:39.555Z"
last_activity: 2026-04-06
progress:
  total_phases: 10
  completed_phases: 7
  total_plans: 28
  completed_plans: 24
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** The visualization accurately tracks audio input in real time with minimal latency
**Current focus:** Phase 08 — codebase-refactoring-and-module-cleanup

## Current Position

Phase: 08 (codebase-refactoring-and-module-cleanup) — EXECUTING
Plan: 5 of 7

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01-codebase-hardening P01 | 2 | 2 tasks | 10 files |
| Phase 01-codebase-hardening P02 | 4 | 2 tasks | 5 files |
| Phase 02-cwt-pipeline-polish P01 | 8 | 2 tasks | 5 files |
| Phase 02-cwt-pipeline-polish P02 | 11 | 2 tasks | 3 files |
| Phase 03-audio-visual-sync P01 | 84 | 2 tasks | 3 files |
| Phase 03-audio-visual-sync P02 | 10 | 2 tasks | 2 files |
| Phase 05-documentation P02 | 2 | 2 tasks | 1 files |
| Phase 05-documentation P03 | 3 | 3 tasks | 3 files |
| Phase 05-documentation P01 | 7 | 2 tasks | 2 files |
| Phase 06 P01 | 10 | 3 tasks | 9 files |
| Phase 06-finalize-example-audio-and-comparison-figures-for-readme P03 | 3 | 2 tasks | 1 files |
| Phase 06 P02 | 8 | 1 tasks | 6 files |
| Phase 05.1-research-toolkit-restructure P01 | 10 | 2 tasks | 4 files |
| Phase 05.1-research-toolkit-restructure P02 | 25 | 2 tasks | 8 files |
| Phase 05.2-benchmark-timing-profiling-and-comparison-grid-polish P01 | 2 | 2 tasks | 3 files |
| Phase 05.2-benchmark-timing-profiling-and-comparison-grid-polish P02 | 1 | 1 tasks | 1 files |
| Phase 07-visual-style-system-and-frequency-range-configuration P02 | 4 | 2 tasks | 4 files |
| Phase 07-visual-style-system-and-frequency-range-configuration P01 | 15 | 2 tasks | 5 files |
| Phase 07-visual-style-system-and-frequency-range-configuration P03 | 217 | 2 tasks | 20 files |
| Phase 07-visual-style-system-and-frequency-range-configuration P04 | 15 | 2 tasks | 4 files |
| Phase 08-codebase-refactoring-and-module-cleanup P01 | 7 | 2 tasks | 7 files |
| Phase 08-codebase-refactoring-and-module-cleanup P04 | 6 | 2 tasks | 4 files |
| Phase 08-codebase-refactoring-and-module-cleanup P02 | 3 | 2 tasks | 6 files |
| Phase 08-codebase-refactoring-and-module-cleanup P03 | 12 | 1 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Hosted demo is v2 — v1 is locally installable only
- [Init]: File-based audio is fine for v1, live capture deferred to v2
- [Init]: Documentation scaffolded by Claude, authored by user in their own voice
- [Init]: GPU fallback belongs in DSP block instantiation, not benchmark code
- [Phase 01-codebase-hardening]: RuntimeError removed from GRACEFUL_EXCEPTIONS — was masking real errors; SubShaderException + KeyboardInterrupt is the correct scope
- [Phase 01-codebase-hardening]: gpu_available() uses lazy cupy import inside try/except — safe on CPU-only machines, never import-time crash
- [Phase 01-codebase-hardening]: AudioConfig default path fixed to assets/audio/daw/a2a3_a4_minor_scale.wav — __main__.py override was compensating for wrong default
- [Phase 01-codebase-hardening]: _validate_texture_data returns None on success (raises on failure) — callers no longer check return value
- [Phase 01-codebase-hardening]: SubShader.__init__ takes ProcessingConfig parameter — removes implicit global config dependency
- [Phase 02-cwt-pipeline-polish]: L1 kernel normalization applied at WaveletKernel construction — bias is structural so fixing at source is cleaner than post-hoc scale correction
- [Phase 02-cwt-pipeline-polish]: normalize_by_scale retained as no-op for interface compatibility with PyWavelet and future backends
- [Phase 02-cwt-pipeline-polish]: cwt_out_type field removed from WaveletConfig — confirmed zero references outside config.py
- [Phase 02-cwt-pipeline-polish]: AUDIO_POLYPHONIC constant updated to overlapping_A3_A4_A5.wav — polyphonic_audio_example.wav was empty 0-frame placeholder
- [Phase 02-cwt-pipeline-polish]: No intensity tracker tuning needed post-normalization — color range looks reasonable in regenerated figures
- [Phase 03-audio-visual-sync]: AudioPlayer stores _data as float32 — PortAudio callback layer expects float32; float64 causes silent type coercion
- [Phase 03-audio-visual-sync]: threading.Lock used for _current_frame — low-contention single-int read/write; queue overhead unnecessary
- [Phase 03-audio-visual-sync]: blocksize=0 in OutputStream — lets PortAudio choose optimal hardware buffer
- [Phase 03-audio-visual-sync]: Audio-clock-driven loop: audio device clock is single source of truth; render loop seeks AudioInput.file_pos to match get_playback_sample() each iteration
- [Phase 03-audio-visual-sync]: 1ms yield when audio clock not advanced avoids busy-wait; frame-skip to current position if render falls behind
- [Phase 03-audio-visual-sync]: audio_player.stop() called first in cleanup to prevent orphaned playback on window close
- [Phase 05-documentation]: Code examples in DSP.md extracted from actual source files (wavelet.py, config.py) — no illustrative stubs
- [Phase 05-documentation]: DSP scaffold uses 'properties' not 'features/patterns' before Section 7 — terminology ladder per discussion_summary.md
- [Phase 05-documentation]: README.md stream-of-consciousness passages flagged as REWRITE (not deleted) — preserves user intent while flagging for authoring
- [Phase 05-documentation]: numpy_vs_cupy_diff.png reference marked as MOVED in README.md — makes the decision visible in-place rather than silently deleting
- [Phase 05-documentation]: Chirp column uses ~215 frames from 10s target rather than capping at NUM_FRAMES — gives complete sweep
- [Phase 05-documentation]: Per-row vmax for each comparison grid column so each representation's dynamic range is independently visible
- [Phase 06]: Bouncing chirp uses CubicSpline in log-frequency space with peak/dip waypoints — matches user sketch naturally
- [Phase 06]: DPI=0 sentinel in generate_comparison_grid() means default naming; dpi>0 produces _Ndpi.png suffix regardless of stub_pywt
- [Phase 06]: Stub suffix only applies to DPI=0 (default) path; explicit --dpi N gets clean comparison_grid_Ndpi.png name
- [Phase 06-finalize-example-audio-and-comparison-figures-for-readme]: _STUB_PYWT suffix replaces _STUB so the stub scope is explicit in the filename
- [Phase 06]: comparison_grid.png copied from comparison_grid_200dpi.png — canonical README name; --dpi 200 always produces _200dpi suffix
- [Phase 06]: Timing bar chart generate_timing_bar_chart() added to benchmark.py with --timing-chart flag — chart can now be reproduced programmatically
- [Phase 06]: Timing bar chart moved from README.md to DSP.md Section 6 — detailed timing analysis belongs in implementation docs
- [Phase 05.1-01]: benchmark.py split into figures.py/timing.py/wav_export.py — all research/ modules use bare 'from utilities import' pattern matching existing CWD convention
- [Phase 05.1-02]: pyproject.toml pythonpath includes src/subshader so conftest helpers are importable in colocated test files without __init__.py
- [Phase 05.1-02]: test_kernel_energy_per_scale rewritten to verify L1 normalization invariant after Phase 2 — L1 norm ~1.0, L2 slope ~+0.5 vs pre-Phase-2 slope ~-0.5
- [Phase 05.1-02]: PyWavelet reliable range capped at bin 90 (~5 kHz) in test_pure_tone_peak_accuracy — above that pywt.cwt() aliases to wrong frequency bin
- [Phase 05.2]: cwt_timed() uses inline time.perf_counter in wavelet.py — production code must not import from research/utilities
- [Phase 05.2]: TimedSubShader now uses 8-method accumulator: get_chunk + 6 cwt sub-stages + push_frame
- [Phase Phase 05.2]: CuWavelet import deferred inside generate_comparison_grid() — avoids unconditional GPU import at module load
- [Phase Phase 05.2]: GPU CWT result discarded in --comparison — timing-table-only per D-07; no new figure rows added
- [Phase 07-02]: @timed placed on concrete overrides not abstract declarations — decorating abstract methods in Python ABCs does not wrap subclass implementations
- [Phase 07-02]: cwt_timed() removed; timing always available via _timing_*_ms instance attributes after each cwt() call
- [Phase 07]: style.py uses plain module-level names — no dicts, no dataclasses per D-03
- [Phase 07]: Backend toggle (set_backend/get_backend/get_active_style) removed — one canonical dark style only per D-04/D-05
- [Phase 07-03]: research/tests added to pythonpath so conftest.py plain helpers are importable via from conftest import
- [Phase 07-03]: cwt_timed tests rewritten to use @timed _timing_*_ms attributes — cwt_timed() removed in 07-02
- [Phase 07-04]: generate_comparison_grid and generate_timing_bar_chart extracted to comparison.py — separation of concerns between per-signal figures (ReadmeFigures) and method-vs-method comparison
- [Phase 07-04]: STUB_DPI = 100 added to style.py — stub_layouts intentionally renders at lower DPI for fast iteration; value lives in style rather than hardcoded
- [Phase 08]: CWTConfig chosen as default from get_default_config() — most param-rich subclass, correct for full pipeline
- [Phase 08]: ProcessingConfig = CWTConfig alias defers __main__.py migration to Plan 08-05 — callers still import without error
- [Phase 08]: Default file_path changed from assets/audio/daw/a2a3 to assets/audio/reference/beltran_sc_rip.wav per D-04
- [Phase 08]: Asset lifecycle: reference/ for committed input files, generated/ for test-suite outputs — asset ownership model settled
- [Phase 08]: AudioReader constructed before AudioPlayer in AudioStream — reader writes config.sample_rate, player reads it; construction order is load-bearing
- [Phase 08]: next_chunk() encapsulates hop-aligned seek + 1ms yield + frame-skip logic from __main__.py — pipeline.py can call audio.next_chunk() cleanly
- [Phase 08]: CWT base class absorbs Wavelet + AntsWavelet shared logic — 7-class wavelet.py hierarchy flattened to CWT + CpuCWT + GpuCWT in cwt.py
- [Phase 08]: WaveletConfig=CWTConfig alias added to config.py — preserves wavelet.py import-time compat until Plan 08-05 completes the migration

### Pending Todos

None yet.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260322-j2b | Fix overlap redundancy in CWT-to-plotter pipeline | 2026-03-22 | 019078b | [260322-j2b-fix-overlap-redundancy-in-cwt-to-plotter](./quick/260322-j2b-fix-overlap-redundancy-in-cwt-to-plotter/) |

### Roadmap Evolution

- Phase 6 added: Finalize example audio and comparison figures for README
- Phase 05.1 inserted after Phase 5: Research toolkit restructure (URGENT) — restructure monolithic benchmark.py into modular research toolkit before 05-04 figure generation
- Phase 05.2 inserted after Phase 5: Benchmark timing profiling and comparison grid polish (URGENT) — sub-stage timing for --timing, NumPy timing in --comparison, PyWavelet normalization, grid label layout
- Phase 7 added: Visual style system and frequency range configuration
- Phase 8 added: Codebase Refactoring and Module Cleanup

### Blockers/Concerns

- [Phase 2 ahead]: EGL headless rendering research flagged for v2 Phase work — not needed for v1
- [Phase 3]: Audio-visual sync latency target is ~100ms perceived lag; may require resolution reduction if not achievable at full resolution

## Session Continuity

Last activity: 2026-04-06
Stopped at: Completed 08-02-PLAN.md
Resume file: None
