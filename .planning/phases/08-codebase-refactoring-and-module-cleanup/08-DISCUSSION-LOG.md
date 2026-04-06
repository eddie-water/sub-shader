# Phase 8: Codebase Refactoring and Module Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 08-codebase-refactoring-and-module-cleanup
**Areas discussed:** Audio module merge, Module & class renaming, Orchestrator simplification, DSP class hierarchy, Config system, Research suite cleanup, Test suite design, Assets organization

---

## Audio Module Merge

| Option | Description | Selected |
|--------|-------------|----------|
| AudioManager facade | New class wraps AudioInput + AudioPlayer | |
| Merge into single class | Combine all audio logic into one class | |
| Keep separate, rename directory | AudioInput and AudioPlayer stay separate | |

**User's choice:** Facade pattern — but not called "AudioManager." User wanted a name indicating "source of audio + controls/syncs playback."

**Follow-up — Name:**

| Option | Description | Selected |
|--------|-------------|----------|
| AudioSource | Emphasizes where audio comes from | |
| AudioStream | Emphasizes real-time flow | ✓ |
| AudioEngine | Broader, implies self-contained subsystem | |
| AudioPipeline | Matches project's pipeline language | |

**User's choice:** AudioStream. Two internal submodules: reader (file I/O) and player (sync).

---

## Config System

Extended discussion over multiple turns. Three approaches presented:

| Option | Description | Selected |
|--------|-------------|----------|
| File-discovered, config stores it (mutable) | AudioStream writes sample_rate back into config | ✓ |
| File-discovered, passed explicitly | Caller threads sample_rate manually | |
| Config immutable, pipeline builder wires it | Factory function handles dependency chain | |

**User's choice:** Mutable config (Option 1). "As long as it is simple, easily overridable and consistent."

**User's note:** "I like the idea of a base class that stores the common configs/params, and then the modules that need more inherit the base and expand on it intelligently."

Design doc created: `config-design.md` with 7 usage contexts (runtime, timing, comparison, per-signal figures, stub layouts, stub pywavelet, custom test).

---

## Module & Class Renaming

| Decision | Before | After |
|----------|--------|-------|
| DSP directory | `dsp/` | `dsp/` (kept) |
| Viz directory | `viz/` | `renderer/` |
| Plotter class | Plotter/ShaderPlot | Renderer |

**User's note:** "The subclasses can have their appropriate naming conventions so long as it makes sense from a high level description."

---

## DSP Class Hierarchy

User wanted to discuss how to organize the DSP module as a whole — not just rename classes but restructure with a facade.

**Decision:** `dsp.py` ABC with `pre()`, `transform()`, `post()`. Each backend inherits and defines its own stages. STFT promoted from research utility to `dsp/stft.py`. PyWavelet extracted to `dsp/pywavelet.py`.

**Backend instantiation:**

| Option | Description | Selected |
|--------|-------------|----------|
| Direct instantiation | `GpuCWT(config)`, `STFT(config)` — classes inherit ABC | ✓ |
| Factory parameter | `DSP(config, backend="gpu_cwt")` — string-to-class registry | |

**User's choice:** Option 1 — "more like a style preference" but preferred explicit class instantiation.

Design doc created: `dsp-design.md`.

---

## Orchestrator Simplification

Discussion clarified that orchestrator simplification IS the config refactor — not a separate concern. With mutable config pattern, init goes from ~20 lines of manual wiring to 3 lines.

**Decision:** Extract SubShader class from `__main__.py` to `pipeline.py`. Entry point becomes ~15 lines.

**User's request:** "I'd like main to almost look like pseudocode — keep it high level, no weird names or sections, minimal comments."

Design doc created: `orchestrator-design.md` with mermaid flow diagram and before/after directory structure.

**Open question noted:** Sleep/yield in render loop — revisit 1ms sleep call and its implications during implementation.

---

## Research Suite Cleanup + Test Suite Design

Extended discussion. User identified that test_suite.py was confusing and wanted to step back and define what the test suite should be.

**Modes decided:**

| Mode | Purpose |
|------|---------|
| `--timing` | Profile pipeline end-to-end, write timestamped results file |
| `--test` | Run pytest |
| `--compare-methods` | Per-signal method comparison figure (STFT, PyWavelet, SubShader) |
| `--figures` | Generate all documentation images |

**Key decisions:**
- `--compare-methods --input-signal "path/to/file.wav"` interface
- Default runs all registry signals, `--input-signal` narrows to one
- DAW reference row gracefully stubs if image missing
- `--figures` reuses `--compare-methods` code, places three figures side by side in README
- 5×3 grid utility kept but not README default
- `--stub` stubs PyWavelet, outputs to `stubs/` dir with `_STUB` suffix
- `@timed` on all pipeline stages, not just DSP
- Timing template file for customizable output format
- Signal registry in `signals.py` — extensible list

Design doc created: `test-suite-design.md`.

---

## Assets Organization

User wanted assets to indicate their consumers — "I can't tell what is meant for what."

**Decision:** `reference/` (committed inputs) and `generated/` (created by test_suite.py) as base subdirs for both audio and images. Structure mirrors test suite modes.

**Key decisions:**
- Default runtime audio changed to `beltran_sc_rip.wav`
- `assets/timing/` for timestamped timing results (format: `timing_YYYY-MM-DD_HH-MM-SS.txt`)
- `assets/archive/` for all unused files (nothing deleted)
- `assets/plots/` stays for architecture diagrams

---

## Claude's Discretion

- AntsWavelet flattening strategy
- `@timed` attribute naming and thread safety
- `timing_template.txt` exact format
- `__init__.py` barrel exports
- `font_showcase_20.py` disposition
- `AudioStream.next_chunk()` blocking/waiting implementation
