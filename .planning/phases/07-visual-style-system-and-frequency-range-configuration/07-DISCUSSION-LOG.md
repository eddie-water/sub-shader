# Phase 7: Visual Style System and Frequency Range Configuration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 07-visual-style-system-and-frequency-range-configuration
**Areas discussed:** Style consolidation scope, Comparison grid header fix, Frequency range bounds, Style theming, Research toolkit architecture, Timing architecture, Research toolkit file organization, Style.py API design, Comparison method extensibility, Unit test placement

---

## Style Consolidation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| research/utilities/style.py | New dedicated module next to plotting.py | ✓ |
| research/utilities/constants.py | Expand existing constants.py | |
| src/subshader/style.py | Main package style module | |

**User's choice:** research/utilities/style.py
**Notes:** User clarified: "I want all these things to be easily configurable wherever they're used" — full centralization, no local style values anywhere.

---

## Comparison Grid Header Fix

| Option | Description | Selected |
|--------|-------------|----------|
| Column titles need more top margin | Titles too close to top edge | |
| Column titles aren't centered | Misaligned relative to columns | |
| Both margin and centering issues | Both problems need fixing | ✓ |

**User's choice:** Both margin and centering issues

---

## Frequency Range Bounds

| Option | Description | Selected |
|--------|-------------|----------|
| Config-level min/max Hz bounds | Add min_freq_hz and max_freq_hz | |
| Octave count + root note | Keep current configurable pattern | ✓ |
| Preset profiles | Named presets (full/balanced/fast) | |

**User's choice:** Root + octave count, keep A0 (27.5Hz) default
**Notes:** User asked about sub-bass audibility. After discussion of 20-60Hz range and subwoofer relevance, decided to keep sub-bass in default range. User confirmed Nyquist trimming already exists in `_generate_chromatic_scale()` at wavelet.py line 127.

---

## Style Theming

| Option | Description | Selected |
|--------|-------------|----------|
| One canonical style | Kill theme switching, single dark style | ✓ |
| Keep theme switching | Maintain DEFAULT_STYLE/SEABORN_STYLE | |
| One default + override mechanism | Single style with dict merge overrides | |

**User's choice:** One canonical style

---

## Timing Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| @timed decorators on methods | Simple, no drift, ~0.2% overhead | ✓ |
| External wrappers + drift test | Production code stays pure, duplication risk | |
| pytest-benchmark | Standard tooling, CI-friendly | |

**User's choice:** @timed decorators
**Notes:** Extended discussion about overhead (1μs per call on 500μs STFT = 0.2%), how external wrappers work, how decorators work, and Python ecosystem best practices. User asked about "benchmark concern in production code" — clarified it's just the @timed line above each method, no body changes.

---

## Research Toolkit File Organization

**User's choice:** Custom organization described by user
**Notes:** User defined four concerns:
1. test_suite.py — dispatcher (--test, --timing, --comparison, --figures)
2. timing.py — thin pipeline profiler
3. comparison.py — method figures + timing
4. utilities/ — reusable library (style, plotting, signals, wav_export, printing, timing decorator, dsp_helpers)
5. archive/ — ants, docs, gpu_basics, misc, python moved out

---

## Dispatcher Name

| Option | Description | Selected |
|--------|-------------|----------|
| toolkit.py | "This is the research toolkit" | |
| harness.py | "Test/benchmark harness" | |
| benchmark.py | Keep current name | |
| test_suite.py | User's choice via "Other" | ✓ |

**User's choice:** test_suite.py

---

## Style.py API Design

| Option | Description | Selected |
|--------|-------------|----------|
| Module constants (style.BG_COLOR) | Flat constants, autocomplete-friendly | ✓ |
| Dataclass (style.grid.bg_color) | Grouped by concern, structured | |
| Dict (STYLE['bg_color']) | Current pattern, easy to merge | |

**User's choice:** Module constants

---

## Comparison Method Extensibility

| Option | Description | Selected |
|--------|-------------|----------|
| Method registry pattern | Auto-discovery via registration | |
| Config list of methods | Explicit list: [{name, function, label}] | ✓ |
| Keep current manual approach | Each method manually wired | |

**User's choice:** Config list of methods

---

## Unit Test Placement

| Option | Description | Selected |
|--------|-------------|----------|
| src/ unit + research/tests/ integration | Split by test type | |
| Move all to research/tests/ | Centralize everything | ✓ |
| Keep colocated in src/ | All tests next to source | |

**User's choice:** Move all to research/tests/, mirror src/ structure

---

## Claude's Discretion

- @timed decorator implementation details
- style.py internal organization
- Exact gridspec adjustments for grid headers
- test_suite.py CLI parsing approach
- Seaborn backend removal strategy

## Deferred Ideas

- pytest-benchmark for CI — separate phase
- .mplstyle file approach — if multi-project needed
- Auto-detect audio frequency content — too complex
- DSP.md foundation figures — Phase 5 plan 05-04 scope
