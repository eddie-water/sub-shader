---
phase: 01-codebase-hardening
verified: 2026-03-21T16:30:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 01: Codebase Hardening Verification Report

**Phase Goal:** The pipeline fails loudly and correctly — no silent blank frames, no swallowed GPU errors, no hardcoded paths
**Verified:** 2026-03-21T16:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Truths are drawn from the combined `must_haves` sections of PLAN 01 and PLAN 02.

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Exception classes defined once in exceptions.py — no duplicates in audio_input.py or plotter.py | VERIFIED | `grep class AudioFileNotFoundError src/subshader/audio/audio_input.py` returns nothing; plotter.py imports from exceptions |
| 2  | audio_input.py raises canonical AudioFileNotFoundError from subshader.exceptions | VERIFIED | Line 22: `from subshader.exceptions import AudioFileNotFoundError, EndOfAudioException`; line 53 raises it |
| 3  | gpu_available() exists in src/subshader/utils/gpu.py and returns bool | VERIFIED | File exists, `def gpu_available() -> bool` confirmed; test passes |
| 4  | Config default audio path points to valid file without a hardcoded override in __main__.py | VERIFIED | AudioConfig.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav" at line 59; no `config.audio.file_path =` assignment in __main__.py |
| 5  | wavelet_kernel.py does not import cupy at module level | VERIFIED | `grep -n "import cupy" wavelet_kernel.py` returns nothing |
| 6  | pytest runs and test_exceptions passes | VERIFIED | 21/21 tests pass including all 11 exception tests |
| 7  | GPU unavailability at startup is detected, logged, and session continues on NumPy | VERIFIED | __main__.py lines 64-68: `if gpu_available(): wavelet_class = CuWavelet else: log.warning(...); wavelet_class = NpWavelet` |
| 8  | GPU fallback logic lives in SubShader.__init__, not in benchmark code | VERIFIED | Logic is in `__main__.py` SubShader.__init__; no benchmark involvement |
| 9  | CuPy can be absent and wavelet.py still imports successfully | VERIFIED | Lines 24-29: try/except guard with `_CUPY_AVAILABLE` flag; `_CUPY_AVAILABLE = False` on exception |
| 10 | Frame render failures crash the app with a clear error instead of being silently swallowed | VERIFIED | plotter.py render_graphic lines 487-489: `except Exception as e: log.error(f"Render exception: {e}"); raise` |
| 11 | Texture validation failures raise ValueError instead of silently returning None | VERIFIED | `_validate_texture_data` (line 393) returns `-> None` and raises `ValueError` on all 5 error branches; test confirms |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/subshader/utils/gpu.py` | gpu_available() function | VERIFIED | Exists, 16 lines, lazy cupy import, returns bool |
| `src/subshader/dsp/wavelet.py` | Guarded CuPy imports with _CUPY_AVAILABLE flag | VERIFIED | Lines 24-29: try/except with flag; CuPyWavelet.__init__ guard at line 577 |
| `src/subshader/__main__.py` | GPU detection and wavelet class selection at init | VERIFIED | Lines 64-68: conditional selection using gpu_available() |
| `src/subshader/viz/plotter.py` | Raising instead of swallowing render/validation errors | VERIFIED | render_graphic re-raises; _validate_texture_data raises ValueError |
| `tests/conftest.py` | Shared test fixtures | VERIFIED | project_root and valid_audio_path fixtures present |
| `tests/test_exceptions.py` | Exception hierarchy validation tests | VERIFIED | 4 test classes, 11 tests, all passing |
| `tests/test_gpu_fallback.py` | GPU fallback behavior tests | VERIFIED | TestGpuAvailable (2 tests), TestGpuFallback (3 tests), all passing |
| `tests/test_plotter.py` | Texture validation raise tests | VERIFIED | TestValidateTextureData (4 tests), TestRenderGraphic (1 test), all passing |
| `tests/__init__.py` | Package marker | VERIFIED | Exists |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/subshader/audio/audio_input.py` | `src/subshader/exceptions.py` | `from subshader.exceptions import AudioFileNotFoundError` | WIRED | Line 22 confirms import; identity test (TestNoDuplicateExceptions) passes |
| `src/subshader/utils/gpu.py` | cupy | lazy import inside try/except | WIRED | Lines 11-14: `import cupy as cp` inside try block |
| `src/subshader/__main__.py` | `src/subshader/utils/gpu.py` | `from subshader.utils.gpu import gpu_available` | WIRED | Line 24 confirmed |
| `src/subshader/__main__.py` | `src/subshader/dsp/wavelet.py` | `wavelet_class = CuWavelet if gpu_available() else NpWavelet` | WIRED | Lines 23 and 64-68 confirmed |
| `src/subshader/viz/plotter.py` | main loop | re-raised exceptions propagate to SubShader.loop() | WIRED | `raise` at line 489; SubShader.loop() does not catch; propagates to main() |
| `src/subshader/utils/__init__.py` | `src/subshader/utils/gpu.py` | `from .gpu import gpu_available` | WIRED | Line 11 of __init__.py; gpu_available in __all__ |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| QUAL-01 | 01-01, 01-02 | Clean, readable code — descriptive function names, well-factored helpers, minimal comments | SATISFIED | Exception hierarchy deduplicated; gpu_available() is a named helper; validate-and-raise pattern factored cleanly |
| QUAL-03 | 01-01, 01-02 | Existing readability maintained — no unnecessary refactoring | SATISFIED | Changes are surgical: imports fixed, dead code removed, guards added without restructuring existing logic |
| PIPE-02 | 01-02 | GPU fallback lives in DSP block instantiation, not benchmark code — auto-detects GPU failure | SATISFIED | SubShader.__init__ lines 63-68 implement detection and fallback; no benchmark code path involved |
| PIPE-03 | 01-02 | GPU availability checked at init — if unavailable, run on NumPy path for the session | SATISFIED | gpu_available() called once in __init__; wavelet_class selected for entire session; TestGpuFallback confirms |

All 4 requirement IDs declared across both plans are accounted for and satisfied. REQUIREMENTS.md confirms all four are marked Phase 1 Complete.

**Orphaned requirements check:** No additional Phase 1 IDs found in REQUIREMENTS.md beyond PIPE-02, PIPE-03, QUAL-01, QUAL-03.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/subshader/viz/plotter.py` | 144 | `TODO-36 : self.ctx #3?` | Info | Pre-existing TODO, not introduced by this phase; unrelated to hardening goal |

No blockers. The single TODO is pre-existing, non-blocking, and unrelated to phase scope.

### Human Verification Required

None. All phase goals are mechanically verifiable:
- Exception identity (same class object) — confirmed by tests
- GPU fallback branching — confirmed by mock tests
- raise propagation — confirmed by test and code inspection
- Config defaults — confirmed by code inspection
- Test pass/fail — confirmed by running pytest

### Gaps Summary

No gaps. All 11 observable truths are verified against the actual codebase, not just claimed in summaries.

---

_Verified: 2026-03-21T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
