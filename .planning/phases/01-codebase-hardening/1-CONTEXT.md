# Phase 1: Codebase Hardening - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

The pipeline fails loudly and correctly — no silent blank frames, no swallowed GPU errors, no hardcoded paths. GPU fallback moves from benchmark code into DSP block instantiation. Code in changed files stays clean and readable.

</domain>

<decisions>
## Implementation Decisions

### Frame failure behavior
- **D-01:** Frame failures crash the app with a clear error — no silent frame drops, no swallowed render exceptions
- **D-02:** Texture validation failures raise an exception instead of silently returning stale data
- **D-03:** GPU cleanup errors use the logger, not print()

### Audio path handling
- **D-04:** Audio file path lives in AudioConfig as the config default — remove the hardcoded override in `__main__.py`
- **D-05:** Config remains overridable before SubShader instantiation (existing pattern: modify `config.audio.file_path` before calling `SubShader(config)`)
- **D-06:** Missing audio file raises AudioFileNotFoundError immediately at init — not a silent blank visualization

### GPU fallback UX
- **D-07:** GPU unavailability detected at init, logged as a clear console message: "GPU unavailable, running on NumPy — expect slower performance"
- **D-08:** Session continues on NumPy path after the message — no crash, no re-check mid-session
- **D-09:** GPU fallback logic lives in DSP block instantiation (`SubShader.__init__`), not in benchmark code

### Claude's Discretion
- Exception hierarchy cleanup (dedup classes, fix base classes, narrow RuntimeError in GRACEFUL_EXCEPTIONS)
- Dead code removal (unused COI mask methods, dead CuPy import in wavelet_kernel.py)
- CuPy import guarding strategy (lazy imports, conditional imports, or restructured modules)
- Cleanup guard safety (`hasattr` checks in `SubShader.cleanup()`)
- Logger configuration timing (console_output=False at module level is currently hiding startup errors)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<canonical_refs>
## Canonical References

No external specs — requirements are fully captured in decisions above and in:

### Project-level
- `.planning/REQUIREMENTS.md` — PIPE-02, PIPE-03, QUAL-01, QUAL-03 define the phase requirements
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (4 items) that must be TRUE after this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Critical Issues Found (scout results)

- `src/subshader/__main__.py:23,64` — Hard `from subshader.dsp.wavelet import CuWavelet`, no fallback. CuWavelet constructed unconditionally
- `src/subshader/dsp/wavelet.py:24-25` — Top-level `import cupy` blocks NumPy-only usage of the entire module
- `src/subshader/dsp/wavelet_kernel.py:6` — Dead `import cupy as cp` (never used, forces CuPy dependency)
- `src/subshader/audio/audio_input.py:33-46` — Duplicate `AudioFileNotFoundError` and `EndOfAudioException` with wrong base class (plain Exception, not SubShaderException)
- `src/subshader/viz/plotter.py:40-45` — Duplicate `WindowCloseException` with wrong base class
- `src/subshader/viz/plotter.py:486-502` — Render exception caught and swallowed, frame silently dropped
- `src/subshader/viz/plotter.py:403-435` — Texture validation returns None on error, caller silently skips frame
- `src/subshader/__main__.py:45` — Hardcoded audio path override at module level
- `src/subshader/config.py:59` — Hardcoded default audio path in config

### Reusable Assets
- `research/utilities/constants.py:50` — `gpu_available()` function ready to promote into `src/subshader/utils/`
- `src/subshader/exceptions.py` — Exception hierarchy exists, just needs dedup and consolidation
- `src/subshader/utils/logging.py` — Logger infrastructure already in place

### Established Patterns
- Dataclass configs with `validate()` method — new validation can follow this pattern
- `ExceptionReporter.report()` dispatches on exception type — new exceptions should have `log_level` attribute
- `GRACEFUL_EXCEPTIONS` tuple controls main loop catch behavior

### Integration Points
- `SubShader.__init__()` in `__main__.py` — where GPU detection and wavelet selection must happen
- `SubShader.loop()` — main loop catches `GRACEFUL_EXCEPTIONS`, needs review after hierarchy cleanup
- `SubShader.cleanup()` — needs `hasattr` guards for partial init failures

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-codebase-hardening*
*Context gathered: 2026-03-21*
