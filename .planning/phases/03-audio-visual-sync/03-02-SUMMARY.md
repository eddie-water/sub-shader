---
phase: 03-audio-visual-sync
plan: 02
subsystem: audio
tags: [sounddevice, argparse, audio-sync, cwt, render-loop]

# Dependency graph
requires:
  - phase: 03-01
    provides: AudioPlayer class with get_playback_sample(), has_looped(), start(), stop()
provides:
  - Audio-clock-driven render loop in SubShader orchestrator
  - CLI argument parsing for audio file path (argparse)
  - AudioPlayer wired into SubShader.__init__ and loop()
  - Clean audio shutdown on window close (audio_player first in cleanup)
  - Seamless loop reset via has_looped() / clear_loop_event()
affects: [04-readme-documentation, future-phases]

# Tech tracking
tech-stack:
  added: [argparse]
  patterns:
    - Audio device clock as single source of truth for render timing
    - Yield-on-not-ready (1ms sleep) instead of fixed sleep in render loop
    - Frame-skip catch-up: seek AudioInput.file_pos to current audio position each iteration

key-files:
  created: []
  modified:
    - src/subshader/__main__.py
    - tests/test_gpu_fallback.py

key-decisions:
  - "Audio-clock-driven loop: render loop checks get_playback_sample() each iteration and seeks AudioInput.file_pos to match; audio device is single source of truth"
  - "1ms yield when audio clock has not advanced past next chunk boundary; avoids busy-wait without introducing fixed frame delay"
  - "Frame-skip logic: target_sample = (playback_pos // hop_size) * hop_size ensures render always shows the most recent chunk if render falls behind"
  - "audio_player.stop() called first in cleanup() before plotter/wavelet to prevent orphaned playback after window close"
  - "test_gpu_fallback updated to mock AudioPlayer and return real numpy array from get_entire_audio() — mock AudioInput was causing TypeError on ndim comparison"

patterns-established:
  - "Audio-clock-driven render loop: check playback_pos > next_expected_sample, yield 1ms if not ready, seek and render if ready"
  - "Loop reset detection: has_looped() + clear_loop_event() pattern for resetting visualization state on audio wrap"

requirements-completed: [AUDIO-01, AUDIO-02]

# Metrics
duration: 10min
completed: 2026-03-22
---

# Phase 03 Plan 02: Wire AudioPlayer into SubShader Orchestrator Summary

**Audio-clock-driven render loop: SubShader now uses sounddevice playback position to drive CWT frame timing, with CLI arg parsing and clean audio shutdown on window close**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-22T01:07:38Z
- **Completed:** 2026-03-22T01:09:21Z
- **Tasks:** 2 (1 auto + 1 checkpoint auto-approved)
- **Files modified:** 2

## Accomplishments
- `__main__.py` rewritten with audio-clock-driven loop — `time.sleep(0.1)` debug artifact removed
- `argparse` CLI arg lets users run `python -m subshader demo.wav` directly
- AudioPlayer wired into `SubShader.__init__` and `loop()` — playback position is the render clock
- Seamless audio loop reset via `has_looped()` / `clear_loop_event()` — visualization resets when file wraps
- Audio stops immediately on window close (`audio_player.stop()` called first in cleanup)
- All 35 tests pass after updating `test_gpu_fallback` to mock `AudioPlayer`

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire AudioPlayer into SubShader orchestrator with CLI arg and sync loop** - `7e28782` (feat)
2. **Task 2: Verify audio-visual sync quality** - auto-approved checkpoint (no code changes)

**Plan metadata:** (docs commit — created after this summary)

## Files Created/Modified
- `src/subshader/__main__.py` — Rewrote loop() with audio-clock-driven design, added argparse, AudioPlayer init, cleanup ordering
- `tests/test_gpu_fallback.py` — Updated all three GPU tests to mock AudioPlayer and return real numpy from get_entire_audio()

## Decisions Made
- Audio device clock is the single source of truth for render timing (D-06 from RESEARCH.md)
- 1ms yield when audio clock has not advanced — avoids busy-wait without adding fixed frame latency
- Frame-skip logic aligns AudioInput.file_pos to current audio position each iteration — if render falls behind, it skips to the most recent chunk rather than playing old frames
- `audio_player.stop()` called first in cleanup to guarantee no orphaned playback after window close

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_gpu_fallback test failure caused by AudioPlayer introduction**
- **Found during:** Task 1 verification (`pytest tests/ -x`)
- **Issue:** Three GPU fallback tests mocked `AudioInput` but not `AudioPlayer`. When `SubShader.__init__` called `get_entire_audio()` on the mock, it returned a `MagicMock` — `AudioPlayer.__init__` then called `audio_data.ndim > 1` which raised `TypeError: '>' not supported between instances of 'MagicMock' and 'int'`
- **Fix:** Added `@patch('subshader.__main__.AudioPlayer')` decorator to all three tests; added helper `_make_mock_audio_input()` that returns a real numpy array from `get_entire_audio()` so AudioPlayer receives valid data even when the test doesn't mock AudioPlayer
- **Files modified:** `tests/test_gpu_fallback.py`
- **Verification:** `pytest tests/ -x` — 35 passed, 0 failed
- **Committed in:** `7e28782` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix)
**Impact on plan:** The fix was necessary for test suite health. The bug was directly caused by the Task 1 changes (introducing AudioPlayer into __init__). No scope creep.

## Issues Encountered
None — the test failure was caught and fixed within Task 1.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Audio-visual sync pipeline is complete for file-based playback
- `python -m subshader` plays audio and renders CWT simultaneously
- `python -m subshader <file.wav>` uses CLI-specified audio file
- Audio loops seamlessly; window close stops audio immediately
- Ready for Phase 04 — README documentation

## Self-Check: PASSED

- src/subshader/__main__.py: FOUND
- tests/test_gpu_fallback.py: FOUND
- .planning/phases/03-audio-visual-sync/03-02-SUMMARY.md: FOUND
- Commit 7e28782: FOUND

---
*Phase: 03-audio-visual-sync*
*Completed: 2026-03-22*
