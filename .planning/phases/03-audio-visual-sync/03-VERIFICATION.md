---
phase: 03-audio-visual-sync
verified: 2026-03-21T00:00:00Z
status: human_needed
score: 5/6 must-haves verified automatically
re_verification: false
human_verification:
  - test: "Audio and visualization play simultaneously — not sequentially"
    expected: "Running `python -m subshader` plays audio through speakers while the CWT shader renders frames in the same window session, both starting at the same moment"
    why_human: "Requires audio hardware. Cannot verify concurrent audio+render from a static grep; only runtime confirms the OutputStream callback and render loop are both active simultaneously."
  - test: "Transient events appear in visualization within ~100ms of being heard"
    expected: "A sharp note attack or drum hit in the audio file produces a visible bright region in the CWT output within approximately one render frame of being audible"
    why_human: "Perceptual latency test. The code structure is correct (1ms yield, audio-clock-driven seek) but whether the actual perceived lag stays under 100ms depends on the audio device, OS scheduler, and GPU render latency — none of which can be measured via grep."
  - test: "No drift over 60 seconds of playback"
    expected: "Visual transients remain aligned with their audio counterparts throughout a 60+ second clip, with no progressive lead or lag accumulating"
    why_human: "Clock drift is a timing property that emerges over time. The implementation anchors the render clock to the audio device clock each iteration (target_sample = (playback_pos // hop_size) * hop_size), which eliminates accumulated drift by design — but this must be confirmed by a human observer."
  - test: "Closing the window stops audio immediately"
    expected: "Audio ceases the moment the window is closed, with no orphaned playback continuing in the background"
    why_human: "Requires audio hardware to observe. cleanup() calls audio_player.stop() first in the ordering, but only a live run confirms the device actually releases."
  - test: "Audio loops seamlessly when file ends"
    expected: "When the audio file finishes, playback restarts from the beginning and the visualization resets without a pause, glitch, or crash"
    why_human: "Perceptual continuity test. The callback wraps _current_frame at EOF and sets _loop_event; the render loop resets file_pos. Whether the crossfade sounds seamless requires listening."
---

# Phase 3: Audio-Visual Sync — Verification Report

**Phase Goal:** Users can play an audio file and watch the CWT visualization track it in real time with no perceptible drift
**Verified:** 2026-03-21
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Success Criteria (from ROADMAP.md)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running the tool with an audio file argument plays audio and renders CWT simultaneously — not sequentially | ? HUMAN | Loop starts `audio_player.start()` then enters while-loop rendering CWT; concurrency is correct by code inspection but requires live audio hardware to confirm |
| 2 | Transient events appear in visualization within ~100ms of being heard | ? HUMAN | 1ms yield + audio-clock-driven seek eliminates fixed lag; actual perceptual latency requires human measurement |
| 3 | Visualization does not drift ahead or behind audio over 60-second playback | ? HUMAN | Render clock re-anchors to audio position every iteration via `target_sample = (playback_pos // hop_size) * hop_size` — drift cannot accumulate by design, but must be confirmed by a human observer |

All three truths have structurally correct implementations. None can be verified without running the application with real audio hardware.

**Automated score: 5/6 must-haves from PLAN frontmatter verified**

---

### Must-Haves: Plan 01

| Truth | Status | Evidence |
|-------|--------|----------|
| AudioPlayer starts a sounddevice OutputStream and plays audio from a float32 array | VERIFIED | `sd.OutputStream(... callback=self._callback)` at line 122–130 of `audio_player.py`; `_data` stored as `float32` at line 48 |
| AudioPlayer.get_playback_sample() returns current playback position via a thread-safe lock | VERIFIED | Lines 90–97 of `audio_player.py`; returns `self._current_frame` under `self._lock` |
| AudioPlayer loops seamlessly when reaching end of audio buffer | VERIFIED | Wrap logic at lines 76–84 of `audio_player.py`; `_current_frame = remaining` on overflow |
| AudioPlayer.stop() stops the stream and releases the audio device | VERIFIED | Lines 138–151 of `audio_player.py`; `_stream = None` in `finally` block |
| sounddevice is a declared dependency in pyproject.toml | VERIFIED | `pyproject.toml` line 15: `"sounddevice"` in dependencies list |

### Must-Haves: Plan 02

| Truth | Status | Evidence |
|-------|--------|----------|
| Running `python -m subshader demo.wav` plays audio and renders CWT simultaneously | ? HUMAN | Structural wiring is complete; requires live run to confirm |
| Running `python -m subshader` without arguments uses default config path | VERIFIED | `nargs="?"` + `default=None` in argparse; config.audio.file_path only overridden when arg provided (lines 199–208 of `__main__.py`) |
| Transient audio events appear in visualization within ~100ms | ? HUMAN | See above |
| Audio loops seamlessly when file ends, visualization resets and continues | ? HUMAN | Structural: `has_looped()` check resets `file_pos` and `next_expected_sample`; requires live run |
| Closing the window stops audio immediately with no orphaned playback | ? HUMAN | `audio_player.stop()` is first in `cleanup()` (line 160); requires live run |
| The time.sleep(0.1) debug artifact is removed from the main loop | VERIFIED | `grep time.sleep(0.1) __main__.py` returns no output; only `time.sleep(0.001)` (1ms yield) present |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/subshader/audio/audio_player.py` | AudioPlayer class with sounddevice OutputStream, callback, thread-safe position counter, loop support | VERIFIED | 152 lines; exports `AudioPlayer`; contains `_callback`, `get_playback_sample`, `has_looped`, `start`, `stop` |
| `pyproject.toml` | sounddevice dependency declaration | VERIFIED | Line 15: `"sounddevice"` present in dependencies list |
| `tests/test_audio_player.py` | Unit tests for AudioPlayer, min 50 lines | VERIFIED | 139 lines; 8 test functions; all 8 pass (`pytest tests/test_audio_player.py -v`: 8 passed) |
| `src/subshader/__main__.py` | CLI arg parsing, AudioPlayer creation, audio-clock-driven render loop, cleanup | VERIFIED | 221 lines; imports `argparse` and `AudioPlayer`; contains all required patterns |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/subshader/audio/audio_player.py` | `sounddevice.OutputStream` | `sd.OutputStream` in `start()` | WIRED | Line 122: `self._stream = sd.OutputStream(...)` with callback wired |
| `src/subshader/audio/audio_player.py` | `threading.Lock` | `self._lock` protecting `self._current_frame` | WIRED | Line 51: `self._lock = threading.Lock()`; used in `_callback` (lines 70, 82, 87) and `get_playback_sample` (line 96) |
| `src/subshader/__main__.py` | `src/subshader/audio/audio_player.py` | `AudioPlayer` instantiation in `__init__`, `get_playback_sample()` in `loop()` | WIRED | Line 68: `self.audio_player = AudioPlayer(...)`; line 121: `self.audio_player.get_playback_sample()` |
| `src/subshader/__main__.py` | `src/subshader/audio/audio_input.py` | `AudioInput.file_pos` seek based on audio clock position | WIRED | Lines 127, 139: `self.audio_input.file_pos = 0` and `= target_sample` |
| `src/subshader/__main__.py` | `argparse` | `argparse.ArgumentParser` for CLI audio file argument | WIRED | Line 17: `import argparse`; line 195: `argparse.ArgumentParser(...)` |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| AUDIO-01 | 03-01-PLAN.md, 03-02-PLAN.md | Audio playback and visualization are synced — file-based audio with real-time CWT rendering | NEEDS HUMAN | Structural wiring fully implemented; simultaneous playback + render requires live confirmation |
| AUDIO-02 | 03-01-PLAN.md, 03-02-PLAN.md | Audio-visual sync with minimal perceptible latency (<100ms perceived lag) | NEEDS HUMAN | `latency='low'` in OutputStream; 1ms yield avoids fixed frame lag; perceptual validation requires human |

No orphaned requirements: both AUDIO-01 and AUDIO-02 are claimed by both plans and both are addressed by concrete implementation.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

Scan covered `src/subshader/__main__.py` and `src/subshader/audio/audio_player.py`. No TODOs, FIXMEs, placeholder returns, `time.sleep(0.1)`, or stub patterns detected.

---

## Human Verification Required

### 1. Simultaneous Audio + Visual Playback

**Test:** Run `python -m subshader` from the project root with speakers or headphones connected.
**Expected:** Audio plays through the output device at the same moment the CWT shader window renders frames. Neither starts before the other.
**Why human:** Requires audio hardware. Code inspection confirms `audio_player.start()` is called before the while-loop, but whether the OS actually produces audio simultaneously with OpenGL frames cannot be verified from source alone.

### 2. Sub-100ms Transient Response (AUDIO-02)

**Test:** Play a file containing sharp transients (e.g., `assets/audio/daw/a2a3_a4_minor_scale.wav`). Listen for note attacks and watch the visualization.
**Expected:** Each note onset produces a visible bright region in the CWT output within approximately one render frame (~16ms at 60fps) of being audible — well within the 100ms target.
**Why human:** Perceptual latency depends on audio device buffer size, OS scheduler, and GPU render time. The code minimizes fixed delays (1ms yield, `blocksize=0`, `latency='low'`) but actual sub-100ms delivery requires a human ear to confirm.

### 3. No Drift Over 60 Seconds (AUDIO-01)

**Test:** Let the default audio file play for at least 60 seconds without intervention. Observe whether musical events that are audible at a specific moment in the audio appear in the visualization at that same moment throughout the duration.
**Expected:** Visual and audio remain perceptually locked throughout; no progressive lead or lag accumulates.
**Why human:** Drift is a cumulative timing property. The implementation re-anchors to the audio clock every iteration (eliminating drift by design), but only a live 60-second observation can confirm no edge case (xruns, system load) breaks the sync.

### 4. Immediate Audio Stop on Window Close

**Test:** Start `python -m subshader`, let audio play, then close the visualization window.
**Expected:** Audio ceases immediately — no continued playback after the window disappears.
**Why human:** Requires audio hardware. `cleanup()` calls `audio_player.stop()` first, but the actual device release must be confirmed by listening.

### 5. Seamless Audio Loop

**Test:** Let the audio file play to completion (or use a short file). Observe the transition from end-of-file back to the start.
**Expected:** The loop restart is smooth — no audible click, gap, or pause; the visualization continues without freezing or crashing.
**Why human:** Perceptual continuity of the loop crossfade. The callback wraps `_current_frame` atomically, but whether the splice is click-free depends on the audio content at the loop boundary.

---

## Gaps Summary

No automated gaps. All code is substantive, wired, and passes the full test suite (35/35). The only open items are the five human verification checks above, which are inherent perceptual tests that cannot be automated — they require audio hardware and observer confirmation.

---

_Verified: 2026-03-21_
_Verifier: Claude (gsd-verifier)_
