---
phase: 03-audio-visual-sync
plan: 01
subsystem: audio
tags: [audio-player, sounddevice, threading, tdd]
dependency_graph:
  requires: []
  provides: [AudioPlayer, sounddevice-dependency]
  affects: [src/subshader/audio/audio_player.py, pyproject.toml]
tech_stack:
  added: [sounddevice==0.5.5]
  patterns: [sounddevice-OutputStream-callback, threading-Lock-position-counter, seamless-loop-wrapping]
key_files:
  created:
    - src/subshader/audio/audio_player.py
    - tests/test_audio_player.py
  modified:
    - pyproject.toml
decisions:
  - AudioPlayer stores _data as float32 — PortAudio callback layer expects float32; float64 causes silent type coercion
  - threading.Lock used for _current_frame — simple, low-contention; no queue overhead needed for single-int read/write
  - blocksize=0 in OutputStream — lets PortAudio choose optimal hardware buffer; avoids overriding device's preferred latency
  - _loop_event is set by callback but cleared by caller — consumer decides when to act on loop boundary
metrics:
  duration_seconds: 84
  completed_date: "2026-03-22"
  tasks_completed: 2
  files_created: 2
  files_modified: 1
---

# Phase 03 Plan 01: AudioPlayer Implementation Summary

AudioPlayer class with sounddevice OutputStream, thread-safe float32 position counter, and seamless loop wrapping — the audio clock source for the sync mechanism.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add sounddevice dependency and install | 71fbd3d | pyproject.toml |
| 2 (RED) | Failing tests for AudioPlayer | 139a577 | tests/test_audio_player.py |
| 2 (GREEN) | Implement AudioPlayer | 55eae22 | src/subshader/audio/audio_player.py |

## What Was Built

`AudioPlayer` is the audio clock source for the entire audio-visual sync mechanism. It plays audio from an in-memory float32 array via a sounddevice `OutputStream`. The callback runs on a dedicated OS thread and atomically updates `_current_frame` under a `threading.Lock`. The main/render thread calls `get_playback_sample()` to read the current position and determine which CWT chunk to display.

Key behaviors:
- Stereo input is silently coerced to mono (first channel) at init
- All input is stored as float32 — PortAudio's native dtype
- At end-of-buffer the callback wraps to frame 0 and sets `_loop_event`
- `has_looped()` / `clear_loop_event()` let the render loop detect and handle boundary crossings
- `start()` / `stop()` manage the OutputStream lifecycle; `stop()` is idempotent

## Test Coverage

8 unit tests — all passing, no actual audio device required:

- `test_init_loads_audio_as_float32` — dtype coercion at init
- `test_init_mono_conversion` — stereo-to-mono at init
- `test_get_playback_sample_initial` — position is 0 before start
- `test_callback_advances_position` — normal callback increments frame counter
- `test_callback_loop_wraps_position` — wrap math is correct
- `test_callback_loop_sets_event` — loop event is set on wrap
- `test_stop_closes_stream` — _stream is None after stop (mocked OutputStream)
- `test_invalid_empty_data_raises` — SubShaderException on empty input

Full suite: 35/35 passed, no regressions.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| float32 internal storage | PortAudio callback layer expects float32; float64 causes silent type coercion or errors |
| threading.Lock for position | Simple, low-contention; a queue would add overhead for a single-integer read/write |
| blocksize=0 | Lets PortAudio choose optimal hardware buffer; overriding can increase latency |
| Consumer clears loop event | Separation of concerns — AudioPlayer signals, render loop decides when to act |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — AudioPlayer is fully wired. No data flows to UI from this module directly (that is Plan 02's responsibility).

## Self-Check: PASSED

- `src/subshader/audio/audio_player.py` — EXISTS
- `tests/test_audio_player.py` — EXISTS
- `pyproject.toml` contains "sounddevice" — CONFIRMED
- Commit 71fbd3d — FOUND
- Commit 139a577 — FOUND
- Commit 55eae22 — FOUND
- 35/35 tests pass — CONFIRMED
