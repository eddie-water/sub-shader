---
phase: 08-codebase-refactoring-and-module-cleanup
plan: 04
subsystem: audio
tags: [facade, audio, refactoring, config-discovery, sync]

requires:
  - phase: 08-codebase-refactoring-and-module-cleanup
    plan: 01
    provides: PipelineConfig base dataclass with sample_rate/total_samples fields

provides:
  - AudioStream facade with start/get_chunk/next_chunk/get_playback_sample/has_looped/cleanup API
  - AudioReader (was AudioInput) accepting PipelineConfig with config.sample_rate writeback
  - AudioPlayer accepting PipelineConfig (reads config.sample_rate after AudioReader init)
  - src/subshader/audio/__init__.py with AudioStream export + deprecated AudioInput/AudioPlayer aliases

affects:
  - 08-05 (orchestrator migration — can now replace AudioInput+AudioPlayer with AudioStream)
  - 08-06 (pipeline — AudioStream is the audio interface)

tech-stack:
  added: []
  patterns:
    - Facade pattern: AudioStream hides reader/player split behind single interface
    - Config writeback: AudioReader discovers sample_rate at file-open and writes back to shared PipelineConfig
    - Clock-driven chunk sync: next_chunk() encapsulates audio-clock alignment (playback_pos // hop_size * hop_size)
    - @timed on get_chunk() and next_chunk() for pipeline profiling

key-files:
  created:
    - src/subshader/audio/reader.py (AudioReader — was AudioInput, accepts PipelineConfig, writes back sample_rate/total_samples)
    - src/subshader/audio/player.py (AudioPlayer — accepts PipelineConfig, reads config.sample_rate for OutputStream)
    - src/subshader/audio/audio_stream.py (AudioStream facade wrapping reader + player)
    - src/subshader/audio/__init__.py (exports AudioStream + deprecated AudioInput/AudioPlayer aliases)
  modified: []

key-decisions:
  - "AudioReader constructed before AudioPlayer — reader writes config.sample_rate, player reads it; construction order is load-bearing"
  - "next_chunk() encapsulates hop-aligned seek, 1ms yield, frame-skip logic from __main__.py loop() — pipeline.py can call audio.next_chunk() cleanly"
  - "clear_loop_event() exposed on AudioStream for callers using get_chunk() directly (vs next_chunk() which clears internally)"
  - "audio_input.py and audio_player.py preserved with no changes — backward compat for __main__.py until Plan 08-05"

metrics:
  duration: 6min
  completed: 2026-04-06
  tasks: 2
  files: 4
---

# Phase 8 Plan 4: AudioStream Facade Summary

**AudioStream facade wrapping AudioReader (file I/O) and AudioPlayer (playback) with PipelineConfig discovery — sample_rate written to shared config on file open, next_chunk() encapsulating audio-clock sync.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-06T23:07:00Z
- **Completed:** 2026-04-06T23:13:00Z
- **Tasks:** 2
- **Files modified:** 4 new files created

## Accomplishments

- Created `reader.py` with `AudioReader` class accepting `PipelineConfig`, opening the audio file, discovering `sample_rate` and `total_samples`, and writing them back to config (D-06 config discovery pattern)
- Created `player.py` with `AudioPlayer` class accepting `PipelineConfig`, loading entire audio into float32 memory, identical threading/callback/loop-detection logic as the original `audio_player.py`
- Created `audio_stream.py` with `AudioStream` facade: constructs reader first (config writeback), then player (reads updated config.sample_rate)
- Implemented `next_chunk()` encapsulating the audio-clock sync loop from `__main__.py`: hop-aligned seek, 1ms yield when audio clock hasn't advanced, frame-skip when render falls behind, loop-wrap reset
- Added `@timed` decorator on `get_chunk()` and `next_chunk()` per D-21
- Created `audio/__init__.py` exporting `AudioStream` as primary export with deprecated `AudioInput`/`AudioPlayer` aliases for backward compatibility

## Task Commits

1. **Task 1: Create reader.py and player.py** — `6776f91`
2. **Task 2: Create AudioStream facade and audio/__init__.py** — `5ebca3c`

## Files Created/Modified

- `/home/eddie-water/dev/python/sub-shader/src/subshader/audio/reader.py` — AudioReader with PipelineConfig, config writeback, has_data(), get_chunk(), get_entire_audio(), cleanup()
- `/home/eddie-water/dev/python/sub-shader/src/subshader/audio/player.py` — AudioPlayer with PipelineConfig, float32 in-memory playback, threading.Lock, seamless loop
- `/home/eddie-water/dev/python/sub-shader/src/subshader/audio/audio_stream.py` — AudioStream facade with full API: start, get_chunk, next_chunk, get_playback_sample, has_looped, clear_loop_event, get_entire_audio, cleanup
- `/home/eddie-water/dev/python/sub-shader/src/subshader/audio/__init__.py` — Module exports: AudioStream (primary), AudioInput/AudioPlayer (deprecated aliases)

## Decisions Made

- AudioReader constructed before AudioPlayer in AudioStream.__init__ — this is load-bearing because reader writes config.sample_rate and player reads it. Reversed order would use the default 44100.0 instead of the file's actual rate.
- `next_chunk()` clears loop_event internally; `clear_loop_event()` is also exposed for callers using the lower-level `get_chunk()` directly.
- `audio_input.py` and `audio_player.py` left completely unchanged — Plan 08-05 (orchestrator migration) handles their removal or replacement.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all methods are fully implemented. `next_chunk()` implementation is complete (not a `...` placeholder).

## Self-Check: PASSED

- `src/subshader/audio/reader.py` — exists, contains `class AudioReader`, `config.sample_rate =`, `config.total_samples =`
- `src/subshader/audio/player.py` — exists, contains `class AudioPlayer`
- `src/subshader/audio/audio_stream.py` — exists, contains `class AudioStream`, all 7 required methods
- `src/subshader/audio/__init__.py` — exists, exports `AudioStream`
- `src/subshader/audio/audio_input.py` — preserved (not deleted)
- `src/subshader/audio/audio_player.py` — preserved (not deleted)
- All imports verified: `from subshader.audio import AudioStream` exits 0, backward compat imports exit 0
- Commits `6776f91` and `5ebca3c` exist

---
*Phase: 08-codebase-refactoring-and-module-cleanup*
*Completed: 2026-04-06*
