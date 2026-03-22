# Phase 3: Audio-Visual Sync - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can play an audio file and watch the CWT visualization track it in real time with no perceptible drift. File-based audio with real-time CWT rendering, synced to sub-100ms perceived lag. Requirements: AUDIO-01 (sync), AUDIO-02 (latency).

</domain>

<decisions>
## Implementation Decisions

### Audio playback library
- **D-01:** Use `sounddevice` for audio playback — PortAudio wrapper with non-blocking streaming via C-level callback
- **D-02:** Audio file loaded entirely into memory as a NumPy array at init — sounddevice callback indexes into it. Demo files are short enough that memory is not a concern
- **D-03:** Playback uses sounddevice's non-blocking mode — C-level audio callback runs on its own thread, main thread runs the render loop

### Audio architecture
- **D-04:** New `AudioPlayer` class in `src/subshader/audio/` — separate from AudioInput. AudioInput stays as the chunk reader for CWT. AudioPlayer handles sounddevice stream, audio clock, and playback state
- **D-05:** SubShader orchestrator coordinates AudioInput and AudioPlayer — both share the same file path and sample rate but have independent position tracking

### Sync mechanism
- **D-06:** Audio device clock is the single source of truth for timing — the sounddevice callback tracks how many samples have been played. Render loop checks this position and renders the CWT frame for where the audio IS
- **D-07:** Remove `time.sleep(0.1)` from main loop entirely — it was a debug artifact, not a pacing mechanism. The audio clock drives render pacing
- **D-08:** Render loop runs freely, checking audio playback position each iteration. When audio has advanced past the next chunk boundary, compute CWT and render. When no new chunk is ready, brief yield to avoid busy-wait
- **D-09:** If render falls behind audio (CWT takes longer than real-time), skip frames to catch up — brief visual gap but audio stays smooth. Audio continuity is prioritized over visual completeness

### Playback UX
- **D-10:** Audio playback and visualization start simultaneously at launch — no user action needed, instant experience
- **D-11:** When audio file ends, loop seamlessly — audio restarts from the beginning, visualization resets and re-renders as if starting fresh. Continuous experience
- **D-12:** When user closes window, audio stops immediately as part of cleanup — no orphaned audio playing after window closes

### CLI argument
- **D-13:** Accept audio file path as a positional CLI argument: `python -m subshader demo.wav` — matches success criterion #1. Falls back to default config path if no argument given

### Claude's Discretion
- sounddevice stream parameters (blocksize, latency settings)
- Exact busy-wait avoidance strategy (vsync, minimal sleep, or yield)
- AudioPlayer internal state machine design
- How circular frame buffer handles loop reset
- Error handling for audio device unavailable

</decisions>

<specifics>
## Specific Ideas

- Audio device clock is THE pacer — render loop follows it, not the other way around
- The `time.sleep(0.1)` was never about pacing, it was slowing things down for testing
- Looping should feel seamless — good for demo purposes

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements
- `.planning/REQUIREMENTS.md` — AUDIO-01 (audio-visual sync), AUDIO-02 (sub-100ms latency)
- `.planning/ROADMAP.md` §Phase 3 — Success criteria (3 items) that must be TRUE after this phase

### Prior phase context
- `.planning/phases/01-codebase-hardening/1-CONTEXT.md` — GPU fallback decisions (D-07 through D-09), exception hierarchy
- `.planning/phases/02-cwt-pipeline-polish/2-CONTEXT.md` — CWT normalization fix, intensity tracker position in pipeline

### Key source files
- `src/subshader/__main__.py` — Main orchestrator, loop(), cleanup() — where sync logic will be integrated
- `src/subshader/audio/audio_input.py` — Current chunk reader, stays as-is for CWT feeding
- `src/subshader/viz/plotter.py` — Renderer, ShaderPlot.update_plot() — render side of sync
- `src/subshader/config.py` — ProcessingConfig, AudioConfig — may need audio playback config additions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AudioInput` class: Already handles file loading, chunk extraction, overlap. AudioPlayer can share the same loaded file data
- `LoopTimer`: Performance monitoring already in the main loop — useful for measuring sync drift
- `soundfile` (sf): Already a dependency for file reading — AudioPlayer can reuse the same file handle or loaded data
- Exception hierarchy: `EndOfAudioException`, `WindowCloseException` already exist for lifecycle events

### Established Patterns
- `SubShader.__init__()` creates components in order: AudioInput → Wavelet → Plotter. AudioPlayer would be added here
- `SubShader.cleanup()` uses `hasattr` guards for safe partial-init teardown — AudioPlayer cleanup follows this pattern
- `GRACEFUL_EXCEPTIONS` tuple controls main loop exit — may need `EndOfAudioException` behavior change for looping

### Integration Points
- `SubShader.__init__()` — create AudioPlayer alongside AudioInput
- `SubShader.loop()` — replace sequential chunk reading + sleep with audio-clock-driven pacing
- `SubShader.cleanup()` — stop sounddevice stream, release audio device
- `AudioInput.get_chunk()` — currently uses sequential `file_pos`. With audio-clock sync, the CWT reader needs to seek to the audio playback position instead

</code_context>

<deferred>
## Deferred Ideas

- Live audio capture from system input — v2 milestone (LIVE-01)
- Playback controls (pause, seek, volume) — not needed for v1 demo
- Adaptive resolution reduction to maintain frame rate — v2 (ENH-06)
- Multiple audio file queue/playlist — out of scope

</deferred>

---

*Phase: 03-audio-visual-sync*
*Context gathered: 2026-03-21*
