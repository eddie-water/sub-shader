# Phase 3: Audio-Visual Sync - Research

**Researched:** 2026-03-21
**Domain:** Real-time audio playback sync with GPU visualization pipeline
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use `sounddevice` for audio playback — PortAudio wrapper with non-blocking streaming via C-level callback
- **D-02:** Audio file loaded entirely into memory as a NumPy array at init — sounddevice callback indexes into it. Demo files are short enough that memory is not a concern
- **D-03:** Playback uses sounddevice's non-blocking mode — C-level audio callback runs on its own thread, main thread runs the render loop
- **D-04:** New `AudioPlayer` class in `src/subshader/audio/` — separate from AudioInput. AudioInput stays as the chunk reader for CWT. AudioPlayer handles sounddevice stream, audio clock, and playback state
- **D-05:** SubShader orchestrator coordinates AudioInput and AudioPlayer — both share the same file path and sample rate but have independent position tracking
- **D-06:** Audio device clock is the single source of truth for timing — the sounddevice callback tracks how many samples have been played. Render loop checks this position and renders the CWT frame for where the audio IS
- **D-07:** Remove `time.sleep(0.1)` from main loop entirely — it was a debug artifact, not a pacing mechanism. The audio clock drives render pacing
- **D-08:** Render loop runs freely, checking audio playback position each iteration. When audio has advanced past the next chunk boundary, compute CWT and render. When no new chunk is ready, brief yield to avoid busy-wait
- **D-09:** If render falls behind audio (CWT takes longer than real-time), skip frames to catch up — brief visual gap but audio stays smooth. Audio continuity is prioritized over visual completeness
- **D-10:** Audio playback and visualization start simultaneously at launch — no user action needed, instant experience
- **D-11:** When audio file ends, loop seamlessly — audio restarts from the beginning, visualization resets and re-renders as if starting fresh. Continuous experience
- **D-12:** When user closes window, audio stops immediately as part of cleanup — no orphaned audio playing after window closes
- **D-13:** Accept audio file path as a positional CLI argument: `python -m subshader demo.wav` — matches success criterion #1. Falls back to default config path if no argument given

### Claude's Discretion
- sounddevice stream parameters (blocksize, latency settings)
- Exact busy-wait avoidance strategy (vsync, minimal sleep, or yield)
- AudioPlayer internal state machine design
- How circular frame buffer handles loop reset
- Error handling for audio device unavailable

### Deferred Ideas (OUT OF SCOPE)
- Live audio capture from system input — v2 milestone (LIVE-01)
- Playback controls (pause, seek, volume) — not needed for v1 demo
- Adaptive resolution reduction to maintain frame rate — v2 (ENH-06)
- Multiple audio file queue/playlist — out of scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUDIO-01 | Audio playback and visualization are synced — file-based audio with real-time CWT rendering | sounddevice `current_frame` counter in callback provides the clock; render loop reads it to select matching CWT chunk |
| AUDIO-02 | Audio-visual sync with minimal perceptible latency (<100ms perceived lag) | With `blocksize=0` (optimal hardware buffer) and `latency='low'`, PortAudio latency is typically 10-30ms; the render loop checks audio position each iteration with no sleep |
</phase_requirements>

---

## Summary

Phase 3 adds `sounddevice` as a new dependency and introduces an `AudioPlayer` class that streams audio through a PortAudio C-level callback while the existing render loop runs on the main thread. The key sync mechanism is a shared atomic counter (`current_frame`) that the callback increments and the render loop reads. When the render loop wakes up, it calculates which CWT chunk corresponds to the current audio position and renders it — if multiple chunks have been skipped due to CWT computation time, it renders the most recent one.

The existing `AudioInput` class is unchanged — it continues to supply overlapping windows to the CWT. The new `AudioPlayer` operates independently of `AudioInput`, loading the same file into memory at init. The orchestrator's `loop()` method replaces its current sequential design (read chunk → CWT → render → sleep) with an audio-clock-driven design (check audio position → if new chunk boundary passed, compute CWT and render → else yield).

Looping is implemented by resetting `current_frame` to 0 inside the callback when the end of the audio buffer is reached, and signaling the render side to reset `AudioInput`'s `file_pos`. No `CallbackStop` is raised — the stream continues indefinitely.

**Primary recommendation:** `sounddevice.OutputStream` with `blocksize=0`, `latency='low'`, a `current_frame` counter (Python `threading.Lock` or `ctypes.c_long` for safe cross-thread read), and a `threading.Event` for start/stop coordination.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sounddevice | 0.5.5 | PortAudio wrapper: non-blocking OutputStream with C-level callback | Only Python audio library that exposes the PortAudio timing struct and runs callback at OS real-time priority |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| threading | stdlib | `Lock` for `current_frame`, `Event` for stop/loop signals | Required — sounddevice callback runs on dedicated OS thread, main thread reads shared state |
| argparse | stdlib | Parse positional CLI argument `python -m subshader demo.wav` | D-13 — simple, no extra dependency |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sounddevice OutputStream | pygame.mixer | pygame adds a large dependency for a single feature; less control over timing |
| sounddevice OutputStream | pyaudio | pyaudio is a lower-level PortAudio binding; sounddevice is the modern successor |
| threading.Lock for counter | ctypes.c_long | `ctypes.c_long` is atomically readable on x86; `Lock` is clearer and portable |

**Installation:**
```bash
pip install sounddevice
```

**Version verification:** Confirmed against PyPI — `sounddevice 0.5.5` is the latest release (verified 2026-03-21).

---

## Architecture Patterns

### Recommended Project Structure
```
src/subshader/
├── audio/
│   ├── audio_input.py        # UNCHANGED — chunk reader for CWT
│   └── audio_player.py       # NEW — sounddevice stream, audio clock
├── __main__.py               # MODIFIED — CLI arg, AudioPlayer init, sync loop
└── config.py                 # UNCHANGED (AudioConfig already has file_path)
```

### Pattern 1: AudioPlayer — File-in-Memory OutputStream with Sample Counter

**What:** Load entire audio file as `np.float32` array at init. Start a `sounddevice.OutputStream`. Callback indexes into the array and increments `current_frame` under a lock. Main thread reads `current_frame` to determine which CWT chunk to render next.

**When to use:** File-based playback where the file fits in memory (confirmed by D-02).

**Example:**
```python
# Source: https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html (play_file.py pattern)
import sounddevice as sd
import threading

class AudioPlayer:
    def __init__(self, audio_data: np.ndarray, sample_rate: float) -> None:
        # audio_data: float32, mono, shape (total_samples,)
        self._data = audio_data.astype(np.float32)
        self._sample_rate = sample_rate
        self._current_frame = 0
        self._lock = threading.Lock()
        self._loop_event = threading.Event()
        self._stream: sd.OutputStream | None = None

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            start = self._current_frame
        end = start + frames
        total = len(self._data)

        if end >= total:
            # Fill to end, then wrap (loop: D-11)
            chunk = self._data[start:]
            remaining = frames - len(chunk)
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = self._data[:remaining]
            with self._lock:
                self._current_frame = remaining
            self._loop_event.set()
        else:
            outdata[:, 0] = self._data[start:end]
            with self._lock:
                self._current_frame = end

    def get_playback_sample(self) -> int:
        """Main thread reads this to know which audio sample is currently playing."""
        with self._lock:
            return self._current_frame

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype='float32',
            blocksize=0,       # Let PortAudio choose optimal buffer size
            latency='low',     # Request low latency (Claude's discretion: D)
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop(ignore_errors=True)
            self._stream.close()
            self._stream = None
```

### Pattern 2: Audio-Clock-Driven Render Loop

**What:** Replace the current sequential `get_chunk → cwt → render → sleep` with a position-check loop. The loop compares `audio_player.get_playback_sample()` against `next_expected_sample` to decide whether to render, skip, or yield.

**When to use:** Always — this is the sync mechanism (D-06, D-07, D-08, D-09).

**Example:**
```python
# Replaces SubShader.loop()
def loop(self):
    next_expected_sample = 0
    hop_size = self.audio_input.hop_size

    while not self.plotter.should_window_close():
        playback_pos = self.audio_player.get_playback_sample()

        if playback_pos < next_expected_sample:
            # Audio has not advanced to the next chunk boundary yet — yield
            time.sleep(0.001)   # 1ms yield, avoids busy-wait (Claude's discretion)
            continue

        # Audio has advanced: may have skipped multiple chunks (D-09)
        # Seek AudioInput to match the current audio position
        target_sample = (playback_pos // hop_size) * hop_size
        self.audio_input.file_pos = target_sample

        audio_data = self.audio_input.get_chunk()
        if audio_data is None:
            # End of file while not yet looped — wait for loop_event
            time.sleep(0.001)
            continue

        coefs = self.wavelet.cwt(audio_data)
        self.plotter.update_plot(coefs)
        self.loop_timer.end_loop_and_report(loop_start)

        next_expected_sample = target_sample + hop_size

    raise exceptions.WindowCloseException("Window closed by user")
```

### Pattern 3: CLI Argument Parsing (D-13)

**What:** Add `argparse` to `main()` to accept an optional positional audio file path.

**Example:**
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="SubShader audio visualizer")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV)")
    args = parser.parse_args()

    if args.audio_file:
        config.audio.file_path = args.audio_file

    subshader = SubShader(config)
    # ... rest of main()
```

### Pattern 4: Loop Reset — AudioInput file_pos

**What:** When `AudioPlayer` loops (wraps `current_frame` to 0), the render loop detects the wrap via a `loop_event` or by observing `playback_pos < next_expected_sample` dramatically, then resets `audio_input.file_pos = 0`.

**When to use:** On every seamless loop boundary (D-11).

**Note:** `CircularFrameBuffer` does not need explicit reset — its circular nature means it continues pushing new frames; old frames age out naturally. The visual effect is a clean restart without a buffer clear.

### Anti-Patterns to Avoid
- **`time.sleep(0.1)` in the render loop:** Directly violates D-07 and makes ~10 FPS maximum. This is the exact artifact being removed.
- **Calling sd.play() instead of OutputStream:** `sd.play()` is a convenience function that cannot be cleanly stopped, does not give callback access, and creates a new stream each call — unsuitable for looping and position tracking.
- **Sharing the sounddevice stream across threads without a lock:** `current_frame` is written inside a C-level callback thread and read from the Python main thread. Even on x86, Python int mutation is not guaranteed atomic without a lock.
- **Keeping EndOfAudioException as a loop terminator:** With looping (D-11), the render side should never terminate on EOF — `AudioInput.get_chunk()` returning `None` is handled by seeking back to 0, not by raising `EndOfAudioException`.
- **Blocking the sounddevice callback:** No I/O, no logging, no locks with contention in the callback. The lock protecting `current_frame` is only held for a single integer write/read — this is acceptable.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Low-latency audio output | Custom PortAudio binding, custom ring buffer | sounddevice OutputStream | PortAudio handles OS-specific ASIO/CoreAudio/ALSA scheduling, buffer sizing, real-time thread priority — none of this is trivial |
| Timing reference for sync | `time.time()` polled in loop | `current_frame` counter in callback | Wall clock drifts from audio hardware clock; callback counter IS the audio clock with sub-frame accuracy |
| Looping via re-opening file | `sf.SoundFile` re-open on each loop | In-memory array + `current_frame` reset | File I/O in the hot path causes audible glitches; array indexing is zero-latency |
| CLI argument parsing | Manual `sys.argv` slicing | `argparse` | Edge cases: paths with spaces, `--help`, missing file error messages |

**Key insight:** The sounddevice callback IS the audio clock. Everything else (render loop, CWT scheduling) is derived from it. Building any alternative timing source creates two clocks that will drift apart.

---

## Common Pitfalls

### Pitfall 1: float64 dtype in OutputStream
**What goes wrong:** `sounddevice.OutputStream` silently fails or produces distorted output if passed float64 data in `outdata`.
**Why it happens:** PortAudio does not support float64 natively. sounddevice's convenience functions auto-convert, but OutputStream callbacks do not.
**How to avoid:** Load audio as `np.float32` at init (`audio_data.astype(np.float32)`). The `outdata` array the callback receives is always float32 when `dtype='float32'` is set on the stream.
**Warning signs:** Silent playback or pitched-wrong output at correct duration.

### Pitfall 2: Python GIL and callback thread contention
**What goes wrong:** `current_frame` read in the main thread races with the write in the callback thread.
**Why it happens:** The callback runs at OS real-time priority on a dedicated thread. Python int assignment appears atomic but is not guaranteed under all CPython implementations.
**How to avoid:** Use `threading.Lock()` around both read and write of `current_frame`. The lock is contended for microseconds — not a performance concern.
**Warning signs:** Render reads a frame number that is ahead of the actual playback position.

### Pitfall 3: Busy-wait without yield
**What goes wrong:** Render loop spins at 100% CPU checking audio position when CWT is faster than audio advancement.
**Why it happens:** Removing `time.sleep(0.1)` without a replacement yield causes the main thread to consume the CPU core.
**How to avoid:** When audio has not advanced to next chunk boundary, `time.sleep(0.001)` (1ms) is sufficient. At 44100 Hz with chunk_size=16384 and overlap=0.5, hop_size=8192 samples = ~185ms per visual frame. The render loop will typically yield many times per visual frame.
**Warning signs:** CPU usage at 100% on one core while visualization runs normally.

### Pitfall 4: AudioInput.file_pos desync on loop
**What goes wrong:** After `AudioPlayer` loops, the render loop's `next_expected_sample` is near the end of the file while `current_frame` has reset to near 0 — the condition `playback_pos < next_expected_sample` is always true, render loop yields forever.
**Why it happens:** The render loop's internal position counter does not know a loop occurred.
**How to avoid:** Detect the wrap: if `playback_pos < previous_playback_pos` (position went backward), reset `next_expected_sample = 0` and `audio_input.file_pos = 0`. Alternatively, use the `loop_event` from `AudioPlayer`.
**Warning signs:** Visualization freezes after first loop while audio continues playing.

### Pitfall 5: sounddevice not installed (pyproject.toml gap)
**What goes wrong:** `import sounddevice` fails at runtime; no clear error.
**Why it happens:** `sounddevice` is not currently in `pyproject.toml` dependencies — it must be added as part of this phase.
**How to avoid:** Add `sounddevice` to `[project] dependencies` in `pyproject.toml` and `pip install sounddevice` in the venv.
**Warning signs:** `ModuleNotFoundError: No module named 'sounddevice'` at launch.

### Pitfall 6: AudioDevice unavailable in WSL2
**What goes wrong:** PortAudio cannot open an audio device under WSL2 without a properly configured audio backend (PulseAudio or PipeWire).
**Why it happens:** WSL2 does not expose audio hardware by default in older configurations. Audio device discovery fails.
**How to avoid:** Catch `sd.PortAudioError` on `stream.start()` and raise a clear `SubShaderException` with a message about WSL audio configuration. The `AudioPlayer.__init__` or `start()` should wrap the stream open in try/except. This is a user environment issue, not a code bug.
**Warning signs:** `PortAudioError: [Errno -9996] Invalid output device` on WSL.

---

## Code Examples

Verified patterns from official sources:

### sounddevice OutputStream callback signature
```python
# Source: https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html
def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
    # outdata shape: (frames, channels) — must be fully populated
    # time.outputBufferDacTime: DAC time for first sample in buffer
    # time.currentTime: callback invocation time
    # status: reports xruns and other flags
    ...
```

### File-in-memory playback with loop (official play_file.py pattern extended)
```python
# Source: https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html
current_frame = 0

def callback(outdata, frames, time, status):
    global current_frame
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()   # <-- for no-loop; omit for looping
    current_frame += chunksize
```

### Stream construction with recommended parameters
```python
# Source: https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html
stream = sd.OutputStream(
    samplerate=44100,
    channels=1,
    dtype='float32',
    blocksize=0,        # Let PortAudio choose optimal; non-zero adds latency
    latency='low',      # 'low' for interactive; 'high' for robustness
    callback=callback,
)
stream.start()
# ... later ...
stream.stop(ignore_errors=True)
stream.close()
```

### argparse for CLI audio file (D-13)
```python
# Source: Python stdlib argparse docs
import argparse, sys

def main():
    parser = argparse.ArgumentParser(prog="subshader")
    parser.add_argument("audio_file", nargs="?", default=None,
                        help="Path to WAV audio file")
    args = parser.parse_args()

    if args.audio_file:
        config.audio.file_path = args.audio_file
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `time.sleep(0.1)` pacing | Audio-clock-driven render (D-07) | Phase 3 | Removes artificial 10 FPS ceiling; visual updates as fast as CWT completes |
| `EndOfAudioException` terminates app | Seamless loop via `current_frame` reset (D-11) | Phase 3 | Demo runs continuously without user intervention |
| Default file path only | CLI positional arg (D-13) | Phase 3 | `python -m subshader demo.wav` as success criterion |
| No audio output (visual only) | `sounddevice` playback synced to CWT | Phase 3 | Core feature of the demo milestone |

**Deprecated/outdated:**
- `time.sleep(0.1)` in `SubShader.loop()`: removed in this phase (D-07). It was never a pacing mechanism.
- Sequential "read → process → render" as loop structure: replaced by clock-driven "check position → maybe render" design.

---

## Open Questions

1. **Audio device unavailable fallback behavior**
   - What we know: `sd.PortAudioError` is raised on `stream.start()` if no device
   - What's unclear: Should the app run in visualization-only mode (no audio) or hard-fail?
   - Recommendation: Hard-fail with a clear `SubShaderException` message. For a demo, silent CWT visualization without audio is confusing. Error message should suggest WSL audio setup if on WSL.

2. **Optimal blocksize for WSL2 + PulseAudio**
   - What we know: `blocksize=0` lets PortAudio choose; on some WSL setups this defaults to a large buffer (512+ frames) adding 10-20ms latency
   - What's unclear: Whether the target machine's WSL audio stack introduces fixed latency regardless of settings
   - Recommendation: Start with `blocksize=0, latency='low'`. Measure actual latency. If > 50ms, try explicit `blocksize=256`.

3. **CircularFrameBuffer behavior during loop reset**
   - What we know: `CircularFrameBuffer` is purely additive — it never reads from a start position, just pushes and wraps
   - What's unclear: Whether a brief visual discontinuity at loop boundary is acceptable, or if `frame_index` should be reset to produce a clean slate
   - Recommendation: Do not reset the buffer. Let frames age out naturally. The visual will show a brief 32-frame scroll-through of old content then clear. This is acceptable for a demo.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, from conftest.py + tests/) |
| Config file | none — pytest discovers tests/ directory automatically |
| Quick run command | `pytest tests/test_audio_player.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUDIO-01 | `AudioPlayer.get_playback_sample()` returns incrementing values while stream is active | unit | `pytest tests/test_audio_player.py::test_playback_position_advances -x` | Wave 0 |
| AUDIO-01 | Render loop reads audio position and selects correct chunk (position-to-chunk mapping) | unit | `pytest tests/test_audio_player.py::test_chunk_selection_from_position -x` | Wave 0 |
| AUDIO-01 | CLI arg sets `config.audio.file_path` correctly | unit | `pytest tests/test_audio_player.py::test_cli_arg_overrides_path -x` | Wave 0 |
| AUDIO-02 | `AudioPlayer` init raises `SubShaderException` on invalid file path | unit | `pytest tests/test_audio_player.py::test_invalid_file_raises -x` | Wave 0 |
| AUDIO-02 | Loop detection: render loop resets `next_expected_sample` when position wraps | unit | `pytest tests/test_audio_player.py::test_loop_wrap_detection -x` | Wave 0 |

**Note:** Actual audio device playback tests are manual-only — CI has no audio device. Unit tests mock the sounddevice stream and test the `AudioPlayer` state machine and position math independently.

### Sampling Rate
- **Per task commit:** `pytest tests/test_audio_player.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_audio_player.py` — covers AUDIO-01, AUDIO-02 (new file needed)
- [ ] `sounddevice` install: `pip install sounddevice` in venv + add to `pyproject.toml`

---

## Sources

### Primary (HIGH confidence)
- https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html — `OutputStream` constructor, callback signature, `time` parameter fields, `stop()`/`abort()` lifecycle
- https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html — `play_file.py` pattern: file-in-memory + `current_frame` counter in callback
- PyPI registry — sounddevice 0.5.5 is current (verified 2026-03-21)

### Secondary (MEDIUM confidence)
- https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html — non-blocking streams overview, callback threading notes
- https://python-sounddevice.readthedocs.io/en/0.5.1/api/convenience-functions.html — float64→float32 conversion behavior confirmed

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — sounddevice 0.5.5 confirmed on PyPI; callback API confirmed from official docs
- Architecture: HIGH — patterns derived directly from official examples and existing codebase inspection
- Pitfalls: HIGH — float64 dtype and loop desync pitfalls confirmed from docs; WSL audio pitfall from known WSL constraints
- Validation: HIGH — existing pytest infrastructure confirmed in `tests/`; new test file identified as Wave 0 gap

**Research date:** 2026-03-21
**Valid until:** 2026-06-21 (sounddevice is a stable API; PortAudio underlying protocol unlikely to change)
