# Audio Module

## Role in the Pipeline

[WRITE: "AudioInput delivers overlapping chunks of audio samples to the DSP stage. AudioPlayer drives the render loop clock by providing real-time playback position."]

[WRITE: "Explain the two distinct roles: data delivery (AudioInput) and timing reference (AudioPlayer)"]

---

## The Overlap Strategy

[WRITE: "Why overlap_factor exists — edge discontinuities and spectral leakage at window boundaries"]

[WRITE: "The key relationship: hop_size = int(chunk_size * (1.0 - overlap_factor))"]

[WRITE: "file_pos advances by hop_size (not chunk_size) — this is what makes overlap work"]

```python
# Source: src/subshader/audio/audio_input.py AudioInput.__init__
# With chunk_size=4096, overlap_factor=0.5:
# hop_size = int(4096 * (1.0 - 0.5)) = 2048
# → each get_chunk() returns 4096 samples; file_pos advances by 2048
self.hop_size = int(chunk_size * (1.0 - self.overlap_factor))
```

[PLACEHOLDER: figure — "4-row overlap visualization showing consecutive chunks with shared regions highlighted — chunk_size=4096, hop_size=2048, 50% overlap"]

---

## AudioInput

[WRITE: "How get_chunk() works — reads chunk_size samples from file_pos, advances by hop_size"]

[WRITE: "Stereo-to-mono conversion if needed — takes first channel (index 0) only"]

[WRITE: "file_pos tracking and end-of-file handling — returns None when file_pos + chunk_size > total_samples"]

```python
# Source: src/subshader/audio/audio_input.py
audio = AudioInput(
    path="assets/audio/daw/a2a3_a4_minor_scale.wav",
    chunk_size=4096,
    overlap_factor=0.5
)
# hop_size = int(4096 * (1 - 0.5)) = 2048 samples
# get_chunk() returns 4096 samples, advances file_pos by 2048
chunk = audio.get_chunk()  # Returns None at end of file
```

Key design detail:
- `get_chunk()` does not validate that `file_pos` is on a chunk boundary — callers seeking to a specific position (e.g., the render loop syncing to audio playback) set `audio_input.file_pos` directly before calling `get_chunk()`

---

## AudioPlayer

[WRITE: "Audio-clock-driven sync design — audio device clock is single source of truth"]

[WRITE: "Callback thread delivers audio to hardware; main thread reads playback position"]

[WRITE: "get_playback_sample() provides the timing reference the render loop uses to seek AudioInput.file_pos"]

```python
# Source: src/subshader/audio/audio_player.py
player = AudioPlayer(audio_data=audio.get_entire_audio(), sample_rate=audio.get_sample_rate())
player.start()
# In render loop — audio clock drives the position
current_sample = player.get_playback_sample()
```

Key design decisions:

- **float32 storage:** `_data` is stored as `float32` — PortAudio does not support `float64` natively; passing float64 causes silent type coercion in the callback layer
- **`blocksize=0` in OutputStream:** lets PortAudio choose the optimal hardware buffer size; this is not a magic number — it hands the decision to the driver
- **`threading.Lock()` on `_current_frame`:** single int read/write, low contention; queue overhead is unnecessary for this access pattern
- **Seamless looping:** when playback reaches end-of-buffer, callback wraps to beginning; `_loop_event` signals the render loop to reset `file_pos`

---

## Configuration

AudioConfig fields (source: `src/subshader/config.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `file_path` | `"assets/audio/daw/a2a3_a4_minor_scale.wav"` | Path to WAV file |
| `chunk_size` | `4096` (`1 << 12`) | Samples per audio window |
| `overlap_factor` | `0.5` | Fraction of overlap between consecutive chunks (0.0–0.9) |

[WRITE: "What each parameter controls and when to change it — e.g., larger chunk_size increases CWT accuracy at lower frequencies but reduces frame rate; higher overlap_factor reduces edge artifacts at cost of more DSP frames per second"]

---

## Usage Example

```python
# Source: src/subshader/__main__.py SubShader.__init__
from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.audio.audio_player import AudioPlayer

config = get_default_config()

# Audio Input — provides overlapping chunks to DSP stage
audio_input = AudioInput(
    path=config.audio.file_path,
    chunk_size=config.audio.chunk_size,
    overlap_factor=config.audio.overlap_factor,
)

# Audio Player — loads full file into memory, drives render loop clock
audio_player = AudioPlayer(
    audio_data=audio_input.get_entire_audio(),
    sample_rate=float(audio_input.get_sample_rate()),
)
audio_player.start()

# In render loop
current_sample = audio_player.get_playback_sample()
audio_input.file_pos = (current_sample // audio_input.hop_size) * audio_input.hop_size
chunk = audio_input.get_chunk()  # Returns None at end of file

# Cleanup
audio_player.stop()
audio_input.cleanup()
```
