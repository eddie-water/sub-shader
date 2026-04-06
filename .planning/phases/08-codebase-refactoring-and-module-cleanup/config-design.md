# Pipeline Config Design

One config object flows through the entire pipeline. AudioStream discovers runtime values (sample_rate) and writes them back. Every module reads what it needs from the same config.

## PipelineConfig

```python
@dataclass
class PipelineConfig:
    # User-settable
    file_path: str = "assets/audio/daw/a2a3_a4_minor_scale.wav"
    chunk_size: int = 1 << 14
    overlap_factor: float = 0.5

    # Discovered at startup by AudioStream
    sample_rate: float = 44100.0
    total_samples: int = 0

    # Derived
    @property
    def hop_size(self) -> int:
        return int(self.chunk_size * (1.0 - self.overlap_factor))

    @property
    def nyquist_freq(self) -> float:
        return self.sample_rate / 2.0
```

### CWT-specific

```python
@dataclass
class CWTConfig(PipelineConfig):
    notes_per_octave: int = 12
    num_octaves: int = 10
    root_note_hz: float = 27.5
    target_width: int = 64
    num_cycles: int = 6
    num_fwhm_cycles: int = 3
```

### Renderer-specific

```python
@dataclass
class RendererConfig(PipelineConfig):
    num_frames: int = 256
    gamma: float = 0.5
    color_norm: ColorNormalizationConfig = field(default_factory=ColorNormalizationConfig)
```

---

## Contexts

### 1. Normal Runtime

Full pipeline: audio → CWT → renderer. Default config, GPU if available.

```python
config = PipelineConfig(file_path="song.wav")

audio = AudioStream(config)       # discovers sample_rate from file
cwt = GpuCWT(config)              # reads sample_rate, chunk_size, overlap
renderer = Renderer(config, cwt.output_shape)

audio.start()
while not renderer.should_close():
    chunk = audio.get_chunk()
    coefs = cwt.cwt(chunk)
    renderer.update(coefs)
```

### 2. Timing Benchmark

Same pipeline, no renderer. Wraps each stage with perf_counter.

```python
config = PipelineConfig(file_path="assets/audio/songs/beltran.wav")

audio = AudioStream(config)
cwt = GpuCWT(config) if gpu_available() else CpuCWT(config)

for i in range(NUM_FRAMES):
    chunk, t_audio = time_call(audio.get_chunk)
    coefs = cwt.cwt(chunk)
    # read cwt._timing_*_ms attributes from @timed decorator
```

### 3. Comparison (multiple backends, same audio)

Same config shared across backends. Each reads sample_rate, chunk_size from config.

```python
config = PipelineConfig(file_path="chirp.wav")
audio = AudioStream(config)

stft   = STFT(config)
pywt   = PywaveletCWT(config)
np_cwt = CpuCWT(config)
gpu_cwt = GpuCWT(config)

chunk = audio.get_chunk()
results = {
    "STFT":              stft.compute(chunk),
    "PyWavelet":         pywt.cwt(chunk),
    "SubShader (NumPy)": np_cwt.cwt(chunk),
    "SubShader (GPU)":   gpu_cwt.cwt(chunk),
}
```

### 4. Per-Signal Figures

One figure per audio signal. Each signal gets its own config + pipeline.

```python
signals = [
    PipelineConfig(file_path="chirp.wav"),
    PipelineConfig(file_path="polyphonic.wav"),
    PipelineConfig(file_path="beltran.wav"),
]

for config in signals:
    audio = AudioStream(config)
    cwt = CpuCWT(config)

    frames = []
    while chunk := audio.get_chunk():
        frames.append(cwt.cwt(chunk))

    render_figure(config.file_path, frames)
```

### 5. Stub Layouts (fast iteration)

No real DSP. Random data in the right shape for layout testing.

```python
config = PipelineConfig(file_path="chirp.wav")
audio = AudioStream(config)
cwt = CpuCWT(config)  # only used for output_shape

stub_frame = np.random.rand(*cwt.output_shape)
render_figure("stub", [stub_frame] * NUM_FRAMES)
```

### 6. Stub PyWavelet (skip slow backend)

Real DSP for fast backends, random stubs for PyWavelet.

```python
config = PipelineConfig(file_path="chirp.wav")
audio = AudioStream(config)

np_cwt = CpuCWT(config)
# pywt skipped — use random stub in its grid cell

chunk = audio.get_chunk()
results = {
    "STFT":              compute_stft(chunk, config),
    "PyWavelet":         np.random.rand(*np_cwt.output_shape),  # stub
    "SubShader (NumPy)": np_cwt.cwt(chunk),
}
```

### 7. Custom Test (single module, override params)

Only need a CWT with specific params. No audio file, synthetic input.

```python
config = PipelineConfig(
    sample_rate=44100.0,   # set directly, no file needed
    chunk_size=4096,
)

cwt = CpuCWT(config)
test_signal = np.sin(2 * np.pi * 440 * np.arange(4096) / 44100)
result = cwt.cwt(test_signal)

assert result.shape == cwt.output_shape
```

---

## Pattern Summary

| Context | Config | AudioStream | CWT | Renderer |
|---------|--------|-------------|-----|----------|
| Normal runtime | default | yes | GPU | yes |
| Timing | default or custom | yes | GPU or CPU | no |
| Comparison | shared across backends | yes | multiple | no |
| Per-signal figures | one per signal | yes | CPU | no (matplotlib) |
| Stub layouts | any | yes (for shape) | shape only | no (matplotlib) |
| Custom test | manual params | no | CPU | no |

Every context uses the same config object. The difference is which modules you construct and which params you override.
