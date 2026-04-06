# DSP Module Design

Two approaches for backend selection. Both use the same ABC.

## The ABC (same either way)

```python
class DSP(ABC):
    def __init__(self, config):
        self.config = config

    def process(self, chunk):
        data = self.pre(chunk)
        raw = self.transform(data)
        return self.post(raw)

    @abstractmethod
    def pre(self, chunk): ...

    @abstractmethod
    def transform(self, data): ...

    @abstractmethod
    def post(self, raw): ...
```

---

## Option 1: Direct instantiation

```python
# Main app
self.dsp = GpuCWT(config)

# Test harness
self.stft = STFT(config)
self.pywt = PywaveletCWT(config)
self.cpu  = CpuCWT(config)

# Comparison — iterate directly
for method in [stft, pywt, cpu_cwt, gpu_cwt]:
    result = method.process(chunk)
```

**Pros:**
- Explicit — you see exactly what you're creating
- No string-to-class mapping to maintain
- Each backend is a real class you can inspect, type-check, autocomplete
- Natural Python — this is how ABCs are meant to be used

**Cons:**
- Caller needs to know class names and import them
- GPU/CPU fallback logic lives in the caller (pipeline.py, test_suite.py)

---

## Option 2: Factory parameter

```python
# Main app
self.dsp = DSP(config, backend="gpu_cwt")

# Test harness
self.stft = DSP(config, backend="stft")
self.pywt = DSP(config, backend="pywavelet")

# Comparison — iterate by name
for name in ["stft", "pywavelet", "cpu_cwt", "gpu_cwt"]:
    method = DSP(config, backend=name)
    result = method.process(chunk)
```

**Pros:**
- One import (`DSP`) covers everything
- Backend selection driven by strings — easy to parameterize from CLI args or config
- Comparison loop can iterate over a list of names

**Cons:**
- String-to-class registry to maintain (breaks if you add a backend but forget the mapping)
- Hides the actual type — harder to inspect, no autocomplete on backend-specific methods
- Factory pattern on an ABC is a bit unusual in Python

---

## How each looks in context

### pipeline.py

```python
# Option 1
from subshader.dsp.cwt import GpuCWT, CpuCWT

class SubShader:
    def __init__(self, config):
        self.audio    = AudioStream(config)
        self.dsp      = GpuCWT(config) if gpu_available() else CpuCWT(config)
        self.renderer = Renderer(config, self.dsp.output_shape)

# Option 2
from subshader.dsp import DSP

class SubShader:
    def __init__(self, config):
        self.audio    = AudioStream(config)
        self.dsp      = DSP(config, backend="gpu_cwt" if gpu_available() else "cpu_cwt")
        self.renderer = Renderer(config, self.dsp.output_shape)
```

### comparison.py

```python
# Option 1
from subshader.dsp.stft import STFT
from subshader.dsp.pywavelet import PywaveletCWT
from subshader.dsp.cwt import CpuCWT, GpuCWT

backends = [STFT(config), PywaveletCWT(config), CpuCWT(config), GpuCWT(config)]
for backend in backends:
    result = backend.process(chunk)

# Option 2
from subshader.dsp import DSP

for name in ["stft", "pywavelet", "cpu_cwt", "gpu_cwt"]:
    backend = DSP(config, backend=name)
    result = backend.process(chunk)
```

### test (single backend, custom params)

```python
# Option 1
from subshader.dsp.cwt import CpuCWT

cwt = CpuCWT(config)
result = cwt.process(test_signal)
assert result.shape == cwt.output_shape

# Option 2
from subshader.dsp import DSP

cwt = DSP(config, backend="cpu_cwt")
result = cwt.process(test_signal)
assert result.shape == cwt.output_shape
```

---

## Directory structure (same either way)

```
dsp/
├── __init__.py         # exports DSP base + all backends
├── dsp.py              # DSP ABC (pre, transform, post)
├── cwt.py              # CpuCWT(DSP), GpuCWT(DSP)
├── stft.py             # STFT(DSP)
├── pywavelet.py        # PywaveletCWT(DSP)
├── wavelet_kernel.py
└── gaussian.py
```
