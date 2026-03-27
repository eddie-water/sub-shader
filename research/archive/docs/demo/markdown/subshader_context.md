# Sub Shader — Project Context & Claude Code Game Plan

## 1. Project Summary

Sub Shader is a real-time audio visualization system that produces DAW-quality spectrograms using continuous wavelet transforms (CWT). The pipeline captures audio, performs CWT analysis, and renders time-frequency representations via GPU-accelerated OpenGL shaders.

**Stack:** Python, CuPy (GPU), ModernGL, GLFW, NumPy, PyWavelets, matplotlib  
**Hardware:** RTX 4060 Ti, AMD Ryzen 7, Windows/WSL  
**Current FPS:** ~40 (bottleneck: CPU→GPU memory transfers)

---

## 2. Architecture — Three Modules

```
AudioInput → Wavelet (CWT) → Renderer (OpenGL/Shader)
```

### 2.1 AudioInput (`subshader/audio/`)
- Loads audio files, streams chunks with configurable overlap
- Overlap windowing reduces edge effects at CWT boundaries
- Pre-loads entire file to eliminate per-frame file I/O bottleneck
- Key params: `chunk_size`, `overlap_factor`, `hop_size`

- Future Ambitions: Live audio stream - from line in, DJ board, aux, something generic - we don't care where it comes from, could be file, could be bluetooth, could line in, the interface handles all that stuff, all we worry about is pure signal data

### 2.2 Wavelet / DSP (`subshader/dsp/`)
- Generates chromatic scale frequencies (A0 root, 12 notes/octave)
- CWT via FFT-based convolution: `ifft(fft(signal) * fft(wavelet_kernel))`
- Inheritance chain: `Wavelet (ABC) → AntsWavelet → NumPyWavelet / CuPyWavelet`
- `PyWavelet` wraps pywt library (incomplete — `normalize_by_scale` and `discard_unreliable_coefs` are `pass`)
- `WaveletKernel` builds complex Morlet wavelets: sinusoid × Gaussian envelope
- Pipeline: raw CWT → scale normalization (√f) → magnitude → edge discard → downsample
- CuPy path uploads wavelet kernels to GPU at init, processes per-frame

- Future Ambitions: allocating memory for the NumPy version too if that speeds things up - then the CuPy could be an almost 1 to 1 class and could use the (x = c or n) xp.fft() or not maybe that's dumb but something where the classes are basically the same but intermediate private functions have the np or cp usage
- Is CuPy good enough or should we straight up perform the convolution using CuDa and writing batch functions and allocating memeroy and sync'ing on the GPU? seems crazy involved for not much gain

### 2.3 Renderer / Viz (`subshader/viz/`)
- ModernGL context with GLFW window management
- Circular frame buffer aggregates CWT frames chronologically
- Texture2D (R32F) stores spectrogram data, written per frame
- Fragment shader: texture sample → inferno colormap → display
- VAO/VBO fullscreen quad, double-buffered rendering

Future Ambitions
- Being able to tune the gamma or the somehow match the color map to phons / human ear filtering

### 2.4 Supporting and Misc
- `config.py` — dataclass-based config (audio, wavelet, viz params)
- `exceptions.py` — custom exception hierarchy with singleton `ExceptionReporter`
- `utils/` — logging, loop timer
- `__main__.py` — `SubShader` orchestrator class with init → loop → cleanup


---

## 3. Topics Discussed (Comprehensive Index)

### Mathematical Foundations
1. **Dot product as signed similarity accumulator** — element-wise multiply + sum, same-sign = alignment, opposite = anti-alignment
2. **Inner product vs dot product** — dot product is concrete (ℝⁿ), inner product is abstract generalization (axioms: linearity, conjugate symmetry, positive-definiteness)
3. **Vector projection / scalar projection** — dot product gives scalar projection when target is unit vector; CWT coefficients = projection of signal onto wavelet
4. **Generalizing from dimensions to elements** — vectors as indexed sequences, not physical axes; extends dot product to N-element arrays and signals
5. **Discrete sum → continuous integral** — Σ aᵢbᵢ generalizes to ∫ a(t)b(t)dt; same operation, continuous domain
6. **Sign accumulation mechanism** — how element-wise products contribute to overall similarity score

### Fourier Analysis
7. **FFT limitations for music** — no temporal information, stationarity assumption, smears transients
8. **STFT as partial fix** — windowed FFT gives time info, but fixed resolution tradeoff
9. **Resolution tradeoff** — short window = good time / poor freq; long window = opposite; fixed window can't serve both
10. **FFT magnitude scaling** — FFT sums over N points, so magnitudes scale with signal length; divide by N to normalize
11. **FFT periodicity assumption** — FFT assumes infinite periodic signal; wavelets trash this via localization

### Continuous Wavelet Transform
12. **CWT as variable-resolution analysis** — template width varies with frequency; low freq = wide, high freq = narrow
13. **CWT performs correlation, not convolution** — measures similarity (pattern matching), not filtering; conjugate handles this
14. **Complex Morlet wavelet construction** — sinusoid × Gaussian; carrier frequency + envelope
15. **Frequency modulation / heterodyning** — multiplying sinusoid × Gaussian in time = shifting Gaussian spectrum to carrier frequency; convolution theorem dual
16. **Wavelet admissibility condition** — zero mean + controlled bandwidth; NOT orthogonality (CWT doesn't require it)
17. **Orthogonality vs redundancy** — DWT needs orthogonality for reconstruction; CWT oversamples for smooth spectrograms
18. **Scale-dependent normalization (√f)** — wider wavelets collect more energy; 1/√s ≈ √f compensates integration length
19. **Edge effects and Cone of Influence** — convolution edges are unreliable; wider wavelets = more edge contamination
20. **Reliable region extraction** — center-keep slice using widest wavelet's time support; uniform rectangular output
21. **WaveletKernel member variables** — which to keep (`conv_n`, `kernel_f`, `time_support_n`, `slice`) vs remove (`kern_n`, `half_kern_n`, `kernel_t`)

### Convolution & LTI Systems
22. **Convolution = sliding inner product** — shift kernel, compute inner product, repeat
23. **Convolution theorem** — time-domain convolution = frequency-domain multiplication; O(N²) → O(N log N)
24. **Complex conjugate in frequency domain** — converts convolution → correlation; preserves causality for pattern matching
25. **LTI systems** — linearity + time-invariance forces convolution form; enables FFT-based computation
26. **Why FFT convolution works for CWT** — zero-padding isolates periodic copies; wavelet localization makes periodicity harmless; edges discarded anyway

### Visualization Pipeline
27. **Magnitude vs Power for audio visualization** — magnitude + dB scale (20·log₁₀|CWT|) matches human perception; power (|CWT|²) for energy analysis
28. **dB scaling: 20 vs 10** — dB defined for power ratios (10·log₁₀); magnitude uses 20 because power ∝ amplitude²; both give same dB value
29. **Fixed dB range normalization** — db_floor/db_ceil prevent per-frame flickering; maps to [0,1] consistently
30. **Shader scaling bug** — `clamp(value / 0.3, ...)` was destroying wavelet normalization; fix: remove per-frame rescaling
31. **Gamma correction** — visual preference, not data accuracy; start with γ=1.0 for testing
32. **Inferno colormap** — perceptually uniform scientific colormap

### GPU / Performance
33. **CPU→GPU memory transfer bottleneck** — primary performance limiter at ~40 FPS
34. **GPU ring buffer optimization** — FIFO on GPU, upload 1 frame instead of 32; ~30x bandwidth reduction; modulo indexing is trivial for GPU
35. **Pre-uploading wavelet kernels** — biggest speedup from eliminating per-frame kernel transfers
36. **Texture vs framebuffer** — texture stores CWT data (shader reads), framebuffer stores final pixels (shader writes)
37. **texture.write vs texture.use** — write uploads data to GPU; use binds texture to shader sampler
38. **OpenGL pipeline** — clear back buffer → write texture → bind → render fullscreen quad → swap buffers

### Rendering Architecture
39. **Circular frame buffer** — aggregates frames chronologically; feeds flattened data to texture
40. **Renderer module naming** — settled on including "Renderer" in name; encompasses GL context, shaders, textures, buffer management
41. **CPU vs GPU flowchart** — architectural diagram showing data flow across PCIe boundary

### Code Quality & Design
42. **Exception handling** — custom `SubShaderException` hierarchy with log levels; singleton `ExceptionReporter`; `GRACEFUL_EXCEPTIONS` tuple
43. **Module organization** — `exceptions.py` at package top level (not utils); config as dataclasses
44. **Naming conventions** — benchmark files (`benchmarks/`), config params (`window_size`, `hop_size`, `overlap_factor`)
45. **Inheritance chain cleanup** — `AntsWavelet` incomplete abstract; mixed GPU/CPU data flow
46. **File I/O elimination** — pre-loading audio removed per-frame file open/close bottleneck
47. **VS Code settings** — disabling inlay type hints in preferences

### Documentation Strategy
48. **Interactive README hierarchy** — 4 sections: overview, AudioInput, Wavelet, Renderer; Jupyter notebooks with plots
49. **Progressive terminology introduction** — start simple, refine with sophisticated language as context advances
50. **Documentation outline** — from dot product → inner product → FFT → STFT → CWT → implementation
51. **Terminology reference table** — maps concepts across vector, signal, Fourier, and wavelet domains
52. **Writing style** — casual but scientific; third person for overview; concise, not verbose
53. **Project overview rewriting** — multiple iterations on concise professional language

### Audio Overlap & Data Flow
54. **Audio overlap windowing** — overlap reduces edge effects; relationship between window size, overlap factor, hop size
55. **Output dimension consistency** — target_width for downsampled output; hop_size / downsample_factor
56. **Config validation** — preventing invalid overlap/input_length/output combinations

---

## 4. Current State & Known Issues

### Working
- Full CWT pipeline (NumPy + CuPy paths)
- Real-time rendering with inferno colormap
- Audio file loading with overlap
- Chromatic scale frequency generation
- Exception handling framework
- Config system

### Incomplete / Broken
- `PyWavelet.normalize_by_scale()` → `pass` (no-op)
- `PyWavelet.discard_unreliable_coefs()` → `pass` (no-op)
- GPU ring buffer (planned, not implemented)
- Documentation (outline exists, content WIP)
- Unit tests (minimal or none) - unit tests should 
- Benchmark suite (naming discussed, not built)
### 2.5 Still to do 
- Go through the config - is everything in here worth keeping? Look for config option combinations that could break subshader or prevent it from running on another system and and validation check to prevent that from happening
- Go through the exceptions and everywhere that they are being used - are we effectively using them? Are they helpful?
- Logging - clean up logging system - right now seems overkill - i think we just need one logging channel for genreal runtime, one for debugging things specifically where it's really easy to add these ones wherever in the code, and for exceptions I guess too, everytime I run subshader it creates a log and time stamps it and keeps it in a file

### Known Tech Debt
- Commented-out matplotlib plotting code in `apply_coi_mask`
- `TODO ISSUE-36` on PyWavelet cmor parameters
- `TODO-37` references for figures and explanations
- `CuWavelet` alias class adds no value
- No type stubs for CuPy imports
- Hardcoded sample rate restriction (44.1kHz only)

---

## 5. Claude Code Game Plan

### Goal
Get the codebase demo-ready: clean, standardized, documented, tested. NOT full CI/CD — just professional-grade code quality.

### Agent Architecture: 2 Agents

**Agent 1: Code Quality Agent (Claude Code — autonomous)**
Handles all the mechanical, well-defined work that doesn't require domain expertise decisions.

**Agent 2: Documentation Agent (You + Claude chat)**  
You drive the README, Jupyter notebooks, and technical writing since that's your focus and requires your voice.

### Agent 1 Scope (Claude Code)

**Phase 1 — Formatting & Standards (low risk, high autonomy)**
- Run and fix `ruff` / `black` formatting across entire codebase
- Standardize docstrings to Google-style or NumPy-style (pick one, be consistent)
- Remove commented-out code (the matplotlib blocks in `apply_coi_mask`, etc.)
- Fix import ordering (stdlib → third-party → local)
- Add `__all__` exports to `__init__.py` files
- Replace magic numbers with named constants where obvious
- Type hint cleanup (consistent use of `np.ndarray` annotations)

**Phase 2 — Unit Tests (medium risk, validation gates)**
- Write tests for pure functions first: `_generate_chromatic_scale`, `downsample`, `compute_mag`
- Test `WaveletKernel` construction at known frequencies
- Test `normalize_by_scale` produces correct √f scaling
- Test `discard_unreliable_coefs` / `slice_for_reliable_region` output shapes
- Test exception hierarchy and reporter
- Test config dataclass defaults and validation
- Compare NumPyWavelet vs CuPyWavelet output on simple signals (sine waves) — should match within tolerance

**Phase 3 — Benchmark Scaffolding (medium risk)**
- Create `benchmarks/` directory structure
- Write timing harness for: CWT computation, texture upload, full frame pipeline
- Profile NumPy vs CuPy path on standardized input
- Output results as JSON or CSV for tracking

**Phase 4 — Cleanup (targeted, present PRs)**
- Implement `PyWavelet.normalize_by_scale` and `discard_unreliable_coefs` (or mark class as deprecated)
- Remove `CuWavelet` alias or justify it
- Resolve TODOs with either implementation or explicit `# FUTURE:` tags
- Add proper `__repr__` / `__str__` to key classes

### Autonomy Rules for Claude Code

```
AUTONOMOUS (just do it):
- Formatting, linting, import ordering
- Adding docstrings to undocumented functions
- Removing dead/commented code
- Creating test file scaffolding
- Running existing tests

ASK FOR VALIDATION:
- Any change to the CWT algorithm or math
- Changing function signatures or public APIs
- Modifying config defaults
- Architectural decisions (moving modules, changing inheritance)
- Anything touching the shader or rendering pipeline logic

PRESENT AS PR:
- Each phase = 1 PR
- Include: what changed, why, what to review carefully
- Tests must pass before PR
```

### Agent 2 Scope (You + Claude Chat)
- README content (the 4-section hierarchy you designed)
- Jupyter notebook interactive demonstrations
- Mathematical foundations document (dot product → CWT progression)
- Terminology reference table
- Project overview writing in your voice

### Suggested Claude Code CLAUDE.md

```markdown
# CLAUDE.md — Sub Shader Project Instructions

## Project
Real-time audio visualization using CWT. Python + CuPy + ModernGL.

## Structure
subshader/
├── __main__.py          # Orchestrator
├── config.py            # Dataclass configs
├── exceptions.py        # Exception hierarchy
├── audio/               # AudioInput module
├── dsp/                 # Wavelet, WaveletKernel
├── viz/                 # ShaderPlot, Renderer, GL context
└── utils/               # Logging, LoopTimer

## Rules
- Do NOT modify CWT math or normalization logic without asking
- Do NOT change shader code without asking
- Do NOT change public API signatures without asking
- DO fix formatting, docstrings, type hints, imports freely
- DO write unit tests for pure/deterministic functions
- DO remove dead commented-out code
- Run `ruff check` and `ruff format` before committing
- Use Google-style docstrings
- Tests go in `tests/` mirroring `subshader/` structure

## Key Domain Knowledge
- Scale normalization uses √f (not 1/√s directly)
- CWT does correlation, not convolution (conjugate matters)
- Magnitude (not power) for visualization: 20·log₁₀(|CWT|)
- Fixed dB range [-60, 0] prevents per-frame flickering
- PyWavelet class is incomplete (normalize_by_scale = pass)
- GPU bottleneck is memory transfer, not computation

## Testing Notes
- NumPy and CuPy CWT paths should produce matching output (within float tolerance) on sine wave inputs
- WaveletKernel frequencies should match chromatic scale
- Downsample output shape must be (num_freqs, target_width)
- Edge discard must preserve (num_freqs, *) shape with fewer columns
```

---

## 6. Milestone Definition: "Demo Ready"

A demo-ready state means:
1. Code runs end-to-end on a test audio file without errors
2. Visualization looks like a DAW spectrogram (no obvious artifacts)
3. Codebase passes linting with zero errors
4. Core functions have unit tests
5. README exists with project overview and basic usage
6. No dead code or unresolved TODOs polluting the main branch
7. Benchmark baseline recorded (FPS, memory, CWT timing)
