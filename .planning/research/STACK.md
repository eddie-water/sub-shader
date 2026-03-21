# Stack Research

**Domain:** Server-rendered GPU visualization with WebSocket streaming and real-time audio capture
**Researched:** 2026-03-21
**Confidence:** MEDIUM-HIGH (core recommendations verified via official docs and PyPI; hosting costs and GPU availability are market-variable)

---

## Context: What This Research Covers

The existing pipeline (CuPy, ModernGL, NumPy, soundfile) is not re-researched here. This document covers only the **new layers** needed for the hosted demo milestone:

1. WebSocket server + async framework (to stream frames from GPU to browser)
2. Headless OpenGL rendering (to run ModernGL without a display)
3. Real-time audio capture (to replace file-based input with live mic/system audio)
4. Frame encoding (to efficiently serialise rendered frames for the wire)
5. Hosting infrastructure (to run the GPU pipeline on a cloud instance)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| FastAPI | 0.115.x | WebSocket server and HTTP endpoints | Native async, built-in WebSocket support via Starlette, near-zero boilerplate for binary frame streaming. The de facto standard for Python async APIs in 2025. |
| Uvicorn | 0.34.x | ASGI server (runs FastAPI) | Uses uvloop under the hood; handles concurrent WebSocket connections without blocking the GPU pipeline thread. Only ASGI server to recommend with FastAPI. |
| sounddevice | 0.5.5 | Real-time audio capture via PortAudio | Callback-based API delivers audio chunks to a Python function on the audio thread. NumPy arrays in, NumPy arrays out — plugs directly into the existing CuPy pipeline with one copy. Updated January 2026. |
| Pillow (PIL) | 11.x | JPEG/PNG encode rendered frames to bytes | Standard library for in-memory image encoding. `io.BytesIO` + `Image.tobytes()` path is the established pattern for encoding OpenGL readback buffers before sending over WebSocket. |
| ModernGL (existing) | 5.12.0 | GPU rendering — now in headless/EGL mode | Already in the stack. The `standalone=True, backend='egl'` context creation path replaces the GLFW window context for server use. No new dependency needed. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| asyncio (stdlib) | 3.12 | Bridge between audio callback thread and WebSocket coroutine | Always. sounddevice callbacks run on a PortAudio thread; use `loop.call_soon_threadsafe(q.put_nowait, frame)` to hand frames to the asyncio event loop safely. |
| threading (stdlib) | 3.12 | Run the GPU render loop on a background thread | Always. The GPU pipeline is CPU-bound/GPU-bound and must not block the uvicorn event loop. A dedicated thread + asyncio.Queue decouples them. |
| numpy (existing) | — | Frame buffer readback and conversion | Already in the stack. `framebuffer.read(components=3)` → `np.frombuffer()` → Pillow encode is the readback chain. |
| glcontext | 3.x | EGL context backend for ModernGL standalone mode | Required on Linux servers. Installed as a transitive dependency of ModernGL but may need explicit install on headless systems. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Docker + nvidia-container-toolkit | Package the app and its CUDA/OpenGL dependencies for cloud deployment | Use `pytorch/pytorch:2.x-cudaXX.X-cudnnX-runtime` or `nvidia/cuda:12.x-runtime-ubuntu22.04` as base image. Both include the CUDA runtime; the latter is smaller if PyTorch is not needed. |
| pytest-asyncio | Test async WebSocket handlers | Needed because standard pytest cannot await coroutines. Add only when writing WebSocket handler tests. |
| websockets (transitive) | FastAPI's underlying WebSocket implementation | Do not import directly — FastAPI manages this. Pin via FastAPI's requirements. |

---

## Installation

```bash
# WebSocket server
pip install "fastapi>=0.115" "uvicorn[standard]>=0.34"

# Real-time audio capture
pip install sounddevice>=0.5.5

# Frame encoding
pip install Pillow>=11.0

# EGL headless context (may already be present as moderngl transitive dep)
pip install glcontext>=3.0

# Dev / test
pip install pytest-asyncio
```

On the server host (Ubuntu/Docker), install EGL system libraries before the Python packages:

```bash
apt-get install -y libegl1-mesa libgl1-mesa-glx
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| FastAPI + Uvicorn | aiohttp | aiohttp has mature WebSocket support but more boilerplate and smaller ecosystem. Only worth it if FastAPI's Starlette overhead becomes measurable, which it hasn't in practice for this workload. |
| FastAPI + Uvicorn | websockets (standalone) | Appropriate for pure WebSocket servers with no HTTP endpoints. Once you need an HTTP health-check endpoint or REST config endpoint (likely for the demo), FastAPI earns its keep. |
| sounddevice | PyAudio | PyAudio gives finer buffer control but installation is more fragile (PortAudio headers required at compile time). sounddevice wraps the same PortAudio library with a NumPy-native API and installs via pip wheel on all platforms. |
| sounddevice | pyaudio + pipewire | Pipewire is the modern audio server on Ubuntu 22.04+ but has no direct Python bindings. Still accessed via PortAudio. No benefit over sounddevice. |
| Pillow JPEG encode | OpenCV imencode | OpenCV adds ~50MB to the Docker image for a task Pillow handles in 3 lines. Avoid unless OpenCV is already present. |
| Pillow JPEG encode | raw RGBA bytes | Sending uncompressed RGBA bytes is simple but ~10x larger on the wire than JPEG at quality=85. At 60fps the bandwidth difference is significant. |
| RunPod (cloud) | Vast.ai | Vast.ai is cheaper but instances can disappear if the host reclaims their machine (spot-market model). RunPod Secure Cloud guarantees availability. Use Vast.ai for development and cost experimentation, RunPod for a stable demo URL. |
| RunPod (cloud) | AWS/GCP GPU VM | AWS/GCP charge minimum 1-hour billing and require more DevOps setup. For a single-GPU demo app, RunPod's per-second billing and Docker-native workflow is faster to ship. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| MJPEG over HTTP (multipart/x-mixed-replace) | One-way streaming only; no backpressure control; poor browser support for canvas integration; effectively deprecated for this use case | WebSocket with binary JPEG frames |
| WebRTC | Correct for peer-to-peer video but heavyweight: requires signalling server, STUN/TURN infrastructure, and codec negotiation. Overkill for a single-server demo where the server controls the frame rate. | WebSocket binary streaming |
| Gunicorn (WSGI) | WSGI is synchronous; WebSocket connections require a persistent async connection. Gunicorn will not handle WebSockets correctly without an ASGI worker, at which point you are just running Uvicorn inside Gunicorn. | Uvicorn directly |
| X11 / GLFW window context on server | Requires a running X server or Xvfb virtual display. Adds an unnecessary process and fails silently in Docker without the right DISPLAY env var. EGL is the correct headless path. | `moderngl.create_context(standalone=True, backend='egl')` |
| base64-encoding frames over WebSocket text messages | Adds ~33% size overhead and a CPU-bound encode/decode step on both sides. WebSocket binary frames (send_bytes) carry raw bytes natively. | `websocket.send_bytes(jpeg_bytes)` |
| PyQt5 / pyqtgraph on server | The existing PyQt5 visualization path requires a display. It is already architecturally separated behind the Plotter abstraction; just don't instantiate it server-side. | ModernGL headless renderer |
| Serverless GPU (RunPod serverless, Lambda Labs serverless) | Serverless functions have cold-start latency (seconds) that is incompatible with real-time streaming. The GPU pipeline must stay warm. | RunPod persistent Pod |

---

## Stack Patterns by Variant

**If the server is the developer's own machine (local demo):**
- Keep the existing GLFW window context; no EGL needed
- sounddevice captures system audio directly
- FastAPI runs on localhost; browser connects to `ws://localhost:8000/ws`

**If the server is a cloud GPU instance (hosted demo):**
- Switch ModernGL to EGL headless context (`standalone=True, backend='egl'`)
- Audio input becomes a pre-recorded file or looped audio asset; live microphone capture from a remote server is not viable (no physical mic)
- FastAPI runs behind RunPod's HTTPS proxy; browser connects to `wss://<pod-id>-8000.proxy.runpod.net/ws`
- Docker image must include `libegl1-mesa` and NVIDIA GPU access via nvidia-container-toolkit

**If audio latency is unacceptable at full resolution:**
- Reduce `target_width` in `VisualizationConfig` (fewer columns per frame = less CWT compute)
- Lower JPEG quality (85 → 70) to reduce encode time and wire size
- The existing `downsample()` step in the CWT pipeline is already the right knob

---

## Architecture of the New Layer

The new components sit between the existing pipeline and the browser. The GPU pipeline runs on a background thread; the WebSocket server runs on the asyncio event loop. They communicate via an `asyncio.Queue`.

```
[sounddevice callback] ---(thread-safe queue)---→ [CuPy CWT] → [ModernGL headless render]
                                                                          ↓
                                                          [framebuffer.read() → numpy]
                                                                          ↓
                                                         [Pillow JPEG encode → bytes]
                                                                          ↓
                                                    [asyncio.Queue.put_nowait(frame)]
                                                                          ↓
                                              [FastAPI WebSocket handler → send_bytes(frame)]
                                                                          ↓
                                                              [Browser canvas drawImage()]
```

Key threading constraint: the sounddevice callback runs on PortAudio's C thread. It must not call asyncio directly. The pattern is:

```python
# In sounddevice callback (C thread):
loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

# In asyncio coroutine (event loop):
chunk = await audio_queue.get()
# → dispatch to GPU pipeline thread
```

---

## Headless ModernGL on Linux Server

Replace the existing GLFW context creation with:

```python
import moderngl
ctx = moderngl.create_context(
    standalone=True,
    backend='egl',
    libgl='libGL.so.1',
    libegl='libEGL.so.1',
)
```

This requires:
- `libegl1-mesa` (or NVIDIA's `libegl1`) installed on the host
- NVIDIA proprietary drivers for GPU-accelerated EGL (Mesa EGL falls back to software rasterization)
- The existing `GLContext` class in `src/subshader/viz/` will need a new code path or a new implementation that returns this context instead of calling GLFW

Confidence: HIGH — verified via ModernGL 5.12.0 official documentation and NVIDIA EGL technical blog.

---

## Hosting: RunPod Persistent Pod

**Recommendation:** RunPod Secure Cloud, persistent Pod, RTX 4090 or A40 tier.

Why RunPod over alternatives:
- Per-second billing — a demo that runs for hours, not weeks, incurs minimal cost
- Docker-native — the deployment artifact is a `Dockerfile`, not a complex Kubernetes manifest
- HTTP proxy included — RunPod exposes `https://<pod-id>-8000.proxy.runpod.net` automatically; no reverse proxy config needed
- NVIDIA GPU access is first-class — unlike general cloud VMs where GPU availability varies

**Approximate cost:** RTX 4090 (24GB) ~$0.39–$0.74/hour on community cloud, ~$0.79/hour on secure cloud (2025 pricing; verify at runpod.io/pricing before budgeting).

The developer's local 4060 Ti (16GB) is the development target. RTX 4090 (24GB) on RunPod is the closest affordable hosted equivalent. The CuPy pipeline will run identically on both.

**Docker base image:** `nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04` — smaller than PyTorch images since the project uses CuPy, not PyTorch.

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| ModernGL 5.12.0 | Python 3.9–3.12 | No known breakage; tested on 3.12 in existing stack |
| sounddevice 0.5.5 | Python 3.7+ | Requires PortAudio; pre-built wheels available for Linux/macOS/Windows |
| FastAPI 0.115.x | Uvicorn 0.34.x | Both follow Starlette's release cycle; always upgrade together |
| CuPy (existing) | CUDA 11.x or 12.x | Match the CUDA version in the Docker base image to the CuPy build. `cupy-cuda12x` for CUDA 12. |
| glcontext 3.x | ModernGL 5.12.x | glcontext is ModernGL's official context backend package; versions must be aligned |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| FastAPI + Uvicorn for WebSocket streaming | HIGH | Official FastAPI docs, widespread production use in 2025, verified on PyPI |
| sounddevice for real-time audio capture | HIGH | PyPI current version 0.5.5 (Jan 2026), official docs confirm callback + NumPy pattern |
| ModernGL EGL headless context | HIGH | Official ModernGL 5.12.0 docs confirm `standalone=True, backend='egl'` |
| JPEG encode via Pillow over WebSocket binary | MEDIUM | Standard pattern; no authoritative benchmark for this specific use case |
| RunPod for hosting | MEDIUM | Pricing and GPU availability are market-variable; deployment pattern is confirmed via RunPod's own FastAPI guide |
| EGL on Docker + NVIDIA | MEDIUM | Pattern is documented by NVIDIA and in production use, but Docker config can be finicky; test early |
| Audio latency end-to-end (capture → browser) | LOW | Highly dependent on buffer sizes, network, and GPU pipeline throughput; no empirical measurement for this stack yet |

---

## Sources

- [FastAPI WebSockets — official docs](https://fastapi.tiangolo.com/advanced/websockets/) — WebSocket handler pattern, binary frame sending
- [ModernGL 5.12.0 headless Ubuntu guide](https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html) — EGL context creation, verified HIGH confidence
- [ModernGL 5.12.0 on PyPI](https://pypi.org/project/moderngl/) — Current version 5.12.0, released October 2024
- [sounddevice 0.5.5 on PyPI](https://pypi.org/project/sounddevice/) — Current version 0.5.5, released January 2026
- [sounddevice real-time processing — DeepWiki](https://deepwiki.com/spatialaudio/python-sounddevice/4.3-real-time-audio-processing) — Callback threading pattern, asyncio queue integration
- [sounddevice GitHub issue #187 — thread safety](https://github.com/spatialaudio/python-sounddevice/issues/187) — `call_soon_threadsafe` pattern confirmed
- [RunPod FastAPI deployment guide](https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud) — Docker + GPU + port proxy pattern
- [NVIDIA EGL blog — headless without X server](https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/) — EGL offscreen rendering on NVIDIA Linux
- [RunPod pricing](https://www.runpod.io/pricing) — Verify current GPU hourly rates before budgeting
- [Python audio tools comparison 2025](https://graphlogic.ai/blog/resources-tools/blog/resources-tools/best-python-tools-audio-manipulation/) — sounddevice vs PyAudio tradeoffs
- [Uvicorn official site](https://www.uvicorn.org/) — ASGI server configuration, WebSocket concurrency

---

*Stack research for: SubShader hosted demo milestone — WebSocket streaming, headless GPU rendering, real-time audio capture*
*Researched: 2026-03-21*
