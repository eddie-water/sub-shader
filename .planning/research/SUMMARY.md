# Project Research Summary

**Project:** SubShader — hosted GPU-accelerated CWT audio visualization demo
**Domain:** Server-rendered GPU visualization with WebSocket streaming
**Researched:** 2026-03-21
**Confidence:** MEDIUM-HIGH

## Executive Summary

SubShader is a server-rendered audio visualization demo where a CuPy GPU pipeline computes a continuous wavelet transform (CWT) on an audio stream, ModernGL renders the result to an offscreen framebuffer, and JPEG-encoded frames are broadcast to browser clients over WebSocket. The architecture is a producer-consumer pipeline with a hard threading boundary: the GPU render loop is synchronous and thread-pinned (OpenGL context affinity), the WebSocket server runs in an asyncio event loop on a daemon thread, and an `asyncio.Queue` bridges them. This is the only viable structure — any deviation (async render loop, blocking sends inside the render loop) will either crash or stall the GPU.

The recommended new stack layers are FastAPI + Uvicorn (WebSocket server), sounddevice (real-time audio capture, callback-based with NumPy output), Pillow/turbojpeg (JPEG frame encoding), and ModernGL's EGL backend (headless context on Linux servers). The existing CuPy, ModernGL, and NumPy pipeline is unchanged by this milestone. Cloud deployment targets RunPod persistent Pods (per-second billing, Docker-native, first-class NVIDIA GPU access), which is the only hosting class compatible with a continuously warm GPU pipeline — serverless is not viable due to cold-start latency.

The dominant risks are all in the transition from local dev to hosted server: GLFW context creation fails on headless servers (fix: EGL backend from day one), CuPy GPU memory accumulates in long-running processes without explicit pool management (fix: periodic `free_all_blocks()` + FFT plan cache clearing), and slow WebSocket clients can stall the GPU render loop if decoupled incorrectly (fix: bounded per-client queue with frame-drop policy). Two existing tech debts must be resolved before any server work begins: the hardcoded audio file path in `__main__.py` and bare `except Exception` handlers in `wavelet.py` and `config.py`.

---

## Key Findings

### Recommended Stack

The existing pipeline (CuPy, ModernGL, NumPy, soundfile) requires no changes. The new stack adds four layers for the hosted demo milestone: an async WebSocket server, headless OpenGL rendering, real-time audio capture, and frame encoding. FastAPI 0.115.x + Uvicorn 0.34.x is the clear choice for WebSocket serving — native async, built-in binary frame support, minimal boilerplate. sounddevice 0.5.5 (January 2026) handles real-time audio via PortAudio callbacks that deliver NumPy arrays, plugging directly into the existing CuPy pipeline with one copy. For cloud hosting, RunPod Secure Cloud persistent Pods with RTX 4090 hardware (~$0.74–$0.79/hour) are the correct tier — per-second billing, Docker-native deployment, and first-class NVIDIA GPU access distinguish it from AWS/GCP for this use case.

**Core technologies:**
- FastAPI 0.115.x + Uvicorn 0.34.x: WebSocket server and ASGI runner — native async, binary frame support, standard for Python async APIs in 2025
- sounddevice 0.5.5: real-time audio capture — NumPy-native callback API, wraps PortAudio, pip-installable without compile-time headers
- Pillow 11.x (then turbojpeg): JPEG frame encoding — standard, sufficient to start; turbojpeg is a drop-in swap if encoding becomes the bottleneck (3-5x faster)
- ModernGL EGL backend (`standalone=True, backend='egl'`): headless GPU rendering — replaces GLFW window context for server deployment; requires `libegl1-mesa` on the host
- RunPod Secure Cloud: GPU hosting — per-second billing, Docker-native, HTTP proxy included, NVIDIA-first

**Critical version constraints:**
- CuPy build must match CUDA version in Docker base image (use `cupy-cuda12x` for CUDA 12)
- FastAPI and Uvicorn must be upgraded together (both follow Starlette's release cycle)
- glcontext version must align with ModernGL 5.12.x

### Expected Features

**Must have — demo is broken without these (P1):**
- WebSocket frame stream from server CWT pipeline to browser canvas — the entire transport layer; everything else depends on this
- Continuous server-side audio loop with frames streamed on page load — demo must be live on arrival, no user input required
- Audio-visual sync within ~100ms — the core value claim of the project; may require reducing resolution before frame rate
- Connection status indicator and auto-reconnect — demo must recover without a page refresh
- Hard connection limit (2-3 concurrent WebSocket clients) and per-IP rate limiting — protects the single GPU from being monopolized
- One-sentence explanation on page of what the visualization shows — essential for non-DSP audiences

**Should have — add after core pipeline is stable (P2):**
- GPU benchmark panel (CuPy vs NumPy timing) — the GPU acceleration story made concrete; infrastructure exists, surfacing it in UI is the work
- Color palette selector — increases perceived interactivity with minimal backend changes; requires GLSL shader parameterization
- Resolution/quality slider — practical latency knob, not cosmetic; lowering resolution directly reduces end-to-end pipeline time
- 3-5 curated track selection — adds variety after user feedback; server-side, no audio pipeline changes

**Defer to v2+ (P3):**
- Browser microphone input — high user value but requires browser-to-server audio upload pipeline; conflicts with single shared GPU pipeline constraint
- Audio file upload — requires input validation, abuse protection; defer until rate limiting is hardened
- Pedagogical annotation overlay — interesting but high effort; only worth building if the demo gains an audience

**Anti-features to avoid:** Client-side WebGL CWT rewrite (defeats the GPU story), per-client GPU pipelines (OOMs the 4060 Ti at 2-3 clients), user-uploaded arbitrary audio files without validation.

**Key competitive insight:** Client-side browser spectrograms (academo.org, borismus/spectrogram) win on latency and zero setup. SubShader's edge is CWT transform quality and the GPU-acceleration story, not latency. The demo should foreground this distinction explicitly.

### Architecture Approach

The system has three layers: a GPU pipeline producing rendered frames (synchronous, thread-pinned), a streaming layer moving JPEG bytes over an `asyncio.Queue` thread boundary, and a browser client displaying frames on a canvas via `createImageBitmap` + `drawImage`. The GPU render loop stays synchronous on its own OS thread; the FastAPI WebSocket server runs in an asyncio event loop on a daemon thread. These two loops never share a thread and communicate only through the bounded queue. All clients receive the same frame from a single shared render loop — no per-client GPU pipelines.

**Major components and build order:**
1. OffscreenRenderer — EGL context + FBO rendering + pixel readback (`fbo.read(components=3)`); replaces GLFW-windowed GLContext; highest-risk unknown, de-risk first
2. FrameEncoder — `numpy bytes → JPEG bytes` via Pillow/turbojpeg; pure function, trivially testable
3. StreamBridge — `asyncio.Queue` + `loop.call_soon_threadsafe()`; thread boundary between render and async layers; verifiable without network
4. WebSocketServer — client registration set, broadcast loop, connection/rate limits; verifiable with a test client
5. Server entry point — wires all components together; replaces `ShaderPlot` with `OffscreenRenderer` in the main loop
6. Browser client — minimal HTML/JS: `new WebSocket(url)`, `onmessage → createImageBitmap → drawImage`
7. Deployment — Docker, nginx/Caddy reverse proxy, TLS, process management

**Key architectural rules:**
- Never await `websocket.send()` inside the render loop — use a bounded queue with frame-drop policy
- Never run OpenGL calls from the asyncio event loop thread — context thread affinity is a hard constraint
- Send JPEG binary frames, never raw pixels (6MB/frame at 1080p) and never base64 (33% size overhead)
- Use `put_nowait()` with `QueueFull` catch to drop frames under backpressure — the render loop must never block on network I/O

### Critical Pitfalls

1. **OpenGL context on wrong thread** — GLFW requires main-thread context creation; web servers want their event loop on the main thread. Resolve by inverting: GPU pipeline runs first (main thread or dedicated thread using EGL, not GLFW), WebSocket server in a daemon thread. Use EGL from the start to eliminate GLFW's main-thread constraint entirely.

2. **GLFW/X11 breaks headless server deployment** — The existing codebase uses GLFW, which requires X11 or explicit EGL config. Servers have neither. Symptom: works locally, exits immediately on server. Fix: `moderngl.create_context(standalone=True, backend='egl')` and `libegl1-mesa` installed in Docker image. Write a headless render smoke-test before any WebSocket code.

3. **CuPy GPU memory accumulation** — Long-running server processes accumulate VRAM from memory pool fragmentation and persistent FFT plan caches. The existing `wavelet.py` already has incomplete GPU cleanup on error paths. Fix: wrap CWT computations in context managers that call `cp.get_default_memory_pool().free_all_blocks()` and `cp.fft.config.get_plan_cache().clear()` on exit/exception; add `nvidia-smi` monitoring; pre-allocate fixed GPU arrays at startup.

4. **Render loop blocking on slow WebSocket clients** — Calling `await websocket.send(frame)` inside the render loop lets slow clients' TCP backpressure stall the GPU. Fix: bounded `asyncio.Queue(maxsize=2-4)` with frame-drop policy; render loop uses `put_nowait()` wrapped in `try/except QueueFull`.

5. **JPEG encoding latency consuming the entire frame budget** — Pillow JPEG encoding at 720p takes 15-25ms on a single CPU core, consuming most of a 33ms frame budget before network transit. Fix: profile the full pipeline with timestamps at each stage before optimizing; switch to turbojpeg (3-5x faster, drop-in replacement); reduce streaming resolution to 640x360 or 960x540 for a heatmap-style visualization.

6. **Hardcoded audio file path breaks server deployment** — `__main__.py:45` hardcodes a local dev path; bare exception handlers in `config.py:46` swallow the resulting `FileNotFoundError`, producing silent blank frames. Fix this before any server work begins: replace with CLI arg or environment variable and add startup validation.

---

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Codebase Hardening and Configuration Cleanup
**Rationale:** Two existing defects will silently sabotage every subsequent phase. The hardcoded audio path produces blank frames with no error. The bare `except Exception` handlers mask GPU and file I/O errors in server contexts. These must be fixed before any server code is written — not after. This phase has no new dependencies and is verifiable in isolation.
**Delivers:** A server-deployable codebase baseline with explicit audio source configuration and honest error propagation.
**Addresses:** Audio source hardcoding (PITFALLS.md Pitfall 7), bare exception handler tech debt (PITFALLS.md Technical Debt Patterns).
**Avoids:** Silent blank-frame failures that waste debugging time across all subsequent phases.
**Research flag:** Standard patterns — no phase research needed.

### Phase 2: Headless GPU Rendering (OffscreenRenderer + EGL)
**Rationale:** EGL headless rendering is the single highest-risk unknown. If it fails (wrong library linkage, VRAM access denied in Docker, WSL2 quirks), every downstream phase is blocked. It must be de-risked before the WebSocket layer is built. This phase produces a smoke-testable artifact: run server without a display, render one frame, assert bytes have correct shape.
**Delivers:** `OffscreenRenderer` class with EGL context + FBO + `fbo.read()` pixel readback; verified working with `DISPLAY` unset.
**Uses:** ModernGL EGL backend (`standalone=True, backend='egl'`), glcontext 3.x, `libegl1-mesa` (STACK.md).
**Implements:** OffscreenRenderer component (ARCHITECTURE.md Component Boundaries).
**Avoids:** GLFW/X11 headless failure (PITFALLS.md Pitfall 2), OpenGL context threading failure (PITFALLS.md Pitfall 1).
**Research flag:** Likely needs phase research — EGL on WSL2 dev + NVIDIA driver interaction has documented quirks; Docker EGL configuration is non-trivial.

### Phase 3: Frame Encoding Pipeline
**Rationale:** FrameEncoder is a pure function with no network or GPU dependencies. It can be built and fully tested immediately after OffscreenRenderer produces pixel bytes. Building it before the WebSocket layer allows the encoding latency to be profiled in isolation and turbojpeg introduced if needed — before the full pipeline makes benchmarking harder.
**Delivers:** `FrameEncoder` module converting raw RGB bytes to JPEG bytes; latency benchmark confirming encoding stays within frame budget at target resolution.
**Uses:** Pillow 11.x (STACK.md); turbojpeg as drop-in if encoding exceeds ~15ms at target resolution.
**Avoids:** JPEG encoding latency trap (PITFALLS.md Pitfall 5), base64 anti-pattern.

### Phase 4: WebSocket Streaming Layer
**Rationale:** With headless rendering and frame encoding validated, the streaming layer can be built on a known-good frame source. Building in order (StreamBridge → WebSocketServer → server entry point) allows each component to be tested independently before wiring the full pipeline. The decoupled queue architecture must be established here — retrofitting it later is expensive.
**Delivers:** Full server-to-browser streaming pipeline at 24+ fps; browser canvas displaying live CWT frames; auto-reconnect on disconnect; connection status indicator.
**Uses:** FastAPI 0.115.x + Uvicorn 0.34.x, asyncio.Queue + call_soon_threadsafe, WebSocket binary frames (STACK.md).
**Implements:** StreamBridge + WebSocketServer + server entry point (ARCHITECTURE.md Build Order).
**Avoids:** Render loop blocking on send (PITFALLS.md Pitfall 4), sending raw pixel bytes (ARCHITECTURE.md Anti-Pattern 3).
**Research flag:** Standard patterns — FastAPI WebSocket docs, websockets library broadcast patterns, asyncio producer-consumer are all well-documented with official sources.

### Phase 5: GPU Pipeline Hardening and Memory Management
**Rationale:** The streaming pipeline must run for hours without degradation before it is shared publicly. CuPy memory pool growth and FFT plan cache accumulation are invisible in short dev sessions but cause OOM crashes after 2-4 hours of continuous operation. This phase also enforces the connection limit and rate limiting that protect the GPU from being monopolized.
**Delivers:** Confirmed stable VRAM usage over 30-minute soak test; per-connection resource accounting; hard limit of 2-3 concurrent WebSocket connections; per-IP rate limiting; WebSocket idle timeouts.
**Uses:** `cp.get_default_memory_pool().free_all_blocks()`, `cp.fft.config.get_plan_cache().clear()`, FastAPI connection middleware (STACK.md, PITFALLS.md).
**Avoids:** CuPy GPU memory accumulation (PITFALLS.md Pitfall 3), single client resource exhaustion (PITFALLS.md Pitfall 6).
**Research flag:** Standard patterns for rate limiting and CuPy memory management — official CuPy docs are authoritative.

### Phase 6: Browser Client and UX Polish
**Rationale:** The first visible user experience is assembled here, on top of a validated and stable streaming backend. This phase includes the loading state, reconnect UI, the one-sentence visualization explanation, and HTTPS/TLS. Deferring UX until the backend is stable prevents UX work from being discarded due to backend changes.
**Delivers:** Polished browser client with loading state, disconnect handling, visualization label; HTTPS secured; tested in Chrome and Firefox.
**Avoids:** UX pitfalls (PITFALLS.md UX Pitfalls section) — blank canvas on load, silent disconnect, no audio feedback.
**Research flag:** Standard patterns — no phase research needed.

### Phase 7: Cloud Deployment
**Rationale:** All previous phases can be validated locally (with `DISPLAY` unset to simulate headless). Cloud deployment is the final step — Docker image, RunPod persistent Pod, nginx/Caddy reverse proxy, TLS, process management. It is last because deployment issues are easier to debug when the application is already known-good.
**Delivers:** Publicly accessible demo URL on RunPod; Docker image with all CUDA/EGL dependencies; production logging configuration; GPU monitoring.
**Uses:** RunPod Secure Cloud RTX 4090 tier (~$0.74-0.79/hour), `nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04` Docker base image (STACK.md Hosting section).
**Avoids:** Serverless cold-start anti-pattern, MJPEG/WebRTC overengineering (STACK.md What NOT to Use).
**Research flag:** Likely needs phase research — RunPod Docker EGL configuration, NVIDIA container toolkit on RunPod, Caddy automatic TLS setup are all narrower patterns with fewer documented examples.

### Phase 8: v1.x Enhancements (Post-Validation)
**Rationale:** After the demo is publicly accessible and stable, add enhancements based on evidence of reception. The GPU benchmark panel, color palette selector, and resolution slider are all independently addable without architectural changes.
**Delivers:** GPU benchmark panel (CuPy vs NumPy), color palette selector, resolution/quality slider; optionally a curated track selection if a single loop feels limiting.
**Implements:** P2 features from FEATURES.md feature prioritization matrix.
**Research flag:** Color palette requires GLSL shader parameterization research; benchmark panel surfacing is straightforward.

### Phase Ordering Rationale

- Phases 1-2 must come before everything else because the whole pipeline is blocked on (1) having a deployable codebase and (2) knowing EGL headless works.
- Phases 3-4 are a natural unit: encode then stream. They could be collapsed if schedule pressure is high, but separating them makes per-component benchmarking easier.
- Phase 5 must come before Phase 7 (public deployment) — releasing a demo with unmanaged GPU memory is a reliability guarantee of needing to restart the server on a user's first evening using it.
- Phase 6 is decoupled from the backend phases and could run in parallel with Phase 5 in a two-developer scenario.
- Phase 7 last: deploy only what is known to work.

---

### Research Flags

Phases needing deeper research during planning:
- **Phase 2 (Headless GPU Rendering):** EGL backend on WSL2 + NVIDIA drivers is documented to have quirks; Docker EGL configuration with nvidia-container-toolkit requires specific flags; the interaction between `libegl1-mesa` and NVIDIA proprietary EGL is non-obvious.
- **Phase 7 (Cloud Deployment):** RunPod-specific Docker configuration for EGL access, Caddy HTTPS on RunPod's HTTP proxy, nvidia-smi monitoring in a container — these are narrower patterns than the core streaming architecture.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Hardening):** Removing hardcoded paths and narrowing exception handlers — straightforward refactoring.
- **Phase 3 (Frame Encoding):** Pillow and turbojpeg JPEG encoding are well-documented; the encode-to-BytesIO pattern has no ambiguity.
- **Phase 4 (WebSocket Streaming):** FastAPI WebSocket docs, asyncio producer-consumer, and websockets library broadcast patterns are all covered by official documentation with high confidence.
- **Phase 5 (GPU Hardening):** CuPy memory management API is authoritative in official docs; connection rate limiting in FastAPI is well-documented.
- **Phase 6 (Browser UX):** Canvas + WebSocket browser patterns are universally documented; no exotic APIs required.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core recommendations (FastAPI, sounddevice, ModernGL EGL) verified via official docs and PyPI; RunPod pricing is market-variable — verify at runpod.io/pricing before budgeting |
| Features | MEDIUM | Feature landscape well understood; GPU-streaming-to-browser is a narrower pattern — latency numbers (50-200ms) are estimates, not empirical measurements for this specific stack |
| Architecture | MEDIUM-HIGH | Threading constraints (OpenGL affinity, asyncio queue pattern) verified via official Khronos and CPython docs; JPEG streaming latency from a single informal experiment (LOW confidence on that specific number) |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls sourced from official docs and confirmed GitHub issues; hosting-specific pitfalls from community sources |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **End-to-end latency:** No empirical measurement exists for this specific stack (CuPy CWT → ModernGL EGL → Pillow JPEG → FastAPI WebSocket → browser canvas). Research cites 50-200ms as a reasonable estimate but this must be profiled during Phase 4. If it consistently exceeds 100ms at target resolution, resolution must be reduced.
- **EGL on WSL2 during development:** All server-side research targets Linux bare metal or Docker. The developer's primary environment is WSL2. EGL on WSL2 has known limitations (GPU passthrough depends on WSL2 GPU driver support). May need Xvfb as a dev fallback with EGL for production — this should be decided in Phase 2 before it becomes a blocking assumption.
- **sounddevice for hosted demo:** sounddevice captures live audio from a physical mic or system audio device. A hosted cloud server has neither. For the hosted milestone, audio input must switch to a looped audio file. The STACK.md variant section documents this correctly; it must be reflected in Phase 1 configuration cleanup — the audio backend selection must be an environment variable, not a code change.
- **JPEG encoding at target resolution:** The 15-25ms Pillow estimate and 3-5x turbojpeg speedup are from benchmarks at various resolutions. Actual numbers for this visualization's target resolution (960x540 or 640x360) need to be measured in Phase 3 before committing to Pillow vs. turbojpeg.

---

## Sources

### Primary (HIGH confidence)
- FastAPI WebSockets official docs — WebSocket handler pattern, binary frame sending
- ModernGL 5.12.0 headless Ubuntu guide (readthedocs) — EGL context creation, FBO API
- sounddevice 0.5.5 PyPI + official docs — callback threading pattern, NumPy integration
- OpenGL and multithreading — Khronos official — thread affinity constraints
- asyncio event loop docs — CPython official — `call_soon_threadsafe` pattern
- CuPy memory management — official CuPy docs — pool API, FFT plan cache
- NVIDIA EGL blog — headless OpenGL without X server
- websockets library official docs — broadcast patterns, backpressure

### Secondary (MEDIUM confidence)
- RunPod FastAPI deployment guide — Docker + GPU + port proxy pattern
- sounddevice GitHub issue #187 — thread safety pattern confirmed
- Python audio tools comparison 2025 (graphlogic.ai) — sounddevice vs PyAudio tradeoffs
- WebSocket rate limiting (OneUptime, 2026) — connection abuse protection patterns
- Wavelet spectrogram CWT advantages (Medium) — CWT differentiator over STFT articulated
- JSMpeg — establishes ~50ms WebSocket streaming latency as achievable for frame streaming

### Tertiary (LOW confidence)
- JPEG streaming latency observations (iimachines/MotionJpegLatencyTest) — single informal experiment, use as directional guidance only
- simplejpeg performance claim (PyPI page) — unverified; benchmark in Phase 3 before committing

---
*Research completed: 2026-03-21*
*Ready for roadmap: yes*
