# Pitfalls Research

**Domain:** Server-rendered GPU streaming — CuPy/OpenGL pipeline hosted as a WebSocket web demo
**Researched:** 2026-03-21
**Confidence:** MEDIUM-HIGH (critical pitfalls from official sources and documented issues; hosting specifics from community sources)

---

## Critical Pitfalls

### Pitfall 1: OpenGL Context Created on the Wrong Thread

**What goes wrong:**
GLFW requires window creation and event processing to happen on the main OS thread. If the server architecture spawns the render loop on a worker thread (to free the main thread for asyncio or HTTP), OpenGL context creation fails silently or with a cryptic platform error. Even if it succeeds on some platforms, context sharing between threads requires explicit make-current/release sequencing that is easy to get wrong, causing GPU corruption or deadlocks.

**Why it happens:**
Web server frameworks (FastAPI, aiohttp) run their event loop on the main thread. Developers assume they can move the GPU render loop to a background thread the same way they offload any CPU task — but OpenGL's threading model is fundamentally different. GLFW's own docs state: "Initialization, termination, event processing and the creation and destruction of windows, cursors and OpenGL contexts are all restricted to the main thread."

**How to avoid:**
Invert the architecture: run the GPU render loop on the main thread and run the WebSocket/HTTP server on a separate thread using `threading.Thread` with `asyncio.run()`. Alternatively, use ModernGL's standalone EGL context (no GLFW at all), which removes the main-thread requirement entirely. The EGL path is the correct choice for headless server deployment.

**Warning signs:**
- Context creation succeeds locally (where a display is attached) but fails on the server
- Errors like `GLFWError: GLFW_NOT_INITIALIZED` or `EGL_NOT_INITIALIZED` on first request
- Works in dev then crashes on first production connection

**Phase to address:**
Server architecture phase — before any WebSocket code is written. The threading model must be decided first.

---

### Pitfall 2: GLFW/X11 Dependency Breaks Headless Server Deployment

**What goes wrong:**
The existing codebase uses GLFW for context creation (via ModernGL + moderngl-window). GLFW requires either a running X11 display server or explicit EGL headless configuration. On a typical Linux server or cloud VM with no GPU desktop environment, OpenGL context creation throws `cannot connect to X server` or `GLFWError: GLFW_PLATFORM_ERROR`. The application appears to work in dev (where DISPLAY is set) and fails immediately in production.

**Why it happens:**
GLFW defaults to X11 on Linux. Servers don't run X11. The NVIDIA GLVND architecture requires linking against `libOpenGL.so` + `libEGL.so` (not the legacy `libGL.so`) for headless EGL rendering. Many tutorials link against the wrong library, which works on desktop but not on headless systems.

**How to avoid:**
Switch to ModernGL's EGL backend for the server path: `moderngl.create_context(standalone=True, backend='egl')`. Install `libegl1-mesa` and `libgl1-mesa-glx`. Do not use `Xvfb` as a crutch — it works but adds unnecessary overhead and is fragile. If GLFW must be retained for local dev, gate it behind an environment variable and use the EGL path in all server contexts. Verify with `glxinfo` and a minimal offscreen render test before building any streaming layer.

**Warning signs:**
- Dev machine has `DISPLAY=:0` set; server does not
- Application runs fine locally but the server process exits immediately on start
- Error: `libGL error: No matching fbConfigs or visuals found`

**Phase to address:**
Server infrastructure phase — first server deployment attempt. Write a headless render smoke-test before wiring up any WebSocket logic.

---

### Pitfall 3: CuPy GPU Memory Accumulates Until the Process Crashes

**What goes wrong:**
CuPy uses a memory pool by default. In a long-running server process, memory fragments and pools grow without bound if allocations aren't returned to the pool correctly. Known CuPy issues document that FFT plan caches do not deallocate during thread cleanup, and multithreaded use exacerbates leaks. For a hosted demo serving visitors over hours, the GPU OOMs and either the process crashes or the server begins rejecting CUDA allocations mid-render.

**Why it happens:**
The existing code already has incomplete GPU memory cleanup on error paths (noted in CONCERNS.md: `wavelet.py:609-620`). In a short local session this is invisible — the process exits and the OS cleans up. In a server process that runs for hours, every exception, every failed render, and every cached FFT plan that isn't explicitly freed accumulates. CuPy's pool holds freed blocks for reuse but does not release them to the OS unless explicitly told to.

**How to avoid:**
- Wrap every CWT computation in a context manager that calls `cp.get_default_memory_pool().free_all_blocks()` and `cp.fft.config.get_plan_cache().clear()` on exit or exception
- Add periodic GPU memory monitoring (e.g., log `cp.get_default_memory_pool().used_bytes()` every N frames)
- Pre-allocate fixed-size GPU arrays at startup rather than dynamically allocating per-frame
- Set a memory pool cap: `cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)`
- Fix the existing bare exception handlers in `wavelet.py` before any server deployment

**Warning signs:**
- `nvidia-smi` shows VRAM climbing over hours with no release
- CuPy raises `OutOfMemoryError` after running fine for several hours
- Server must be restarted every N hours to recover

**Phase to address:**
GPU pipeline hardening phase — before the server is made publicly accessible.

---

### Pitfall 4: Render Loop Blocks the WebSocket Send, Causing Head-of-Line Backpressure

**What goes wrong:**
The GPU render loop produces frames faster than slow clients can consume them. If the server awaits `websocket.send(frame)` synchronously inside the render loop, a slow client's TCP backpressure propagates back through the entire pipeline, blocking the GPU. All other clients starve. With a single-GPU server this is especially severe — one slow connection pauses the entire visualization.

**Why it happens:**
Developers wire the pipeline naively: `render_frame() → encode_jpeg() → await ws.send()`. The send blocks when the client's receive buffer is full. Because asyncio is single-threaded, this blocks every other coroutine. The GPU sits idle waiting for a TCP ACK from a client on a slow mobile connection.

**How to avoid:**
Decouple the render loop from the send loop using a per-client frame queue with a maximum depth of 1-2 frames. Use drop-newest or drop-oldest policy — for a visual demo, dropping stale frames is always correct. Never block the render loop on network I/O. The pattern:

```python
# Render loop (main thread or dedicated thread)
latest_frame = frame  # always overwrite, never queue

# Send loop (per WebSocket connection, asyncio task)
while connected:
    frame = await frame_event.wait()
    await ws.send(frame)  # slow client only hurts itself
```

This ensures no single client can stall the GPU pipeline.

**Warning signs:**
- Frame rate drops when a second browser tab is opened
- GPU utilization drops to near zero while a client is connected
- `websockets` library warns about slow consumers or write buffer growth

**Phase to address:**
WebSocket streaming phase — wire the send/render decoupling from the first frame sent.

---

### Pitfall 5: JPEG Encoding Latency Eats the Entire Latency Budget

**What goes wrong:**
Each rendered frame must be read from the GPU (framebuffer readback), encoded to JPEG or PNG, and sent over the wire. GPU-to-CPU transfer + CPU JPEG encoding is typically 10-40ms per frame on a CPU encoder. At 30 FPS, the total latency budget is ~33ms per frame. JPEG encoding alone can consume 15-25ms of that, leaving no headroom for CWT computation, WebSocket framing, or network jitter. The result is a visualization that feels laggy even though the GPU renders fast.

**Why it happens:**
Pillow's `Image.save()` to a BytesIO buffer is single-threaded CPU work. Developers benchmark the GPU pipeline in isolation, see 5ms frame times, and are surprised when end-to-end latency is 80ms. Framebuffer readback (`gl.read()` → CPU numpy array → encode) is the invisible bottleneck.

**How to avoid:**
- Use `turbojpeg` (libjpeg-turbo bindings) instead of Pillow — typically 3-5x faster JPEG encoding
- Reduce output resolution: a 640x360 stream encodes 4x faster than 1280x720 with similar visual quality for a color-mapped spectrogram
- Profile the full pipeline with timestamps at: CWT done, render done, readback done, encode done, send done — identify the actual bottleneck before optimizing
- Consider PNG only if lossless is required; for this visualization, JPEG at quality 85 is indistinguishable and significantly faster
- Move encoding to a thread pool via `asyncio.get_event_loop().run_in_executor()` to avoid blocking the event loop

**Warning signs:**
- Frame timestamps show <5ms GPU time but >40ms total pipeline time
- CPU usage is high on a single core while GPU sits at low utilization
- Adding more clients doesn't change per-frame GPU time but increases per-frame wall time

**Phase to address:**
Streaming pipeline phase — profile before optimizing, but use turbojpeg from the start.

---

### Pitfall 6: A Single Connected Client Can Exhaust the GPU

**What goes wrong:**
Without connection limits or per-client resource accounting, multiple simultaneous connections each trigger independent CWT pipelines or force the GPU to process multiple streams. On a 4060 Ti with 16GB VRAM, 2-3 simultaneous clients running full CWT pipelines can OOM the GPU. Even without OOM, compute throughput drops and all clients receive degraded frame rates.

**Why it happens:**
For a personal hosted demo, developers assume "it's just friends" and skip resource limits. But public demos get crawled, shared on social media, or hit by curiosity bots. A single malicious or accidental connection that reconnects in a tight loop can keep the GPU pegged at 100%.

**How to avoid:**
- Enforce a hard maximum concurrent connection limit (1-3 for a single 4060 Ti running full CWT)
- Implement connection rate limiting: max N connections per IP per minute
- Serve all clients from a **single shared render loop** — one CWT computation, one frame, broadcast to all connected clients. Do not fork per-client GPU pipelines
- Queue new connections if at capacity and reject after a timeout with a useful HTTP 503 message
- Set WebSocket idle timeouts — disconnect clients that haven't acknowledged frames in >30 seconds

**Warning signs:**
- GPU utilization spikes to 100% when a second connection opens
- Memory usage grows linearly with connections
- Server becomes unresponsive when more than 2 clients connect

**Phase to address:**
Security and reliability hardening phase — before public URL is shared with anyone.

---

### Pitfall 7: Audio Source Hardcoding Breaks Server Deployment

**What goes wrong:**
The existing codebase hardcodes an audio file path in `__main__.py:45`. In a server context, this path either doesn't exist or references a local dev asset. The server silently fails to load audio, the CWT receives silence or zeros, and the visualization renders a blank or static frame with no obvious error.

**Why it happens:**
The path is a dev shortcut that was never cleaned up (already flagged in CONCERNS.md). On a server, file paths relative to the working directory, or absolute paths pointing to a dev machine's filesystem, are both invalid.

**How to avoid:**
- Replace the hardcoded path with a configurable source before any server work begins — CLI argument, environment variable, or config file
- For a hosted demo streaming a pre-selected audio track: bundle the audio file with the server deployment and reference it via an absolute path set in deployment config
- For a hosted demo with live audio: treat audio input as a swappable backend (file vs. system audio) resolved at startup, not hardcoded in source
- Add startup validation that verifies the audio source is readable before binding the WebSocket port

**Warning signs:**
- Server starts, accepts connections, but streams static or empty frames
- No error is logged because the bare exception handlers in `config.py:46` swallow the FileNotFoundError

**Phase to address:**
Configuration cleanup phase — fix before the first server run.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Bare `except Exception` handlers | Prevents crashes during dev | Masks real errors in production; audio source failures silently produce blank frames | Never in production code |
| Per-frame `np.percentile()` in IntensityTracker | Simple, correct | Expensive at high frame rates; adds 5-15ms CPU overhead per frame | Only if frame rate is low (<15 FPS) |
| `log.info()` inside the per-frame CWT loop | Useful during dev | Fills disk and adds 2-5ms per frame at production logging volumes | Never in server deployment |
| GLFW context (dev-only path) kept in server code | Easier to test locally | Context creation fails silently on headless server; EGL path never exercised | Only if dev/prod paths are clearly separated by env flag |
| Xvfb virtual display for headless | Gets OpenGL working fast | Adds a fragile process dependency; crashes if Xvfb dies; EGL is strictly better | Never — use EGL instead |
| Circular frame buffer flattened every frame | Correct, simple | GPU texture upload latency accumulates; ~1-3ms of unnecessary copy per frame | MVP only; replace with GPU-side ring buffer |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| CuPy + asyncio | Running CuPy ops inside `async def` without thread isolation | CuPy blocks the event loop; run all GPU work in a dedicated thread or process, communicate results via asyncio Queue |
| ModernGL EGL on NVIDIA | Linking against `libGL.so` (legacy) | Link against `libOpenGL.so` + `libEGL.so` (GLVND); requires NVIDIA driver 361.28+ |
| WebSocket + slow clients | Awaiting send inside render loop | Use per-client bounded queue with drop policy; render loop never waits on network |
| CuPy FFT plan cache | Assuming memory is freed when arrays are deleted | Explicitly call `cp.fft.config.get_plan_cache().clear()` periodically; FFT plans persist in cache indefinitely |
| GLFW + server | Creating window on a server without `DISPLAY` set | Use `moderngl.create_context(standalone=True, backend='egl')` — no GLFW, no display required |
| FastAPI/aiohttp + GPU render | Running render loop as asyncio task | GPU work is synchronous and CPU-blocking; use `run_in_executor()` or a dedicated thread with `asyncio.Queue` for frame delivery |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-frame framebuffer readback without timing | Total latency 5x what GPU time suggests | Measure readback separately; consider async PBO readback | At any resolution above ~512x256 |
| Pillow JPEG encoding on main thread | Single CPU core saturated; frame rate drops under load | Use turbojpeg or run encoding in thread pool | Above ~15 FPS at 720p |
| CuPy memory pool growth unchecked | VRAM usage climbs 100MB/hour; eventual OOM after N hours | Monitor pool size; call `free_all_blocks()` periodically | After ~2-4 hours of continuous operation |
| Broadcasting frames to slow clients without drop policy | One slow client causes all clients to lag | Bounded per-client queue with drop-newest policy | When any client's RTT exceeds frame interval |
| Per-frame `np.percentile()` on full frame data | CPU usage high even when GPU is idle | Cache or approximate percentile; update every N frames | At >20 FPS with large frame arrays |
| WebSocket sending text-encoded base64 frames | 33% larger payload than binary; visible bandwidth waste | Send raw binary frames using `ws.send(bytes)` | Immediately; no threshold |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| No connection limit | Single visitor or bot can consume 100% GPU, crashing the demo for everyone | Hard cap of 2-3 simultaneous WebSocket connections; HTTP 503 with retry-after header when at capacity |
| No connection rate limit per IP | Reconnect loops exhaust TCP connection overhead even without GPU load | Rate limit new connections: max 5 per IP per minute using a token bucket in the WebSocket handshake handler |
| Exposing the raw Python process directly on port 80/443 | Process crash takes down the entire server; no TLS | Put Nginx or Caddy in front; use Caddy for automatic TLS; Python process listens only on localhost |
| No WebSocket idle timeout | Clients that navigate away leave connections open indefinitely, consuming GPU broadcast slots | Close connections that haven't sent a pong in >30 seconds |
| Audio file path user-configurable without validation | Path traversal if user can influence the audio source parameter | For a read-only demo, fix the audio source at deployment time; never accept user-supplied file paths |
| Debug logging left enabled in production | Leaks internal paths, memory addresses, and timing data | Set log level to WARNING or ERROR in production; use environment variable to control level |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No loading state while first CWT frame renders | User sees blank canvas for 1-3 seconds; assumes the page is broken | Show a "Initializing..." or animated placeholder until the first frame arrives |
| No error state when WebSocket disconnects | Stream silently stops; user has no idea why | Detect `onclose` and show a reconnect button with a brief explanation |
| Streaming at full GPU resolution (e.g., 1920x1080) | Bandwidth-limited users see stuttering; no benefit over 720p for a spectrogram | Default to a sensible streaming resolution (640x360 or 960x540); full resolution is wasteful for this content type |
| No visual indication that audio is live vs. silence | User cannot tell if the visualization is working or if audio input is just quiet | Add a simple RMS level indicator or a "listening" badge |
| Page blocks WebSocket behind an HTTP auth wall | Demo URL shared in a tweet; visitors hit an auth prompt and leave | For a public demo, no auth on the stream endpoint; IP rate limiting is sufficient protection |

---

## "Looks Done But Isn't" Checklist

- [ ] **EGL headless mode:** Works locally with `DISPLAY` set — verify it also works with `DISPLAY` unset on the same machine before deploying
- [ ] **GPU memory:** Runs for 30 minutes without memory growth — check `nvidia-smi` before declaring stable
- [ ] **Frame drop policy:** Second browser tab opens — verify first tab does not degrade
- [ ] **Audio source:** Server starts without assets directory present — verify it fails clearly, not silently
- [ ] **Connection cleanup:** WebSocket client navigates away — verify GPU resources are released within 5 seconds
- [ ] **JPEG encoding latency:** Full pipeline latency measured end-to-end with timestamps — not just GPU compute time
- [ ] **Exception handler breadth:** All `except Exception` blocks converted to specific types — bare handlers mask server-critical errors
- [ ] **Logging level:** `log.info()` inside CWT loop removed or gated behind `DEBUG` — verify it is off in production config

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| GLFW/X11 context failure in production | MEDIUM | Switch render context to EGL backend; remove GLFW dependency from server path; requires testing on real server before re-deploy |
| CuPy OOM after memory leak | LOW | Restart server process; implement `free_all_blocks()` calls; deploy monitoring before restarting |
| Render loop blocking on slow client | MEDIUM | Refactor send loop to use per-client queue; no GPU code changes needed, only WebSocket layer |
| JPEG encoding bottleneck | LOW | Drop-in turbojpeg replacement for Pillow encode calls; no architecture change |
| Audio hardcode breaks server | LOW | Set environment variable or config file for audio path; no code change beyond removing hardcoded string |
| Single client monopolizing GPU | MEDIUM | Add connection limit in WebSocket handshake handler; switch to single shared render loop broadcast pattern |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| OpenGL context on wrong thread | Server architecture design (before code) | Render loop and WebSocket server start cleanly with no threading errors |
| GLFW/X11 breaks headless | Server infrastructure setup | `DISPLAY=` (unset) python server.py starts and serves a frame successfully |
| CuPy memory accumulation | GPU pipeline hardening | `nvidia-smi` shows stable VRAM after 30-minute soak test |
| Render loop blocks on send | WebSocket streaming layer | Opening a second tab does not degrade first tab frame rate |
| JPEG encoding latency | Streaming pipeline profiling | Full pipeline latency measured; P95 < 50ms end-to-end |
| Client resource exhaustion | Security hardening (before public URL) | Connection limit enforced; fourth connection receives HTTP 503 |
| Audio source hardcoding | Configuration cleanup (first server task) | Server starts cleanly with audio path set via env var; missing path fails loudly |

---

## Sources

- NVIDIA Technical Blog — Linking OpenGL for Server-Side Rendering: https://developer.nvidia.com/blog/linking-opengl-server-side-rendering/
- ModernGL headless Ubuntu server guide: https://moderngl.readthedocs.io/en/5.10.0/techniques/headless_ubuntu_18_server.html
- GLFW threading constraints (official docs): https://www.glfw.org/docs/3.3/context_guide.html
- CuPy memory management (official docs): https://docs.cupy.dev/en/stable/user_guide/memory.html
- CuPy multithreaded cufft memory leak (GitHub issue #6355): https://github.com/cupy/cupy/issues/6355
- websockets library backpressure documentation: https://websockets.readthedocs.io/en/stable/topics/memory.html
- Managing WebSocket backpressure in FastAPI (HexShift): https://hexshift.medium.com/managing-websocket-backpressure-in-fastapi-applications-893c049017d4
- EGL headless without X server (NVIDIA blog): https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
- CuPy/OpenGL CUDA memory sharing (GitHub issue #5711): https://github.com/cupy/cupy/issues/5711
- WebSocket rate limiting: https://oneuptime.com/blog/post/2026-01-24-websocket-rate-limiting/view
- Armin Ronacher — async backpressure: https://lucumr.pocoo.org/2020/1/1/async-pressure/
- Codebase CONCERNS.md (2026-03-21) — known issues in this specific codebase

---
*Pitfalls research for: server-rendered GPU streaming, CuPy/OpenGL/WebSocket hosted demo*
*Researched: 2026-03-21*
