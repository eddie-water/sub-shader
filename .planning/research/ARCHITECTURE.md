# Architecture Research

**Domain:** Server-rendered GPU visualization streaming (audio CWT → OpenGL → WebSocket → browser)
**Researched:** 2026-03-21
**Confidence:** MEDIUM-HIGH

---

## Standard Architecture

### System Overview

The canonical structure for this domain is a three-layer system: a GPU pipeline that produces rendered frames, a streaming layer that moves those frames over the network, and a browser client that decodes and displays them.

```
┌──────────────────────────────────────────────────────────────┐
│                      SERVER PROCESS                          │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  Audio Input │──▶│  CWT (CuPy)  │──▶│ ShaderPlot /    │  │
│  │  (file/live) │   │  GPU compute │   │ Offscreen FBO   │  │
│  └──────────────┘   └──────────────┘   └────────┬────────┘  │
│                                                  │           │
│                                          fbo.read()          │
│                                          → bytes             │
│                                                  │           │
│                                         ┌────────▼────────┐  │
│                                         │  Frame Queue    │  │
│                                         │  asyncio.Queue  │  │
│                                         └────────┬────────┘  │
│                                                  │           │
│                                         ┌────────▼────────┐  │
│                                         │  WebSocket      │  │
│                                         │  Server         │  │
│                                         │  (websockets /  │  │
│                                         │   FastAPI)      │  │
│                                         └────────┬────────┘  │
└──────────────────────────────────────────────────┼───────────┘
                                                   │ WS binary frames
                                                   │ (JPEG bytes)
                            ┌──────────────────────▼─────────────────────┐
                            │               BROWSER CLIENT               │
                            │                                            │
                            │  WebSocket.onmessage → createImageBitmap   │
                            │  → ctx2d.drawImage() on <canvas>           │
                            └────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Boundary |
|-----------|---------------|----------|
| AudioInput | Deliver overlapping audio chunks | Unchanged from current pipeline |
| CuWavelet | GPU CWT, returns float32 numpy array | Unchanged from current pipeline |
| OffscreenRenderer | Render CWT result to FBO, read pixels back as bytes | Replaces GLFW-windowed GLContext |
| FrameQueue | asyncio.Queue bridging the render thread and async WS server | Thread boundary |
| WebSocketServer | Accept browser connections, broadcast JPEG frames to all clients | Network boundary |
| BrowserClient | Receive binary WebSocket frames, decode JPEG, draw to canvas | Client-only |

---

## Key Architectural Decisions

### Decision 1: Render thread stays synchronous; WebSocket server is async

OpenGL contexts have strict thread affinity — the context must be used on the same OS thread it was created on. The existing pipeline already runs synchronously. The correct split is:

- **Render thread (existing Python main thread):** runs the audio→CWT→render loop, reads back pixels, puts JPEG bytes into a queue.
- **asyncio event loop (separate thread):** runs the WebSocket server, drains the queue, broadcasts to clients.

These communicate via `asyncio.Queue` with `loop.call_soon_threadsafe()` to put items from the render thread into the async queue safely.

**Why not run_in_executor for rendering?** OpenGL context cannot move between threads. run_in_executor works for CPU-bound work without thread affinity constraints; it does not work here.

### Decision 2: Offscreen framebuffer, not GLFW window

For a hosted server demo there is no display. The existing `GLContext` creates a GLFW window, which requires a display (X11 or Wayland). The server path needs an offscreen render target instead.

ModernGL supports two headless approaches:
- **EGL backend** (`backend='egl'`) — GPU-accelerated, no X server required. Correct for production on NVIDIA hardware.
- **X11 with Xvfb** — virtual display, adds overhead, useful as fallback during development on WSL2.

The existing `Renderer` class already renders to the back buffer and reads nothing back. For the streaming path, render to a `moderngl.Framebuffer` (FBO), then call `fbo.read(components=3)` to get RGB bytes after each frame.

### Decision 3: JPEG as the wire format

Raw pixel bytes at typical visualization resolutions (e.g. 1920×1080, RGB) are ~6MB per frame — too large for practical WebSocket streaming. JPEG encoding at quality 70–85 reduces this to ~50–200KB with negligible visual degradation on a heatmap-style visualization.

The encode step is: `fbo.read(components=3)` → `numpy array` → `PIL.Image` → `io.BytesIO` → JPEG bytes → WebSocket binary message.

For higher throughput, `simplejpeg` is a faster JPEG encoder than PIL for this pattern (reported to handle 100k+ small encodes/second), but PIL is simpler to introduce first.

### Decision 4: Drop frames under backpressure, never block the render loop

The render loop timing must not be held hostage to network conditions. If the frame queue is full (slow clients), drop the new frame rather than blocking the render thread. This is a `asyncio.Queue(maxsize=N)` with a non-blocking `put_nowait()` wrapped in a try/except `QueueFull`.

---

## Component Boundaries (what talks to what)

```
AudioInput ──────────────────────────────────────────────▶ (unchanged)
                                                            CuWavelet
                                                               │
                                                    float32 ndarray
                                                               │
                                                               ▼
OffscreenRenderer.update_and_capture(coefs) ──▶  fbo.read() → RGB bytes
                                                               │
                                             JPEG encode (PIL/simplejpeg)
                                                               │
                                             queue.put_nowait(jpeg_bytes)
                                                               │
                              ┌────────────────────────────────┘
                              │  asyncio queue (thread boundary)
                              ▼
WebSocketServer.broadcast_loop() ──▶ websocket.send(jpeg_bytes)
                                         (to all connected clients)
                              ▲
                              │
BrowserClient connects ───────┘
```

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Render thread ↔ asyncio event loop | `asyncio.Queue` + `loop.call_soon_threadsafe()` | One-directional push; render produces, WS server consumes |
| OffscreenRenderer ↔ existing CuWavelet | numpy float32 array, same as today | No change to DSP layer |
| WebSocketServer ↔ browser | Binary WebSocket frames (JPEG bytes) | No base64; send raw bytes for performance |
| WebSocketServer ↔ connected clients | Set of WebSocket connection objects | Registration pattern: add on connect, remove on disconnect |

---

## Data Flow

### Full Frame Pipeline

```
[Audio file / live mic]
        │
        ▼ float64 ndarray (chunk_size,)
[AudioInput.get_chunk()]
        │
        ▼ float64 ndarray (chunk_size,)
[CuWavelet.cwt()]
        │ GPU: FFT convolution, normalize, downsample
        ▼ float32 ndarray (num_freqs, target_width)
[OffscreenRenderer.update_and_capture()]
        │ push frame to CircularFrameBuffer
        │ render flattened buffer to FBO
        │ fbo.read(components=3)
        ▼ bytes (width × height × 3, uint8)
[JPEG encode]
        ▼ bytes (~50-200KB)
[queue.put_nowait(jpeg_bytes)]
        │ crosses thread boundary
        ▼
[asyncio event loop]
[broadcast to all WebSocket clients]
        │ websocket.send(binary)
        ▼
[Browser: WebSocket.onmessage]
        │ event.data → Blob → createImageBitmap
        ▼
[canvas 2D context: drawImage()]
```

### State Management (added components)

- **Client registry:** Set of active WebSocket connections, managed by WS server. Thread-safe only within the asyncio event loop.
- **Frame queue:** `asyncio.Queue(maxsize=2–4)`. One producer (render thread), N consumers (one broadcast coroutine drains it for all clients).
- **Render loop state:** Unchanged — AudioInput file position, CircularFrameBuffer, IntensityTracker.

---

## Recommended Project Structure (additions only)

The existing `src/subshader/` tree is extended, not restructured:

```
src/subshader/
├── server/                  # New: streaming server components
│   ├── __init__.py
│   ├── websocket_server.py  # WebSocket broadcast server (websockets library)
│   ├── frame_encoder.py     # JPEG encode: numpy bytes → JPEG bytes
│   └── stream_bridge.py     # Thread-safe queue + event loop hand-off
├── viz/
│   ├── plotter.py           # Existing (keep ShaderPlot for local dev)
│   ├── offscreen_renderer.py  # New: headless FBO-based renderer (no GLFW window)
│   └── ...
└── __main__.py              # Existing local mode
└── __main_server__.py       # New: server entry point (streaming mode)
```

**Rationale for separation:**
- `server/` contains all network-facing code. The DSP/viz layers don't import from it.
- `offscreen_renderer.py` is a sibling of `plotter.py` — same role, different context (no window).
- A separate `__main_server__.py` keeps the local dev entry point clean. A `--mode server` CLI flag could unify them later.

---

## Architectural Patterns

### Pattern 1: Producer-Consumer with Thread Boundary

**What:** Render loop (synchronous, GPU-thread-pinned) produces JPEG bytes. WebSocket broadcast coroutine (asyncio) consumes them. They communicate via an `asyncio.Queue` with thread-safe insertion.

**When to use:** Whenever a blocking, thread-affine loop (OpenGL render) needs to feed an async network layer.

**Trade-offs:** Simple and correct. The queue provides backpressure visibility. Drop policy on full queue protects render loop timing.

```python
# In render thread (synchronous)
jpeg_bytes = encode_frame(fbo.read(components=3), width, height)
try:
    asyncio.run_coroutine_threadsafe(
        frame_queue.put(jpeg_bytes), event_loop
    )
except asyncio.QueueFull:
    pass  # drop frame, never block render

# In asyncio event loop
async def broadcast_loop(clients, frame_queue):
    while True:
        jpeg_bytes = await frame_queue.get()
        for ws in list(clients):
            try:
                await ws.send(jpeg_bytes)
            except Exception:
                clients.discard(ws)
```

### Pattern 2: Client Registration Set

**What:** Maintain a `set` of active WebSocket connections. Add on connect, remove on disconnect or send error.

**When to use:** Broadcasting to multiple browser tabs / viewers.

**Trade-offs:** Correct for a single-process server. Does not span multiple processes (no Redis pub/sub needed for this use case — single GPU, single process).

### Pattern 3: Offscreen FBO with EGL Backend

**What:** Replace GLFW windowed context with `moderngl.create_context(standalone=True, backend='egl')` and render to a `ctx.simple_framebuffer((width, height))`. Call `fbo.read()` after each render.

**When to use:** Server with no display manager (production deployment). NVIDIA 4060 Ti supports EGL natively.

**Trade-offs:** Requires EGL libraries installed (`libegl1-mesa` or NVIDIA EGL). On WSL2 dev environment, Xvfb is an easier fallback.

---

## Anti-Patterns

### Anti-Pattern 1: Blocking the render loop on network I/O

**What people do:** Call `await websocket.send(frame)` directly inside the render loop, or use a synchronous socket send.

**Why it's wrong:** Network latency (slow clients, TCP backpressure) stalls the GPU render loop. The visualization freezes or falls behind audio.

**Do this instead:** Put frames into a bounded queue and let the async layer drain it independently. Drop frames if the queue is full.

### Anti-Pattern 2: Moving the OpenGL context to the asyncio thread

**What people do:** Try to run the render loop as an asyncio coroutine or pass rendering into `run_in_executor`.

**Why it's wrong:** OpenGL contexts are thread-local. ModernGL will error (or silently misbehave) if called from a thread other than the one that created the context.

**Do this instead:** Keep the render loop synchronous on its own thread. Run the asyncio event loop in a daemon thread. Bridge with a queue.

### Anti-Pattern 3: Sending raw pixel bytes over WebSocket

**What people do:** Send `fbo.read()` bytes directly without encoding.

**Why it's wrong:** A 1280×720 RGB frame is ~2.8MB. At 30fps that is ~84MB/s. LAN is fine; internet connections are not. Browser decode of raw pixel bytes also requires manual ImageData construction.

**Do this instead:** JPEG encode each frame before sending. The browser can decode JPEG natively via `createImageBitmap(blob)`.

### Anti-Pattern 4: GLFW window on a headless server

**What people do:** Keep the existing `GLContext` (which calls `glfw.create_window`) and try to run it on a server.

**Why it's wrong:** GLFW requires a display server (X11/Wayland). On a headless host this fails immediately unless Xvfb is running, which adds process management overhead and is fragile.

**Do this instead:** Create a separate `OffscreenRenderer` that uses EGL directly via `moderngl.create_context(standalone=True, backend='egl')`. Keep `ShaderPlot` for local dev.

---

## Build Order (Dependencies Between Components)

The components have clear dependency ordering. Build in this sequence:

1. **OffscreenRenderer** — EGL context + FBO rendering + pixel readback. No network. Verifiable in isolation: render one frame, assert bytes come back with correct shape. This is the riskiest unknown (EGL on WSL2/Linux) and should be de-risked first.

2. **FrameEncoder** — `numpy bytes → JPEG bytes` via PIL or simplejpeg. Pure function, no dependencies, trivial to test. Fold in alongside OffscreenRenderer.

3. **StreamBridge** — `asyncio.Queue` + thread-safe insertion (`loop.call_soon_threadsafe`). No network yet. Verifiable: synthetic producer thread puts items, async consumer drains them.

4. **WebSocketServer** — Client registry, connection handler, broadcast loop. Verifiable: connect a test client, assert it receives frames.

5. **Server Entry Point** — Wire OffscreenRenderer + StreamBridge + WebSocketServer together. This replaces `ShaderPlot` with `OffscreenRenderer` in the main loop and runs the asyncio server in a daemon thread.

6. **Browser Client** — HTML canvas + WebSocket consumer. Minimal JS: `new WebSocket(url)`, `onmessage → createImageBitmap → drawImage`.

7. **Deployment** — Reverse proxy (nginx), process management (systemd or Docker), rate limiting, TLS.

---

## Scaling Considerations

| Scale | Architecture Adjustment |
|-------|------------------------|
| 1 viewer | No changes needed from the above design |
| 2-10 viewers | Same design; broadcast sends N copies of the same JPEG bytes. GPU is the bottleneck before the network is. |
| 10+ viewers | JPEG encoding once per frame (not per viewer) already handles this well. Possible bottleneck: TCP send throughput for many concurrent clients on a single NIC. Not a concern for a demo. |
| Multi-GPU | Out of scope; single 4060 Ti is sufficient for a demo. |

**First bottleneck:** EGL pixel readback (`fbo.read()`) is a GPU→CPU copy. At high resolution or high frame rates this can become a bottleneck. Mitigation: reduce output resolution (e.g. stream at 960×540 rather than native). The CWT computation is the more likely bottleneck at high frequency resolution.

**Second bottleneck:** JPEG encode on CPU. At 30fps and moderate resolution (960×540), PIL encode takes roughly 5–15ms. simplejpeg can cut this to ~2ms if needed.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Browser | WebSocket binary frames | Standard WebSocket API, no library needed on client |
| nginx (optional) | Reverse proxy to WebSocket server | Needed for TLS termination in production |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| OffscreenRenderer ↔ CuWavelet | numpy float32 array | Same contract as ShaderPlot.update_plot() — drop-in compatible |
| OffscreenRenderer ↔ FrameEncoder | raw RGB bytes (fbo.read result) | Pure function, no state |
| FrameEncoder ↔ StreamBridge | JPEG bytes | Queue put is the only interface |
| StreamBridge ↔ WebSocketServer | asyncio.Queue | Server owns the event loop; bridge writes into it |

---

## Sources

- ModernGL headless documentation (EGL backend): https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html (MEDIUM confidence — confirmed EGL and X11 backend options exist, confirmed `fbo.read()` API)
- ModernGL framebuffer API: https://moderngl.readthedocs.io/en/latest/reference/framebuffer.html (HIGH confidence — official docs, confirmed `read(components=N)` method)
- OpenGL multithreading constraints: https://www.khronos.org/opengl/wiki/OpenGL_and_multithreading (HIGH confidence — Khronos official, thread affinity is a fundamental constraint)
- websockets library patterns (broadcast, producer-consumer): https://websockets.readthedocs.io/en/stable/howto/patterns.html (HIGH confidence — official library docs)
- NVIDIA EGL for headless rendering: https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/ (HIGH confidence — official NVIDIA developer blog)
- asyncio thread safety (call_soon_threadsafe): https://docs.python.org/3/library/asyncio-eventloop.html (HIGH confidence — CPython official docs)
- JPEG streaming latency observations: https://github.com/iimachines/MotionJpegLatencyTest (LOW confidence — single informal experiment)
- simplejpeg performance: https://pypi.org/project/simplejpeg/ (LOW confidence — PyPI page claim, unverified benchmark)

---

*Architecture research for: SubShader server-rendered streaming demo*
*Researched: 2026-03-21*
