# Feature Research

**Domain:** Real-time audio visualization — hosted GPU-streaming demo
**Researched:** 2026-03-21
**Confidence:** MEDIUM (ecosystem well-understood; GPU-streaming-to-browser is a narrower pattern with fewer direct analogues)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Visible visualization on page load | Demo sites open to something moving, not a blank canvas waiting for input | LOW | Serve a pre-selected audio loop so the demo is live on arrival |
| Smooth continuous rendering | Choppy frames read as "broken" regardless of accuracy | MEDIUM | Target 24–30 fps to canvas; pipeline already exists, bottleneck is transport |
| Perceptible audio-visual sync | If the visual lags behind audio by more than ~80ms, users notice. Core project value statement. | HIGH | The hardest constraint. May require dropping resolution before frame rate |
| Clear indication of what they're looking at | Users need one sentence: "CWT spectrogram of live audio." No DSP jargon required. | LOW | UI label or tooltip |
| Graceful degradation on connection issues | WebSocket drops should show a reconnect state, not a white screen | MEDIUM | Client-side reconnection logic with status indicator |
| Page works in Chrome and Firefox | These are the two browsers with full Web Audio API + WebSocket support | LOW | No exotic APIs needed; canvas + WebSocket is universally supported |
| HTTPS / secure connection | Browsers block mixed content; microphone access requires secure context | LOW | Prerequisite for any deployment |

### Differentiators (Competitive Advantage)

Features that set this demo apart. Not required for launch but elevate the impression.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| CWT (not FFT) spectrogram | Time-frequency resolution is qualitatively better than a standard STFT spectrogram — transients resolve cleanly at high frequencies | LOW (already built) | The pipeline exists. The differentiator is making it legible to viewers who don't know what CWT means |
| Pedagogical annotation layer | A callout overlay that explains one interesting thing happening in the current frame ("high-frequency transient = drum hit") turns a demo into a teaching tool | HIGH | Scope risk — defer unless compelling |
| GPU benchmark panel | Side-by-side CuPy vs NumPy timing visible to the user — shows the GPU acceleration story concretely | MEDIUM | Benchmark infrastructure already exists; surfacing it in the UI is the work |
| Multiple audio input modes | Switch between a curated demo track, live microphone (browser), or audio file upload | HIGH | Microphone and upload both require browser-side capture + WebSocket audio upload pipeline. Significant scope. |
| Color palette controls | Let users change the shader color map — increases perceived interactivity with minimal backend work | MEDIUM | Requires parameterizing the GLSL shader and exposing controls in the UI |
| Resolution / quality slider | Trade frame rate for resolution or vice versa — makes transport tradeoffs visible and lets users self-select their experience | MEDIUM | Useful as a self-serve latency knob; scoped to adjusting server-side downsampling factor |
| Embeddable iframe | Researchers and educators can drop the demo into their own pages | LOW | Mostly a deployment configuration concern (CORS, frame-ancestors) |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create disproportionate cost or risk for this specific demo.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Client-side WebGL/WebAssembly CWT | "Why not run it in the browser?" | Requires porting the entire CuPy/NumPy pipeline to WASM or JS — a rewrite. Loses the GPU acceleration story. PROJECT.md explicitly ruled this out. | Keep pipeline server-side. That's the point. |
| Multi-user concurrent sessions with isolated GPU pipelines | Scale looks good on paper | One NVIDIA 4060 Ti is the server. Concurrent GPU pipelines will saturate VRAM and memory bandwidth fast. | One active pipeline with a viewer queue or connection cap. Document this as intentional single-GPU demo, not a production service. |
| User-uploaded arbitrary audio files (unscreened) | Makes the demo personal and interactive | File upload opens a surface area: malicious files, storage costs, abuse. CPU/GPU cost is unbounded per upload. | Offer a curated set of 3–5 demo tracks server-side. Add upload only after rate limiting and input validation are hardened. |
| Real-time MIDI / OSC parameter control | Power users expect it from visualizer tools | Significant protocol overhead for a hosted demo with no local software. Out of scope for this milestone. | Keyboard shortcuts or URL parameters for preset switching are simpler and sufficient. |
| User accounts / save state | "Remember my settings" | Auth infrastructure, database, session management — not a demo-stage concern | LocalStorage for ephemeral client-side preference persistence only |
| Mobile support | Broader reach | PROJECT.md explicitly deferred mobile. Touch UI, smaller canvas, and bandwidth constraints are all non-trivial. | Mark demo as "desktop browser recommended." |
| Persistent chat or comments | Community engagement | Moderation burden, infra, off-topic for the demo's purpose | Link to GitHub Discussions for feedback |

---

## Feature Dependencies

```
[WebSocket frame stream]
    └──required by──> [Visible visualization on page load]
    └──required by──> [Smooth continuous rendering]
    └──required by──> [Perceptible audio-visual sync]
                          └──requires──> [Server-side audio playback loop]
                          └──requires──> [Frame timestamp discipline on server]

[HTTPS / secure context]
    └──required by──> [Browser microphone access] (if added)
    └──required by──> [WebSocket connection in modern browsers]

[Graceful degradation]
    └──requires──> [WebSocket reconnection logic on client]
    └──requires──> [Server heartbeat / ping-pong]

[GPU benchmark panel]
    └──enhances──> [CWT differentiator story]
    └──requires──> [Existing benchmark infrastructure surfaced in UI]

[Color palette controls]
    └──requires──> [Shader parameter passing from server or client config]

[Resolution / quality slider]
    └──requires──> [Server-side downsampling factor exposed as configurable]
    └──enhances──> [Perceptible audio-visual sync] (lower res = lower latency)

[Multiple audio input modes]
    └──conflicts──> [Single active GPU pipeline constraint]
    └──requires──> [Browser audio capture + WebSocket upload pipeline]
```

### Dependency Notes

- **Audio-visual sync requires server-side audio playback loop:** The server must be the clock. If the browser plays audio independently, drift is guaranteed over any connection with variable latency. Server plays audio, server renders frames, frames carry timestamps. Client renders on arrival.
- **WebSocket frame stream is the load-bearing transport:** Everything visible to the user depends on this working reliably. It must be the first thing built in the hosted demo phase.
- **Resolution/quality slider enhances sync:** Lowering resolution reduces per-frame compute and serialization time, which directly reduces end-to-end latency. This is a practical knob, not just a cosmetic one.
- **Multiple audio input modes conflicts with single GPU pipeline:** Concurrent user microphone sessions each need their own pipeline instance. That's not feasible on one 4060 Ti at demo scale. Resolve by treating the server as a single-pipeline broadcaster, not a multi-tenant service.

---

## MVP Definition

### Launch With (v1)

Minimum viable demo — validates that the server-rendered GPU streaming approach works and is compelling to viewers.

- [ ] WebSocket frame stream from server CWT pipeline to browser canvas — the entire transport layer
- [ ] Pre-selected audio loop playing server-side, frames streamed continuously — no user input required to see something
- [ ] Smooth rendering at 24+ fps with audio-visual sync within ~100ms — the core value claim
- [ ] Connection status indicator and auto-reconnect — demo should recover without user refresh
- [ ] One-sentence explanation visible on page of what the visualization is — essential for non-DSP viewers
- [ ] Rate limiting on WebSocket connections — protects the single GPU server from abuse

### Add After Validation (v1.x)

Add once the streaming pipeline is stable and the demo is publicly accessible.

- [ ] GPU benchmark panel — add when benchmark infrastructure is confirmed stable; shows the CuPy story
- [ ] Color palette selector — add when the basic demo is well-received; increases perceived interactivity
- [ ] Resolution / quality slider — add if latency complaints surface; gives users a self-service knob
- [ ] 3–5 curated track selection — add if a single loop feels limiting after user feedback

### Future Consideration (v2+)

Defer until the demo-ready milestone is complete and there's evidence of user interest.

- [ ] Browser microphone input — requires browser-to-server audio pipeline; high complexity, high user value, but scope risk
- [ ] Audio file upload — requires input validation, abuse protection, and storage; defer until rate limiting is hardened
- [ ] Pedagogical annotation overlay — interesting but high effort; only worth building if the demo gains audience

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| WebSocket frame stream | HIGH | HIGH | P1 |
| Continuous audio loop server-side | HIGH | LOW | P1 |
| Audio-visual sync (<100ms) | HIGH | HIGH | P1 |
| Connection status + auto-reconnect | HIGH | LOW | P1 |
| Rate limiting on connections | HIGH (safety) | LOW | P1 |
| Page-load explanation label | MEDIUM | LOW | P1 |
| GPU benchmark panel | MEDIUM | MEDIUM | P2 |
| Color palette controls | MEDIUM | MEDIUM | P2 |
| Resolution / quality slider | MEDIUM | MEDIUM | P2 |
| Curated track selection | LOW | MEDIUM | P2 |
| Browser microphone input | HIGH | HIGH | P3 |
| Audio file upload | MEDIUM | HIGH | P3 |
| Pedagogical annotations | HIGH | HIGH | P3 |

**Priority key:**
- P1: Must have for launch — demo is broken or unsafe without it
- P2: Should have — add when core pipeline is stable
- P3: Nice to have — future milestone, only if there's evidence of demand

---

## Competitor Feature Analysis

Context: SubShader is not competing commercially. The comparison is about what viewers of technical audio demos expect when they arrive at the page.

| Feature | Web Audio API spectrograms (academo.org, borismus/spectrogram) | Synesthesia / Magic Music Visuals (desktop VJ) | SubShader approach |
|---------|--------------------------------------------------------------|------------------------------------------------|-------------------|
| Audio source | Browser microphone | Local audio device | Server-side audio loop (curated tracks) |
| Transform | FFT / STFT | FFT / STFT | CWT (continuous wavelet transform) — higher time-frequency resolution |
| Rendering | Client-side canvas or WebGL | Client-side GPU | Server GPU (CuPy + ModernGL), streamed as frames |
| Latency | ~10–50ms (all client-side) | ~10ms (local) | ~50–200ms over WebSocket (network-dependent) |
| Setup required | Zero (browser) | Install software | Zero (browser, server-rendered) |
| Customization | Minimal | Extensive | Minimal at launch; expandable |
| Differentiator | Accessibility | Power and customization | GPU-accelerated CWT — academically interesting, accessible via browser |

**Key insight:** Client-side browser spectrograms win on latency and zero-setup. SubShader's edge is not latency — it is the CWT transform quality and the GPU-acceleration story. The demo should foreground that distinction rather than compete on latency numbers it cannot win.

---

## Sources

- [sndpeek: real-time audio visualization (Princeton Sound Lab)](https://soundlab.cs.princeton.edu/software/sndpeek/) — reference for what real-time spectrogram tools expose to users
- [borismus/spectrogram (GitHub)](https://github.com/borismus/spectrogram) — canonical browser spectrogram; sets baseline expectations
- [Real-Time Audio Spectrograms in the Browser — DEV Community](https://dev.to/hexshift/real-time-audio-spectrograms-in-the-browser-using-web-audio-api-and-canvas-4b2d) — latency discussion for browser-based spectrograms
- [JSMpeg — HTML5 live video streaming via WebSockets (phoboslab)](https://github.com/phoboslab/jsmpeg) — ~50ms WebSocket video streaming; establishes achievable latency for frame streaming to canvas
- [Web Audio API: audio output latency (web.dev)](https://web.dev/articles/audio-output-latency) — authoritative source on audio/video sync in the browser
- [WebSocket Rate Limiting (OneUptime, 2026)](https://oneuptime.com/blog/post/2026-01-24-websocket-rate-limiting/view) — rate limiting strategies for persistent connections
- [WebSocket Security Hardening Guide (websocket.org)](https://websocket.org/guides/security/) — connection abuse protection patterns
- [Wavelet spectrogram: CWT advantages (Medium)](https://medium.com/@harshivs2002/wavelet-spectrogram-leveraging-wavelet-transform-for-spectrogram-interpretation-a866786561e3) — articulates the CWT differentiator over STFT for audio
- [Best Practices for Optimizing WebSockets Performance (PixelFree Studio)](https://blog.pixelfreestudio.com/best-practices-for-optimizing-websockets-performance/) — binary encoding, heartbeat, compression guidance

---

*Feature research for: SubShader — real-time GPU-accelerated CWT audio visualization, hosted demo*
*Researched: 2026-03-21*
