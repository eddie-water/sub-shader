# Circular Frame Buffer

Displaying a scrolling history of the last N CWT frames without reallocating memory every frame requires a fixed-size ring buffer: a new frame overwrites the oldest slot in place, tracked by a single write-pointer index. This avoids the per-frame allocation and copy cost of naive approaches (rebuilding a list, `np.roll`) on the real-time hot path.

**Key numbers** (defaults from [`config.py`](../../../src/subshader/config.py))

- Ring: `(num_frames, height, width)` = **(32, 116, 64)** — 32 frames of CWT history, ~5.9 s of audio at one frame per 185.8 ms hop
- `frame_index` advances modulo 32; iterating from it yields oldest-to-newest order without moving any data
- A pre-allocated `flattened_buffer` of shape (116, 64 × 32 = 2,048) float32 (~0.95 MB) is refilled in place each frame — that single array is what uploads to the GPU as one texture

## In SubShader

`CircularFrameBuffer` in [`frame_buffer.py`](../../../src/subshader/renderer/frame_buffer.py); chronological order is copied into column-slices of the existing flattened buffer, so nothing is allocated per frame. Size controlled by `RendererConfig.num_frames`. Documented in [RENDERER.md](../../../src/subshader/renderer/RENDERER.md#circular-frame-buffer).

---

**Related:** [Real-Time Frame Budget](real-time-frame-budget.md) · [Intensity Normalization](intensity-normalization.md) · [Colormap and Gamma Correction](colormap-and-gamma-correction.md) · [RENDERER](../../../src/subshader/renderer/RENDERER.md) · [MAP](../MAP.md)
