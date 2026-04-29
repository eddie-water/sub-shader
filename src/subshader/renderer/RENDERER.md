# Renderer Module

## Role in the Pipeline

[WRITE: "Receives CWT frames from the DSP stage, stores recent history in a circular buffer, and renders the buffer as a GPU texture via fragment shader. The result is the scrolling spectrogram you see on screen."]

[WRITE: "Three components and their responsibilities: CircularFrameBuffer (stores frame history on the CPU), GPURenderer (uploads texture, runs shader, draws), GLContext (window creation, OpenGL context, buffer swap). Renderer orchestrates the three each frame."]

---

## Circular Frame Buffer

[WRITE: "What problem this solves — we need to display the last N frames as a scrolling history. Naive approaches (rebuild a list each frame, np.roll, list of arrays) reallocate memory on the hot path and don't scale."]

[WRITE: "How CircularFrameBuffer stores frames: a fixed-size 3D array `(num_frames, height, width)`, with a single integer `frame_index` pointing at the next write slot. Writing a new frame overwrites the oldest one in place — no allocation, no copy of older frames."]

```python
# Source: src/subshader/renderer/frame_buffer.py CircularFrameBuffer.push_frame
self.frames[self.frame_index] = frame_data
self.frame_index = (self.frame_index + 1) % self.num_frames
frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]
```

[WRITE: "Walk through the frame_order line — after the increment, frame_index points at the oldest slot (next to be overwritten). Iterating from there modulo num_frames yields oldest-to-newest order, which is what we hand to the GPU."]

Key performance detail:
- `flattened_buffer` is pre-allocated at init as a `(height, width * num_frames)` array — it is NOT rebuilt each frame. After updating `frames`, the chronological order is *copied into column-slices* of the existing buffer in place. This eliminates per-frame memory allocation on the hot path.

```python
# Source: src/subshader/renderer/frame_buffer.py CircularFrameBuffer.__init__
self.flattened_buffer = np.zeros((self.height, self.width * num_frames), dtype=np.float32)
```

[PLACEHOLDER: figure — "circular buffer diagram: ring of N frame slots with a write pointer; flattened_buffer below showing how the ring unrolls into chronological column-slices"]

---

## Why a Shader

[WRITE: "Python plotting libraries (matplotlib, pyqtgraph) cannot sustain real-time frame rates with this data volume. Each frame is height × width pixels and we want to draw `num_frames` of them every render cycle — that's millions of points per frame at 60 FPS. CPU-side rendering loops cannot keep up."]

[WRITE: "The GPU is purpose-built for this: it can sample a texture at every pixel of the output viewport in parallel. Once the frame history is uploaded as a single texture, drawing the entire spectrogram is one draw call instead of millions of CPU operations."]

candidate analogy: "Each frame is a column of the image. Instead of plotting columns one by one in Python, the entire image is pre-assembled in `flattened_buffer` and handed to the GPU as a texture in a single upload. The shader does the rest."

**What goes to the GPU once (at init):**
- Quad geometry — two triangles covering the full viewport, uploaded to a vertex buffer
- Texture allocation — fixed shape `(height, width × num_frames)`, written to each frame

**What goes to the GPU every frame:**
- The flattened buffer's bytes via `texture.write(...)` — a single CPU→GPU memcpy
- The current `intensity_max` uniform value

[WRITE: "Design rationale: minimize per-frame CPU↔GPU transfers. Geometry and shader programs are static — uploaded once. Only the pixel data and one uniform change per frame."]

```python
# Source: src/subshader/renderer/renderer.py Renderer.update — one render cycle
self.frame_buffer.push_frame(plot_values)                                   # 1. CPU buffer update
self.gpu_renderer.update_texture(self.frame_buffer.get_flattened_buffer())  # 2. CPU→GPU upload
self.gpu_renderer.set_intensity_max(self.intensity_tracker.global_max)      # 3. Uniform
self.gl_context.clear_graphic()                                              # 4. Clear back buffer
self.gpu_renderer.render_graphic()                                           # 5. Run shader
self.gl_context.display_graphic()                                            # 6. Buffer swap
```

[WRITE: "Why this order matters — texture must be updated before the shader reads it; intensity_max must be set before the shader executes; the back buffer must be cleared before drawing."]

---

## Color Mapping

[WRITE: "What the fragment shader does at each pixel: sample the texture, normalize the sampled coefficient by `intensity_max`, look up a color in the colormap, apply gamma correction. The output is the on-screen pixel color."]

**Normalization (the `intensity_max` uniform):**

[WRITE: "Why a normalization reference matters at all — raw CWT coefficient magnitudes vary wildly across signals (a quiet recording vs. a loud one). Without a reference, the same colormap range maps to wildly different brightness levels. We want the brightest part of the signal to map to the top of the colormap, consistently."]

[WRITE: "Current strategy: a fixed pre-scan computes the percentile-th coefficient magnitude across the entire audio file before playback starts. That value is held constant for every frame — the same coefficient magnitude always maps to the same color, regardless of which frame it appears in."]

```python
# Source: src/subshader/renderer/intensity.py IntensityTracker
# Constructed once before the render loop with fixed_max from pipeline pre-scan.
# update() is a no-op — global_max never changes during playback.
self.global_max = max(fixed_max, floor_value)
```

> **Polish backlog (revisit after DSP.md):** the current fixed-pre-scan reference may be solving the wrong problem. The original goal was: *frame N's largest coefficient should map to a color the same way frame M's largest does — so frames are consistent with each other.* The pre-scan max satisfies "consistent absolute reference" but may not be what the visualization actually needs. Re-examine after authoring is done.

**Colormap and gamma:**

[WRITE: "Fragment shader pseudocode: sample texture → divide by intensity_max → clamp to [0, 1] → colormap lookup → gamma correction → output. The colormap and gamma curve are defined in the GLSL itself."]

[WRITE: "`gamma` uniform: non-linear brightness curve. Values < 1.0 brighten midtones (perceptually-accurate visualization where mid-magnitude features are visible alongside loud peaks)."]

Key detail:
- `shader['intensity_max'] = max(intensity_max, 1e-8)` — floor prevents division-by-zero on silent or warmup frames

Shader files:
- `src/subshader/renderer/shaders/vertex.glsl` — full-screen quad
- `src/subshader/renderer/shaders/fragment.glsl` — texture sample + colormap + gamma

[PLACEHOLDER: figure — "before/after of intensity normalization on the same frame at three magnitudes; same colormap, different brightness"]

---

## Configuration

`RendererConfig` (source: `src/subshader/config.py`) extends `PipelineConfig` with:

| Field | Default | Description |
|-------|---------|-------------|
| `num_frames` | `256` | Frames stored in the circular buffer (controls horizontal history width on screen) |
| `color_norm` | `ColorNormalizationConfig()` | Nested config for color mapping |

`ColorNormalizationConfig` (source: `src/subshader/config.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `gamma` | `0.5` | Gamma correction in the fragment shader. Values < 1.0 brighten midtones |
| `global_intensity_percentile` | `99.0` | Percentile used by the pipeline pre-scan to compute the fixed `intensity_max` reference (robust to spike values) |

[WRITE: "What each parameter controls and when to change it — increase `num_frames` to show more history at the cost of horizontal resolution per frame; lower `gamma` further if quiet passages are too dark; raise `global_intensity_percentile` toward 99.5–100 for very dynamic signals where the 99th percentile undersells peaks."]

---

## Diagram

[PLACEHOLDER: diagram — "Renderer pipeline: DSP frame → CircularFrameBuffer.push → flattened_buffer → texture.write → fragment shader (sample, normalize, colormap, gamma) → screen pixel"]
