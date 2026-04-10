# Renderer Module

## Role in the Pipeline

[WRITE: "Receives CWT frames from DSP stage, stores history in circular buffer, renders as GPU texture via fragment shader"]

[WRITE: "Three components and their responsibilities: CircularFrameBuffer (stores history), Renderer (GPU texture + shader), GLContext (window + OpenGL context)"]

---

## Why Shaders

[WRITE: "Python plotting libraries (matplotlib, pyqtgraph) cannot sustain real-time frame rates with this data volume"]

[WRITE: "GPU renders entire frame history as a single texture in one draw call — orders of magnitude faster than per-frame Python rendering"]

candidate analogy: "Each frame is a column of the image. Instead of redrawing the image column by column on the CPU, the entire image is pre-assembled in the circular buffer and handed to the GPU as a texture in one operation."

---

## Circular Frame Buffer

[WRITE: "How CircularFrameBuffer stores frames, maintains chronological order, pre-allocates flattened_buffer"]

[WRITE: "frame_index points to the NEXT write slot; chronological ordering is computed from there — oldest frame is at frame_index after the write, newest is just before"]

```python
# Source: src/subshader/viz/plotter.py CircularFrameBuffer.push_frame
# After writing frame at self.frame_index and incrementing:
self.frame_index = (self.frame_index + 1) % self.num_frames
frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]
```

[WRITE: "Walk through the frame_order line — frame_index is the oldest slot after the write; iterating from there in order gives oldest-to-newest sequence"]

[WRITE: "Explain why circular buffer is necessary — fixed memory allocation, no reallocation per frame, O(1) insert"]

Key performance detail:
- `flattened_buffer` is pre-allocated at init as a `(height, width * num_frames)` array — it is NOT rebuilt each frame. Frames are copied into their column-slices in place. This eliminates per-frame memory allocation on the hot path.

```python
# Source: src/subshader/viz/plotter.py CircularFrameBuffer.__init__
# Pre-allocated at init — shape never changes
self.flattened_buffer = np.zeros((self.height, self.width * num_frames), dtype=np.float32)
```

---

## Intensity Normalization

[WRITE: "IntensityTracker: percentile-based global_max with exponential decay"]

[WRITE: "Why this instead of per-frame min/max — per-frame normalization causes flickering as intensity varies frame-to-frame; global tracking with decay adapts smoothly to changing signal levels"]

[WRITE: "warmup_frames: initial frames may have atypical intensity (ramp-up artifacts, empty buffer); warmup period prevents early values from skewing the normalization baseline"]

```python
# Source: src/subshader/renderer/intensity.py IntensityTracker.update
self.global_max = self.retention_rate * self.global_max
self.global_max = max(self.global_max, self.floor_value)
self.global_max = max(self.global_max, frame_max)
```

[WRITE: "Walk through the three lines — slow decay so max doesn't collapse on quiet passages, floor prevents zero (avoids division-by-zero in shader), new frame_max can raise it immediately"]

Key detail:
- `frame_max` is computed as the `percentile`-th percentile of the frame (default: 99th) — not the absolute maximum. This makes the normalization robust to occasional spike values that would otherwise compress the visible range.

---

## Init: CPU-GPU Transfers

[WRITE: "What goes to GPU at init: quad geometry vertices uploaded to VBO, texture allocation (shape fixed to height × width × num_frames)"]

[WRITE: "What goes every frame: texture data write only — the flattened_buffer bytes are uploaded via texture.write()"]

[WRITE: "Design rationale: minimize per-frame CPU-GPU transfers — quad geometry is static (uploaded once), only pixel data changes each frame"]

Key detail:
- `shader['intensity_max'] = max(intensity_max, 1e-8)` — the floor of `1e-8` prevents division-by-zero in the fragment shader during warmup or silent passages

---

## Runtime: The Render Loop

[WRITE: "Sequence: push_frame → update_texture → set_intensity_max → clear → render → display"]

[WRITE: "Why this order matters — texture must be updated before render_graphic() reads it; intensity_max must be set before the shader executes; clear must happen before rendering to blank the back buffer"]

```python
# Source: src/subshader/viz/plotter.py ShaderPlot.update_plot
# Each call to update_plot is one full render cycle:
self.frame_buffer.push_frame(plot_values)                         # 1. Update CPU-side buffer
self.renderer.update_texture(self.frame_buffer.get_flattened_buffer())  # 2. CPU→GPU texture upload
self.renderer.set_intensity_max(self.frame_buffer.get_intensity_max())  # 3. Set normalization uniform
self.gl_context.clear_graphic()                                    # 4. Clear back buffer
self.renderer.render_graphic()                                     # 5. Execute shader
self.gl_context.display_graphic()                                  # 6. Swap buffers (display)
```

---

## Shader Pipeline

[WRITE: "Vertex shader: full-screen quad geometry — two triangles (TRIANGLE_STRIP) covering the full viewport from (-1,-1) to (1,1)"]

[WRITE: "Fragment shader: samples the texture at each pixel, applies colormap, applies gamma correction"]

[WRITE: "`intensity_max` uniform: shader divides sampled texture value by intensity_max to normalize pixel brightness to [0, 1] range before colormap"]

[WRITE: "`gamma` uniform: non-linear brightness curve applied after colormap for perceptual accuracy — values < 1.0 brighten midtones"]

Key detail:
- `shader['intensity_max'] = max(intensity_max, 1e-8)` — floor prevents division-by-zero in the fragment shader on silent or warmup frames
- `shader['gamma'] = gamma` — set at init from `VisualizationConfig.gamma`; not updated per frame

Shader files:
- `src/subshader/viz/shaders/vertex_shader.glsl` — full-screen quad
- `src/subshader/viz/shaders/fragment_shader.glsl` — colormap + gamma

---

## Configuration

VisualizationConfig fields (source: `src/subshader/config.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `num_frames` | `256` | Number of frames stored in circular buffer (controls horizontal history width) |
| `gamma` | `0.5` | Gamma correction factor — values < 1.0 brighten midtones |
| `color_norm` | `ColorNormalizationConfig()` | Nested config for IntensityTracker |

ColorNormalizationConfig fields (source: `src/subshader/config.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `percentile` | `99.0` | Percentile of frame data used as frame_max (robust to spikes) |
| `retention_rate` | `0.95` | Fraction of global_max retained per frame (0.95 = retain 95%) |
| `floor_value` | `1e-8` | Minimum value for global_max (prevents division-by-zero) |
| `warmup_frames` | `10` | Frames before IntensityTracker is considered "ready" |

[WRITE: "What each parameter controls and when to change it — e.g., lower retention_rate means faster adaptation to quieter passages; higher percentile is more robust to transient spikes; increase num_frames to show more history"]

---

## Diagram

[PLACEHOLDER: diagram — "User has a renderer pipeline diagram; insert here when available"]
