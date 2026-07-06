# Handoff Brief — SubShader Timing/Pipeline Visual Cohesion

## Goal
Make the **timing Gantt charts, the pipeline profile, and the draw.io flowcharts**
read as **one visual system**, so a viewer can follow: when each module runs, what
runs on CPU vs GPU, where data lives, and when it crosses the CPU↔GPU boundary.
The current diagrams confuse because **swimlanes and color both encode the same
thing (CPU/GPU)** and every edge looks alike (control vs data vs transfer are
indistinguishable).

## The core insight (the fix)
Make every visual channel carry **exactly one meaning**, and crucially make
**lane and color orthogonal**:
- **Lane = MODULE** (Audio / DSP / Renderer / Memory) — software structure
- **Color = HARDWARE** (purple = CPU, orange = GPU) — where it executed

Payoff: a DSP-lane bar that's orange instantly means "DSP offloaded this to GPU"
(`multiply`/`ifft` glow orange inside an otherwise-purple DSP row).

## The encoding strategy (lock this)
| Question | Channel | Encoding |
|---|---|---|
| Which module? | **lane (Y)** | Audio · DSP · Renderer · Memory |
| CPU or GPU? | **fill hue** | purple `#7b6fe1` = CPU · orange `#ff5a1f` = GPU |
| Code or memory? | **shape** | rounded bar = method/code · cylinder/rail = memory |
| Startup or runtime? | **fill style** | hollow/outline = init (once) · solid = loop (per-frame) |
| Control vs data? | **edge** | gray thin = sequence/control · colored = data |
| Crosses CPU↔GPU? | **cyan `#22d3ee`** | cyan arrow = transfer — *only* use of cyan |

**Palette = 3 meaningful hues** (purple/orange/cyan) + neutrals (white text,
gray control/deemphasis, black bg) + gold `#FFD27D` reserved for the audio/disk
**source** only. Background `#000000`.

## Single source of truth (recommended, not yet built)
A `stages.json` registry — one canonical **stage id** per pipeline step, shared
verbatim across code instrumentation, CSV, Gantt, and flowchart. Each entry:
`{id, label, domain (cpu/gpu/disk), kind (compute/transfer/control/alloc), module, order}`.
Both renderers key off it → colors/order/labels can't drift.

## Stage vocabulary (already exists in the data)
`timing_results.csv` `stage` column is a controlled vocabulary. Loop stages →
flowchart boxes:

| stage id | flowchart box | module | hw | kind |
|---|---|---|---|---|
| `audio_read` | Fetch Audio Samples | Audio | CPU | compute |
| `fft_cpu` | FFT | DSP | CPU | compute |
| `upload` | (cyan edge) | DSP | CPU→GPU | **transfer** |
| `multiply` | Freq-Domain Multiply | DSP | GPU | compute |
| `ifft` | IFFT | DSP | GPU | compute |
| `download` | (cyan edge) | DSP | GPU→CPU | **transfer** |
| `magnitude` | Compute Magnitude | DSP | CPU | compute |
| `edge_trim` | Discard Edges | DSP | CPU | compute |
| `hop_center` | Advance Audio Hop | DSP | CPU | compute |
| `downsample` | Down-sample | DSP | CPU | compute |
| `buf_push` | Store → Frame Buffer | Renderer | CPU | compute (write CPU RAM) |
| `tex_upload` | Upload → Frame Texture | Renderer | CPU→GPU | **transfer** |
| `gl_clear` | Clear Previous | Renderer | GPU | control |
| `gl_draw` | Shader Draw | Renderer | GPU | compute |
| `gl_swap` | Update Display Buffer | Renderer | GPU | present |
| `wait:next_chunk` | (loop-back) | — | — | control |

Startup stages are `init:*` namespaced (`init:dsp_kernels`,
`init:render_glcontext`, …).

**Three transfers per loop** — `upload`, `download`, `tex_upload` — are already
separate timing bars. These are exactly the cyan crossings.

## Key factual finding (corrects a mislabel)
`CircularFrameBuffer.frames` is `np.zeros(...)` → **Frame Buffer lives in CPU RAM**,
not VRAM. The GPU Texture (`ctx.texture(...)`) is VRAM. So the *single* end-of-loop
host→device crossing is `tex_upload`. The current flowchart colors Frame Buffer
orange (GPU) — **should be purple (CPU)**.

## Diagram-type ideas explored
- **Gantt with module lanes** (the chosen primary) — lane=module, color=hardware,
  cyan transfer arrows into/out of a GPU Memory rail. Same ms X-axis as timing data.
- **Memory-residence timeline** — one frame's data as a polyline; Y=which memory
  it's in, cyan vertical jumps = transfers crossing the PCIe boundary. Good
  companion for "where does the data live."
- (Rejected: id-tag box↔bar matching alone — too abstract, didn't show data movement.)

## Assets (in `assets/timing/`)
- `horizontal_pipeline_sw_flowchart.drawio` — current pipeline flowchart (good
  units/spacing — **preserve**; only fills/edges/lane-meaning change). 3 bands,
  Init|Loop, cylinders for persistent memory, cyan upload/readback edges already present.
- `timing_results.csv` (per-stage loop timing), `timing_methods.csv` (method
  benchmark), plus PNGs: `timing_pipeline.png`, `timing_methods.png`,
  `timing_lifecycle.png`, `timing_startup_breakdown.png`, `timing_sweep.png`,
  `timing_config.png`.
- `dsp_method_timing.drawio`, `TIMING.md`.
- draw.io conventions in use: `shape=cylinder3` (memory),
  `edgeStyle=orthogonalEdgeStyle`, `light-dark(a,b)`, `exitX/Y/entryX/Y` anchors,
  `dashed=1` for one-time/alloc links, `strokeWidth` for emphasis.

## Encoding channels recap (for a fresh renderer)
- `#7b6fe1` purple = CPU · `#ff5a1f` orange = GPU · `#22d3ee` cyan = transfer (only)
- `#FFD27D` gold = disk/audio source · `#8c8c8c` gray = control/deemphasis
- `#FFFFFF` white = text/structure · `#000000` black = background
- shape: rounded rect = code/method · cylinder = memory resource
- fill: solid = runtime/loop · hollow(outline) = init/startup
- edge: gray thin = control/sequence · colored = data read/write · cyan = CPU↔GPU transfer · dashed = one-time alloc link

## Status & next steps
1. **Encoding strategy is proposed, awaiting final confirmation** (the table above).
   Design session should pressure-test it visually.
2. On confirm: build `stages.json` registry → propagate encoding to
   **Gantt + pipeline profile first**, then flowcharts.
3. Flowchart fix list: recolor Frame Buffer cylinder CPU/purple; relabel the cyan
   edges with exact stage ids (`upload`/`download`/`tex_upload`); make control
   edges gray, data edges colored.
4. Eventually **split Init vs Loop into two separate diagrams** (currently combined
   via Init|Loop divider).

## Constraints (carry over)
- Preserve the draw.io units/spacing — don't re-layout.
- Don't overwrite existing PNGs — new files with descriptive iteration names.
- Propose before applying; work on copies (v2/v3 untouched until accepted).
- Minimal color — color only where it carries meaning.
- User authors final prose; Claude scaffolds and suggests.
