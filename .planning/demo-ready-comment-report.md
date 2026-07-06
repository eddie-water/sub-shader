# SubShader Comment Report

Companion to [demo-ready-cleanup-plan.md](demo-ready-cleanup-plan.md) · Workstream C.
Read-only audit of every `.py` under `src/subshader/`. **No edits applied** — this is the
punch-list the cleanup plan executes against.

---

## Verdict

The app source is **clean and above average**. The DSP core (`cwt.py`, `wavelet_kernel.py`,
`gaussian.py`, `pywavelet.py`, `stft.py`) and the pipeline/audio layer carry exactly the
load-bearing math/DSP and design-decision comments the standard wants (`D-xx`, `Pitfall x`,
L1-normalization rationale) — these read well and **stay**.

Two real problem files, plus a handful of stale breadcrumbs and typos:

- **`utils/signal_generator.py`** — notebook-export scratch; commented-out `print`/`linspace`
  blocks. (Also: not imported by the app — only appears in build-artifact `SOURCES.txt`.)
- **`dsp/wavelet_kernel.py`** — `_plot_kernel()` is a dead method with a fully commented-out body.

**Confirmed clean (no findings):** `cwt.py`, `pipeline.py`, `dsp.py`, `stft.py`, `pywavelet.py`,
`audio_stream.py`, `player.py`, `reader.py`, `intensity.py`, `timing.py`, `gpu.py`, `__main__.py`,
`os_env_setup.py`, all `__init__.py`.

---

## Priority cleanup list (highest value first)

| # | Location | Action |
|---|---|---|
| 1 | `wavelet_kernel.py:83–94` | Delete commented-out body of `_plot_kernel()` (and likely the dead method). |
| 2 | `signal_generator.py:55,61–62,68–71` | Delete commented-out `print`/`linspace` scratch. |
| 3 | `renderer.py:43` | Resolve or delete `# TODO-36 : self.ctx #3?` (cryptic, malformed). |
| 4 | `renderer.py:246` | Fix stale comment — `intensity_max` is set once before the loop, not "each frame". |
| 5 | `frame_buffer.py:65–70` | Fix `pop_frame` docstring/signature mismatch (ignores `index`). |
| 6 | `gaussian.py:16 & 37` | Typos: "the how steep", "defualt", double space. |
| 7 | `renderer.py:78–89, 130–143` | Tighten GLFW/ModernGL "what is this library" docstrings to project intent. |
| 8 | `signal_generator.py` (whole file) | Decide: belongs in `src/`? Reads as research scratch, unimported. |
| 9 | `renderer.py:102–104` | Collapse duplicated "Core Profile" comments into one line. |
| 10 | banner blocks | Decide once whether `# ===== SECTION =====` blocks stay; if not, strip uniformly. |

---

## Per-file findings

### `utils/signal_generator.py` — worst offender (research scratch)

| line(s) | category | action |
|---|---|---|
| 8 | stale | bare `pass` after `__init__` assignments → delete |
| 55 | commented-out code | `# print("The Sampling Frequency:...")` → delete |
| 61–62 | commented-out code | `# ta = np.linspace(...)` experiment → delete |
| 68–71 | commented-out code | 4 commented `# print(...)` debug lines → delete |
| 60, 64 | scratch | notebook-style "when using linspace/arange…" notes → tighten/delete |

### `dsp/wavelet_kernel.py`

| line(s) | category | action |
|---|---|---|
| 83–93 | commented-out code | entire `_plot_kernel()` body commented matplotlib + `pass` → delete (dead method) |
| 42–44 | keep | Time-Support explanation → keep |
| 61–64 | keep | L1 unit-area normalization rationale → keep (clear) |
| 10 | trivial | trailing whitespace → fix |

### `renderer/renderer.py`

| line(s) | category | action |
|---|---|---|
| 43 | malformed TODO | `# TODO-36 : self.ctx #3?` → fix-format or delete |
| 246 | stale | "updated each frame" contradicts `set_fixed_intensity_max()` (set once, never updates) → fix |
| 78–89 | wordy | `_init_window` docstring explains GLFW generally → tighten |
| 130–143 | wordy | `_init_opengl_context` docstring explains ModernGL generally → tighten |
| 102–104 | redundant | duplicated "Core Profile" comments → collapse to one line |
| 65 | redundant | `glfw.poll_events()  # Process window events` → delete (mild) |
| 156–158 | keep | face-culling explanation → keep |
| 262–263 | keep | VBO `tobytes()` comment → keep / lightly tighten |

### `dsp/gaussian.py`

| line(s) | category | action |
|---|---|---|
| 13–21 | wordy + keep | FWHM explanation load-bearing but verbose → keep, light trim |
| 16 | typo | "the how steep", double space → fix |
| 37 | typo | "defualt" → fix |

### `renderer/frame_buffer.py`

| line(s) | category | action |
|---|---|---|
| 65–70 | stale | `pop_frame` docstring says "specific frame" but ignores `index` and decrements `frame_index` → fix or remove |
| 30 | redundant (mild) | `# Store full frames (no overlap)` → keep/trim |

### `config.py`

| line(s) | category | action |
|---|---|---|
| 106 | keep | `# TODO-45 Fix the overlap and plot overlap relationship` → correct format, keep |
| 249–255 | keep | DEPRECATED shim comments → clear and temporary, keep |
| field comments | keep | per-field config comments → genuinely useful, keep |

### Cross-cutting

| files | item | action |
|---|---|---|
| `config.py`, `audio/*.py`, `renderer.py`, `frame_buffer.py` | `# ===== SECTION =====` banner blocks | judgment call — consistent but borderline litter; decide once |
| `logging.py`, `loop_timer.py`, `exceptions.py` | trailing whitespace, missing final newline (`exceptions.py:55`) | trivial formatting fix |
