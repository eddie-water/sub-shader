# SubShader Timing Plan

[Goal](#goal) · [Status](#status) · [How it works](#how-it-works) · [Data](#data) · [Decisions & order](#decisions--order) · [Worklog](#worklog)

---

## Goal

Profile the **live** pipeline (real song, real-time paced, incl. renderer) per SW
module/stage, save numbers to one CSV, show them as a GitHub table.
Answer: (1) fast enough live? FPS + keeps up with audio, (2) where's the time going?
per-module/stage **+ GPU transfer bottlenecks** (upload/compute/download),
(3) did a change slow it down? vs past runs, (4) which setup wins? GPU/CPU × params + plot.

Don't touch: the DSP notebook, dsplot figures, READMEs. `src/` only gets light timing
(see Decision 1). No commits unless asked.

---

## Status

- [x] **1. Storage layer** — `timing_results.py`: CSV + TIMING.md _(built + tested w/ fake data)_
- [x] **2. Live wire-up** — `research/timing_live.py` + `--live-timing`; real run verified on device → CSV + TIMING.md. Software-paced fallback for no-audio boxes (Parsec).
- [x] **3. GPU transfer profiling** — `research/timing_gpu.py` `InstrumentedGpuCWT` (synced legs); `raw_cwt` splits into fft/upload/multiply/ifft/download. Verified on device.
- [x] **4. Sweep + plot** — `research/timing_sweep.py` OFAT over backend × chunk × overlap × **octaves** → CSV + `timing_sweep.png`. (num_octaves wired via a small `src/pipeline.py` propagation fix.)
- [x] **5. Cleanup** — retired old offline `--timing` path (→ `research/archive/`), deleted stale `.txt`. _(README_pipeline_timing.md left for its owning session.)_
- [x] **6. Visual report** — `research/timing_report.py` `timing_pipeline.png` (proportional frame + zoom + RT-budget) and a narrative `TIMING.md` (verdict, mermaid flow, embedded figures, findings). Auto-regenerates from the CSV.

---

## How it works

Two layers, kept separate. App stays clean; benchmark tools do all the file work.

```
src/        @timed tag → records "_timing_x_ms" on the object. No files. Only timing thing in src.
                |
research/   reads those numbers → stats → writes CSV + Markdown + plots.
            (research may import src; src never imports research)
```

```
run pipeline N frames
  for each frame: time every step  ──► arrays of ms
  TimingRecorder.record_run(meta, arrays)  ──► append rows to timing_results.csv
  render_markdown()                        ──► TIMING.md
```

---

## Data

**Steps timed each frame** (plus a `TOTAL`, and one-time `init:*` rows):

```
coarse:  get_chunk → raw_cwt → normalize → magnitude → edge_trim → hop_center → downsample → push_frame → render

fine:    raw_cwt splits into →  fft_cpu → gpu_upload → gpu_multiply → gpu_download → ifft_cpu
         (the gpu_upload / gpu_download legs = the transfer bottlenecks; needs GPU sync — step 3)
```

**CSV** `assets/timing/timing_results.csv` — one row per (run, step), append-only:

```
run_id, timestamp, git_sha, backend, chunk_size, num_freqs, num_frames,
stage, mean_ms, std_ms, min_ms, max_ms, pct_of_total, rt_margin
```

`rt_margin` (on TOTAL row only) = how many times faster than real-time. **>1 = fast enough.**

**How it's viewed** — `assets/timing/TIMING.md` (example):

```
Latest: GpuCWT · chunk 16384 · 100 freqs · 64 frames

Stage         Mean   %Total
raw_cwt       1.89   38%
magnitude     0.79   16%
downsample    0.60   12%
...
TOTAL         5.0 ms · 200 FPS · RT 18× ✅

All runs
Backend  Chunk   Total   FPS    RT×
GpuCWT   16384   6.1ms   164    30×
CpuCWT   16384   15.0ms  66     12×
```

---

## Decisions & order

1. **Live, full pipeline, real-time paced.** Run the real `src/pipeline.py` end-to-end
   (audio → CWT → renderer) on a few seconds of a real song, paced to the audio clock —
   parity with actual live performance. Needs light timing in `src/` (a per-frame total
   timer in the loop; `@timed` tags already exist there incl. renderer). User OK'd this.
2. **GPU sync = required, sequenced.** GPU is async, so a plain stopwatch reads ~0 per
   GPU leg (time lands on the wrong step) — the *total* stays honest but transfer
   bottlenecks are invisible. Needed for the GPU-transfer goal. Order: get sensible
   end-to-end live numbers first (step 2), then add the synced "wait until done"
   subclass (step 3) to expose upload/compute/download.
3. **Sweep axes:** backend × chunk_size (4k/8k/16k/32k) × num_octaves × overlap (.25/.5/.75).
4. **CSV append-all** (keeps history); TIMING.md shows latest + summary.

---

## Worklog

- **2026-06-26** — ✅ Step 8: report simplified to 4 sections + method comparison.
  **Report** (`render_markdown`) rebuilt to **Overview · Pipeline profiling · Config
  performance · Conclusion** (user spec). Overview consolidates title + a headline
  table (start-up init · runtime loop processes one chunk_size window · processing
  rate · deadline/headroom) + a **transform-method table**. Config section is now
  **tabular** (no embedded sweep plot — user disliked those colors). **New benchmark**
  `research/timing_methods.py` + `--methods`: times **STFT / PyWavelet / CWT-CPU /
  CWT-GPU** on real audio (pure DSP, no window) with each method's **native
  resolution** (STFT linear sample_rate/nperseg ≈ 2.7 Hz uniform vs CWT family
  log/constant-Q semitone). Storage/render helpers (`METHODS_CSV`,
  `append_method_rows`, `latest_method_rows`, `methods_table_md`) live in the
  utilities layer so the report folds the table in with no src import. **Figure**
  (`timing_pipeline.png`) re-themed to the **optichrome v52 palette** on a dark
  ground and reworked for clarity: runtime panel now uses **3 strong module colors**
  (no confusing within-module tints) labeled audio/DSP/render; a **per-stage detail**
  panel names every DSP+render stage (identity from labels, not shades).
  **Per-module init done** (user OK'd src change): cosmetic `@timed` on
  AudioStream/CpuCWT/GpuCWT/Renderer `__init__` → startup panel now splits into
  audio 95 · CWT kernels 402 · renderer 300 · pre-scan 163 · other 100 ms (~1.1 s).
  **Method numbers (clean, solo runs):** STFT 3.3 ms · CWT-GPU 6.2 ms · CWT-CPU
  82 ms · **PyWavelet 40 s** (reference only, ~6500× slower). PyWavelet's `pywt.cwt`
  is ~50 s/call, so `--methods` uses N_CHUNKS=2. **Report re-voiced to plain language**
  per user ("What Was Tested" list, "What It Measures" table, simple section intros).
  Clean live frame: 13.4 ms, 13.9× headroom.
- **2026-06-27** — ✅ Step 9: **visuals-first rewrite** (user: "let the visuals do the
  talking — too wordy"). TIMING.md is now a one-line verdict + three diagrams, no
  tables/bullets/prose. Two new figures in `timing_report.py`: **`timing_methods.png`**
  (four-method log-scale bars; color = resolution story: orange GPU-CWT "runs live",
  purple CWT-log, gray STFT-linear) and **`timing_config.png`** (per-config compute,
  GPU orange / CPU gray, labeled with RT×). `config_summary_rows` helper added;
  verbose `render_markdown` body deleted. `update_report` renders all three figures.
- **2026-06-26** — ✅ Step 7: per-module profile + restructured report. **Src (cosmetic
  @timed):** decorated the renderer's internal calls (push_frame, update_texture,
  clear_graphic, render_graphic, display_graphic) + `_prescan_intensity` — render is no
  longer a black box. **Driver:** captures render legs (from frame_buffer/gpu_renderer/
  gl_context sub-objects), audio read (via @timed get_chunk, split from RT wait), and init
  split (build vs prescan); `profile_render` flag (on for live, off for sweep blobs).
  **Report restructured** to Overview · Main takeaways · Pipeline profile (audio/dsp/render
  with subtotals) · Pipeline diagrams (startup mermaid + runtime mermaid + deadline-check
  table) · DSP config timings. Anatomy always uses the latest *detailed* run
  (`select_anatomy_run`, marker=gl_swap) so a blob-only sweep doesn't flatten it.
  **Key finding:** the biggest single line in the whole pipeline is **swap + vsync
  3.2 ms (25%)** — the renderer blocking on the display refresh, not GPU work; draw 1.8ms
  next. DSP 6.9ms · render 5.7ms · audio 0.14ms. Deadline reframed per user: real-time
  audio keep-up at full freq resolution (186 ms/hop, 15× headroom); noted on-screen update
  = chunk rate (~5.4 FPS), not the 78 FPS compute ceiling.
- **2026-06-26** — ✅ Steps 4(octaves)+5+6. **Octaves axis:** fixed `src/pipeline.py`
  to propagate num_octaves/notes_per_octave (was silently dropped — latent bug); added
  octaves to the sweep (6/8/10 → f72/96/116 → 9.8/11.0/11.6 ms). **Cleanup:** retired the
  old offline `--timing` framework — `research/timing.py` → `archive/timing_offline.py`,
  `timing_template.txt` → archive, removed the `--timing` CLI branch, deleted the stale
  `*_timing.txt`. (`research/README_pipeline_timing.md` documents the old path — left for
  its owning session.) **Visual report:** `research/timing_report.py` builds
  `timing_pipeline.png` (3 panels: proportional frame · zoom on sub-0.5ms slivers ·
  compute-vs-deadline budget) and upgraded `render_markdown` into a narrative `TIMING.md`
  (verdict line, TL;DR, mermaid pipeline flow, embedded figures, sweep findings, capped
  recent-runs table). Drivers now call `update_report()` so figure + MD regenerate from
  the CSV together. Detailed live run confirms render 4.1ms(37%) + iFFT 2.4(22%) dominate,
  11.0ms/frame, **16.9× RT headroom**.
- **2026-06-25** — ✅ Step 4: config sweep + plot (`research/timing_sweep.py`,
  `--sweep`). OFAT around a baseline (gpu/16384/0.5), **un-paced** (raw compute,
  not RT slack). `run_live_timing` refactored to take chunk/overlap/backend/paced/
  quiet knobs — no-arg `--live-timing` unchanged. **Subprocess-per-config:** each
  config runs in a fresh `test_suite.py --live-timing` process (cold GPU — no CuPy
  pool / cuFFT cache / CUDA context bleed between runs); child prints `RUN_ID=`,
  parent collects + plots. _(First cut ran all 9 in one process — warm-state
  contamination made overlap look like it cost compute: 12.7→14.7→15.4ms. Cold
  starts show it flat ~11.9ms, which is correct — overlap changes the hop, not the
  per-frame work. Isolation fixed it.)_ Findings: **CPU 87.7ms ≈ 7.3× GPU 12.0ms**;
  bigger chunk → more compute (7.2→14.0ms) but **higher RT margin** (6.5×→26×, hop
  grows too); overlap flat compute, **RT margin falls** with overlap (23×→8×).
  Stacked bars = raw_cwt + magnitude + render dominate. Saved
  `assets/timing/timing_sweep.png`.
  **num_octaves omitted:** `SubShader` rebuilds CWTConfig internally and drops it, so
  sweeping it cleanly needs a ~2-line src/pipeline.py change (propagate num_octaves/
  notes_per_octave) — flagged, not done (src is @timed-only).
- **2026-06-25** — Plan drafted. Inventoried existing timing code (LoopTimer, `@timed`,
  TimingAccumulator, old `.txt` reports). Built + tested `research/utilities/timing_results.py`
  (TimingRecorder → CSV, render_markdown → TIMING.md) with synthetic data — schema, RT
  margin, latest+summary tables all verified.
- **2026-06-25** — ✅ Step 3: GPU transfer profiling (`research/timing_gpu.py`
  `InstrumentedGpuCWT`, synced legs; swapped into the live driver via
  `profile_gpu_transfers`). Real run: render 4.18ms(38%), ifft 2.39(22%),
  **download 1.74(16%) ≫ upload 0.36(3%)** — download dominates transfers (full
  116×16384 matrix vs 1-D input). Total 11.08ms (vs 10.68 unsynced = sync overhead).
  Confirmed **normalize is a deliberate no-op** (L1 at kernel build) → dropped from
  table. Console summary reworked: arrows, bars, plain labels, sorted by cost.
  **Bottleneck = render + ifft, not transfers.** Next → step 4 sweep + plot.
- **2026-06-25** — ✅ First real live run (beltran_sc_rip, 27 frames, GpuCWT, chunk 16384,
  116 freqs): **compute 10.68 ms/frame, ~94 FPS, RT 17.4×**. Stage split (untrusted until
  step 3): raw_cwt 5.5ms(51%), render 4.3ms(40%, max 17ms spike), magnitude 0.75ms;
  **normalize reads 0.0** (folded into transform on GPU? confirm). Software-paced fallback
  added for no-audio boxes (Parsec). Wrote `assets/timing/{timing_results.csv, TIMING.md}`.
  Next → step 3 GPU sync to fix the breakdown + expose transfer legs.
- **2026-06-25** — Built live driver `research/timing_live.py` (runs real `SubShader`,
  timed copy of the 3-line loop → src untouched; `next_chunk` recorded as `wait:` slack,
  excluded from compute total). Wired `--live-timing` into `test_suite.py`. Recorder
  extended to exclude `wait:`/`init:` from totals — verified. Default clip:
  beltran_sc_rip, 5s (~27 frames). **Pending real run on device.**
- **2026-06-25** — Pivoted Decision 1 offline→**live**: full pipeline + renderer,
  real-time paced on a real song clip (parity w/ live perf). Added goals: per-module/stage
  profiling **+ GPU transfer bottlenecks**. GPU sync moved from deferred → required step 3
  (split `raw_cwt` into upload/compute/download). Next: live wire-up (step 2).
