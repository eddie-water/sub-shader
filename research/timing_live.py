"""Live timing driver — profile the real SubShader pipeline on a song clip.

Runs the actual production pipeline (AudioStream -> CWT -> Renderer) for a few
seconds of a real song, paced by the audio device clock exactly like a normal
run, and records per-stage timings to the standardized CSV.

Why this lives in research/ and not src/
----------------------------------------
SubShader.run() is a 3-line loop: ``next_chunk -> dsp.process -> renderer.update``.
The per-stage ``@timed`` tags already exist inside src/ (CWT stages + renderer).
So we construct the real ``SubShader`` and run a *timed copy* of that loop around
its public stages — src/ is untouched, and the numbers have parity with live
performance because it IS the live pipeline (same objects, audio-clock paced).

The ``next_chunk()`` call blocks on the audio clock, so its time is real-time
*slack*, not compute. It is recorded as ``wait:next_chunk`` and excluded from the
per-frame compute total (and therefore from FPS / RT margin).

Usage:
    python research/test_suite.py --live-timing
"""

import os
import sys
import time

import numpy as np

from subshader.config import CWTConfig
from subshader.pipeline import SubShader
from subshader.exceptions import SubShaderException

from timing_gpu import InstrumentedGpuCWT, GPU_LEG_LABELS

from utilities import (
    AUDIO_BELTRAN,
    TimingRecorder,
    RESULTS_CSV,
    RESULTS_MD,
    time_call,
)

from timing_report import update_report

DEFAULT_SECONDS = 5.0

# CWT stages AFTER the transform — each is @timed in src (attr -> stage label).
# The transform itself is handled separately: one `raw_cwt` blob normally, or the
# split GPU legs when transfer profiling is on.
_POST_STAGES = {
    # normalize omitted: _normalize_by_scale is a deliberate no-op (energy bias is
    # corrected at kernel construction), not @timed — it would only add a 0.00 row.
    "magnitude":  "_timing__compute_mag_ms",
    "edge_trim":  "_timing_discard_unreliable_coefs_ms",
    "hop_center": "_timing_extract_hop_center_ms",
    "downsample": "_timing_downsample_ms",
}

# Render sub-stages — each is @timed in src, but the attrs live on the renderer's
# sub-objects (frame_buffer / gpu_renderer / gl_context). Read after update().
# label -> (sub-object attr on Renderer, timing attr on that sub-object)
_RENDER_LEGS = {
    "buf_push":   ("frame_buffer", "_timing_push_frame_ms"),
    "tex_upload": ("gpu_renderer", "_timing_update_texture_ms"),
    "gl_clear":   ("gl_context",   "_timing_clear_graphic_ms"),
    "gl_draw":    ("gpu_renderer", "_timing_render_graphic_ms"),
    "gl_swap":    ("gl_context",   "_timing_display_graphic_ms"),
}
_RENDER_LEG_LABELS = list(_RENDER_LEGS)


# Plain-language stage labels for the console summary.
_FRIENDLY = {
    "audio_read": "read chunk",
    "raw_cwt":    "wavelet transform",
    "fft_cpu":    "fft (cpu)",
    "upload":     "upload  → gpu",
    "multiply":   "multiply (gpu)",
    "ifft":       "ifft (gpu)",
    "download":   "download → cpu",
    "magnitude":  "magnitude",
    "edge_trim":  "trim edges",
    "hop_center": "center hop",
    "downsample": "downsample",
    "render":     "render frame",
    "buf_push":   "buffer push",
    "tex_upload": "texture upload",
    "gl_clear":   "clear",
    "gl_draw":    "draw",
    "gl_swap":    "swap + vsync",
}


def _bar(frac, width=14):
    """Unicode bar for a 0..1 fraction."""
    n = int(round(frac * width))
    return "█" * n + " " * (width - n)


def _print_summary(backend, audio_path, num_frames, stages, period_ms, run_id):
    """Clean, plain-language readout of where the per-frame time goes."""
    legs = [(lbl, float(np.mean(stages[lbl])))
            for lbl in stages if not lbl.startswith("wait:")]
    legs.sort(key=lambda x: x[1], reverse=True)
    total = sum(m for _, m in legs) or 1e-9
    fps = 1000.0 / total
    head = period_ms / total
    verdict = ("keeps up easily" if head >= 2
               else "keeps up" if head >= 1 else "TOO SLOW")
    mark = "ok" if head >= 1 else "!!"

    rule = "  " + "─" * 54
    print()
    print(rule)
    print(f"   RESULTS   {backend}  ·  {num_frames} frames  ·  {os.path.basename(audio_path)}")
    print(rule)
    print()
    print("   where the time goes   (per frame)")
    print()
    for lbl, m in legs:
        frac = m / total
        print(f"     {_FRIENDLY.get(lbl, lbl):<18}  →  {m:6.2f} ms   {_bar(frac)} {frac*100:3.0f}%")
    print(f"     {'-' * 48}")
    print(f"     {'TOTAL':<18}  →  {total:6.2f} ms")
    print()
    print("   speed")
    print()
    print(f"     frame rate      →  {fps:5.0f} FPS")
    print(f"     audio budget    →  {period_ms:6.1f} ms per frame")
    print(f"     headroom        →  {head:5.1f}x   [{mark}] {verdict}")
    print()
    print("   saved")
    print()
    print(f"     numbers   →  {RESULTS_CSV}")
    print(f"     table     →  {RESULTS_MD}")
    print(f"     run id    →  {run_id}")
    print()


def _target_frames(sample_rate, hop_size, seconds):
    """Number of frames that fit in `seconds` of audio at this hop size."""
    frame_period_s = hop_size / sample_rate
    return max(1, int(round(seconds / frame_period_s)))


def _cwt_config_from(config):
    """Rebuild a CWTConfig the same way SubShader does internally (for dsp swaps).

    Carries num_octaves so a swapped backend keeps the same num_freqs / output
    shape the renderer was built with.
    """
    return CWTConfig(
        file_path=config.file_path,
        chunk_size=config.chunk_size,
        overlap_factor=config.overlap_factor,
        sample_rate=config.sample_rate,
        total_samples=config.total_samples,
        num_octaves=config.num_octaves,
        notes_per_octave=config.notes_per_octave,
    )


def run_live_timing(audio_path=AUDIO_BELTRAN, seconds=DEFAULT_SECONDS,
                    profile_gpu_transfers=True, profile_render=True,
                    chunk_size=None, overlap_factor=None, num_octaves=None,
                    backend=None, paced=True, quiet=False):
    """Profile the live pipeline on `seconds` of `audio_path`; record to CSV + MD.

    When `profile_gpu_transfers` and the backend is GpuCWT, the CWT is swapped for
    a synced InstrumentedGpuCWT so `raw_cwt` splits into fft/upload/multiply/ifft/
    download legs (upload+download = the transfer bottlenecks).

    Sweep knobs (defaults preserve the plain `--live-timing` behaviour):
        chunk_size / overlap_factor  override the config (both propagate through
            SubShader natively, including the renderer's output shape).
        backend  "cpu" forces CpuCWT, "gpu" forces GpuCWT, None auto-selects.
        paced    True paces to the audio clock (real-time parity); False runs the
            frames back-to-back to measure raw compute throughput (used by the sweep).
        quiet    suppress the verbose readout; print one compact line instead.

    Returns the recorded `run_id` (or None if no frames were collected).
    """
    log = (lambda *a, **k: None) if quiet else print
    log(f"\nSubShader Live Timing — {seconds:.0f}s of {audio_path}\n")

    cfg_kwargs = {"file_path": audio_path}
    if chunk_size is not None:
        cfg_kwargs["chunk_size"] = chunk_size
    if overlap_factor is not None:
        cfg_kwargs["overlap_factor"] = overlap_factor
    if num_octaves is not None:
        cfg_kwargs["num_octaves"] = num_octaves
    config = CWTConfig(**cfg_kwargs)

    # Construct the real pipeline (audio, dsp, renderer, intensity pre-scan).
    # AudioStream writes sample_rate/total_samples back into `config`.
    log("  [1/3] Building pipeline (audio + CWT + renderer + intensity pre-scan)...")
    log("        this opens the render window and can take a few seconds.", flush=True)
    shader, init_ms = time_call(SubShader, config)
    sample_rate = config.sample_rate
    hop_size = config.hop_size
    # Per-module init breakdown. Each constructor (AudioStream / GpuCWT|CpuCWT /
    # Renderer) is @timed, and _prescan_intensity is @timed separately. Read here,
    # before the backend swap (which would overwrite shader.dsp) and before
    # cleanup() nulls the sub-objects. `init:other` is the leftover glue/config.
    init_audio_ms = getattr(shader.audio, "_timing___init___ms", 0.0)
    init_dsp_ms = getattr(shader.dsp, "_timing___init___ms", 0.0)
    init_renderer_ms = getattr(shader.renderer, "_timing___init___ms", 0.0)
    init_prescan_ms = getattr(shader, "_timing__prescan_intensity_ms", 0.0)
    init_other_ms = max(0.0, init_ms - (init_audio_ms + init_dsp_ms
                                        + init_renderer_ms + init_prescan_ms))

    # Per-module sub-stage breakdown (timed_block context managers inside each
    # constructor). Read before the backend swap recreates shader.dsp. Each dict
    # maps a sub-stage key (e.g. "dsp_kernels") to its construction time in ms.
    init_substages = {}
    for module in (shader.audio, shader.dsp, shader.renderer):
        init_substages.update(getattr(module, "_init_substages", {}) or {})

    # Force a backend if requested (the renderer's output shape is identical across
    # backends — (num_freqs, target_width) — so the swap is safe). Mirrors how
    # SubShader builds its internal CWTConfig.
    if backend == "cpu" and type(shader.dsp).__name__ != "CpuCWT":
        from subshader.dsp.cwt import CpuCWT
        shader.dsp.cleanup()
        shader.dsp = CpuCWT(_cwt_config_from(config))

    backend_name = type(shader.dsp).__name__
    num_freqs = shader.dsp.num_freqs  # capture before cleanup() nulls shader.dsp

    # Swap in the synced subclass to expose GPU transfer legs (research-only).
    transfer_profiling = False
    if profile_gpu_transfers and backend_name == "GpuCWT":
        shader.dsp.cleanup()
        shader.dsp = InstrumentedGpuCWT(_cwt_config_from(config))
        transfer_profiling = True
        log("        GPU transfer profiling ON (synced legs; total slightly pessimistic)")
    num_frames = _target_frames(sample_rate, hop_size, seconds)
    expected_s = num_frames * hop_size / sample_rate
    log(f"        ready in {init_ms/1000:.1f}s  ·  backend={backend_name}  "
        f"sample_rate={sample_rate:.0f}  hop={hop_size}")
    pace_note = "real-time paced" if paced else "un-paced (raw compute)"
    log(f"  [2/3] Profiling {num_frames} frames ({pace_note})...", flush=True)

    # Per-stage timing collectors, in pipeline order:
    #   audio_read → (transform: raw_cwt blob or split GPU legs) → dsp post stages
    #   → (render: one blob or split GL legs)
    transform_labels = GPU_LEG_LABELS if transfer_profiling else ["raw_cwt"]
    render_labels = _RENDER_LEG_LABELS if profile_render else ["render"]
    stages = {"wait:next_chunk": [], "audio_read": []}
    for label in transform_labels:
        stages[label] = []
    for label in _POST_STAGES:
        stages[label] = []
    for label in render_labels:
        stages[label] = []

    # Pacing. With `paced`, the audio device clock drives the loop (real-time
    # parity); on a remote/headless box with no output device we fall back to a
    # silent software clock — compute timings are identical, only audible playback
    # + device-clock pacing drop. With `paced=False` (sweep) frames run back-to-back
    # to measure raw compute throughput.
    frame_period_s = hop_size / sample_rate
    use_audio_clock = False
    if paced:
        try:
            shader.audio.start()
            use_audio_clock = True
        except SubShaderException as e:
            print(f"        no audio output device; silent software-paced timing.\n"
                  f"        ({e})", flush=True)

    reader = shader.audio._reader
    collected = 0
    loop_start = time.perf_counter()
    while collected < num_frames and not shader.renderer.should_close():
        if use_audio_clock:
            # next_chunk() blocks on the audio clock — its time is real-time slack,
            # not the read compute (which it does internally, not via @timed get_chunk).
            t0 = time.perf_counter()
            chunk = shader.audio.next_chunk()
            wait_ms = (time.perf_counter() - t0) * 1000.0
            audio_read_ms = 0.0
        else:
            # Route through the @timed AudioStream.get_chunk so the read compute
            # (file read + slice) is captured separately from any pacing slack.
            chunk = shader.audio.get_chunk()
            audio_read_ms = getattr(shader.audio, "_timing_get_chunk_ms", 0.0)
            wait_ms = 0.0  # software-paced: set below; un-paced: stays 0
        if chunk is None:
            continue  # EOF/loop boundary — reader/audio handles reset, don't count it

        coefs = shader.dsp.process(chunk)
        shader.renderer.update(coefs)

        if paced and not use_audio_clock:
            # Pace to the audio frame period with a software clock; record the
            # leftover sleep as real-time slack (excluded from the compute total).
            slack = (loop_start + (collected + 1) * frame_period_s) - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            wait_ms = max(0.0, slack) * 1000.0

        stages["wait:next_chunk"].append(wait_ms)
        stages["audio_read"].append(audio_read_ms)
        if transfer_profiling:
            for label in GPU_LEG_LABELS:
                stages[label].append(shader.dsp.leg_times_ms.get(label, 0.0))
        else:
            stages["raw_cwt"].append(getattr(shader.dsp, "_timing_transform_ms", 0.0))
        for label, attr in _POST_STAGES.items():
            stages[label].append(getattr(shader.dsp, attr, 0.0))
        if profile_render:
            for label, (sub, attr) in _RENDER_LEGS.items():
                stages[label].append(getattr(getattr(shader.renderer, sub), attr, 0.0))
        else:
            stages["render"].append(getattr(shader.renderer, "_timing_update_ms", 0.0))

        collected += 1
        if not quiet:
            elapsed = time.perf_counter() - loop_start
            pct = int(40 * collected / num_frames)
            sys.stdout.write(
                f"\r        [{'#'*pct}{'.'*(40-pct)}] "
                f"{collected}/{num_frames} frames  {elapsed:4.1f}s"
            )
            sys.stdout.flush()

    log()  # finish the progress line
    log("  [3/3] Writing results...", flush=True)
    shader.cleanup()

    if collected == 0:
        print("No frames collected (window closed immediately?). Nothing recorded.")
        return None

    stage_arrays = {
        "init:audio": np.array([init_audio_ms]),
        "init:dsp": np.array([init_dsp_ms]),
        "init:renderer": np.array([init_renderer_ms]),
        "init:prescan": np.array([init_prescan_ms]),
        "init:other": np.array([init_other_ms]),
    }
    # Fine-grained init sub-stages (e.g. init:dsp_kernels, init:render_glcontext).
    # Emitted alongside the coarse module rows: the coarse rows remain the
    # authoritative per-module totals; these break each module down.
    for key, ms in init_substages.items():
        stage_arrays[f"init:{key}"] = np.array([float(ms)])
    stage_arrays.update({k: np.asarray(v, dtype=float) for k, v in stages.items()})

    meta = dict(
        backend=backend_name,
        chunk_size=config.chunk_size,
        num_freqs=num_freqs,
        num_frames=collected,
        sample_rate=sample_rate,
        overlap_factor=config.overlap_factor,
    )

    recorder = TimingRecorder()
    run_id = recorder.record_run(meta, stage_arrays)
    update_report()

    period_ms = hop_size / sample_rate * 1000.0
    if quiet:
        total = sum(float(np.mean(stages[s])) for s in stages if not s.startswith("wait:"))
        rt = period_ms / total if total else 0.0
        print(f"        {backend_name}  c{config.chunk_size}  ov{config.overlap_factor:g}  "
              f"f{num_freqs}  →  {total:5.2f} ms/frame   RT {rt:4.1f}x   ({collected} frames)")
    else:
        _print_summary(backend_name, audio_path, collected, stages, period_ms, run_id)
    return run_id


if __name__ == "__main__":
    run_live_timing()
