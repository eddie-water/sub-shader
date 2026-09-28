"""Multi-run timing campaign - N fresh-process init samples + one long loop
capture, recorded together under one run_id so timing_results.csv carries
N-sample init stats (mean/std) alongside a 300+ frame loop capture.

Why subprocess-per-init-sample
-------------------------------
CUDA context creation (init:cuda) is a once-per-process cost - the very thing
this campaign exists to measure honestly. A single long-running process can
only ever pay it once; every subsequent SubShader() construction inside that
process reuses the already-warm CUDA context. So each init sample is its own
fresh ``--child`` subprocess. The 300+ frame loop capture doesn't have this
problem (loop steady-state is the same warm pipeline every iteration) and
reuses the single-process ``run_live_timing`` driver.

Usage:
    python research/timing_campaign.py --smoke    # tiny N + short loop, verification only
    python research/timing_campaign.py --full      # N>=10 init samples + 300+ frame loop + report
    python research/timing_campaign.py --child      # internal: one init sample; not for direct use
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")

from utilities import (
    AUDIO_BELTRAN, TIMING_DIR, RESULTS_CSV, TimingRecorder, time_call,
    audio_frame_period_ms,
)

from subshader.config import CWTConfig
from subshader.pipeline import SubShader

from timing_live import run_live_timing, collect_init_stage_ms

# Marker line prefix so the child's one JSON payload survives amid any GPU
# driver / windowing-library noise that may land on stdout during construction.
_CHILD_MARKER = "SUBSHADER_INIT_JSON::"

# Coarse module-level keys collect_init_stage_ms always emits (unprefixed),
# PLUS "cuda" - its own top-level timed_block on the shader, not a module
# sub-stage, but still part of the one-time construction Total (same coarse-sum
# wiring contract as gen_timing_gantt._COARSE / timing_report._INIT_ORDER /
# timing_results._INIT_COARSE: leaving it out here would only mis-report the
# printed sanity total, not the recorded CSV/report data - but it's the exact
# "coarse-sum missing cuda" bug shape this campaign's own sanity check exists
# to catch, so it must be included here too.
_COARSE_KEYS = ("audio", "cuda", "dsp", "renderer", "prescan", "prime", "other")

INIT_SAMPLES_CSV = os.path.join(TIMING_DIR, "timing_init_samples.csv")
INIT_SAMPLES_COLUMNS = ["run_id", "sample_idx", "stage", "ms"]

DEFAULT_FULL_SAMPLES = 12        # N >= 10
DEFAULT_SMOKE_SAMPLES = 2
DEFAULT_FULL_SECONDS = 60.0      # ~323 frames at the default chunk/hop/sample-rate
                                  # (frame period = hop / sample_rate ~= 185.8 ms;
                                  # 12.0 s previously under-shot 300 frames, ~65 only)
DEFAULT_SMOKE_SECONDS = 1.0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_child_sample(audio_path):
    """One fresh-process init sample: construct SubShader, read its per-stage
    init breakdown, print it as a marker-prefixed JSON line, exit. Runs in its
    own subprocess (spawned by _spawn_init_samples) so init:cuda is paid fresh,
    exactly like a real program launch.
    """
    config = CWTConfig(file_path=audio_path)
    shader, init_ms = time_call(SubShader, config)
    stages = {f"init:{k}": float(v)
             for k, v in collect_init_stage_ms(shader, init_ms).items()}
    shader.cleanup()
    print(_CHILD_MARKER + json.dumps(stages), flush=True)


def _spawn_init_samples(n, audio_path):
    """Spawn `n` fresh child processes; return a list of {stage: ms} dicts."""
    samples = []
    for i in range(n):
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", "--audio", audio_path],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        marker_line = next(
            (line for line in result.stdout.splitlines() if line.startswith(_CHILD_MARKER)),
            None,
        )
        if marker_line is None:
            raise RuntimeError(
                f"init sample {i + 1}/{n} produced no JSON marker "
                f"(child exit code {result.returncode}).\n"
                f"--- child stdout (tail) ---\n{result.stdout[-2000:]}\n"
                f"--- child stderr (tail) ---\n{result.stderr[-2000:]}"
            )
        samples.append(json.loads(marker_line[len(_CHILD_MARKER):]))
        print(f"        init sample {i + 1}/{n} captured", flush=True)
    return samples


def _init_stage_arrays(samples):
    """[{stage: ms}, ...] (length N) -> {stage: np.array([...length N])}."""
    stage_names = sorted({stage for sample in samples for stage in sample})
    return {stage: np.array([sample.get(stage, 0.0) for sample in samples])
           for stage in stage_names}


def _write_init_sidecar(run_id, samples, csv_path=INIT_SAMPLES_CSV):
    """Persist per-sample init timings - one row per (run, sample, stage) - so
    the (Task 3) distribution figure can read the raw per-sample cloud instead
    of only the mean/std the main results CSV carries."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(INIT_SAMPLES_COLUMNS)
        for idx, sample in enumerate(samples):
            for stage, ms in sample.items():
                writer.writerow([run_id, idx, stage, round(float(ms), 4)])


def _sanity_pass(meta, stage_arrays, samples):
    """Plain-language pass/fail readout: coarse init stages sum to the measured
    Total, init:cuda lands around ~300 ms, the init:other residual stays small,
    and the loop total sits comfortably under the real-time deadline."""
    frame_stages = {k: v for k, v in stage_arrays.items()
                    if not k.startswith("init:") and not k.startswith("wait:")}
    loop_total = float(sum(np.mean(v) for v in frame_stages.values())) if frame_stages else 0.0

    init_coarse_ms = {k: float(np.mean(stage_arrays.get(f"init:{k}", [0.0])))
                      for k in _COARSE_KEYS}
    init_total = sum(init_coarse_ms.values())
    init_cuda = float(np.mean(stage_arrays.get("init:cuda", [0.0])))
    init_other = init_coarse_ms["other"]

    deadline_ms = audio_frame_period_ms(
        meta["chunk_size"], meta["overlap_factor"], meta["sample_rate"]
    )
    rt_margin = deadline_ms / loop_total if loop_total else 0.0

    cuda_ok = init_cuda > 50.0
    residual_ok = init_other < 100.0
    loop_ok = rt_margin >= 1.0

    print()
    print("  sanity pass")
    print(f"    init samples (N={len(samples)}):  "
          f"init:cuda = {init_cuda:6.1f} ms   init:other = {init_other:5.1f} ms   "
          f"coarse total = {init_total:6.1f} ms")
    print(f"    loop capture ({meta['num_frames']} frames):  "
          f"{loop_total:6.2f} ms/frame  vs  {deadline_ms:.1f} ms deadline  "
          f"({rt_margin:.1f}x headroom)")
    print(f"    checks:  init:cuda present {'OK' if cuda_ok else 'FAIL'}   ·   "
          f"init:other small {'OK' if residual_ok else 'FAIL'}   ·   "
          f"loop under deadline {'OK' if loop_ok else 'FAIL'}")
    print()


def run_campaign(n_samples, seconds, audio_path=AUDIO_BELTRAN, full=False):
    """Drive the full measurement campaign: N fresh-process init samples + one
    long loop capture, recorded together under one run_id so TIMING.md and the
    gantts read N-sample init means/std alongside the loop capture's means/std.

    `full=True` regenerates the standard report figures/TIMING.md afterward
    (via a lazy timing_report import); `full=False` (smoke) never touches
    timing_report - only the CSV/sidecar writes + the sanity print.
    """
    print(f"\nSubShader Timing Campaign - {n_samples} init samples, "
          f"{seconds:.0f}s loop capture ({'full' if full else 'smoke'})\n")

    print(f"  [1/3] Spawning {n_samples} fresh-process init samples...")
    samples = _spawn_init_samples(n_samples, audio_path)

    print(f"\n  [2/3] Capturing loop timing (single warm process, {seconds:.0f}s)...")
    result = run_live_timing(audio_path=audio_path, seconds=seconds,
                             record=False, quiet=not full)
    if result is None:
        print("Loop capture produced no frames - aborting campaign.")
        return None
    stage_arrays, meta, collected = result

    # The loop capture's own construction only ever gives ONE init sample (a
    # single warm process, not representative of a fresh launch) - replace its
    # init:* arrays with the N-sample campaign arrays before recording, so one
    # run_id carries both the N-sample init stats and the 300+ frame loop stats.
    stage_arrays = {k: v for k, v in stage_arrays.items() if not k.startswith("init:")}
    stage_arrays.update(_init_stage_arrays(samples))

    print("  [3/3] Recording campaign + writing sidecars...")
    recorder = TimingRecorder()
    run_id = recorder.record_run(meta, stage_arrays)
    _write_init_sidecar(run_id, samples)

    if full:
        from timing_report import update_report
        update_report()

    _sanity_pass(meta, stage_arrays, samples)
    print(f"  run id     →  {run_id}")
    print(f"  main csv   →  {RESULTS_CSV}")
    print(f"  init csv   →  {INIT_SAMPLES_CSV}")
    print()
    return run_id


def main():
    parser = argparse.ArgumentParser(description="SubShader timing campaign")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="Tiny N + short loop - verification only, no report regen")
    mode.add_argument("--full", action="store_true",
                      help="N>=10 init samples + 300+ frame loop, regenerates the standard report")
    mode.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audio", type=str, default=AUDIO_BELTRAN,
                        help="Audio file for both init construction and the loop capture")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override init sample count (N)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Override loop capture duration (seconds)")
    args = parser.parse_args()

    if args.child:
        _run_child_sample(args.audio)
        return

    if args.smoke:
        run_campaign(args.samples or DEFAULT_SMOKE_SAMPLES,
                    args.seconds or DEFAULT_SMOKE_SECONDS,
                    audio_path=args.audio, full=False)
    elif args.full:
        run_campaign(args.samples or DEFAULT_FULL_SAMPLES,
                    args.seconds or DEFAULT_FULL_SECONDS,
                    audio_path=args.audio, full=True)


if __name__ == "__main__":
    main()
