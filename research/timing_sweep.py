"""Config sweep — compare live timing across backend / chunk_size / overlap.

One-factor-at-a-time (OFAT) around a baseline: each config is a *real* live run
(same pipeline as ``timing_live``) recorded to the shared CSV, then rendered as a
stacked-bar comparison PNG. Runs are **un-paced** — we want raw compute-per-frame
to answer "which setup is cheapest?", not real-time slack.

Axes covered here (all propagate through ``SubShader`` with no src changes):
    backend      gpu vs cpu
    chunk_size   4k / 8k / 16k / 32k
    overlap      0.25 / 0.5 / 0.75

``num_octaves`` is intentionally omitted: it changes ``num_freqs`` (and the
renderer's output shape) but ``SubShader`` rebuilds its CWTConfig internally and
drops it, so sweeping it cleanly needs a small src/pipeline.py change (flagged,
not done here).

Usage:
    python research/test_suite.py --sweep
"""

import os
import sys
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities import AUDIO_BELTRAN, RESULTS_CSV, TIMING_DIR
from utilities.timing_results import _read_rows, TOTAL_STAGE, _is_frame_stage

from timing_report import update_report

_TEST_SUITE = os.path.join("research", "test_suite.py")
SWEEP_PNG = os.path.join(TIMING_DIR, "timing_sweep.png")
SWEEP_SECONDS = 3.0

# Baseline held fixed while one axis varies.
_BASE = {"chunk_size": 16384, "overlap_factor": 0.5, "num_octaves": 10, "backend": "gpu"}

# Stage colors (data-flow order). Anything unseen falls back to gray.
_STAGE_COLORS = {
    "raw_cwt":    "#4c72b0",
    "magnitude":  "#dd8452",
    "edge_trim":  "#55a868",
    "hop_center": "#c44e52",
    "downsample": "#8172b3",
    "render":     "#937860",
}
_STAGE_ORDER = list(_STAGE_COLORS)


def _sweep_configs():
    """OFAT config list as (group, label, overrides)."""
    configs = []
    for b in ["gpu", "cpu"]:
        configs.append(("backend", b, {**_BASE, "backend": b}))
    for c in [4096, 8192, 16384, 32768]:
        configs.append(("chunk_size", str(c), {**_BASE, "chunk_size": c}))
    for ov in [0.25, 0.5, 0.75]:
        configs.append(("overlap", f"{ov:g}", {**_BASE, "overlap_factor": ov}))
    for oc in [6, 8, 10]:
        configs.append(("octaves", str(oc), {**_BASE, "num_octaves": oc}))
    return configs


def _run_one(overrides, seconds):
    """Run one config in a fresh process (cold GPU) and return its run_id.

    Each config gets its own `python test_suite.py --live-timing ...` process so
    no warm GPU state (CuPy pool, cuFFT plan cache, CUDA context) carries between
    configs — the comparison is fair. The child prints `RUN_ID=<id>`; we parse it.
    """
    argv = [
        sys.executable, _TEST_SUITE, "--live-timing", "--quiet", "--no-pace",
        "--seconds", str(seconds),
        "--chunk-size", str(overrides["chunk_size"]),
        "--overlap", str(overrides["overlap_factor"]),
        "--octaves", str(overrides["num_octaves"]),
        "--backend", overrides["backend"],
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    run_id = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("RUN_ID="):
            run_id = s.split("=", 1)[1]
        elif "ms/frame" in s:  # the child's compact result line
            print(f"     {s}")
    if proc.returncode != 0 or run_id is None:
        print(f"     ⚠ run failed (rc={proc.returncode})")
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        for t in tail:
            print(f"       {t}")
    return run_id


def run_sweep(audio_path=AUDIO_BELTRAN, seconds=SWEEP_SECONDS):
    """Run the OFAT sweep, append to the CSV, and render the comparison PNG."""
    configs = _sweep_configs()
    print(f"\nSubShader Config Sweep — {len(configs)} runs, "
          f"{seconds:.0f}s each, un-paced, fresh process per config\n")

    results = []  # (run_id, group, label)
    for i, (group, label, overrides) in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] {group} = {label}  (cold start)")
        run_id = _run_one(overrides, seconds)
        if run_id:
            results.append((run_id, group, label))

    if results:
        _plot(results)
    update_report()
    if results:
        print(f"\n  sweep plot  →  {SWEEP_PNG}")
        print(f"  numbers     →  {RESULTS_CSV}")
        print(f"  table       →  {os.path.join(TIMING_DIR, 'TIMING.md')}\n")
    else:
        print("\n  No runs recorded — nothing to plot.\n")


def _run_lookup(rows):
    """run_id -> {stage: mean_ms} (frame stages) plus total/rt/meta."""
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], {})
        stage = r["stage"]
        if stage == TOTAL_STAGE:
            by_run[r["run_id"]]["_total"] = float(r["mean_ms"])
            by_run[r["run_id"]]["_rt"] = r["rt_margin"]
        elif _is_frame_stage(stage):
            by_run[r["run_id"]][stage] = float(r["mean_ms"])
    return by_run


def _plot(results):
    """Stacked-bar comparison: one subplot per swept axis, bars = configs."""
    lookup = _run_lookup(_read_rows(RESULTS_CSV))

    groups = []
    for _, group, _label in results:
        if group not in groups:
            groups.append(group)

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(9, 2.4 * len(groups) + 0.6), squeeze=False,
    )
    axes = axes[:, 0]
    fig.suptitle("SubShader config sweep — per-frame compute (ms)",
                 fontsize=13, fontweight="bold")

    seen_stages = []
    for ax, group in zip(axes, groups):
        runs = [(rid, label) for rid, g, label in results if g == group]
        y = np.arange(len(runs))
        labels = [label for _, label in runs]

        for row, (rid, _label) in enumerate(runs):
            data = lookup.get(rid, {})
            left = 0.0
            for stage in _STAGE_ORDER:
                val = data.get(stage, 0.0)
                if val <= 0:
                    continue
                ax.barh(row, val, left=left, color=_STAGE_COLORS[stage],
                        edgecolor="white", linewidth=0.5)
                left += val
                if stage not in seen_stages:
                    seen_stages.append(stage)
            total = data.get("_total", left)
            rt = data.get("_rt", "")
            rt_txt = f"  RT {float(rt):.1f}x" if rt not in ("", None) else ""
            ax.text(total, row, f"  {total:.2f} ms{rt_txt}",
                    va="center", ha="left", fontsize=8.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, len(runs) - 0.4)
        ax.invert_yaxis()
        ax.set_title(f"{group}", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("ms / frame")
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_STAGE_COLORS[s]) for s in seen_stages]
    fig.legend(handles, seen_stages, loc="lower center", ncol=len(seen_stages),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    os.makedirs(TIMING_DIR, exist_ok=True)
    fig.savefig(SWEEP_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_sweep()
