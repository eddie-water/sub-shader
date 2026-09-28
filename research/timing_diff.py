"""Timing diff - compare two recorded timing runs stage by stage.

Every performance fix gets a before/after pair of timing runs (see
assets/timing/LEDGER.md). This tool turns that pair into a readable markdown
delta table showing exactly which pipeline stages got faster, by how much,
and what it means for the real-time headroom.

Usage (from the repo root, same as timing_campaign.py):
    python research/timing_diff.py --list            # show recorded run ids
    python research/timing_diff.py <run_id_before> <run_id_after>
    python research/timing_diff.py <run_id_before> <run_id_after> --out delta.md

Reads:
    assets/timing/timing_results.csv     per-stage summary rows (init + loop)
    assets/timing/timing_iterations.csv  per-frame samples (loop stages) -
                                         used to compute medians, which the
                                         summary csv does not carry
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from utilities.timing_results import (
    RESULTS_CSV, ITERATIONS_CSV, _PIPE_ORDER, _PIPE_LABEL, _STAGE_MODULE,
    _md_table,
)

# Loop pacing: 16384-sample chunks, 50% overlap -> 8192 new samples per frame.
HOP_SAMPLES = 8192
SAMPLE_RATE = 44100
DEADLINE_MS = HOP_SAMPLES / SAMPLE_RATE * 1000.0
REALTIME_SAMPLES_PER_S = float(SAMPLE_RATE)


def read_summary_rows(run_id, csv_path=RESULTS_CSV):
    """Return {stage: row_dict} for one run from the summary csv."""
    stages = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run_id:
                stages[row["stage"]] = row
    return stages


def read_frame_samples(run_id, csv_path=ITERATIONS_CSV):
    """Return {stage: np.array of per-frame ms} for one run."""
    samples = defaultdict(list)
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run_id:
                samples[row["stage"]].append(float(row["ms"]))
    return {stage: np.asarray(ms) for stage, ms in samples.items()}


def list_run_ids(csv_path=RESULTS_CSV):
    """All recorded run ids with timestamp and git sha, in file order."""
    seen = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seen.setdefault(row["run_id"], (row["timestamp"], row["git_sha"]))
    return seen


def stage_stats(stage, summary, frames):
    """(mean, median, min, max) in ms for one stage, preferring per-frame data."""
    if stage in frames and len(frames[stage]):
        arr = frames[stage]
        return arr.mean(), float(np.median(arr)), arr.min(), arr.max()
    if stage in summary:
        row = summary[stage]
        return (float(row["mean_ms"]), None,
                float(row["min_ms"]), float(row["max_ms"]))
    return None


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def fmt_delta(before, after):
    """Delta as 'signed ms (signed %)'; negative = faster."""
    if before is None or after is None:
        return "-"
    delta = after - before
    pct = (delta / before * 100.0) if before else 0.0
    return f"{delta:+.2f} ({pct:+.0f}%)"


# Module color dots matching the timing diagrams (dsplot palette-3):
# Audio = TERTIARY #ffd27d, DSP = SECONDARY #7b6fe1, Render = PRIMARY #ff5a1f,
# Setup = SPINE #444444 - same assignment as gen_timing_gantt.py.
MODULE_DOT = {"audio": "🟡", "dsp": "🟣", "render": "🟠", "setup": "⚫"}

# Start-up stage → module, mirroring gen_timing_gantt's color assignment
# (prescan is renderer-colored there: it calibrates the color map).
_INIT_MODULE = {
    "audio": "audio", "audio_reader": "audio", "audio_player": "audio",
    "cuda": "dsp", "dsp": "dsp", "dsp_kernels": "dsp", "dsp_fft": "dsp",
    "dsp_upload": "dsp",
    "renderer": "render", "render_buffer": "render",
    "render_glcontext": "render", "render_shader": "render",
    "prescan": "render",
    "prime": "setup", "other": "setup",
}


def module_dot(module):
    return MODULE_DOT.get(module, "")


# Ornament thresholds: a change must be both material in absolute terms
# (SAME_MS) and in relative terms (SAME_PCT) to count as movement at all;
# the dot count then scales with the relative size of the change.
SAME_MS, SAME_PCT = 0.25, 5.0
MODERATE_PCT, BIG_PCT = 25.0, 60.0


def ornament(before, after):
    """Direction + magnitude glyphs: 🟢 faster / 🔴 slower / ⚪ same-ish.

    One dot = noticeable (>5%), two = substantial (>25%), three = huge (>60%).
    Tiny absolute moves (<0.25 ms) read as same-ish regardless of percent.
    """
    if before is None or after is None:
        return ""
    delta = after - before
    pct = abs(delta / before * 100.0) if before else 0.0
    if abs(delta) < SAME_MS or pct < SAME_PCT:
        return "⚪"
    dots = 1 + (pct >= MODERATE_PCT) + (pct >= BIG_PCT)
    return ("🟢" if delta < 0 else "🔴") * dots


def build_delta_tables(run_before, run_after):
    """Return (loop_table_md, init_table_md, headline_md) for the run pair."""
    summary_b = read_summary_rows(run_before)
    summary_a = read_summary_rows(run_after)
    if not summary_b:
        sys.exit(f"run id not found in {RESULTS_CSV}: {run_before}")
    if not summary_a:
        sys.exit(f"run id not found in {RESULTS_CSV}: {run_after}")
    frames_b = read_frame_samples(run_before)
    frames_a = read_frame_samples(run_after)

    module_totals = {}
    loop_rows = []
    for stage in _PIPE_ORDER + ["TOTAL"]:
        stats_b = stage_stats(stage, summary_b, frames_b)
        stats_a = stage_stats(stage, summary_a, frames_a)
        if stats_b is None and stats_a is None:
            continue
        mean_b, med_b = (stats_b[0], stats_b[1]) if stats_b else (None, None)
        mean_a, med_a = (stats_a[0], stats_a[1]) if stats_a else (None, None)
        if stage == "TOTAL":
            label = "**Whole frame**"
        else:
            label = (module_dot(_STAGE_MODULE.get(stage)) + " "
                     + _PIPE_LABEL.get(stage, stage))
        loop_rows.append([
            ornament(mean_b, mean_a), label,
            fmt(mean_b), fmt(mean_a), fmt_delta(mean_b, mean_a),
            fmt(med_b), fmt(med_a),
        ])
        module = _STAGE_MODULE.get(stage)
        if module and mean_b is not None and mean_a is not None:
            acc = module_totals.setdefault(module, [0.0, 0.0])
            acc[0] += mean_b
            acc[1] += mean_a
    loop_table = _md_table(
        ["", "Stage", "before mean", "after mean", "Δ mean ms (%)",
         "before median", "after median"],
        loop_rows)

    module_rows = []
    for module, title in (("audio", "Audio"), ("dsp", "DSP"), ("render", "Render")):
        if module not in module_totals:
            continue
        mean_b, mean_a = module_totals[module]
        module_rows.append([ornament(mean_b, mean_a),
                            f"{module_dot(module)} {title}",
                            fmt(mean_b), fmt(mean_a), fmt_delta(mean_b, mean_a)])
    total_b = stage_stats("TOTAL", summary_b, frames_b)
    total_a = stage_stats("TOTAL", summary_a, frames_a)
    if total_b and total_a:
        module_rows.append([ornament(total_b[0], total_a[0]), "**Whole frame**",
                            fmt(total_b[0]), fmt(total_a[0]),
                            fmt_delta(total_b[0], total_a[0])])
    module_table = _md_table(
        ["", "Module", "before mean", "after mean", "Δ mean ms (%)"],
        module_rows)

    init_stages = sorted(s for s in set(summary_b) | set(summary_a)
                         if s.startswith("init:"))
    init_rows = []
    for stage in init_stages:
        stats_b = stage_stats(stage, summary_b, frames_b)
        stats_a = stage_stats(stage, summary_a, frames_a)
        mean_b = stats_b[0] if stats_b else None
        mean_a = stats_a[0] if stats_a else None
        short = stage.removeprefix("init:")
        init_rows.append([ornament(mean_b, mean_a),
                          f"{module_dot(_INIT_MODULE.get(short))} {short}".strip(),
                          fmt(mean_b), fmt(mean_a), fmt_delta(mean_b, mean_a)])
    init_table = _md_table(
        ["", "Start-up stage", "before mean", "after mean", "Δ mean ms (%)"],
        init_rows)

    headline = headline_md(summary_b, summary_a, frames_b, frames_a)
    return module_table, loop_table, init_table, headline


def headline_md(summary_b, summary_a, frames_b, frames_a):
    """Deadline / headroom / throughput summary for the TOTAL row."""
    lines = []
    for name, summary, frames in (("before", summary_b, frames_b),
                                  ("after", summary_a, frames_a)):
        stats = stage_stats("TOTAL", summary, frames)
        if stats is None:
            continue
        mean, med, lo, hi = stats
        headroom = DEADLINE_MS / mean
        samples_per_s = HOP_SAMPLES / (mean / 1000.0)
        lines.append(
            f"- **{name}**: {mean:.1f} ms work per {DEADLINE_MS:.0f} ms frame "
            f"→ {headroom:.1f}× headroom, ~{samples_per_s/1000:.0f}k samples/s "
            f"(real time = {REALTIME_SAMPLES_PER_S/1000:.1f}k) "
            f"[median {fmt(med, 1)}, worst {hi:.1f}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_before", nargs="?", help="run id before the fix")
    parser.add_argument("run_after", nargs="?", help="run id after the fix")
    parser.add_argument("--list", action="store_true",
                        help="list recorded run ids and exit")
    parser.add_argument("--out", help="write markdown to this file")
    args = parser.parse_args()

    if args.list or not (args.run_before and args.run_after):
        print(f"{'run_id':44s} {'timestamp':20s} git_sha")
        for run_id, (ts, sha) in list_run_ids().items():
            print(f"{run_id:44s} {ts:20s} {sha}")
        return

    module_table, loop_table, init_table, headline = build_delta_tables(
        args.run_before, args.run_after)

    legend = ("🟢 faster · 🔴 slower · ⚪ same-ish — "
              "one dot >5%, two >25%, three >60% change\n"
              "Modules (diagram colors): 🟡 Audio · 🟣 DSP · 🟠 Render · ⚫ Setup")
    md = "\n\n".join([
        f"**Runs compared:** `{args.run_before}` → `{args.run_after}`",
        headline,
        legend,
        "### Modules (per frame)", module_table,
        "### Process loop stages (per frame)", loop_table,
        "### Start-up (one time)", init_table,
    ]) + "\n"

    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
