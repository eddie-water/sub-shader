"""Standardized timing-result persistence and rendering.

One tidy CSV (``assets/timing/timing_results.csv``) is the single source of truth:
one row per ``(run, stage)``. ``render_markdown`` turns it into a GitHub-viewable
table (``assets/timing/TIMING.md``).

Layering contract
-----------------
The in-``src`` timing primitive is the ``@timed`` decorator
(``subshader.utils.timing``), which writes ``self._timing_{method}_ms`` and nothing
else. This module never imports from ``src`` and never measures — it only records
the timing arrays the drivers collect, and renders them. ``research/`` may read
``src``; ``src`` never reads ``research/``.

Schema (tidy / long, append-only)
---------------------------------
    run_id, timestamp, git_sha, backend, chunk_size, num_freqs, num_frames,
    stage, mean_ms, std_ms, min_ms, max_ms, pct_of_total, rt_margin

``rt_margin`` is populated only on the synthetic ``TOTAL`` row:
``rt_margin = audio_frame_period_ms / total_mean_ms`` (``> 1`` ⇒ real-time capable).
``audio_frame_period_ms = hop_size / sample_rate * 1000`` where
``hop_size = chunk_size * (1 - overlap_factor)``.
"""

import csv
import os
import subprocess
import datetime

import numpy as np

from .constants import TIMING_DIR

RESULTS_CSV = os.path.join(TIMING_DIR, "timing_results.csv")
RESULTS_MD = os.path.join(TIMING_DIR, "TIMING.md")

# Method-comparison results (STFT / PyWavelet / CWT-CPU / CWT-GPU) live in their
# own small CSV — different shape from the per-stage table. The measurement lives
# in research/timing_methods.py; storage + rendering stay here (no src imports) so
# render_markdown can fold the table in.
METHODS_CSV = os.path.join(TIMING_DIR, "timing_methods.csv")
METHODS_COLUMNS = ["timestamp", "git_sha", "method", "backend", "chunk_size",
                   "num_freqs", "sample_rate", "freq_lo", "freq_hi",
                   "mean_ms", "std_ms", "native_res", "res_kind"]

COLUMNS = [
    "run_id",
    "timestamp",
    "git_sha",
    "backend",
    "chunk_size",
    "num_freqs",
    "num_frames",
    "stage",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "pct_of_total",
    "rt_margin",
]

TOTAL_STAGE = "TOTAL"

# Stage-name prefixes excluded from the per-frame compute total and % breakdown:
#   init:*  one-time setup costs
#   wait:*  real-time-paced slack (e.g. blocking on the audio clock) — not work
_EXCLUDED_PREFIXES = ("init:", "wait:")


def _is_frame_stage(stage):
    """True if a stage counts toward the per-frame compute total."""
    return not stage.startswith(_EXCLUDED_PREFIXES)


# =============================================================================
# DERIVED TIMING QUANTITIES
# =============================================================================

def audio_frame_period_ms(chunk_size, overlap_factor, sample_rate):
    """Wall-clock duration of audio consumed per frame, in milliseconds.

    A frame advances by ``hop_size = chunk_size * (1 - overlap_factor)`` samples,
    so the pipeline must produce a frame within ``hop_size / sample_rate`` seconds
    to keep up with real-time playback.
    """
    hop_size = chunk_size * (1.0 - overlap_factor)
    return hop_size / sample_rate * 1000.0


def git_sha():
    """Short git SHA of HEAD, or ``"nogit"`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _timestamp():
    """Current local timestamp as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_id(timestamp, backend, chunk_size, num_freqs):
    """Compact, human-scannable run identifier."""
    compact = timestamp.replace("-", "").replace(":", "").replace(" ", "_")
    return f"{compact}_{backend}_c{chunk_size}_f{num_freqs}"


def _stats(arr):
    """mean/std/min/max of a timing array in ms (handles empty arrays)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    )


# =============================================================================
# RECORDER
# =============================================================================

class TimingRecorder:
    """Append timing runs to the standardized tidy CSV.

    One ``record_run`` call writes one row per stage plus a synthetic ``TOTAL``
    row (elementwise sum of the per-frame stage arrays, so its std reflects the
    real per-frame total distribution rather than a sum of independent stds).
    """

    def __init__(self, csv_path=RESULTS_CSV):
        self.csv_path = csv_path

    def record_run(self, meta, stage_arrays):
        """Record one benchmark run.

        Args:
            meta: dict with keys ``backend``, ``chunk_size``, ``num_freqs``,
                ``num_frames``, ``sample_rate``, ``overlap_factor``. Optional
                ``timestamp``, ``git_sha``, ``run_id`` are filled if absent.
            stage_arrays: ordered dict mapping stage name -> per-frame timing
                array (ms). Init/one-shot timings may use a length-1 array.

        Returns:
            The ``run_id`` string used for the recorded rows.
        """
        timestamp = meta.get("timestamp") or _timestamp()
        sha = meta.get("git_sha") or git_sha()
        backend = meta["backend"]
        chunk_size = meta["chunk_size"]
        num_freqs = meta["num_freqs"]
        num_frames = meta["num_frames"]
        run_id = meta.get("run_id") or _run_id(timestamp, backend, chunk_size, num_freqs)

        # Per-frame total = elementwise sum of stages that vary per frame.
        # init:* (setup) and wait:* (real-time slack) rows are excluded.
        frame_stages = {k: np.asarray(v, dtype=float)
                        for k, v in stage_arrays.items()
                        if _is_frame_stage(k)}
        if frame_stages:
            width = min(len(v) for v in frame_stages.values())
            total_arr = np.sum([v[:width] for v in frame_stages.values()], axis=0)
        else:
            total_arr = np.array([], dtype=float)
        total_mean = float(np.mean(total_arr)) if total_arr.size else 0.0

        period_ms = audio_frame_period_ms(
            chunk_size, meta["overlap_factor"], meta["sample_rate"]
        )

        rows = []
        base = {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_sha": sha,
            "backend": backend,
            "chunk_size": chunk_size,
            "num_freqs": num_freqs,
            "num_frames": num_frames,
        }

        for stage, arr in stage_arrays.items():
            mean, std, mn, mx = _stats(arr)
            is_frame = _is_frame_stage(stage)
            pct = (100.0 * mean / total_mean) if (is_frame and total_mean > 0) else ""
            rows.append({
                **base,
                "stage": stage,
                "mean_ms": round(mean, 4),
                "std_ms": round(std, 4),
                "min_ms": round(mn, 4),
                "max_ms": round(mx, 4),
                "pct_of_total": round(pct, 2) if pct != "" else "",
                "rt_margin": "",
            })

        # Synthetic TOTAL row carries the RT margin.
        t_mean, t_std, t_min, t_max = _stats(total_arr)
        rt_margin = (period_ms / t_mean) if t_mean > 0 else ""
        rows.append({
            **base,
            "stage": TOTAL_STAGE,
            "mean_ms": round(t_mean, 4),
            "std_ms": round(t_std, 4),
            "min_ms": round(t_min, 4),
            "max_ms": round(t_max, 4),
            "pct_of_total": 100.0 if t_mean > 0 else "",
            "rt_margin": round(rt_margin, 3) if rt_margin != "" else "",
        })

        self._append(rows)
        return run_id

    def _append(self, rows):
        """Append rows to the CSV, writing the header if the file is new."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        new_file = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)


# =============================================================================
# RENDERING (CSV -> Markdown)
# =============================================================================

def _read_rows(csv_path):
    """Read all rows from the results CSV as a list of dicts."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# A "detailed" run splits render into legs (sweep runs use blobs). The anatomy
# section/figure always use the latest detailed run so the leg breakdown survives
# a subsequent blob-only sweep.
_DETAIL_MARKER = "gl_swap"


def select_anatomy_run(rows):
    """run_id for the anatomy: latest run with render legs, else latest overall."""
    ids = []
    for r in rows:
        if r["run_id"] not in ids:
            ids.append(r["run_id"])
    for rid in reversed(ids):
        if any(r["stage"] == _DETAIL_MARKER for r in rows if r["run_id"] == rid):
            return rid
    return ids[-1] if ids else None


def _md_table(headers, rows):
    """Render a list of row-lists as a GitHub Markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# Pipeline order + friendly labels for the flow diagram (matches timing_report.py).
_PIPE_ORDER = ["audio_read",
               "fft_cpu", "upload", "multiply", "ifft", "download", "raw_cwt",
               "magnitude", "edge_trim", "hop_center", "downsample",
               "buf_push", "tex_upload", "gl_clear", "gl_draw", "gl_swap", "render"]
# Labels track the software flowchart (palette 3) box wording. `upload`/`download`
# are the CPU↔GPU lane-crossing legs the diagram draws as arrows, not boxes;
# `gl_clear` is folded into `gl_draw` ("Shader Draw") at the report layer so the
# renderer keeps issuing clear+draw as distinct GL calls while the figure shows
# one box. `raw_cwt`/`render` are the coarse-run aggregate blobs (never present
# alongside their split legs).
_PIPE_LABEL = {
    "audio_read": "Fetch Audio Samples",
    "fft_cpu": "FFT",
    "upload": "Transfer → GPU",
    "multiply": "Freq-Domain Multiply",
    "ifft": "IFFT",
    "download": "Transfer ← GPU",
    "raw_cwt": "Wavelet Transform",
    "magnitude": "Compute Magnitude",
    "edge_trim": "Discard Edges",
    "hop_center": "Extract New Hop",
    "downsample": "Down-sample",
    "buf_push": "Store Onto Frame Buffer",
    "tex_upload": "Upload To Texture",
    "gl_clear": "Clear Previous Display Buffer",
    "gl_draw": "Shader Draw",
    "gl_swap": "Update Display Buffer",
    "render": "Render Frame",
}
# Logical module each stage belongs to (for the grouped Pipeline Profile section).
_STAGE_MODULE = {
    "audio_read": "audio",
    "fft_cpu": "dsp", "upload": "dsp", "multiply": "dsp", "ifft": "dsp",
    "download": "dsp", "raw_cwt": "dsp", "magnitude": "dsp", "edge_trim": "dsp",
    "hop_center": "dsp", "downsample": "dsp",
    "buf_push": "render", "tex_upload": "render", "gl_clear": "render",
    "gl_draw": "render", "gl_swap": "render", "render": "render",
}


def append_method_rows(rows, csv_path=METHODS_CSV):
    """Append method-comparison rows to the methods CSV (header if new)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METHODS_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def latest_method_rows(csv_path=METHODS_CSV):
    """Rows from the most recent method benchmark (by timestamp), or []."""
    rows = _read_rows(csv_path)
    if not rows:
        return []
    latest_ts = rows[-1]["timestamp"]
    return [r for r in rows if r["timestamp"] == latest_ts]


def _fmt_ms(v):
    """Readable ms: thousands-separated whole numbers once we're past a second."""
    v = float(v)
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}"


def methods_table_md(csv_path=METHODS_CSV):
    """Latest method comparison as a Markdown table, or '' if none recorded."""
    rows = latest_method_rows(csv_path)
    if not rows:
        return ""
    body = [[r["method"], r["backend"], _fmt_ms(r["mean_ms"]),
             r["num_freqs"], r["native_res"]] for r in rows]
    return _md_table(
        ["Method", "Backend", "ms / frame", "Output bins", "Native resolution"],
        body,
    )


def _fmt_hz(hz):
    """Hz below 1000, kHz above — one decimal."""
    hz = float(hz)
    return f"{hz / 1000:.1f} kHz" if hz >= 1000 else f"{hz:g} Hz"


def methods_params(csv_path=METHODS_CSV):
    """Shared test parameters for the method comparison, or None if unavailable.

    Reads the latest method run (all rows share one config). Returns the test
    setup behind the figure: samples per frame, sampling rate, frequency range,
    resolution (frequencies per octave), and frequency count.
    """
    rows = latest_method_rows(csv_path)
    if not rows:
        return None
    r = rows[0]

    def _num(key):
        try:
            return float(r.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    lo, hi, sr = _num("freq_lo"), _num("freq_hi"), _num("sample_rate")
    nfreq, chunk = int(_num("num_freqs")), int(_num("chunk_size"))
    per_octave = int(round(nfreq / np.log2(hi / lo))) if hi > lo > 0 else None
    return {
        "samples": chunk, "sample_rate": sr, "freq_lo": lo, "freq_hi": hi,
        "num_freqs": nfreq, "per_octave": per_octave,
    }


def config_summary_rows(csv_path=RESULTS_CSV):
    """One summary row per unique (backend, chunk, freqs), newest value wins.

    Returns dicts with backend / chunk / freqs / mean_ms / fps / rt — the data
    behind the config chart and (historically) the config table.
    """
    rows = _read_rows(csv_path)
    run_ids = []
    for r in rows:
        if r["run_id"] not in run_ids:
            run_ids.append(r["run_id"])
    out, seen = [], set()
    for rid in reversed(run_ids):
        run = [r for r in rows if r["run_id"] == rid]
        tot = next((r for r in run if r["stage"] == TOTAL_STAGE), None)
        if tot is None:
            continue
        m = run[0]
        key = (m["backend"], m["chunk_size"], m["num_freqs"])
        if key in seen:
            continue
        seen.add(key)
        mean = float(tot["mean_ms"])
        rt = tot["rt_margin"]
        out.append({
            "backend": m["backend"], "chunk": m["chunk_size"], "freqs": m["num_freqs"],
            "mean_ms": mean, "fps": (1000.0 / mean if mean > 0 else 0.0),
            "rt": (float(rt) if rt not in ("", None) else 0.0),
        })
    return out


def _fmt_dur(ms):
    """Human duration: seconds once past 1 s, else whole milliseconds."""
    ms = float(ms)
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def render_markdown(csv_path=RESULTS_CSV, md_path=RESULTS_MD):
    """Render the timing report.

    Overview (headline numbers) → Pipeline Profiling (test params, then startup
    and runtime each as high-level + breakdown) → method comparison → settings
    sweep. Returns the path written, or ``None`` if there are no results yet.
    """
    rows = _read_rows(csv_path)
    if not rows:
        return None
    latest_id = select_anatomy_run(rows)
    latest = [r for r in rows if r["run_id"] == latest_id]
    total_row = next((r for r in latest if r["stage"] == TOTAL_STAGE), None)
    total = float(total_row["mean_ms"]) if total_row else 0.0
    rt = total_row["rt_margin"] if total_row else ""
    rt_val = float(rt) if rt not in ("", None) else 0.0
    deadline_ms = rt_val * total if rt_val else total
    fps = 1000.0 / total if total else 0.0
    init_total = sum(float(r["mean_ms"]) for r in latest if r["stage"].startswith("init:"))
    ok = rt_val >= 1
    params = methods_params(csv_path=METHODS_CSV)

    p = ["# SubShader Timing", ""]

    # --- Overview ------------------------------------------------------------
    p.append("## Overview")
    p.append("")
    p.append(f"{'✅' if ok else '⚠️'} **Real-Time Performance**")
    p.append("")
    if params and params["sample_rate"] and rt_val:
        speed = params["sample_rate"] * rt_val
        p.append(f"- **Processing Speed** — ~{speed:,.0f} samples per second")
    p.append(f"- **FPS** — ~{fps:.0f} frames per second")
    p.append(f"- **Deadline** — {rt_val:.0f}× under the {deadline_ms:.0f} ms deadline")
    p.append("")

    # --- Pipeline Profiling --------------------------------------------------
    p.append("## Pipeline Profiling")
    p.append("")
    if params:
        p.append("**Test parameters**")
        p.append("")
        param_rows = [("Number of Samples", f"{params['samples']:,}")]
        if params["sample_rate"]:
            param_rows.append(("Sampling Rate", f"{params['sample_rate'] / 1000:.1f} kHz"))
        if params["freq_lo"]:
            param_rows.append(("Lowest Frequency", f"{params['freq_lo']:g} Hz"))
        if params["freq_hi"]:
            param_rows.append(("Highest Frequency", _fmt_hz(params["freq_hi"])))
        if params["per_octave"]:
            param_rows.append(("Frequency Resolution",
                               f"{params['per_octave']} per octave"))
        if params["num_freqs"]:
            param_rows.append(("Number of Frequencies", f"{params['num_freqs']}"))
        p.append("| " + " | ".join(name for name, _ in param_rows) + " |")
        p.append("| " + " | ".join("---" for _ in param_rows) + " |")
        p.append("| " + " | ".join(value for _, value in param_rows) + " |")
        p.append("")

    p.append("### Pipeline Structure")
    p.append("")
    p.append("![Start up — one-time construction, CPU and GPU lanes]"
             "(pipeline%20start%20up%20init.drawio.png)")
    p.append("")
    p.append("![Runtime loop — the per-frame pipeline, CPU and GPU lanes]"
             "(pipeline%20runtime%20loop%20process.drawio.png)")
    p.append("")
    p.append("### Start Up vs Runtime — Measured Per-Stage Timing")
    p.append("")
    p.append("![Pipeline timing — startup construction and runtime loop, each to "
             "scale](timing_pipeline.png)")
    p.append("")

    # --- Method comparison ---------------------------------------------------
    if latest_method_rows():
        p.append("## Fourier vs Wavelet Implementations")
        p.append("")
        p.append("![Fourier vs Wavelet — time per frame (log scale) and frequency "
                 "resolution](timing_methods.png)")
        p.append("")

    # --- Settings sweep ------------------------------------------------------
    if config_summary_rows(csv_path):
        p.append("## Performance Across Settings")
        p.append("")
        p.append("![Compute per frame for each backend, chunk size, and resolution]"
                 "(timing_config.png)")
        p.append("")

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(p))
    return md_path
