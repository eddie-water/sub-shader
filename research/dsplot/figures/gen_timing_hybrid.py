"""Hybrid design+timing figures for the README Design section.

Three figures share one visual grammar (rounded square boxes, black fill,
module-color strokes, white wires):

  - runtime hybrid: 5-stage flowchart (CPU/GPU columns) + per-stage gantt
  - startup hybrid: same grid, one-shot wire, 5 consolidated init buckets
  - deadline strip: one frame of work vs the 186 ms deadline window

Stage means come from the newest run in assets/timing/timing_results.csv,
bucketed to the consolidated tier, so the figures regenerate truthfully
after each performance fix. Geometry locked via mocks v10/v11 (runtime),
v4 (startup), s3b (deadline) in .planning/mocks/.

Run from repo root:
    python -c "from research.dsplot.figures.gen_timing_hybrid import render_all; render_all()"
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from .. import style
from ..export import crop_to_content

AUDIO, DSP, REND = style.AUDIO_COLOR, style.DSP_COLOR, style.RENDER_COLOR
WHITE, BG = style.NEUTRAL_COLOR, style.BG_COLOR
CHUNK_SAMPLES, SAMPLE_RATE_K = 8192, 44.1
LW, RADIUS = style.README_STROKE_PT, style.README_BOX_RADIUS

REPO = Path(__file__).resolve().parents[3]
RESULTS_CSV = REPO / "assets/timing/timing_results.csv"
OUT_DIR = REPO / "assets/timing"

DEADLINE_MS = CHUNK_SAMPLES / SAMPLE_RATE_K  # 185.76 -> shown as 186

RUNTIME_BUCKETS = [
    ("Fetch\nAudio",   AUDIO, "CPU", ["audio_read"]),
    ("CWT",            DSP,   "GPU", ["fft_cpu", "upload", "multiply",
                                      "ifft", "download"]),
    ("Post-\nprocess", DSP,   "GPU", ["magnitude", "edge_trim",
                                      "hop_center", "downsample"]),
    ("Draw",           REND,  "GPU", ["buf_push", "tex_upload",
                                      "gl_clear", "gl_draw"]),
    ("Sync\nDisplay",  REND,  "CPU", ["gl_swap"]),
]

STARTUP_BUCKETS = [
    ("Audio",         AUDIO, "CPU", 13, ["init:audio"]),
    ("CUDA",          DSP,   "CPU", 13, ["init:cuda"]),
    ("Wavelet\nBank", DSP,   "CPU", 10, ["init:dsp"]),
    ("OpenGL",        REND,  "CPU", 10, ["init:renderer"]),
    ("Color\nMap",    REND,  "GPU", 12, ["init:prescan", "init:prime"]),
]
STARTUP_EXTRA = ["init:other"]               # counted in the total bar only


def load_latest_run(csv_path=RESULTS_CSV):
    """Stage -> mean_ms for the newest run_id in the campaign CSV."""
    rows = list(csv.DictReader(open(csv_path)))
    latest = max(rows, key=lambda r: r["timestamp"])["run_id"]
    return ({r["stage"]: float(r["mean_ms"]) for r in rows
             if r["run_id"] == latest}, latest)


def _new_axes(figsize):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(style.FONT_STACK)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, out_path):
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return crop_to_content(str(out_path))


class _HybridGrid:
    """Shared strict-grid geometry: GAP (half a box) drives every distance."""

    BH = W = 16.0
    GAP = 8.0
    PITCH = BH + 2 * GAP
    RAIL_X = 8.0
    CHART_SPAN = 280.0

    def __init__(self, ax, n_rows):
        self.ax = ax
        self.CPU_X = self.RAIL_X + 2 * self.GAP + self.W / 2
        self.GPU_X = self.CPU_X + self.PITCH
        self.RULE_X = (self.CPU_X + self.GPU_X) / 2
        self.CHART_X0 = self.GPU_X + self.W / 2 + 2 * self.GAP
        self.CHART_X1 = self.CHART_X0 + self.CHART_SPAN
        self.ROW_Y = [n_rows * self.PITCH - k * self.PITCH
                      for k in range(n_rows)]
        self.LANE_Y = self.ROW_Y[-1] - self.PITCH
        self.TOP_Y = self.ROW_Y[0] + self.BH / 2 + self.GAP
        self.BOT_Y = self.LANE_Y - self.BH / 2

    def col(self, lane):
        return self.CPU_X if lane == "CPU" else self.GPU_X

    def rbox(self, cx, cy, w, h, edge, lw=LW):
        self.ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad=0,rounding_size={RADIUS * _HybridGrid.BH}",
            facecolor=BG, edgecolor=edge, linewidth=lw, zorder=3))

    def label(self, cx, cy, text, size=13):
        self.ax.text(cx, cy, text, color=WHITE, fontsize=size,
                     fontweight="bold", ha="center", va="center",
                     zorder=4, linespacing=1.0)

    def wire(self, pts, lw=2.4):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=WHITE, linewidth=lw,
                     solid_capstyle="projecting", zorder=2)
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                         arrowprops=dict(arrowstyle="-|>", color=WHITE,
                                         lw=lw, mutation_scale=18,
                                         shrinkA=0, shrinkB=0), zorder=3)

    def double_rule(self):
        for dx in (-0.8, 0.8):
            self.ax.plot([self.RULE_X + dx] * 2,
                         [self.LANE_Y + self.BH / 2 + self.GAP, self.TOP_Y],
                         color=WHITE, linewidth=1.4, zorder=1)

    def lanes(self):
        for x, name in [(self.CPU_X, "CPU"), (self.GPU_X, "GPU")]:
            self.rbox(x, self.LANE_Y, self.W, self.BH, WHITE)
            self.label(x, self.LANE_Y, name, 15)

    def gantt(self, entries, total_ms, total_label, fmt, ticks):
        scale = self.CHART_SPAN / total_ms
        start = 0.0
        for k, (color, dur) in enumerate(entries):
            x0 = self.CHART_X0 + start * scale
            self.ax.add_patch(FancyBboxPatch(
                (x0, self.ROW_Y[k] - self.BH / 2), dur * scale, self.BH,
                boxstyle=f"round,pad=0,rounding_size={RADIUS * _HybridGrid.BH}",
                facecolor=BG, edgecolor=color, linewidth=LW, zorder=3))
            self.ax.text(x0 + dur * scale + self.GAP / 2, self.ROW_Y[k],
                         fmt(dur), color=WHITE, fontsize=16,
                         fontweight="bold", ha="left", va="center")
            start += dur
        self.ax.add_patch(FancyBboxPatch(
            (self.CHART_X0, self.LANE_Y - self.BH / 2), total_ms * scale,
            self.BH, boxstyle=f"round,pad=0,rounding_size={RADIUS * _HybridGrid.BH}",
            facecolor=BG, edgecolor=WHITE, linewidth=LW, zorder=3))
        self.ax.text(self.CHART_X1 + self.GAP / 2, self.LANE_Y, total_label,
                     color=WHITE, fontsize=16, fontweight="bold",
                     ha="left", va="center")
        self.ax.plot([self.CHART_X0, self.CHART_X1],
                     [self.BOT_Y - 2] * 2, color=WHITE, linewidth=2.6)
        self.ax.plot([self.CHART_X0] * 2, [self.BOT_Y - 2, self.TOP_Y],
                     color=WHITE, linewidth=2.6)
        for t in ticks:
            x = self.CHART_X0 + t * scale
            self.ax.plot([x, x], [self.BOT_Y - 2, self.BOT_Y - 5],
                         color=WHITE, linewidth=2.6)
            self.ax.text(x, self.BOT_Y - 7, f"{t:g}", color=WHITE,
                         fontsize=16, fontweight="bold", ha="center", va="top")


def build_runtime_hybrid(stages, out_path):
    """stages: list of (name, color, lane, dur_ms, label_size)."""
    fig, ax = _new_axes((26.0, 11.0))
    g = _HybridGrid(ax, len(stages))
    ax.set_xlim(0, g.CHART_X1 + 30)
    ax.set_ylim(g.BOT_Y - 14, g.TOP_Y + 6)
    g.double_rule()
    first_x, last_x = g.col(stages[0][2]), g.col(stages[-1][2])
    g.wire([(last_x, g.ROW_Y[-1] - g.BH / 2),
            (last_x, g.ROW_Y[-1] - g.BH / 2 - g.GAP),
            (g.RAIL_X, g.ROW_Y[-1] - g.BH / 2 - g.GAP),
            (g.RAIL_X, g.TOP_Y), (first_x, g.TOP_Y),
            (first_x, g.ROW_Y[0] + g.BH / 2)])
    for k, (name, color, lane, _, size) in enumerate(stages):
        cx = g.col(lane)
        g.rbox(cx, g.ROW_Y[k], g.W, g.BH, color)
        g.label(cx, g.ROW_Y[k], name, size)
        if k < len(stages) - 1:
            nx = g.col(stages[k + 1][2])
            g.wire([(cx, g.ROW_Y[k] - g.BH / 2),
                    (cx, g.ROW_Y[k] - g.BH / 2 - g.GAP),
                    (nx, g.ROW_Y[k] - g.BH / 2 - g.GAP),
                    (nx, g.ROW_Y[k + 1] + g.BH / 2)])
    g.lanes()
    total = sum(s[3] for s in stages)
    g.gantt([(s[1], s[3]) for s in stages], total, f"{total:.1f} ms",
            lambda d: f"{d:.1f} ms", range(0, int(total), 2))
    return _save(fig, out_path)


def build_startup_hybrid(stages, total_ms, out_path):
    fig, ax = _new_axes((26.0, 11.0))
    g = _HybridGrid(ax, len(stages))
    ax.set_xlim(0, g.CHART_X1 + 46)
    ax.set_ylim(g.BOT_Y - 14, g.TOP_Y + 6)
    g.double_rule()
    first_x, last_x = g.col(stages[0][2]), g.col(stages[-1][2])
    g.wire([(first_x, g.TOP_Y), (first_x, g.ROW_Y[0] + g.BH / 2)])
    for k, (name, color, lane, _, size) in enumerate(stages):
        cx = g.col(lane)
        g.rbox(cx, g.ROW_Y[k], g.W, g.BH, color)
        g.label(cx, g.ROW_Y[k], name, size)
        if k < len(stages) - 1:
            nx = g.col(stages[k + 1][2])
            g.wire([(cx, g.ROW_Y[k] - g.BH / 2),
                    (cx, g.ROW_Y[k] - g.BH / 2 - g.GAP),
                    (nx, g.ROW_Y[k] - g.BH / 2 - g.GAP),
                    (nx, g.ROW_Y[k + 1] + g.BH / 2)])
    g.wire([(last_x, g.ROW_Y[-1] - g.BH / 2),
            (last_x, g.ROW_Y[-1] - g.BH / 2 - g.GAP),
            (g.RAIL_X, g.ROW_Y[-1] - g.BH / 2 - g.GAP),
            (g.RAIL_X, g.BOT_Y - 2)])
    g.lanes()
    scale_total = round(total_ms)
    g.gantt([(s[1], s[3]) for s in stages], scale_total,
            f"{scale_total:.0f} ms", lambda d: f"{d:.0f} ms",
            range(0, int(scale_total), 200))
    return _save(fig, out_path)


def build_deadline_strip(work_ms, out_path, dead_ms=186.0):
    """One solid frame of work + a continuous dashed lane to the deadline."""
    MS = 0.5
    BH = work_ms * MS                            # square work box
    RS = RADIUS * BH
    fig, ax = _new_axes((27.0, 2.4))

    def rbox(x, w, ls="solid", lw=LW):
        ax.add_patch(FancyBboxPatch(
            (x, 0), w, BH, boxstyle=f"round,pad=0,rounding_size={RS}",
            facecolor=BG, edgecolor=WHITE, linewidth=lw, linestyle=ls,
            zorder=3))

    rbox(0, work_ms * MS)
    rbox(work_ms * MS, (dead_ms - work_ms) * MS, ls=(0, (4, 4)), lw=2.0)
    n = int(dead_ms // work_ms)
    ax.text((work_ms + dead_ms) / 2 * MS, BH / 2, f"{n}× within deadline",
            color=WHITE, fontsize=20, fontweight="bold", ha="center",
            va="center", zorder=5)
    AXY = -1.6
    ax.plot([0, dead_ms * MS], [AXY, AXY], color=WHITE, lw=2.4, zorder=2)
    marks = [(0, "0", "center"), (work_ms, f"{work_ms:.1f} ms", "left"),
             (50, "50", "center"), (100, "100", "center"),
             (150, "150", "center"), (dead_ms, f"{dead_ms:.0f} ms", "center")]
    for x, lab, ha in marks:
        ax.plot([x * MS] * 2, [AXY, AXY - 0.6], color=WHITE, lw=2.4)
        tx = x * MS + (0.8 if ha == "left" else 0)
        ax.text(tx, AXY - 1.2, lab, color=WHITE, fontsize=17,
                fontweight="bold", ha=ha, va="top")
    ax.set_xlim(-1.5, dead_ms * MS + 1.5)
    ax.set_ylim(-4.8, BH + 1.2)
    return _save(fig, out_path)


def build_rate_check(work_ms, out_path, play_k=SAMPLE_RATE_K):
    """Playback rate vs pipeline throughput, both bars to scale (K samples/s)."""
    pipe_k = CHUNK_SAMPLES / work_ms
    factor = int(pipe_k / play_k)
    MS = 0.1
    BH = play_k * MS                             # square playback box
    RS, GAP = RADIUS * BH, 0.7 * BH
    fig, ax = _new_axes((27.0, 3.4))

    def rbox(x, y, w, lw=LW):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, BH, boxstyle=f"round,pad=0,rounding_size={RS}",
            facecolor=BG, edgecolor=WHITE, linewidth=lw, zorder=3))

    y_play = BH + GAP
    rbox(0, y_play, play_k * MS)
    ax.text(play_k * MS + 0.35 * BH, y_play + BH / 2,
            f"{play_k:.1f}K s/s - playback", color=WHITE, fontsize=20,
            fontweight="bold", ha="left", va="center", zorder=5)
    rbox(0, 0, pipe_k * MS)
    ax.text(pipe_k * MS / 2, BH / 2, f"{factor}× faster than playback",
            color=WHITE, fontsize=20, fontweight="bold", ha="center",
            va="center", zorder=5)
    AXY = -0.35 * BH
    ax.plot([0, pipe_k * MS], [AXY, AXY], color=WHITE, lw=2.4, zorder=2)
    marks = [(0, "0", "center"), (500, "500K", "center"),
             (1000, "1M", "center"), (pipe_k, f"{pipe_k / 1000:.1f}M s/s", "right")]
    for x, lab, ha in marks:
        ax.plot([x * MS] * 2, [AXY, AXY - 0.13 * BH], color=WHITE, lw=2.4)
        ax.text(x * MS, AXY - 0.27 * BH, lab, color=WHITE, fontsize=17,
                fontweight="bold", ha=ha, va="top")
    ax.set_xlim(-0.35, pipe_k * MS + 0.35)
    ax.set_ylim(AXY - 0.75 * BH, y_play + BH + 0.3)
    return _save(fig, out_path)


def render_all(runtime_name="timing_runtime_hybrid_v12.png",
               startup_name="timing_startup_hybrid_v5.png",
               deadline_name="timing_runtime_deadline_v7.png",
               rate_name="timing_rate_check_v4.png"):
    means, run_id = load_latest_run()
    runtime = [(name, color, lane, sum(means[k] for k in keys), 13)
               for name, color, lane, keys in RUNTIME_BUCKETS]
    startup = [(name, color, lane, sum(means[k] for k in keys), size)
               for name, color, lane, size, keys in STARTUP_BUCKETS]
    startup_total = (sum(s[3] for s in startup)
                     + sum(means[k] for k in STARTUP_EXTRA))
    outs = [
        build_runtime_hybrid(runtime, OUT_DIR / runtime_name),
        build_startup_hybrid(startup, startup_total, OUT_DIR / startup_name),
        build_deadline_strip(round(means["TOTAL"], 1),
                             OUT_DIR / deadline_name),
        build_rate_check(round(means["TOTAL"], 2), OUT_DIR / rate_name),
    ]
    print(f"run: {run_id}")
    for o in outs:
        print(o)
    return outs


if __name__ == "__main__":
    render_all()
