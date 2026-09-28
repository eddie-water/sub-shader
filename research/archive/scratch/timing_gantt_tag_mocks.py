"""Mockups: letter-tagged timing gantt variants (real numbers, v52 palette)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, offset_copy

OUT = "/tmp/claude-1000/-home-eddie-water-dev-python-sub-shader/e860b56b-0c9f-46b5-843a-f80e97daa56e/scratchpad"

_BG, _FG, _SPINE, _TICK = "#1A1A1A", "#EEEEEE", "#444444", "#888888"
_AUDIO, _DSP, _RENDER = "#ffd27d", "#7b6fe1", "#ff5a1f"
COL = {"audio": _AUDIO, "dsp": _DSP, "render": _RENDER}

RUNTIME = [  # (tag, name, ms, module)
    ("A", "Fetch Audio Samples", 0.2, "audio"),
    ("B", "FFT", 1.2, "dsp"),
    ("C", "Transfer → GPU", 0.4, "dsp"),
    ("D", "Freq-Domain Multiply", 0.4, "dsp"),
    ("E", "IFFT", 2.5, "dsp"),
    ("F", "Transfer ← GPU", 1.6, "dsp"),
    ("G", "Compute Magnitude", 0.7, "dsp"),
    ("H", "Discard Edges", 0.05, "dsp"),
    ("I", "Extract New Hop", 0.05, "dsp"),
    ("J", "Down-sample", 0.1, "dsp"),
    ("K", "Store Onto Frame Buffer", 0.1, "render"),
    ("L", "Upload To Texture", 0.3, "render"),
    ("M", "Shader Draw", 2.6, "render"),
    ("N", "Update Display Buffer", 3.1, "render"),
]
STARTUP = [  # seconds
    ("1", "Open Audio File", 0.0017, "audio"),
    ("2", "Audio Output Init", 0.502, "audio"),
    ("3", "Build Wavelet Kernels", 0.097, "dsp"),
    ("4", "Generate FFT Kernel Bank", 0.153, "dsp"),
    ("5", "Transfer Kernel Bank → GPU", 0.143, "dsp"),
    ("6", "Allocate GPU Buffer", 0.0001, "render"),
    ("7", "Create Window + GL Context", 0.366, "render"),
    ("8", "Compile Shader + Texture", 0.011, "render"),
    ("9", "Color Map Init", 0.358, "render"),
]

FIG_W = 13.0
ROW = 0.46

# stage codes: roman numerals - order only, module comes from the color
CODES = {"A": "I", "B": "II", "C": "III", "D": "IV", "E": "V", "F": "VI",
         "G": "VII", "H": "VIII", "I": "IX", "J": "X", "K": "XI", "L": "XII",
         "M": "XIII", "N": "XIV",
         "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI",
         "7": "VII", "8": "VIII", "9": "IX"}


BADGE_SIDE = 25.0   # pt - exact square side, sized so XIII fits unshrunk
BADGE_FONT = 10.5   # one size for every tag: T, I ... XIV
BADGE_LW = 2.5      # same stroke weight as the flowchart boxes
_BADGE_FP = FontProperties(family="sans-serif", weight="bold")  # DejaVu Sans
                    # Bold - the same face as every other label in the gantt


# bars match the badge's outer height exactly (side + stroke), in row units
BAR_H = (BADGE_SIDE + BADGE_LW) / 72.0 / ROW


def badge(ax, xy, tag, color, dx=0.0, dy=0.0, shape="square", font=None):
    """The one shared identifier style: an exact BADGE_SIDE x BADGE_SIDE
    square (Rectangle, not a text bbox), colored outline, black fill, and
    the glyph placed as a TextPath centered on its own geometric bounds -
    not on font ascent/descent metrics, which sit capital letters high."""
    s = BADGE_SIDE
    da = DrawingArea(s, s, 0, 0)
    if shape == "round":
        box = FancyBboxPatch((0, 0), s, s, fc="black", ec=color, lw=BADGE_LW,
                             boxstyle="round,pad=0,rounding_size=4.5")
    else:
        box = Rectangle((0, 0), s, s, fc="black", ec=color,
                        lw=BADGE_LW, joinstyle="miter")
    box.set_snap(True)
    da.add_artist(box)
    glyph = TextPath((0, 0), tag, size=font or BADGE_FONT, prop=_BADGE_FP)
    gb = glyph.get_extents()
    # wide tags (VIII, XIII) shrink uniformly to fit - the square never grows
    inner = s - BADGE_LW - 2.5
    scale = min(1.0, inner / gb.width) if gb.width else 1.0
    center = (Affine2D()
              .translate(-(gb.x0 + gb.x1) / 2, -(gb.y0 + gb.y1) / 2)
              .scale(scale)
              .translate(s / 2, s / 2))
    da.add_artist(PathPatch(glyph.transformed(center), fc="white", ec="none"))
    ab = AnnotationBbox(da, xy, xybox=(dx, dy), xycoords="data",
                        boxcoords="offset points", box_alignment=(0.5, 0.5),
                        frameon=False, pad=0, annotation_clip=False)
    ab.set_zorder(5)
    ax.add_artist(ab)


def fmt(v, unit):
    if unit == "ms":
        return "< 0.1 ms" if v < 0.1 else f"{v:g} ms"
    return f"{v*1000:.1f} ms" if v < 0.05 else f"{v:g} s"


def waterfall(ax, rows, unit, tick_mode="letter"):
    """tick_mode: 'letter' (tag ticks), 'badge' (no ticks, chip at bar), 'full'"""
    total = sum(v for _, _, v, _ in rows)
    n = len(rows) + 1
    start = 0.0
    for i, (tag, name, v, mod) in enumerate(rows):
        ax.barh(i, v, left=start, height=BAR_H, color=COL[mod])
        ax.text(start + v + total * 0.015, i, fmt(v, unit), va="center",
                color=_FG, fontsize=12, fontweight="bold")
        if tick_mode == "badge":
            badge(ax, (0, i), CODES.get(tag, tag), COL[mod], dx=-20)
        start += v
    # total bar sits at the bottom
    tot_row = len(rows)
    ax.barh(tot_row, total, left=0, height=BAR_H, color="white")
    ax.text(total + total * 0.015, tot_row, fmt(total, unit), va="center",
            color=_FG, fontsize=12, fontweight="bold")
    if tick_mode == "badge":
        badge(ax, (0, tot_row), "T", "white", dx=-20)
        # concept 9: chain the chips with arrows - the axis IS the pipeline
        tr = offset_copy(ax.transData, fig=ax.figure, x=-20, y=0,
                         units="points")
        for i in range(len(rows) - 1):
            ax.annotate("", xy=(0, i + 1), xytext=(0, i),
                        xycoords=tr, textcoords=tr,
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=1.2, shrinkA=15, shrinkB=15,
                                        mutation_scale=8))
    ticks = list(range(n))
    ax.set_yticks(ticks)
    if tick_mode == "letter":
        ax.set_yticklabels([t for t, _, _, _ in rows] + ["Total"],
                           fontsize=13, fontweight="bold", family="monospace")
    elif tick_mode == "badge":
        ax.set_yticklabels([""] * n)
    else:
        ax.set_yticklabels([f"{n}" for _, n, _, _ in rows] + ["Total"], fontsize=11)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(colors=_TICK, labelcolor=_FG, length=0)
    ax.margins(x=0.14)
    ax.grid(axis="x", alpha=0.18, color=_SPINE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_SPINE)
    ax.set_facecolor(_BG)
    ax.set_xlabel(unit, color=_FG, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)


def block_title(fig, y, text):
    fig.text(0.055, y, text, color=_FG, fontsize=14, fontweight="bold", va="bottom")


# ---- Mock A: both blocks, tag ticks (startup 1-9, runtime A-N), tiny margin ----
def mock_a():
    left_in, right_in = 0.75, 0.5
    su_h, rt_h = ROW * (len(STARTUP) + 1), ROW * (len(RUNTIME) + 1)
    fig_h = 0.55 + su_h + 0.85 + 0.55 + rt_h + 0.9
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=_BG)
    aw = 1 - (left_in + right_in) / FIG_W
    ax_x = left_in / FIG_W
    y = 1 - 0.55 / fig_h
    ax1 = fig.add_axes([ax_x, y - su_h / fig_h, aw, su_h / fig_h])
    block_title(fig, y + 0.06 / fig_h, "Start Up - Pipeline Construction   (1–9 → Init flowchart)")
    waterfall(ax1, STARTUP, "s", "letter")
    y2 = y - (su_h + 0.85 + 0.1) / fig_h
    ax2 = fig.add_axes([ax_x, y2 - rt_h / fig_h, aw, rt_h / fig_h])
    block_title(fig, y2 + 0.06 / fig_h, "Runtime Loop   (A–N → Runtime flowchart)")
    waterfall(ax2, RUNTIME, "ms", "letter")
    fig.savefig(f"{OUT}/mock_a_tag_ticks.png", dpi=100, facecolor=_BG)
    plt.close(fig)


# ---- Mock B: no title, chips only (T + codes), plot maximised ----
def mock_b():
    rt_h = ROW * (len(RUNTIME) + 1)
    fig_h = 0.18 + rt_h + 0.8
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=_BG)
    left_in = 0.58
    aw = 1 - (left_in + 0.45) / FIG_W
    y = 1 - 0.18 / fig_h
    ax = fig.add_axes([left_in / FIG_W, y - rt_h / fig_h, aw, rt_h / fig_h])
    waterfall(ax, RUNTIME, "ms", "badge")
    fig.savefig(f"{OUT}/mock_b_badge_chips.png", dpi=200, facecolor=_BG)
    plt.close(fig)


# ---- Mock C: runtime only, tag ticks + compact key strip under the chart ----
def mock_c():
    rt_h = ROW * (len(RUNTIME) + 1)
    key_h = 0.85
    fig_h = 0.55 + rt_h + 0.85 + key_h
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=_BG)
    left_in = 0.75
    aw = 1 - (left_in + 0.5) / FIG_W
    y = 1 - 0.55 / fig_h
    ax = fig.add_axes([left_in / FIG_W, y - rt_h / fig_h, aw, rt_h / fig_h])
    block_title(fig, y + 0.06 / fig_h, "Runtime Loop")
    waterfall(ax, RUNTIME, "ms", "letter")
    half = (len(RUNTIME) + 1) // 2
    lines = []
    for chunk in (RUNTIME[:half], RUNTIME[half:]):
        lines.append("    ".join(f"$\\bf{{{t}}}$ {n}" for t, n, _, _ in chunk))
    fig.text(0.5, key_h * 0.55 / fig_h, "\n".join(lines), ha="center", va="center",
             color=_TICK, fontsize=10.5, linespacing=1.8)
    fig.savefig(f"{OUT}/mock_c_ticks_plus_key.png", dpi=100, facecolor=_BG)
    plt.close(fig)


# ---- Mock D: exploration sheet - ways to link a flowchart box to its tag ----
def _flow_box(ax, x, by, name, color, bw=1.5, bh=1.0, pad=0.08):
    ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                                boxstyle=f"round,pad={pad},rounding_size=0.18",
                                fc="black", ec=color, lw=2.5, zorder=1))
    ax.text(x + bw / 2, by + bh / 2, name, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", zorder=2)


def mock_d():
    fig, ax = plt.subplots(figsize=(14, 6.2), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    bw, bh, pad = 1.5, 1.0, 0.08
    tiles = [
        ("1  outside corner, gap",       "out_gap"),
        ("2  inside corner",             "in_corner"),
        ("3  above, left-aligned",       "above"),
        ("4  tab fused into top border", "tab_top"),
        ("5  fused on the corner",       "on_corner"),
        ("6  fused on left edge",        "tab_left"),
        ("7  naked letter, no box",      "naked"),
        ("8  rounded chip (box-like)",   "round_chip"),
    ]
    for i, (cap, kind) in enumerate(tiles):
        col, row = i % 4, i // 4
        x = 0.75 + col * 3.5
        by = 4.0 - row * 3.0
        _flow_box(ax, x, by, "FFT", _DSP, bw, bh, pad)
        ax.text(x + bw / 2, by - 0.45, cap, ha="center", va="center",
                color="#888888", fontsize=10)
        left, top = x - pad, by + bh + pad          # visible outer edges
        if kind == "out_gap":
            badge(ax, (left, top), "B", _DSP, dx=-13, dy=13)
        elif kind == "in_corner":
            badge(ax, (left, top), "B", _DSP, dx=15, dy=-15)
        elif kind == "above":
            badge(ax, (left, top), "B", _DSP, dx=11, dy=15)
        elif kind == "tab_top":
            # half-in / half-out, sitting on the top border like a tab
            badge(ax, (left + 0.32, top), "B", _DSP)
        elif kind == "on_corner":
            # centered on the corner vertex; fill hides the stroke beneath
            badge(ax, (left + 0.04, top - 0.04), "B", _DSP)
        elif kind == "tab_left":
            # straddling the left border, vertically centered
            badge(ax, (left, by + bh / 2), "B", _DSP)
        elif kind == "naked":
            ax.text(x + 0.12, by + bh - 0.12, "B", ha="left", va="top",
                    color=_DSP, fontsize=13, fontweight="bold", zorder=3)
        elif kind == "round_chip":
            badge(ax, (left, top), "B", _DSP, dx=-13, dy=13, shape="round")
    fig.savefig(f"{OUT}/mock_d_flowchart_badge.png", dpi=200, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)


# ---- Calibration: isolated badges for pixel-measuring square/centering ----
def mock_cal():
    fig, ax = plt.subplots(figsize=(6, 1.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    for i, tag in enumerate(["A", "I", "M", "N", "J", "1"]):
        badge(ax, (0.5 + i, 0.6), tag, _DSP)
    fig.savefig(f"{OUT}/mock_cal_badges.png", dpi=200, facecolor=_BG)
    plt.close(fig)


# ---- Mock E: outside-the-box linking concepts (numbering continues d6) ----
def mock_e():
    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def cap(x, y, t):
        ax.text(x, y, t, ha="center", va="center", color="#888888",
                fontsize=10, linespacing=1.5)

    # 9 - gantt chips chained with arrows: the y-axis becomes the pipeline
    chips = [("A", _AUDIO, 0.2), ("B", _DSP, 1.2),
             ("C", _DSP, 0.4), ("D", _DSP, 0.4)]
    for i, (t, c, ms) in enumerate(chips):
        y = 6.9 - i * 0.62
        badge(ax, (1.0, y), t, c)
        ax.add_patch(Rectangle((1.4, y - 0.13), 0.25 + ms * 1.05, 0.26,
                               fc=c, ec="none"))
        if i:
            ax.annotate("", xy=(1.0, y + 0.17), xytext=(1.0, y + 0.46),
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=1.4))
    cap(1.9, 4.5, "9  chips chained with arrows -\nthe gantt axis IS the pipeline")

    # 10 - merged figure: funnels tie each box to its slice of frame time
    stages = [("FFT", _DSP, 1.2), ("IFFT", _DSP, 2.5),
              ("Magnitude", _DSP, 0.7), ("Draw", _RENDER, 2.6)]
    total = sum(s[2] for s in stages)
    bx0, bw2, gap2 = 4.3, 0.95, 0.28
    span = len(stages) * bw2 + (len(stages) - 1) * gap2
    segx = bx0
    for i, (nm, c, ms) in enumerate(stages):
        x = bx0 + i * (bw2 + gap2)
        ax.add_patch(FancyBboxPatch((x, 5.7), bw2, 0.62,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc="black", ec=c, lw=2.0, zorder=2))
        ax.text(x + bw2 / 2, 6.01, nm, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold", zorder=3)
        sw = ms / total * span
        ax.add_patch(Rectangle((segx, 4.72), sw, 0.2, fc=c, ec="none"))
        ax.add_patch(Polygon([(x - 0.05, 5.65), (x + bw2 + 0.05, 5.65),
                              (segx + sw, 4.92), (segx, 4.92)],
                             closed=True, fc=c, ec="none", alpha=0.22))
        segx += sw
    cap(6.6, 4.25, "10  merged figure - translucent funnels tie each box "
                   "to its slice of frame time")

    # 11 - ghost letter watermark behind the label
    _flow_box(ax, 10.9, 5.4, "FFT", _DSP)
    ax.text(11.65, 5.93, "B", ha="center", va="center", color=_DSP,
            alpha=0.30, fontsize=46, fontweight="bold", zorder=1.5)
    cap(11.65, 4.6, "11  ghost letter watermark\nbehind the label")

    # 12 - superscript letter fused to the label itself
    _flow_box(ax, 1.0, 1.7, "FFT", _DSP)
    ax.text(2.08, 2.36, "B", ha="left", va="center", color=_DSP,
            fontsize=10.5, fontweight="bold", zorder=3)
    cap(1.75, 0.95, "12  superscript letter fused\nto the label itself")

    # 13 - module-order code on the fused corner (their pick, style 5)
    _flow_box(ax, 5.2, 1.7, "FFT", _DSP)
    badge(ax, (5.16, 2.74), "D1", _DSP, font=9.5)
    badge(ax, (5.5, 1.05), "D1", _DSP, font=9.5)
    ax.add_patch(Rectangle((5.9, 0.92), 1.4, 0.26, fc=_DSP, ec="none"))
    cap(6.4, 0.45, "13  module-order code: D1 = 1st DSP stage "
                   "(A1, D1–D9, R1–R5)\nsame code on box corner and gantt row")

    # 14 - time-bar under each box: flowchart doubles as the timing figure
    for x, nm, ms in ((9.9, "FFT", 1.2), (11.75, "IFFT", 2.5)):
        _flow_box(ax, x, 1.9, nm, _DSP, bw=1.35, bh=0.85)
        ax.add_patch(Rectangle((x - 0.08, 1.5), (ms / 2.5) * (1.35 + 0.16),
                               0.14, fc=_DSP, ec="none"))
    cap(11.5, 0.85, "14  time-bar under each box -\nthe flowchart doubles "
                    "as the timing figure")

    fig.savefig(f"{OUT}/mock_e_link_concepts.png", dpi=200, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)


# ---- Mock F: naked corner code - position sweep on worst-case labels ----
def mock_f():
    fig, ax = plt.subplots(figsize=(14, 6.6), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    demo = [("Freq-\nDomain\nMultiply", "IV", _DSP),
            ("Sync\nDisplay", "XIV", _RENDER)]
    bw = bh = 1.15
    pad = 0.08
    spots = [("top-left", "tl"), ("top-center", "tc"), ("top-right", "tr"),
             ("bottom-left", "bl"), ("bottom-center", "bc"),
             ("bottom-right", "br")]
    for i, (label, k) in enumerate(spots):
        col, row = i % 3, i // 3
        tx = 0.7 + col * 4.7
        ty = 4.35 - row * 2.95
        for j, (name, code, c) in enumerate(demo):
            x = tx + j * 1.75
            _flow_box(ax, x, ty, name, c, bw, bh, pad)
            in_l, in_r = x - pad + 0.1, x + bw + pad - 0.1
            in_b, in_t = ty - pad + 0.09, ty + bh + pad - 0.09
            cx = x + bw / 2
            px, py, ha, va = {
                "tl": (in_l, in_t, "left", "top"),
                "tc": (cx, in_t, "center", "top"),
                "tr": (in_r, in_t, "right", "top"),
                "bl": (in_l, in_b, "left", "bottom"),
                "bc": (cx, in_b, "center", "bottom"),
                "br": (in_r, in_b, "right", "bottom")}[k]
            ax.text(px, py, code, ha=ha, va=va, color=c, fontsize=9.5,
                    fontweight="bold", zorder=3)
        ax.text(tx + 1.45, ty - 0.45, label, ha="center", color="#888888",
                fontsize=10)
    fig.savefig(f"{OUT}/mock_f_large_text_proof.png", dpi=200,
                facecolor="black", bbox_inches="tight")
    plt.close(fig)


mock_a()
mock_b()
mock_c()
mock_d()
mock_cal()
mock_e()
mock_f()
print("done")
