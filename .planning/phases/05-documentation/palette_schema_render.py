"""
Comprehensive color schema PNG for the SubShader plot lib.

Shows every color we've considered, organized two ways for cross-referencing:

  1. NEUTRALS strip — all structural colors (bg, axes, text, spines)
  2. By-tier strips — all 6 families' SPOTLIGHT swatches in one row,
     same for BASE, same for MUTED. Lets you compare across families
     within a tier.
  3. By-family matrix — 6 rows × 3 cols, showing each family across
     its 3 tiers. Lets you compare across tiers within a family.
  4. Spotlight rotation demo — same vector figure 3x with the spotlight
     rotating, plus default and anti-pattern panels.

Nothing locked. This is the "see everything before deciding" view.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys

OUT = "/tmp/palette_schema_v4_no_proj_label.png"

BG = "#1A1A1A"
TEXT_LIGHT = "#dddddd"
TEXT_DIM = "#aaaaaa"


def hsl_to_hex(h, s, l):
    h = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, l / 100.0, s / 100.0)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def hex_to_hsl(hexc):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (round(h * 360), round(s * 100), round(l * 100))


# =============================================================
# COLOR DEFINITIONS
# =============================================================

# ----- NEUTRALS (locked, structural) -----
NEUTRALS = [
    ("#1A1A1A", "BG_COLOR",                "background"),
    ("#444444", "SPINE_COLOR",             "axis spines"),
    ("#606060", "WAVEFORM_COLOR",          "neutral waveform"),
    ("#888888", "TICK_LABEL_COLOR",        "tick labels"),
    ("#AAAAAA", "FREQ_LINE_COLOR",         "freq reference lines"),
    ("#cccccc", "VECTOR_AXIS_COLOR",       "vector-fig axes"),
    ("#dddddd", "VECTOR_PANEL_RESULT_COLOR","light text on dark"),
    ("#FFFFFF", "axis grid (α 0.08)",      "barely-visible grid"),
]

# ----- CHROMATIC FAMILIES — 3 tiers each -----
# Locked primaries
PRIMARY_SPOTLIGHT  = "#F5A83D"
PRIMARY_BASE       = "#E79E55"
PRIMARY_MUTED      = "#E0A26C"

CONTRAST_SPOTLIGHT = "#8878DD"
CONTRAST_BASE      = "#9189D2"
CONTRAST_MUTED     = "#9C98CD"

# Accent candidates (all 4 — user picks one later)
SAGE_SPOTLIGHT    = "#52E081"
SAGE_BASE         = "#79D297"
SAGE_MUTED        = hsl_to_hex(140, 30, 70)

ROSE_SPOTLIGHT    = "#F075A8"
ROSE_BASE         = "#DF90B1"
ROSE_MUTED        = hsl_to_hex(335, 32, 75)

MUSTARD_SPOTLIGHT = "#F2CC5A"
MUSTARD_BASE      = "#DBBD70"
MUSTARD_MUTED     = hsl_to_hex(43, 35, 70)

TEAL_SPOTLIGHT    = "#47D1CD"
TEAL_BASE         = "#70C2C2"
TEAL_MUTED        = hsl_to_hex(180, 25, 67)

# Family table: (label, sub-label, spotlight, base, muted, locked?)
FAMILIES = [
    ("PRIMARY",  "warm orange",         PRIMARY_SPOTLIGHT,  PRIMARY_BASE,  PRIMARY_MUTED,  True),
    ("CONTRAST", "medium slate blue",   CONTRAST_SPOTLIGHT, CONTRAST_BASE, CONTRAST_MUTED, True),
    ("ACCENT 1", "sage (triadic)",      SAGE_SPOTLIGHT,     SAGE_BASE,     SAGE_MUTED,     False),
    ("ACCENT 2", "rose (split-comp)",   ROSE_SPOTLIGHT,     ROSE_BASE,     ROSE_MUTED,     False),
    ("ACCENT 3", "mustard (analogous)", MUSTARD_SPOTLIGHT,  MUSTARD_BASE,  MUSTARD_MUTED,  False),
    ("ACCENT 4", "teal (analogous)",    TEAL_SPOTLIGHT,     TEAL_BASE,     TEAL_MUTED,     False),
]

ACCENT_FOR_DEMO_SPOTLIGHT = SAGE_SPOTLIGHT
ACCENT_FOR_DEMO_BASE      = SAGE_BASE
ACCENT_FOR_DEMO_MUTED     = SAGE_MUTED


# =============================================================
# DRAWING HELPERS
# =============================================================
def draw_swatch_strip(ax, items, title, subtitle=None,
                      hex_color=TEXT_LIGHT, label_color=TEXT_DIM,
                      include_hsl=False):
    """items = [(hex, label, subdesc), ...]"""
    ax.set_facecolor(BG)
    n = len(items)
    ax.set_xlim(0, max(n, 1))
    ax.set_ylim(0, 2.4 if include_hsl else 2.0)
    ax.axis("off")
    ax.text(0, 2.20 if include_hsl else 1.85, title, color=TEXT_LIGHT,
            fontsize=14, fontweight="bold", ha="left", va="center")
    if subtitle:
        ax.text(0, 1.95 if include_hsl else 1.65, subtitle, color=label_color,
                fontsize=10, ha="left", va="center", style="italic")
    sw_top = 1.55 if include_hsl else 1.40
    sw_bot = 0.75
    for i, item in enumerate(items):
        hexc = item[0]
        lab = item[1] if len(item) > 1 else ""
        sub = item[2] if len(item) > 2 else ""
        ax.add_patch(mpatches.Rectangle((i + 0.06, sw_bot), 0.88,
                                        sw_top - sw_bot,
                                        facecolor=hexc, edgecolor="#666666",
                                        lw=0.6))
        ax.text(i + 0.5, sw_bot - 0.09, hexc, color=hex_color, fontsize=8.5,
                ha="center", va="top", family="monospace")
        if include_hsl:
            h, s, l = hex_to_hsl(hexc)
            ax.text(i + 0.5, sw_bot - 0.25, f"hsl {h},{s},{l}",
                    color="#777777", fontsize=7, ha="center", va="top",
                    family="monospace")
        ax.text(i + 0.5, sw_bot - (0.42 if include_hsl else 0.25),
                lab, color=label_color, fontsize=8.5,
                ha="center", va="top")
        if sub:
            ax.text(i + 0.5, sw_bot - (0.58 if include_hsl else 0.41), sub,
                    color="#777777", fontsize=7.5, ha="center", va="top",
                    style="italic")


def draw_family_matrix(ax):
    """6 family rows × 3 tier cols.

    Layout:
      col headers: SPOTLIGHT, BASE, MUTED
      row labels: PRIMARY, CONTRAST, ACCENT 1..4
      each cell: large swatch + hex + hsl
    """
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(FAMILIES) * 1.4 + 1.2)
    ax.axis("off")

    col_centers = [3.5, 6.5, 9.5]
    col_labels = [
        ("SPOTLIGHT", "saturated · the focused element"),
        ("BASE",      "considered · default rendering"),
        ("MUTED",     "dusty · de-emphasized"),
    ]
    head_y = len(FAMILIES) * 1.4 + 0.85
    for cx, (cname, csub) in zip(col_centers, col_labels):
        ax.text(cx, head_y, cname, color=TEXT_LIGHT, fontsize=12,
                fontweight="bold", ha="center", va="center")
        ax.text(cx, head_y - 0.40, csub, color=TEXT_DIM, fontsize=9,
                ha="center", va="center", style="italic")
    ax.text(0, head_y, "family / role", color=TEXT_LIGHT, fontsize=12,
            fontweight="bold", ha="left", va="center")

    # Rows
    for i, (label, sub, sp, ba, mu, locked) in enumerate(FAMILIES):
        y = (len(FAMILIES) - i - 1) * 1.4 + 0.25
        # Row label
        col = TEXT_LIGHT if locked else TEXT_DIM
        ax.text(0, y + 0.55, label, color=col, fontsize=11,
                fontweight="bold" if locked else "normal",
                ha="left", va="center")
        ax.text(0, y + 0.20, sub, color="#777777", fontsize=9,
                ha="left", va="center", style="italic")
        if not locked:
            ax.text(0, y - 0.10, "(candidate)", color="#666666",
                    fontsize=8, ha="left", va="center")
        # Swatches
        for cx, hexc in zip(col_centers, [sp, ba, mu]):
            ax.add_patch(mpatches.Rectangle((cx - 1.1, y + 0.10),
                                            2.2, 0.85,
                                            facecolor=hexc,
                                            edgecolor="#666666", lw=0.6))
            ax.text(cx, y - 0.04, hexc, color=TEXT_LIGHT, fontsize=9.5,
                    ha="center", va="top", family="monospace")
            h, s, l = hex_to_hsl(hexc)
            ax.text(cx, y - 0.30, f"hsl({h}, {s}%, {l}%)", color="#888888",
                    fontsize=8, ha="center", va="top", family="monospace")


def draw_vector_panel(ax, a_color, b_color, proj_color, title, subtitle):
    NEUTRAL_AXIS = "#cccccc"
    NEUTRAL_DROP = "#888888"
    ax.set_facecolor(BG)
    LIM = 1.85
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    INSET = 0.07
    ax.annotate("", xy=(LIM - INSET, 0), xytext=(-LIM + INSET, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=NEUTRAL_AXIS,
                                alpha=0.6, lw=1.2))
    ax.annotate("", xy=(0, LIM - INSET), xytext=(0, -LIM + INSET),
                arrowprops=dict(arrowstyle="<|-|>", color=NEUTRAL_AXIS,
                                alpha=0.6, lw=1.2))
    # a: mag 1, angle 45°  →  (0.707, 0.707)  — the smaller vector
    # b: mag 1.5, angle 0° →  (1.5, 0)        — the longer vector
    # proj_a_on_b:  (a·b / b·b) * b  =  0.4714 * b  ≈  (0.707, 0)
    # Dropline drops vertically from a's tip down to its shadow on b.
    a = np.array([1.0 * np.cos(np.radians(45)),
                  1.0 * np.sin(np.radians(45))])
    b = np.array([1.5 * np.cos(np.radians(0)),
                  1.5 * np.sin(np.radians(0))])
    proj_scalar = np.dot(a, b) / np.dot(b, b)
    proj = proj_scalar * b

    def vec(start, v, color, lw, label):
        ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]),
                    xytext=(start[0], start[1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=18))
        if label:
            tip = (start[0] + v[0], start[1] + v[1])
            off = v / (np.linalg.norm(v) + 1e-9) * 0.16
            ax.text(tip[0] + off[0], tip[1] + off[1], label, color=color,
                    fontsize=12, fontweight="bold", ha="center", va="center")

    # b drawn first (longest, behind), then proj (shadow on b), then dropline
    # from a's tip down to proj, then a on top.
    vec(np.zeros(2), b, b_color, 2.4, "b")
    vec(np.zeros(2), proj, proj_color, 3.0, "")  # no label on projection
    ax.plot([a[0], proj[0]], [a[1], proj[1]], color=NEUTRAL_DROP, ls="--",
            lw=1.0, alpha=0.7)
    vec(np.zeros(2), a, a_color, 3.6, "a")
    ax.set_title(title, color=TEXT_LIGHT, fontsize=11, fontweight="bold",
                 pad=4, loc="left")
    ax.text(-LIM + 0.05, -LIM + 0.05, subtitle, color=TEXT_DIM,
            fontsize=8.5, ha="left", va="bottom", style="italic")


# =============================================================
# LAYOUT
# =============================================================
fig = plt.figure(figsize=(20, 40), facecolor=BG)
gs = fig.add_gridspec(
    nrows=15, ncols=3,
    height_ratios=[
        0.5,    # 0: header
        2.0,    # 1: neutrals strip
        0.4,    # 2: divider — by-tier strips
        2.2,    # 3: spotlight strip
        2.2,    # 4: base strip
        2.2,    # 5: muted strip
        0.4,    # 6: divider — by-family matrix
        9.0,    # 7: family matrix
        0.4,    # 8: divider — trio juxtaposition (NEW)
        2.6,    # 9: trio sage
        2.6,    # 10: trio rose
        2.6,    # 11: trio mustard
        2.6,    # 12: trio teal
        0.4,    # 13: divider — spotlight rotation in context
        2.6,    # 14: spotlight rotation
    ],
    hspace=0.35, wspace=0.18,
    left=0.03, right=0.98, top=0.975, bottom=0.012,
)

# --- ROW 0: Header ---
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor(BG); ax_hdr.axis("off")
ax_hdr.text(0.5, 0.65,
    "SubShader plot lib — full color schema",
    color=TEXT_LIGHT, fontsize=20, fontweight="bold", ha="center", va="center")
ax_hdr.text(0.5, 0.10,
    "Tiers: SPOTLIGHT (saturated · focus) · BASE (considered · default) · MUTED (dusty · de-emphasized).   "
    "Roles: PRIMARY · CONTRAST · ACCENT (4 candidates).   "
    "Plus structural neutrals.",
    color=TEXT_DIM, fontsize=11, ha="center", va="center", style="italic")

# --- ROW 1: Neutrals ---
ax_neut = fig.add_subplot(gs[1, :])
draw_swatch_strip(
    ax_neut, NEUTRALS,
    "NEUTRALS — locked structural colors",
    subtitle="background, axes, spines, text, freq lines — already in style.py",
    include_hsl=False,
)

# --- ROW 2: Divider ---
ax_d1 = fig.add_subplot(gs[2, :])
ax_d1.set_facecolor(BG); ax_d1.axis("off")
ax_d1.text(0, 0.50,
    "ALL FAMILIES BY TIER — read across to compare colors within the same tier",
    color=TEXT_LIGHT, fontsize=14, fontweight="bold", ha="left", va="center")

# --- ROW 3: SPOTLIGHT strip across all 6 families ---
spotlight_items = [(f[2], f[0], f[1]) for f in FAMILIES]
ax_sp = fig.add_subplot(gs[3, :])
draw_swatch_strip(
    ax_sp, spotlight_items,
    "SPOTLIGHT tier — saturated stops",
    subtitle="use when this color is the focused element of the panel",
    include_hsl=True,
)

# --- ROW 4: BASE strip ---
base_items = [(f[3], f[0], f[1]) for f in FAMILIES]
ax_bs = fig.add_subplot(gs[4, :])
draw_swatch_strip(
    ax_bs, base_items,
    "BASE tier — considered stops",
    subtitle="default rendering when no element is singled out",
    include_hsl=True,
)

# --- ROW 5: MUTED strip ---
muted_items = [(f[4], f[0], f[1]) for f in FAMILIES]
ax_mu = fig.add_subplot(gs[5, :])
draw_swatch_strip(
    ax_mu, muted_items,
    "MUTED tier — dusty stops",
    subtitle="present but de-emphasized; differentiable but quiet",
    include_hsl=True,
)

# --- ROW 6: Divider ---
ax_d2 = fig.add_subplot(gs[6, :])
ax_d2.set_facecolor(BG); ax_d2.axis("off")
ax_d2.text(0, 0.50,
    "FAMILY × TIER MATRIX — read across each row to see one family's three tiers",
    color=TEXT_LIGHT, fontsize=14, fontweight="bold", ha="left", va="center")

# --- ROW 7: Family matrix ---
ax_mx = fig.add_subplot(gs[7, :])
draw_family_matrix(ax_mx)

# --- ROW 8: Divider — trio juxtaposition ---
ax_d3 = fig.add_subplot(gs[8, :])
ax_d3.set_facecolor(BG); ax_d3.axis("off")
ax_d3.text(0, 0.50,
    "TRIO JUXTAPOSITION — primary + contrast + each accent candidate, in context",
    color=TEXT_LIGHT, fontsize=14, fontweight="bold", ha="left", va="center")
ax_d3.text(0, 0.05,
    "Each row: locked orange + slate-blue + ONE accent candidate.   "
    "Vector panel uses the BASE tier (default rendering).",
    color=TEXT_DIM, fontsize=9.5, ha="left", va="center", style="italic")


def draw_trio_swatches(ax, label, c1, c2, c3, c1_name, c2_name, c3_name):
    """Three swatches stacked vertically with hex+role labels."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(0.05, 3.75, label, color=TEXT_LIGHT, fontsize=12,
            fontweight="bold", ha="left", va="center")
    triples = [
        (3.0, c1, c1_name, "PRIMARY"),
        (2.0, c2, c2_name, "CONTRAST"),
        (1.0, c3, c3_name, "ACCENT"),
    ]
    for y, hexc, name, role in triples:
        ax.add_patch(mpatches.Rectangle((0.05, y - 0.32), 0.55, 0.65,
                                        facecolor=hexc, edgecolor="#666666",
                                        lw=0.6))
        ax.text(0.65, y + 0.05, hexc, color=TEXT_LIGHT, fontsize=10,
                ha="left", va="center", family="monospace")
        ax.text(0.65, y - 0.18, f"{role} · {name}", color=TEXT_DIM,
                fontsize=9, ha="left", va="center")


# Trio rows: (gridspec_row, accent_label, accent_subname, accent_hex)
TRIOS = [
    (9,  "Trio A — sage accent",    "sage",    SAGE_BASE),
    (10, "Trio B — rose accent",    "rose",    ROSE_BASE),
    (11, "Trio C — mustard accent", "mustard", MUSTARD_BASE),
    (12, "Trio D — teal accent",    "teal",    TEAL_BASE),
]

for gs_row, label, accent_name, accent_hex in TRIOS:
    sub = gs[gs_row, :].subgridspec(1, 3, width_ratios=[1.0, 1.4, 1.4],
                                    wspace=0.10)
    ax_sw = fig.add_subplot(sub[0, 0])
    draw_trio_swatches(ax_sw, label,
                       PRIMARY_BASE, CONTRAST_BASE, accent_hex,
                       "warm orange", "slate blue", accent_name)
    ax_v1 = fig.add_subplot(sub[0, 1])
    draw_vector_panel(ax_v1,
        a_color=PRIMARY_BASE,
        b_color=CONTRAST_BASE,
        proj_color=accent_hex,
        title="vector example — BASE tier",
        subtitle=f"a=orange · b=slate blue · proj={accent_name}")
    # Second variant: same trio at SPOTLIGHT tier (so user sees both)
    accent_spotlight = {
        "sage": SAGE_SPOTLIGHT, "rose": ROSE_SPOTLIGHT,
        "mustard": MUSTARD_SPOTLIGHT, "teal": TEAL_SPOTLIGHT,
    }[accent_name]
    ax_v2 = fig.add_subplot(sub[0, 2])
    draw_vector_panel(ax_v2,
        a_color=PRIMARY_SPOTLIGHT,
        b_color=CONTRAST_SPOTLIGHT,
        proj_color=accent_spotlight,
        title="vector example — SPOTLIGHT tier",
        subtitle="all three at saturated stops")

# --- ROW 13: Divider — spotlight rotation ---
ax_d4 = fig.add_subplot(gs[13, :])
ax_d4.set_facecolor(BG); ax_d4.axis("off")
ax_d4.text(0, 0.50,
    "SPOTLIGHT ROTATION — same panel, different element in focus per scene",
    color=TEXT_LIGHT, fontsize=14, fontweight="bold", ha="left", va="center")
ax_d4.text(0, 0.05,
    "Demo accent: sage.   Spotlit element at SPOTLIGHT, others at MUTED.",
    color=TEXT_DIM, fontsize=9.5, ha="left", va="center", style="italic")

# --- ROW 14: Spotlight rotation (3 panels) ---
ax_s1 = fig.add_subplot(gs[14, 0])
draw_vector_panel(ax_s1,
    a_color=PRIMARY_SPOTLIGHT,
    b_color=CONTRAST_MUTED,
    proj_color=ACCENT_FOR_DEMO_MUTED,
    title="Scene 1 — explain vector a",
    subtitle="a=SPOTLIGHT · b=MUTED · proj=MUTED")
ax_s2 = fig.add_subplot(gs[14, 1])
draw_vector_panel(ax_s2,
    a_color=PRIMARY_MUTED,
    b_color=CONTRAST_SPOTLIGHT,
    proj_color=ACCENT_FOR_DEMO_MUTED,
    title="Scene 2 — explain vector b",
    subtitle="a=MUTED · b=SPOTLIGHT · proj=MUTED")
ax_s3 = fig.add_subplot(gs[14, 2])
draw_vector_panel(ax_s3,
    a_color=PRIMARY_MUTED,
    b_color=CONTRAST_MUTED,
    proj_color=ACCENT_FOR_DEMO_SPOTLIGHT,
    title="Scene 3 — explain the projection",
    subtitle="a=MUTED · b=MUTED · proj=SPOTLIGHT")

plt.savefig(OUT, dpi=150, facecolor=BG, bbox_inches="tight")
print(f"wrote {OUT}")
