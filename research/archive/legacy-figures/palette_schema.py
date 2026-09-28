"""
Palette schema renderer for SubShader figures.

Two outputs:
- v5 (pruned): the locked-down 4-family x 2-tier reference
- v6 (candidates): side-by-side primary / contrast / tertiary comparison
  between the v4-derived hexes and the v5 alternatives, so we can pick
  one set before locking style.py.

v4 hexes were sampled directly from
  assets/images/claude/palette_schema_v4_no_proj_label.png
via PIL pixel sampling at known swatch centers (see /tmp/extract_v4_precise.py
in the development log). v5 hexes were hand-tuned alternatives.

Run from repo root:
    python research/palette_schema.py            # writes v8 colormap-derived candidates
    python research/palette_schema.py --v7       # writes v7 tertiary candidates
    python research/palette_schema.py --v6       # writes v6 anchor + tertiary
    python research/palette_schema.py --v5       # writes v5 pruned ref

Output:
    assets/images/claude/palette_schema_v8_colormap_candidates.png
    assets/images/claude/palette_schema_v7_tertiary_candidates.png
    assets/images/claude/palette_schema_v6_candidates.png
    assets/images/claude/palette_schema_v5_pruned.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ============================================================================
# Constants
# ============================================================================

BG = "#1a1a1a"
PANEL_BG = "#1a1a1a"
TITLE_COLOR = "#f0f0f0"
LABEL_COLOR = "#cccccc"
DIM_COLOR = "#888888"
DIVIDER_COLOR = "#333333"

NEUTRALS = [
    ("#0a0a0a", "background"),
    ("#1f1f1f", "panel"),
    ("#2c2c2c", "axis-bg"),
    ("#4a4a4a", "spine"),
    ("#6e6e6e", "tick"),
    ("#9a9a9a", "axis-label"),
    ("#cfcfcf", "text"),
    ("#ffffff", "highlight"),
]


# v4-derived hexes (sampled from palette_schema_v4_no_proj_label.png)
V4 = {
    "primary":    {"spotlight": "#f5a83d", "muted": "#c18e62"},
    "contrast":   {"spotlight": "#8878dd", "muted": "#8986b1"},
    "sage":       {"spotlight": "#52e081", "muted": "#89ae95"},
    "teal":       {"spotlight": "#47d1cd", "muted": "#84a7a7"},
}

# v5 alternatives (hand-tuned in 2026-05-09 prune pass)
V5 = {
    "primary":    {"spotlight": "#ffa552", "muted": "#a87545"},
    "contrast":   {"spotlight": "#b6a4ff", "muted": "#7a72b0"},
    "sage":       {"spotlight": "#7ee787", "muted": "#558a5b"},
    "teal":       {"spotlight": "#5fc8d8", "muted": "#3e8a96"},
}


@dataclass(frozen=True)
class Family:
    name: str
    role: str
    spotlight: str
    muted: str


def _families_v5() -> list[Family]:
    return [
        Family("primary",  "PRIMARY  - default rendering",  V5["primary"]["spotlight"],  V5["primary"]["muted"]),
        Family("contrast", "CONTRAST - default annotation", V5["contrast"]["spotlight"], V5["contrast"]["muted"]),
        Family("sage",     "ACCENT 1 - sage support",       V5["sage"]["spotlight"],     V5["sage"]["muted"]),
        Family("teal",     "ACCENT 2 - teal support",       V5["teal"]["spotlight"],     V5["teal"]["muted"]),
    ]


# ============================================================================
# Drawing primitives
# ============================================================================

def _section_title(ax, x, y, text, size=10):
    ax.text(x, y, text, color=TITLE_COLOR, fontsize=size, family="sans-serif",
            fontweight="bold", ha="left", va="bottom")


def _section_subtitle(ax, x, y, text, size=7.5):
    ax.text(x, y, text, color=DIM_COLOR, fontsize=size, family="sans-serif",
            ha="left", va="bottom", style="italic")


def _hex_swatch(ax, x, y, w, h, color, sub_label=None, version_tag=None):
    rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="none")
    ax.add_patch(rect)
    ax.text(x + w * 0.04, y + h - 0.10, color,
            color=BG, fontsize=7.5, family="monospace",
            ha="left", va="top")
    if sub_label:
        ax.text(x + w * 0.04, y + 0.06, sub_label,
                color=BG, fontsize=6.2, family="monospace",
                ha="left", va="bottom", alpha=0.85)
    if version_tag:
        ax.text(x + w - 0.05, y + h - 0.10, version_tag,
                color=BG, fontsize=6.5, family="monospace", fontweight="bold",
                ha="right", va="top", alpha=0.95)


def _divider(ax, x0, y, x1):
    ax.plot([x0, x1], [y, y], color=DIVIDER_COLOR, lw=0.6)


def _draw_neutrals(ax, x0, y0, width, row_height):
    _section_title(ax, x0, y0 + row_height + 0.40, "NEUTRALS - locked structural colors")
    _section_subtitle(ax, x0, y0 + row_height + 0.15,
                      "background, axes, labels - never tinted")
    n = len(NEUTRALS)
    cell_w = width / n
    for i, (hex_, role) in enumerate(NEUTRALS):
        x = x0 + i * cell_w
        rect = patches.Rectangle((x, y0), cell_w * 0.97, row_height,
                                 facecolor=hex_, edgecolor="none")
        ax.add_patch(rect)
        text_color = "#ffffff" if i < 4 else "#1a1a1a"
        ax.text(x + 0.04 * cell_w, y0 + row_height - 0.12,
                hex_, color=text_color, fontsize=7, family="monospace",
                ha="left", va="top")
        ax.text(x + 0.04 * cell_w, y0 + 0.10,
                role, color=text_color, fontsize=6, family="monospace",
                ha="left", va="bottom", alpha=0.85)


def _draw_candidate_row(ax, x0, y0, width, row_height,
                        title, subtitle, candidates):
    """
    candidates: list of (label, version_tag, spotlight_hex, muted_hex)
    Renders one tier-pair (spotlight + muted) per candidate, side-by-side.
    """
    _section_title(ax, x0, y0 + row_height + 0.40, title)
    _section_subtitle(ax, x0, y0 + row_height + 0.15, subtitle)

    label_w = width * 0.16
    cand_w = (width - label_w) / len(candidates)
    swatch_w = (cand_w - 0.10) / 2  # spotlight + muted side by side, 0.10 gap

    # tier mini-headers above the swatches
    ax.text(x0 + label_w + swatch_w * 0.5, y0 + row_height + 0.05,
            "spotlight", color=DIM_COLOR, fontsize=6.5, ha="center")
    ax.text(x0 + label_w + swatch_w * 1.5 + 0.10, y0 + row_height + 0.05,
            "muted", color=DIM_COLOR, fontsize=6.5, ha="center")

    for i, (label, tag, sp, mu) in enumerate(candidates):
        cx = x0 + label_w + i * cand_w
        # candidate label
        ax.text(x0, y0 + row_height * 0.5, "" if i > 0 else "",
                color=LABEL_COLOR, fontsize=8, ha="left", va="center")
        # We want one label per CANDIDATE row, not per family row
        # so we stack candidates horizontally; the label_w gutter on the
        # very first candidate carries the family/role text.
        if i == 0:
            ax.text(x0, y0 + row_height * 0.62, label,
                    color=LABEL_COLOR, fontsize=8.5, fontweight="bold",
                    ha="left", va="center")
            ax.text(x0, y0 + row_height * 0.30, "v4 vs v5",
                    color=DIM_COLOR, fontsize=6.5, ha="left", va="center",
                    style="italic")
        # spotlight swatch
        _hex_swatch(ax, cx, y0, swatch_w, row_height, sp,
                    sub_label=f"{label}-spot", version_tag=tag)
        # muted swatch
        _hex_swatch(ax, cx + swatch_w + 0.10, y0, swatch_w, row_height, mu,
                    sub_label=f"{label}-muted", version_tag=tag)


def _draw_candidate_block(ax, x0, y0, width, row_height,
                          title, subtitle, options):
    """
    Multi-row candidate block: each option on its own row.
    options: list of (label, version_tag, spotlight, muted)
    """
    _section_title(ax, x0, y0 + len(options) * (row_height + 0.10) + 0.30, title)
    _section_subtitle(ax, x0, y0 + len(options) * (row_height + 0.10) + 0.10,
                      subtitle)

    label_w = width * 0.22
    swatch_w = (width - label_w - 0.20) / 2

    # tier mini-headers above the rows
    header_y = y0 + len(options) * (row_height + 0.10) - 0.04
    ax.text(x0 + label_w + swatch_w * 0.5, header_y,
            "spotlight", color=DIM_COLOR, fontsize=7, ha="center")
    ax.text(x0 + label_w + swatch_w * 1.5 + 0.20, header_y,
            "muted", color=DIM_COLOR, fontsize=7, ha="center")

    for i, (label, tag, sp, mu) in enumerate(options):
        # bottom-up
        ry = y0 + (len(options) - 1 - i) * (row_height + 0.10)
        ax.text(x0, ry + row_height * 0.62, label,
                color=LABEL_COLOR, fontsize=8.5, fontweight="bold",
                ha="left", va="center")
        ax.text(x0, ry + row_height * 0.30, tag,
                color=DIM_COLOR, fontsize=7, ha="left", va="center",
                style="italic", family="monospace")
        _hex_swatch(ax, x0 + label_w, ry, swatch_w, row_height, sp,
                    sub_label="spotlight")
        _hex_swatch(ax, x0 + label_w + swatch_w + 0.20, ry, swatch_w,
                    row_height, mu, sub_label="muted")


def _draw_vector_example(ax, x0, y0, w, h, primary, contrast, accent, scene_label):
    panel = patches.Rectangle((x0, y0), w, h, facecolor=PANEL_BG,
                              edgecolor="#333333", linewidth=0.6)
    ax.add_patch(panel)

    cx = x0 + w * 0.5
    cy = y0 + h * 0.5
    span = min(w, h) * 0.38

    ax.annotate("", xy=(cx + span, cy), xytext=(cx - span * 0.6, cy),
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))
    ax.annotate("", xy=(cx, cy + span), xytext=(cx, cy - span * 0.6),
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))

    ax_end = (cx + span * 0.85, cy + span * 0.55)
    ax.annotate("", xy=ax_end, xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=primary,
                                lw=1.6, mutation_scale=10))
    bx_end = (cx + span * 0.45, cy + span * 0.85)
    ax.annotate("", xy=bx_end, xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=contrast,
                                lw=1.6, mutation_scale=10))
    proj_end = (cx + span * 0.62, cy + span * 0.40)
    ax.annotate("", xy=proj_end, xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=accent,
                                lw=2.0, mutation_scale=11))
    ax.plot([bx_end[0], proj_end[0]], [bx_end[1], proj_end[1]],
            linestyle=(0, (2, 2)), color="#888", lw=0.6, alpha=0.7)

    ax.text(x0 + w * 0.04, y0 + h - 0.05, scene_label,
            color=LABEL_COLOR, fontsize=6, ha="left", va="top",
            family="monospace")


def _draw_trio_grid(ax, x0, y0, width, total_height, title, subtitle, combos):
    """
    combos: list of (label, primary_hex, contrast_hex, accent_hex)
    Rendered as a vertical stack: label + 3-swatch row + vector example.
    """
    _section_title(ax, x0, y0 + total_height + 0.20, title)
    _section_subtitle(ax, x0, y0 + total_height, subtitle)

    n = len(combos)
    block_h = total_height / n - 0.15
    label_w = width * 0.22
    swatch_col_w = width * 0.40
    example_col_w = width - label_w - swatch_col_w - 0.20

    for ci, (label, p, c, a) in enumerate(combos):
        ry = y0 + (n - 1 - ci) * (block_h + 0.15)
        ax.text(x0, ry + block_h * 0.92, label,
                color=LABEL_COLOR, fontsize=8.5, fontweight="bold",
                ha="left", va="top")

        sub_h = (block_h - 0.10) / 3
        sx = x0 + label_w
        for si, (color, role) in enumerate([
            (p, "primary"),
            (c, "contrast"),
            (a, "tertiary"),
        ]):
            sy = ry + (2 - si) * (sub_h + 0.05)
            _hex_swatch(ax, sx, sy, swatch_col_w, sub_h, color, sub_label=role)

        ex_x = x0 + label_w + swatch_col_w + 0.20
        _draw_vector_example(ax, ex_x, ry, example_col_w, block_h,
                             p, c, a, "vector example")


# ============================================================================
# Figures
# ============================================================================

def render_v5(out_path: Path) -> Path:
    """Locked-down v5 reference (single set, 4 fams x 2 tiers)."""
    fams = _families_v5()

    fig: Figure = plt.figure(figsize=(11, 14), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    ax.text(0.45, 13.55, "SubShader plot lib - full color schema (v5 pruned)",
            color=TITLE_COLOR, fontsize=14, fontweight="bold",
            ha="left", va="bottom")
    ax.text(0.45, 13.30,
            "tiers: SPOTLIGHT (saturated) + MUTED (dusty).  "
            "families: PRIMARY, CONTRAST, sage, teal.",
            color=DIM_COLOR, fontsize=8, ha="left", va="bottom",
            style="italic")

    margin_x = 0.45
    content_w = 11 - 2 * margin_x

    # Neutrals
    _draw_neutrals(ax, margin_x, 12.20, content_w, 0.85)

    # Family x tier
    label_w = content_w * 0.20
    n_tiers = 2
    col_w = (content_w - label_w) / n_tiers
    matrix_row_h = 0.80
    matrix_gap = 0.18
    matrix_y = 6.50

    title_y = matrix_y + len(fams) * (matrix_row_h + matrix_gap) + 0.45
    _section_title(ax, margin_x, title_y,
                   "FAMILY x TIER MATRIX - 4 families x 2 tiers")
    _section_subtitle(ax, margin_x, matrix_y +
                      len(fams) * (matrix_row_h + matrix_gap) + 0.20,
                      "spotlight = saturated, muted = dusty")

    for ci, h in enumerate(["SPOTLIGHT", "MUTED"]):
        cx = margin_x + label_w + ci * col_w
        ax.text(cx + col_w * 0.5,
                matrix_y + len(fams) * (matrix_row_h + matrix_gap) + 0.05,
                h, color=LABEL_COLOR, fontsize=8, fontweight="bold",
                ha="center", va="bottom")

    for fi, fam in enumerate(fams):
        ry = matrix_y + (len(fams) - 1 - fi) * (matrix_row_h + matrix_gap)
        ax.text(margin_x, ry + matrix_row_h * 0.55, fam.name,
                color=LABEL_COLOR, fontsize=8.5, fontweight="bold",
                ha="left", va="center")
        ax.text(margin_x, ry + matrix_row_h * 0.20, fam.role,
                color=DIM_COLOR, fontsize=6.5, ha="left", va="center",
                style="italic")
        for ci, color in enumerate([fam.spotlight, fam.muted]):
            cx = margin_x + label_w + ci * col_w
            _hex_swatch(ax, cx, ry, col_w * 0.97, matrix_row_h, color,
                        sub_label=fam.name)

    # Spotlight rotation at the bottom
    rotation_y = 1.50
    rotation_h = 2.30
    _section_title(ax, margin_x, rotation_y + rotation_h + 0.30,
                   "SPOTLIGHT ROTATION - same panel, different element in focus")
    _section_subtitle(ax, margin_x, rotation_y + rotation_h + 0.05,
                      "rotate which element gets the spotlight tier")
    n = 3
    gap = 0.10
    panel_w = (content_w - (n - 1) * gap) / n
    p_s = fams[0].spotlight
    c_s = fams[1].spotlight
    a_s = fams[2].spotlight  # sage
    p_m = fams[0].muted
    c_m = fams[1].muted
    a_m = fams[2].muted
    scenes = [
        ("Scene 1 - explain a",          p_s, c_m, a_m),
        ("Scene 2 - explain b",          p_m, c_s, a_m),
        ("Scene 3 - explain projection", p_m, c_m, a_s),
    ]
    for i, (label, p, c, a) in enumerate(scenes):
        x = margin_x + i * (panel_w + gap)
        _draw_vector_example(ax, x, rotation_y, panel_w, rotation_h, p, c, a, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_path


def render_v6(out_path: Path) -> Path:
    """v6 candidate comparison: v4 vs v5 hexes for primary, contrast, tertiary."""
    fig: Figure = plt.figure(figsize=(12, 17), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 17)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    # Title
    ax.text(0.45, 16.55, "SubShader plot lib - color candidates (v6)",
            color=TITLE_COLOR, fontsize=14, fontweight="bold",
            ha="left", va="bottom")
    ax.text(0.45, 16.30,
            "v4 hexes (sampled from palette_schema_v4) vs v5 hexes (hand-tuned).  "
            "Pick one set per slot before locking style.py.",
            color=DIM_COLOR, fontsize=8, ha="left", va="bottom",
            style="italic")

    margin_x = 0.45
    content_w = 12 - 2 * margin_x

    # --- Neutrals (locked) -------------------------------------------------
    _draw_neutrals(ax, margin_x, 15.10, content_w, 0.75)

    # --- PRIMARY candidates ------------------------------------------------
    primary_options = [
        ("v4-primary",  "v4 / #f5a83d",
         V4["primary"]["spotlight"], V4["primary"]["muted"]),
        ("v5-primary",  "v5 / #ffa552",
         V5["primary"]["spotlight"], V5["primary"]["muted"]),
    ]
    _draw_candidate_block(ax, margin_x, 12.40, content_w, 0.70,
                          "PRIMARY CANDIDATES - default rendering color (orange)",
                          "v4 is warmer / amber, v5 is cooler / saturated",
                          primary_options)

    # --- CONTRAST candidates -----------------------------------------------
    contrast_options = [
        ("v4-contrast", "v4 / #8878dd",
         V4["contrast"]["spotlight"], V4["contrast"]["muted"]),
        ("v5-contrast", "v5 / #b6a4ff",
         V5["contrast"]["spotlight"], V5["contrast"]["muted"]),
    ]
    _draw_candidate_block(ax, margin_x, 9.60, content_w, 0.70,
                          "CONTRAST CANDIDATES - default annotation color (lavender)",
                          "v4 is deeper violet, v5 is lighter / pastel lavender",
                          contrast_options)

    # --- TERTIARY candidates (4 options: sage v4/v5, teal v4/v5) -----------
    tertiary_options = [
        ("sage-v4", "v4 / #52e081",
         V4["sage"]["spotlight"], V4["sage"]["muted"]),
        ("sage-v5", "v5 / #7ee787",
         V5["sage"]["spotlight"], V5["sage"]["muted"]),
        ("teal-v4", "v4 / #47d1cd",
         V4["teal"]["spotlight"], V4["teal"]["muted"]),
        ("teal-v5", "v5 / #5fc8d8",
         V5["teal"]["spotlight"], V5["teal"]["muted"]),
    ]
    _draw_candidate_block(ax, margin_x, 5.10, content_w, 0.70,
                          "TERTIARY CANDIDATES - sage and teal accent options",
                          "two hue families x two saturation tunings - pick one tertiary per scene",
                          tertiary_options)

    # --- Trio combos in context (4 corners: v4/v5 anchors x sage/teal) ----
    trio_combos = [
        ("v4 anchors + sage v4",
         V4["primary"]["spotlight"], V4["contrast"]["spotlight"], V4["sage"]["spotlight"]),
        ("v5 anchors + sage v5",
         V5["primary"]["spotlight"], V5["contrast"]["spotlight"], V5["sage"]["spotlight"]),
        ("v4 anchors + teal v4",
         V4["primary"]["spotlight"], V4["contrast"]["spotlight"], V4["teal"]["spotlight"]),
        ("v5 anchors + teal v5",
         V5["primary"]["spotlight"], V5["contrast"]["spotlight"], V5["teal"]["spotlight"]),
    ]
    _draw_trio_grid(ax, margin_x, 0.50, content_w, 4.00,
                    "TRIO JUXTAPOSITION - candidates in context",
                    "see how each candidate set reads when placed against panel + axes",
                    trio_combos)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_path


# ============================================================================
# v7 - 8 tertiary candidates against locked PRIMARY + CONTRAST anchors
# ============================================================================

# Anchors locked from v5 (user liked the v5 sage, holding v5 anchors as default;
# easy to flip to v4 hexes by editing this dict).
LOCKED_ANCHORS = {
    "primary":  V5["primary"]["spotlight"],
    "contrast": V5["contrast"]["spotlight"],
}

# 8 tertiary candidates - sage v5 anchored as #1 (user's current favorite),
# then 7 hues we haven't explored. Each one is a SPOTLIGHT-tier candidate;
# muted-tier tuning happens once a winner is picked.
TERTIARY_CANDIDATES = [
    ("sage v5",     "#7ee787", "current pick - mint-green"),
    ("emerald",     "#2ecc71", "deeper, more saturated green"),
    ("chartreuse",  "#c8e668", "yellow-green - high luminance"),
    ("mint-cyan",   "#4fd9b6", "green->cyan transition"),
    ("sky-blue",    "#6cc4ff", "clean blue, distinct from lavender"),
    ("coral",       "#ff8a6b", "warm coral - complementary to teal range"),
    ("lemon",       "#f5e266", "pale warm yellow"),
    ("amethyst",    "#b878e8", "deeper violet - more saturated than contrast"),
]


def _draw_locked_anchor_strip(ax, x0, y0, width, height):
    """Two narrow swatches showing the locked PRIMARY + CONTRAST anchors."""
    label_w = width * 0.18
    swatch_w = (width - label_w - 0.10) / 2

    ax.text(x0, y0 + height * 0.5, "LOCKED",
            color=LABEL_COLOR, fontsize=8.5, fontweight="bold",
            ha="left", va="center")
    ax.text(x0, y0 + height * 0.18, "primary + contrast",
            color=DIM_COLOR, fontsize=6.5, ha="left", va="center",
            style="italic")

    _hex_swatch(ax, x0 + label_w, y0, swatch_w, height,
                LOCKED_ANCHORS["primary"], sub_label="primary")
    _hex_swatch(ax, x0 + label_w + swatch_w + 0.10, y0, swatch_w, height,
                LOCKED_ANCHORS["contrast"], sub_label="contrast")


def _draw_tertiary_row(ax, x0, y0, width, height, label, hex_, note):
    """One candidate row: small swatch + larger vector example beside it."""
    label_w = width * 0.10
    swatch_w = width * 0.16
    gap = 0.15
    example_w = width - label_w - swatch_w - gap

    # label gutter
    ax.text(x0, y0 + height * 0.65, label,
            color=LABEL_COLOR, fontsize=9, fontweight="bold",
            ha="left", va="center")
    ax.text(x0, y0 + height * 0.30, note,
            color=DIM_COLOR, fontsize=6.5, ha="left", va="center",
            style="italic", wrap=True)

    # small swatch with hex
    sx = x0 + label_w
    rect = patches.Rectangle((sx, y0), swatch_w, height,
                             facecolor=hex_, edgecolor="none")
    ax.add_patch(rect)
    ax.text(sx + 0.05, y0 + height - 0.08,
            hex_, color=BG, fontsize=8, family="monospace",
            fontweight="bold", ha="left", va="top")
    ax.text(sx + 0.05, y0 + 0.06,
            "tertiary", color=BG, fontsize=6.5, family="monospace",
            ha="left", va="bottom", alpha=0.85)

    # larger vector example
    ex_x = x0 + label_w + swatch_w + gap
    _draw_vector_example(ax, ex_x, y0, example_w, height,
                         LOCKED_ANCHORS["primary"],
                         LOCKED_ANCHORS["contrast"],
                         hex_,
                         f"primary + contrast + {label}")


def render_v7(out_path: Path) -> Path:
    """v7: 8 tertiary candidates juxtaposed against locked anchors."""
    fig: Figure = plt.figure(figsize=(13, 18), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 18)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    margin_x = 0.45
    content_w = 13 - 2 * margin_x

    # Title
    ax.text(margin_x, 17.55, "SubShader plot lib - tertiary candidates (v7)",
            color=TITLE_COLOR, fontsize=14, fontweight="bold",
            ha="left", va="bottom")
    ax.text(margin_x, 17.25,
            "primary + contrast locked from v5.  "
            "8 tertiary options - sage v5 anchored as #1, 7 fresh hues below.",
            color=DIM_COLOR, fontsize=8, ha="left", va="bottom",
            style="italic")

    # Locked anchor strip
    _draw_locked_anchor_strip(ax, margin_x, 16.20, content_w, 0.65)
    _divider(ax, margin_x, 16.05, margin_x + content_w)

    # 8 candidate rows - each row generous so vector example reads well
    n = len(TERTIARY_CANDIDATES)
    rows_top = 15.85
    rows_bottom = 0.40
    total_rows_h = rows_top - rows_bottom
    gap = 0.18
    row_h = (total_rows_h - (n - 1) * gap) / n

    for i, (label, hex_, note) in enumerate(TERTIARY_CANDIDATES):
        ry = rows_top - (i + 1) * row_h - i * gap
        _draw_tertiary_row(ax, margin_x, ry, content_w, row_h,
                           label, hex_, note)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_path


# ============================================================================
# v8 - colormap-derived palette candidates
# ============================================================================

# Use case: 10 points in vector_a correspond to 10 points in vector_b,
# colored by index so correspondence reads at a glance. Pick a colormap
# whose 10 evenly-spaced samples are visually distinguishable AND can
# also serve as our PRIMARY / CONTRAST / TERTIARY / ... anchors.

CMAP_CANDIDATES = [
    ("inferno",  "matplotlib sequential - black -> purple -> red -> orange -> yellow"),
    ("plasma",   "matplotlib sequential - purple -> magenta -> orange -> yellow"),
    ("magma",    "matplotlib sequential - black -> purple -> pink -> cream"),
    ("viridis",  "matplotlib sequential - purple -> blue -> green -> yellow"),
    ("turbo",    "matplotlib rainbow-ish - blue -> green -> yellow -> red"),
    ("cividis",  "matplotlib sequential - colorblind-safe blue -> yellow"),
    ("twilight", "matplotlib cyclic - blue -> purple -> red -> blue"),
    ("tab10",    "matplotlib categorical - designed for 10 distinct classes"),
]

N_PALETTE = 10


def _sample_cmap(name: str, n: int) -> list[str]:
    """Sample n colors from a matplotlib colormap, return as hex strings."""
    import numpy as np
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(name)
    if name == "tab10":
        # categorical - take the first 10 entries directly
        rgba = [cmap(i) for i in range(n)]
    else:
        # sequential / cyclic - skip the very dark end (unreadable on #1a1a1a)
        positions = np.linspace(0.10, 0.95, n)
        rgba = [cmap(p) for p in positions]
    return ["#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for (r, g, b, _a) in rgba]


def _draw_correspondence_example(ax, x0, y0, w, h, colors):
    """
    Two horizontal rows of N_PALETTE markers each, same colors top and bottom,
    with thin connecting lines so 'point i in a corresponds to point i in b'
    reads clearly.
    """
    panel = patches.Rectangle((x0, y0), w, h, facecolor=PANEL_BG,
                              edgecolor="#333333", linewidth=0.6)
    ax.add_patch(panel)

    n = len(colors)
    pad = 0.18
    inner_w = w - 2 * pad
    cy_a = y0 + h * 0.72
    cy_b = y0 + h * 0.28

    # row labels
    ax.text(x0 + 0.06, cy_a, "a:", color=DIM_COLOR, fontsize=7,
            family="monospace", ha="left", va="center")
    ax.text(x0 + 0.06, cy_b, "b:", color=DIM_COLOR, fontsize=7,
            family="monospace", ha="left", va="center")

    # Permute b's positions a bit so the connecting lines aren't perfectly
    # vertical - sells the "correspondence by color" idea more clearly.
    # Use a fixed shuffle so it's deterministic.
    perm = [3, 7, 1, 9, 5, 0, 8, 2, 6, 4]

    xs = [x0 + pad + 0.18 + (i + 0.5) * (inner_w - 0.36) / n for i in range(n)]

    # connecting lines (drawn first, behind markers)
    for i in range(n):
        ax.plot([xs[i], xs[perm[i]]], [cy_a, cy_b],
                color=colors[i], lw=0.8, alpha=0.45)

    # markers
    marker_r = min(h * 0.10, (inner_w - 0.36) / n * 0.30)
    for i, color in enumerate(colors):
        circ_a = patches.Circle((xs[i], cy_a), marker_r,
                                facecolor=color, edgecolor="none")
        ax.add_patch(circ_a)
        circ_b = patches.Circle((xs[perm[i]], cy_b), marker_r,
                                facecolor=color, edgecolor="none")
        ax.add_patch(circ_b)


def _draw_cmap_candidate_row(ax, x0, y0, width, height,
                             cmap_name, description, colors):
    """One candidate row: name + 10 swatches with hex + correspondence panel."""
    label_w = width * 0.10
    swatch_band_w = width * 0.42
    gap = 0.12
    example_w = width - label_w - swatch_band_w - gap

    # --- label gutter ---
    ax.text(x0, y0 + height * 0.65, cmap_name,
            color=LABEL_COLOR, fontsize=9.5, fontweight="bold",
            ha="left", va="center", family="monospace")
    ax.text(x0, y0 + height * 0.30, description,
            color=DIM_COLOR, fontsize=6.2, ha="left", va="center",
            style="italic")

    # --- swatch band: 10 swatches stacked vertically with hex labels ---
    band_x = x0 + label_w
    n = len(colors)
    sw = swatch_band_w / n
    sh = height - 0.30  # leave room above for hex labels
    sy = y0
    role_labels = ["primary", "contrast", "tertiary",
                   "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    for i, color in enumerate(colors):
        sx = band_x + i * sw
        rect = patches.Rectangle((sx + 0.02, sy), sw - 0.04, sh,
                                 facecolor=color, edgecolor="none")
        ax.add_patch(rect)
        # hex below swatch (above swatch is too crowded)
        ax.text(sx + sw * 0.5, y0 + sh + 0.18, color,
                color=LABEL_COLOR, fontsize=6.2, family="monospace",
                ha="center", va="bottom")
        # role label inside swatch (small)
        ax.text(sx + sw * 0.5, y0 + sh * 0.5, role_labels[i],
                color=BG, fontsize=5.5, family="monospace",
                ha="center", va="center", alpha=0.85)

    # --- correspondence example ---
    ex_x = x0 + label_w + swatch_band_w + gap
    _draw_correspondence_example(ax, ex_x, y0, example_w, height, colors)


def render_v8(out_path: Path) -> Path:
    """v8: sample 10 colors from each candidate colormap, see correspondence."""
    fig: Figure = plt.figure(figsize=(14, 18), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    margin_x = 0.45
    content_w = 14 - 2 * margin_x

    # --- Title ---
    ax.text(margin_x, 17.55,
            "SubShader plot lib - colormap-derived palette candidates (v8)",
            color=TITLE_COLOR, fontsize=14, fontweight="bold",
            ha="left", va="bottom")
    ax.text(margin_x, 17.20,
            "10 colors sampled from each colormap.  Correspondence panel shows "
            "vector_a vs vector_b colored by index - point i in a links to point "
            "i in b by color.  Pick a map whose 10 samples are distinguishable "
            "AND whose first 3 work as primary / contrast / tertiary.",
            color=DIM_COLOR, fontsize=8, ha="left", va="bottom",
            style="italic", wrap=True)

    # --- Neutrals strip ---
    _draw_neutrals(ax, margin_x, 15.80, content_w, 0.65)
    _divider(ax, margin_x, 15.65, margin_x + content_w)

    # --- Candidate rows ---
    n = len(CMAP_CANDIDATES)
    rows_top = 15.40
    rows_bottom = 0.40
    total = rows_top - rows_bottom
    gap = 0.20
    row_h = (total - (n - 1) * gap) / n

    for i, (name, desc) in enumerate(CMAP_CANDIDATES):
        ry = rows_top - (i + 1) * row_h - i * gap
        try:
            colors = _sample_cmap(name, N_PALETTE)
        except Exception as exc:
            print(f"!! could not sample {name}: {exc}")
            continue
        _draw_cmap_candidate_row(ax, margin_x, ry, content_w, row_h,
                                 name, desc, colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "assets" / "images" / "claude"
    if "--v5" in sys.argv:
        out = out_dir / "palette_schema_v5_pruned.png"
        saved = render_v5(out)
    elif "--v6" in sys.argv:
        out = out_dir / "palette_schema_v6_candidates.png"
        saved = render_v6(out)
    elif "--v7" in sys.argv:
        out = out_dir / "palette_schema_v7_tertiary_candidates.png"
        saved = render_v7(out)
    else:
        out = out_dir / "palette_schema_v8_colormap_candidates.png"
        saved = render_v8(out)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
