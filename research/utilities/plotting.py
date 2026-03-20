"""
Reusable rendering helpers for the benchmark suite.

All figure layout, spectrogram rendering, and top-row rendering logic
lives here so that stub_layouts() and _generate_comparison_figure()
share a single code path. Backend dispatch (matplotlib vs seaborn)
is handled via set_backend() / get_backend().
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample as scipy_resample

from . import constants

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# =============================================================================
# STYLE CONSTANTS
# =============================================================================

DEFAULT_STYLE = {
    "waveform_color": "#606060",
    "waveform_alpha": 0.75,
    "freq_line_color": "#AAAAAA",
    "freq_bg": "#1A1A1A",
    "freq_line_width": 2,
    "title_fontsize": 24,
    "tick_labelsize": 14,
    "axis_label_fontsize": 18,
    "suptitle_fontsize": 32,
    "subtitle_fontsize": 24,
    "subtitle_color": "black",
    "figsize_w": 20,
    "row_height": 4,
    "hspace": 0.22,
    "left": 0.06,
    "right": 0.94,
    "bottom": 0.04,
    "top": 0.90,
    "suptitle_y": 0.975,
    "subtitle_y": 0.925,
    "dpi": 150,
}

SEABORN_STYLE = {
    "waveform_color": "#FF8C42",
    "waveform_alpha": 0.85,
    "freq_line_color": "#FFD700",
    "freq_bg": "#2D1B4E",
    "freq_line_width": 2.5,
    "title_fontsize": 24,
    "tick_labelsize": 16,
    "axis_label_fontsize": 9,
    "suptitle_fontsize": 13,
    "subtitle_fontsize": 13,
    "subtitle_color": "#C9D1D9",
    "figsize_w": 22,
    "row_height": 4.5,
    "hspace": 0.15,
    "left": 0.07,
    "right": 0.92,
    "bottom": 0.04,
    "top": 0.93,
    "suptitle_y": 0.99,
    "subtitle_y": None,  # subtitle baked into suptitle for seaborn
    "dpi": 150,
    "bg_color": "#0D1117",
}

# =============================================================================
# BACKEND DISPATCH
# =============================================================================

_BACKEND = "matplotlib"


def set_backend(backend):
    """Set the active plot backend: 'matplotlib' or 'seaborn'."""
    global _BACKEND
    if backend not in ("matplotlib", "seaborn"):
        raise ValueError(f"Unknown backend: {backend!r}")
    _BACKEND = backend
    if backend == "seaborn":
        if not SEABORN_AVAILABLE:
            raise RuntimeError("seaborn is not installed")
        sns.set_theme(style="dark", rc={
            "axes.facecolor": "#0D1117",
            "figure.facecolor": "#0D1117",
            "text.color": "#C9D1D9",
            "axes.labelcolor": "#C9D1D9",
            "xtick.color": "#8B949E",
            "ytick.color": "#8B949E",
        })
    else:
        if SEABORN_AVAILABLE:
            sns.reset_orig()


def get_backend():
    """Return the current backend name."""
    return _BACKEND


def get_active_style():
    """Return SEABORN_STYLE when seaborn backend is active, else DEFAULT_STYLE."""
    return SEABORN_STYLE if _BACKEND == "seaborn" else DEFAULT_STYLE


# =============================================================================
# HELPERS
# =============================================================================

def compute_freq_yticks(cwt_freqs, tick_freqs=(20, 200, 2000, 20000)):
    """Map log-spaced frequencies to bin positions.

    Returns (ytick_bins, ytick_labels).
    """
    freq_min, freq_max = cwt_freqs[0], cwt_freqs[-1]
    n = len(cwt_freqs)
    ytick_bins, ytick_labels = [], []
    for f in tick_freqs:
        if freq_min <= f <= freq_max:
            ytick_bins.append(float(np.interp(f, cwt_freqs, np.arange(n))))
            ytick_labels.append(f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}')
    return ytick_bins, ytick_labels


def placeholder_ax(ax, label="reference not available"):
    """Render a polished placeholder for missing reference images."""
    ax.set_facecolor("#1A1A1A")
    for x in np.linspace(0, 1, 20):
        ax.axline((x, 0), slope=1, color="#333333", linewidth=0.5,
                  transform=ax.transAxes)
    ax.text(0.5, 0.5, f"[ {label} ]",
            ha="center", va="center", color="#555555",
            fontsize=11, transform=ax.transAxes)
    ax.axis("off")


def create_figure_scaffold(title, subtitle, n_top_rows, style=None):
    """Create the standard figure layout.

    Returns (fig, gs, ax_stft, ax_pywt, ax_npwt).
    """
    if style is None:
        style = get_active_style()
    s = {**DEFAULT_STYLE, **style}
    n_total = n_top_rows + 3

    fig = plt.figure(figsize=(s["figsize_w"], s["row_height"] * n_total))
    fig.suptitle(title, fontsize=s["suptitle_fontsize"], y=s["suptitle_y"])
    if s.get("subtitle_y") is not None and subtitle:
        fig.text(0.5, s["subtitle_y"], subtitle,
                 ha='center', fontsize=s["subtitle_fontsize"],
                 color=s["subtitle_color"])
    gs = gridspec.GridSpec(n_total, 1, figure=fig,
                           height_ratios=[1] * n_total, hspace=s["hspace"])
    fig.subplots_adjust(left=s["left"], right=s["right"],
                        bottom=s["bottom"], top=s["top"])

    ax_stft = fig.add_subplot(gs[n_top_rows])
    ax_pywt = fig.add_subplot(gs[n_top_rows + 1], sharex=ax_stft, sharey=ax_stft)
    ax_npwt = fig.add_subplot(gs[n_top_rows + 2], sharex=ax_stft, sharey=ax_stft)

    return fig, gs, ax_stft, ax_pywt, ax_npwt


def render_top_row(fig, gs, idx, row, ax_stft, *,
                   t_audio, y_min, y_max, cwt_freqs, duration_s,
                   ytick_bins, ytick_labels, style=None):
    """Render one top row (waveform / freq_line / image) into gs[idx]."""
    if style is None:
        style = get_active_style()
    s = {**DEFAULT_STYLE, **style}
    rtype = row["type"]
    n_cwt_freqs = len(cwt_freqs)

    if rtype == "waveform":
        ax = fig.add_subplot(gs[idx], sharex=ax_stft)
        if style and "bg_color" in style:
            ax.set_facecolor(style["bg_color"])
        ax.fill_between(t_audio, y_min, y_max,
                        color=s["waveform_color"], alpha=s["waveform_alpha"])
        if style and "bg_color" in style:
            ax.set_ylim([np.min([y_min, y_max]), np.max([y_min, y_max])])
            ax.set_xlim([0, duration_s])
            ax.margins(x=0, y=0)
        # ax.set_ylabel("Amplitude", fontsize=s["axis_label_fontsize"])
        ax.set_title(row["title"], fontsize=s["title_fontsize"], loc="left")
        ax.tick_params(labelsize=s["tick_labelsize"])
        plt.setp(ax.get_xticklabels(), visible=False)

    elif rtype == "freq_line":
        ax = fig.add_subplot(gs[idx], sharex=ax_stft, sharey=ax_stft)
        t_curve = np.linspace(0, duration_s, 500)
        f_curve = row['f0'] + (row['f1'] - row['f0']) * t_curve / duration_s
        bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
        ax.plot(t_curve, bin_curve, color=s["freq_line_color"],
                linewidth=s["freq_line_width"])
        ax.set_facecolor(s["freq_bg"])
        ax.set_title(row["title"], fontsize=s["title_fontsize"], loc="left")
        ax.set_yticks(ytick_bins)
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(labelsize=s["tick_labelsize"])
        if style and "bg_color" in style:
            ax.tick_params(colors=s.get("subtitle_color", "#C9D1D9"))
        plt.setp(ax.get_xticklabels(), visible=False)

    elif rtype == "image":
        ax = fig.add_subplot(gs[idx])
        img_path = row.get("path")
        if img_path and os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img, aspect="auto", origin="upper",
                      cmap='gray' if style is None else None)
            ax.set_title(row["title"], fontsize=s["title_fontsize"], loc="left")
            ax.axis("off")
        else:
            placeholder_ax(ax)
            ax.set_title(row["title"], fontsize=s["title_fontsize"], loc="left")


def render_spectrogram_row(ax, data, *, title, extent, vmax,
                           ytick_bins, ytick_labels,
                           is_bottom=False, cmap="inferno",
                           n_cwt_freqs=None, duration_s=None, style=None):
    """Render one spectrogram row, dispatching to the active backend.

    n_cwt_freqs and duration_s are required by the seaborn backend for tick
    scaling; matplotlib ignores them.
    """
    if style is None:
        style = get_active_style()

    if _BACKEND == "seaborn":
        _render_spectrogram_seaborn(
            ax, data, title=title, vmax=vmax,
            ytick_bins=ytick_bins, ytick_labels=ytick_labels,
            is_bottom=is_bottom, cmap=cmap,
            n_cwt_freqs=n_cwt_freqs, duration_s=duration_s, style=style)
    else:
        _render_spectrogram_matplotlib(
            ax, data, title=title, extent=extent, vmax=vmax,
            ytick_bins=ytick_bins, ytick_labels=ytick_labels,
            is_bottom=is_bottom, cmap=cmap, style=style)


def _render_spectrogram_matplotlib(ax, data, *, title, extent, vmax,
                                    ytick_bins, ytick_labels,
                                    is_bottom=False, cmap="inferno", style=None):
    """Render one spectrogram row with imshow + labels + yticks."""
    if style is None:
        style = get_active_style()
    s = {**DEFAULT_STYLE, **style}

    ax.imshow(data, cmap=cmap, aspect="auto", origin="lower",
              extent=extent, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=s["title_fontsize"], loc="left")
    # ax.set_ylabel("Freq", fontsize=s["axis_label_fontsize"])
    ax.set_yticks(ytick_bins)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(labelsize=s["tick_labelsize"])
    if is_bottom:
        ax.set_xlabel("Time (s)", fontsize=s["axis_label_fontsize"])
    else:
        plt.setp(ax.get_xticklabels(), visible=False)


def _render_spectrogram_seaborn(ax, data, *, title, vmax,
                                 ytick_bins, ytick_labels,
                                 is_bottom=False, cmap="inferno",
                                 n_cwt_freqs=None, duration_s=None, style=None):
    """Render one spectrogram row as a seaborn heatmap."""
    if style is None:
        style = get_active_style()
    s = {**DEFAULT_STYLE, **style}

    ds = downsample_spec(data)
    sns.heatmap(ds, ax=ax, cmap=cmap, vmin=0, vmax=vmax,
                xticklabels=False, yticklabels=False,
                cbar=True, cbar_kws={"shrink": 0.8, "pad": 0.01,
                                     "label": "Intensity"})
    ax.invert_yaxis()
    if duration_s is not None:
        ax.set_xlim([0, duration_s])
    if n_cwt_freqs is not None:
        scale = ds.shape[0] / n_cwt_freqs
        ax.set_yticks([b * scale for b in ytick_bins])
    else:
        ax.set_yticks(ytick_bins)
    ax.set_yticklabels(ytick_labels, fontsize=s["tick_labelsize"])
    ax.set_title(title, fontsize=s["title_fontsize"], loc="left")

    if is_bottom and duration_s is not None:
        desired_s = np.linspace(0, duration_s, 6)
        ax.set_xticks(desired_s)
        ax.set_xticklabels([f"{t:.1f}" for t in desired_s], fontsize=s["tick_labelsize"])
        ax.set_xlabel("Time (s)", fontsize=s["axis_label_fontsize"])
    elif not is_bottom:
        plt.setp(ax.get_xticklabels(), visible=False)


def downsample_spec(arr, max_rows=None, max_cols=None):
    """Downsample spectrogram to fit heatmap constraints."""
    if max_rows is None:
        max_rows = constants.HEATMAP_MAX_ROWS
    if max_cols is None:
        max_cols = constants.HEATMAP_MAX_COLS

    h, w = arr.shape
    target_w = min(w, max_cols)
    target_h = min(h, max_rows)
    if w != target_w:
        arr = scipy_resample(arr, target_w, axis=1)
    if h != target_h:
        arr = scipy_resample(arr, target_h, axis=0)
    return np.clip(arr, 0, None).astype(np.float32)
