"""
Reusable rendering helpers for the benchmark suite.

All figure layout, spectrogram rendering, and top-row rendering logic
lives here so that stub_layouts() and _generate_comparison_figure()
share a single code path. Visual constants are imported from style.py.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample as scipy_resample

from . import constants
from . import style


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
    ax.set_facecolor(style.BG_COLOR)
    for x in np.linspace(0, 1, 20):
        ax.axline((x, 0), slope=1, color="#333333", linewidth=0.5,
                  transform=ax.transAxes)
    ax.text(0.5, 0.5, f"[ {label} ]",
            ha="center", va="center", color="#555555",
            fontsize=11, transform=ax.transAxes)
    ax.axis("off")


def create_figure_scaffold(title, subtitle, n_top_rows):
    """Create the standard figure layout.

    Returns (fig, gs, ax_stft, ax_pywt, ax_npwt).
    """
    n_total = n_top_rows + 3

    fig = plt.figure(figsize=(style.FIGURE_WIDTH, style.ROW_HEIGHT * n_total))
    fig.suptitle(title, fontsize=style.SUPTITLE_FONT_SIZE, y=style.SUPTITLE_Y)
    if subtitle:
        fig.text(0.5, style.SUBTITLE_Y, subtitle,
                 ha='center', fontsize=style.SUBTITLE_FONT_SIZE,
                 color=style.SUBTITLE_COLOR)
    gs = gridspec.GridSpec(n_total, 1, figure=fig,
                           height_ratios=[1] * n_total, hspace=style.HSPACE)
    fig.subplots_adjust(left=style.LEFT_MARGIN, right=style.RIGHT_MARGIN,
                        bottom=style.BOTTOM_MARGIN, top=style.TOP_MARGIN)

    ax_stft = fig.add_subplot(gs[n_top_rows])
    ax_pywt = fig.add_subplot(gs[n_top_rows + 1], sharex=ax_stft, sharey=ax_stft)
    ax_npwt = fig.add_subplot(gs[n_top_rows + 2], sharex=ax_stft, sharey=ax_stft)

    return fig, gs, ax_stft, ax_pywt, ax_npwt


def render_top_row(fig, gs, idx, row, ax_stft, *,
                   t_audio, y_min, y_max, cwt_freqs, duration_s,
                   ytick_bins, ytick_labels):
    """Render one top row (waveform / freq_line / image) into gs[idx]."""
    rtype = row["type"]
    n_cwt_freqs = len(cwt_freqs)

    if rtype == "waveform":
        ax = fig.add_subplot(gs[idx], sharex=ax_stft)
        ax.set_facecolor(style.BG_COLOR)
        ax.fill_between(t_audio, y_min, y_max,
                        color=style.WAVEFORM_COLOR, alpha=style.WAVEFORM_ALPHA)
        ax.set_ylim([np.min([y_min, y_max]), np.max([y_min, y_max])])
        ax.set_xlim([0, duration_s])
        ax.margins(x=0, y=0)
        ax.set_title(row["title"], fontsize=style.TITLE_FONT_SIZE, loc="left")
        ax.tick_params(labelsize=style.TICK_LABEL_SIZE)
        plt.setp(ax.get_xticklabels(), visible=False)

    elif rtype == "freq_line":
        ax = fig.add_subplot(gs[idx], sharex=ax_stft, sharey=ax_stft)
        t_curve = np.linspace(0, duration_s, 500)
        f_curve = row['f0'] + (row['f1'] - row['f0']) * t_curve / duration_s
        bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
        ax.plot(t_curve, bin_curve, color=style.FREQ_LINE_COLOR,
                linewidth=style.FREQ_LINE_WIDTH)
        ax.set_facecolor(style.BG_COLOR)
        ax.set_title(row["title"], fontsize=style.TITLE_FONT_SIZE, loc="left")
        ax.set_yticks(ytick_bins)
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(labelsize=style.TICK_LABEL_SIZE)
        plt.setp(ax.get_xticklabels(), visible=False)

    elif rtype == "image":
        ax = fig.add_subplot(gs[idx])
        img_path = row.get("path")
        if img_path and os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img, aspect="auto", origin="upper")
            ax.set_title(row["title"], fontsize=style.TITLE_FONT_SIZE, loc="left")
            ax.axis("off")
        else:
            placeholder_ax(ax)
            ax.set_title(row["title"], fontsize=style.TITLE_FONT_SIZE, loc="left")


def render_spectrogram_row(ax, data, *, title, extent, vmax,
                           ytick_bins, ytick_labels,
                           is_bottom=False, cmap="inferno",
                           n_cwt_freqs=None, duration_s=None):
    """Render one spectrogram row using matplotlib imshow."""
    ax.imshow(data, cmap=cmap, aspect="auto", origin="lower",
              extent=extent, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=style.TITLE_FONT_SIZE, loc="left")
    ax.set_yticks(ytick_bins)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE)
    if is_bottom:
        ax.set_xlabel("Time (s)", fontsize=style.AXIS_LABEL_FONT_SIZE)
    else:
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
