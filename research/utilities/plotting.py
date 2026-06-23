"""
Reusable rendering helpers for the benchmark suite.

All figure layout, spectrogram rendering, and top-row rendering logic
lives here so that stub_layouts() and _generate_comparison_figure()
share a single code path. Visual constants are imported from style.py.
"""

import math
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


# =============================================================================
# ATOMIC PLOTTERS — single-Axes renderers for standalone figure layouts
# =============================================================================

def plot_time_series(ax, signal, sr, *,
                     color=style.WAVEFORM_COLOR, alpha=style.WAVEFORM_ALPHA):
    """Plot raw audio waveform vs time on a single Axes."""
    t = np.arange(len(signal)) / sr
    peak = max(float(np.abs(signal).max()), 1e-6)
    ax.fill_between(t, signal, 0, color=color, alpha=alpha)
    ax.set_xlim(0, t[-1] if len(t) > 0 else 0)
    ax.set_ylim(-peak * style.WAVEFORM_YLIM_PADDING,
                peak * style.WAVEFORM_YLIM_PADDING)
    ax.set_facecolor(style.BG_COLOR)
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE, colors=style.TICK_LABEL_COLOR)


def plot_inst_freq(ax, inst_freq_hz, t, cwt_freqs, *,
                   color=style.GRID_WAVEFORM_COLOR,
                   linewidth=style.INST_FREQ_LINEWIDTH,
                   alpha=style.INST_FREQ_ALPHA):
    """Plot instantaneous frequency curve in bin-space matching a spectrogram y-axis."""
    bins = np.interp(inst_freq_hz, cwt_freqs, np.arange(len(cwt_freqs)))
    ax.plot(t, bins, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlim(0, t[-1] if len(t) > 0 else 0)
    ax.set_ylim(0, len(cwt_freqs))
    ax.set_facecolor(style.BG_COLOR)
    ax.grid(True, color=style.AXIS_GRID_COLOR,
            alpha=style.AXIS_GRID_ALPHA, linewidth=style.AXIS_GRID_LINEWIDTH)
    ytick_bins, ytick_labels = compute_freq_yticks(cwt_freqs)
    ax.set_yticks(ytick_bins)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE, colors=style.TICK_LABEL_COLOR)


def plot_fft_magnitude(ax, signal, sr, *,
                       log_x=True, log_y=True,
                       color=style.GRID_WAVEFORM_COLOR,
                       linewidth=style.FFT_LINEWIDTH):
    """Plot one-sided FFT magnitude spectrum on a single Axes."""
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    ax.plot(freqs[1:], spec[1:], color=color, linewidth=linewidth)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_facecolor(style.BG_COLOR)
    ax.grid(True, color=style.AXIS_GRID_COLOR,
            alpha=style.AXIS_GRID_ALPHA, linewidth=style.AXIS_GRID_LINEWIDTH,
            which="both")
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE, colors=style.TICK_LABEL_COLOR)


def plot_stft_spectrogram(ax, signal, sr, *,
                          nperseg=None, noverlap=None,
                          vmax=None, cmap=style.GRID_CMAP,
                          freq_lim_hz=None, duration_s=None):
    """Plot STFT magnitude spectrogram on log-frequency y-axis.

    Mirrors plot_cwt_spectrogram visually so the two can be compared in a
    side-by-side figure. Default nperseg adapts to signal length: largest
    power of 2 <= len(signal)/4, capped at 1024.

    When `duration_s` is set, x is clamped to [0, duration_s] so that the
    panel aligns vertically with time-series and CWT panels (scipy zero-pads
    + pcolormesh half-cell extension push the last bin past signal end).
    When `freq_lim_hz` is set, y-ticks are placed at the standard
    log-octave grid (20/200/2k/20k) for parity with plot_cwt_spectrogram.
    """
    from scipy.signal import stft as scipy_stft

    n = len(signal)
    if nperseg is None:
        nperseg = min(1024, max(64, 1 << int(math.log2(max(64, n // 4)))))
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Zxx = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    if vmax is None:
        vmax = float(np.percentile(mag, style.VMAX_PERCENTILE))

    # Drop DC bin so log y-scale is well-defined.
    f_plot = f[1:]
    mag_plot = mag[1:]

    ax.pcolormesh(t, f_plot, mag_plot, cmap=cmap,
                  vmin=0, vmax=vmax, shading="auto")
    ax.set_yscale("log")
    if freq_lim_hz is not None:
        ax.set_ylim(freq_lim_hz[0], freq_lim_hz[1])
        f_min, f_max = freq_lim_hz
        ytick_freqs, ytick_labels = [], []
        for f_tick in (20, 200, 2000, 20000):
            if f_min <= f_tick <= f_max:
                ytick_freqs.append(f_tick)
                ytick_labels.append(f'{f_tick/1000:.0f}k' if f_tick >= 1000 else f'{int(f_tick)}')
        ax.set_yticks(ytick_freqs)
        ax.set_yticklabels(ytick_labels)
        ax.minorticks_off()
    if duration_s is not None:
        ax.set_xlim(0, duration_s)
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE, colors=style.TICK_LABEL_COLOR)


def plot_cwt_spectrogram(ax, cwt_data, duration_s, cwt_freqs, *,
                         vmax=None, cmap=style.GRID_CMAP):
    """Plot 2D CWT magnitude spectrogram with consistent colormap and log-freq y-axis."""
    if vmax is None:
        vmax = float(np.percentile(cwt_data, style.VMAX_PERCENTILE))
    n_freqs = len(cwt_freqs)
    extent = [0, duration_s, 0, n_freqs]
    ax.imshow(cwt_data, cmap=cmap, aspect="auto", origin="lower",
              extent=extent, vmin=0, vmax=vmax)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, n_freqs)
    ytick_bins, ytick_labels = compute_freq_yticks(cwt_freqs)
    ax.set_yticks(ytick_bins)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(labelsize=style.TICK_LABEL_SIZE, colors=style.TICK_LABEL_COLOR)


def render_image_row(ax, img_path, *,
                     fallback_label="reference not available", extent=None):
    """Render image into ax, falling back to placeholder if file missing.

    Atomic version usable from both row-based renderers and standalone Axes.
    """
    if img_path and os.path.exists(img_path):
        img = plt.imread(img_path)
        if extent is not None:
            ax.imshow(img, aspect="auto", extent=extent)
        else:
            ax.imshow(img, aspect="auto", origin="upper")
    else:
        placeholder_ax(ax, label=fallback_label)


# =============================================================================
# GRID SCAFFOLD — standalone N×M layout (use create_figure_scaffold for the
# stacked-rows-with-shared-spectrogram pattern instead)
# =============================================================================

def create_grid_scaffold(n_rows, n_cols, *,
                         figsize=None, hspace=None, wspace=None, dpi=None,
                         width_ratios=None, height_ratios=None):
    """Create an N×M grid of axes with consistent dark-background styling.

    Returns (fig, axes) where axes[r][c] is the Axes at row r, col c.
    All axes use style.BG_COLOR background and gray spines.

    `width_ratios` / `height_ratios` are forwarded to `add_gridspec` and
    let callers carve out a narrower label column or a taller header row
    (mirrors the comparison-grid pattern in research/comparison.py).
    """
    if figsize is None:
        figsize = (style.GRID_FIGSIZE_W, style.GRID_FIGSIZE_H)
    if hspace is None:
        hspace = style.GRID_HSPACE
    if wspace is None:
        wspace = style.GRID_WSPACE
    if dpi is None:
        dpi = style.DEFAULT_DPI

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(style.BG_COLOR)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=hspace, wspace=wspace,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios)
    axes = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(style.BG_COLOR)
            for spine in ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.SPINE_LINEWIDTH)
            row.append(ax)
        axes.append(row)
    return fig, axes


# =============================================================================
# FOUNDATION-FIGURE SCAFFOLDS — small, square-panel layouts for the §2 vector
# figures (and future polar / sequence figures). Distinct from the comparison
# grid scaffold above: smaller per-panel size, square aspect, one shared
# suptitle. Use create_grid_scaffold for spectrogram-sized panels.
# =============================================================================

def create_panel_row(n_panels, *, panel_size=None, height=None,
                     suptitle=None, suptitle_y=0.98, dpi=None):
    """1×N row of square axes for vector / polar / small-figure layouts.

    Returns (fig, axes) where axes is a flat list of length n_panels.
    All axes share the dark background and gray spines from style.py.
    """
    if panel_size is None:
        panel_size = style.VECTOR_FIGSIZE_PER_PANEL
    if height is None:
        height = style.VECTOR_FIGSIZE_HEIGHT
    if dpi is None:
        dpi = style.DEFAULT_DPI

    fig = plt.figure(figsize=(panel_size * n_panels, height), dpi=dpi)
    fig.patch.set_facecolor(style.BG_COLOR)
    if suptitle is not None:
        fig.suptitle(suptitle, color=style.TICK_LABEL_COLOR,
                     fontsize=style.SUBTITLE_FONT_SIZE, y=suptitle_y)
    gs = fig.add_gridspec(1, n_panels, wspace=style.GRID_WSPACE)
    axes = []
    for c in range(n_panels):
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor(style.BG_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(style.SPINE_COLOR)
            spine.set_linewidth(style.SPINE_LINEWIDTH)
        axes.append(ax)
    return fig, axes


# =============================================================================
# VECTOR PLOTTERS — atomic helpers for §2 foundation figures. All visual
# constants routed through style.py.
# =============================================================================

def setup_vector_axes(ax, *, lim=None, panel_title=None, result_text=None,
                       show_border=True, axis_style="line", axis_labels=False,
                       x_color=None, y_color=None,
                       axis_alpha=None, axis_linewidth=None):
    """Square axes with origin crosshair, no ticks, equal aspect.

    `panel_title` lands above the panel (left-aligned). `result_text` lands
    bottom-center as the dot-product result annotation (e.g. `a · b = +1.0`).

    `show_border=False` hides the four spines for a clean axes-only look.
    `axis_style="arrow"` replaces axhline/axvline with +x and +y arrows from
    origin to axis limit (negative axes hidden).
    `axis_labels=True` places "x" in Q-IV and "y" in Q-I near the limits.

    `x_color` / `y_color` override the per-axis crosshair color (default both
    use style.VECTOR_AXIS_COLOR). `axis_alpha` and `axis_linewidth` adjust
    the crosshair weight uniformly. Useful when the figure assigns each axis
    a per-dimension hue (e.g. orange = x, purple = y).
    """
    if lim is None:
        lim = style.VECTOR_DEFAULT_LIM
    if x_color is None:
        x_color = style.VECTOR_AXIS_COLOR
    if y_color is None:
        y_color = style.VECTOR_AXIS_COLOR
    if axis_alpha is None:
        axis_alpha = style.VECTOR_AXIS_ALPHA
    if axis_linewidth is None:
        axis_linewidth = 0.8

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    if not show_border:
        for spine in ax.spines.values():
            spine.set_visible(False)

    inset = style.VECTOR_AXIS_ARROW_INSET
    if axis_style == "arrow":
        for (tail, head), color in (
            (((-lim + inset, 0.0), (lim - inset, 0.0)), x_color),
            (((0.0, -lim + inset), (0.0, lim - inset)), y_color),
        ):
            ax.annotate(
                "", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="<|-|>",
                                color=color,
                                alpha=axis_alpha,
                                linewidth=max(axis_linewidth, 0.9),
                                mutation_scale=14,
                                shrinkA=0, shrinkB=0),
                zorder=0,
            )
    else:
        ax.axhline(0, color=x_color,
                   alpha=axis_alpha, linewidth=axis_linewidth, zorder=0)
        ax.axvline(0, color=y_color,
                   alpha=axis_alpha, linewidth=axis_linewidth, zorder=0)

    if axis_labels:
        offset = style.VECTOR_AXIS_LABEL_OFFSET
        ax.text(lim - inset, -offset, "x",
                color=style.VECTOR_AXIS_COLOR, alpha=0.9,
                fontsize=style.VECTOR_AXIS_LABEL_SIZE,
                ha="center", va="top", style="italic")
        ax.text(offset, lim - inset, "y",
                color=style.VECTOR_AXIS_COLOR, alpha=0.9,
                fontsize=style.VECTOR_AXIS_LABEL_SIZE,
                ha="left", va="center", style="italic")

    if panel_title is not None:
        ax.set_title(panel_title, color=style.TICK_LABEL_COLOR,
                     fontsize=style.VECTOR_PANEL_TITLE_SIZE, loc="center", pad=8)
    if result_text is not None:
        ax.text(0.5, -0.08, result_text, transform=ax.transAxes,
                ha="center", va="top",
                fontsize=style.VECTOR_PANEL_RESULT_SIZE,
                color=style.VECTOR_PANEL_RESULT_COLOR,
                family="monospace")


def plot_vector(ax, vec, *, origin=(0.0, 0.0), color=None, label=None,
                linewidth=None, alpha=1.0, linestyle="-",
                label_offset=None, zorder=2):
    """Draw an arrow from `origin` along `vec` with optional label at the tip.

    Uses ax.annotate so the head scales with linewidth and renders crisply.
    Label is placed slightly past the tip in the direction of `vec`.
    """
    if color is None:
        color = style.VECTOR_NEUTRAL_COLOR
    if linewidth is None:
        linewidth = style.VECTOR_LINEWIDTH
    if label_offset is None:
        label_offset = style.VECTOR_LABEL_OFFSET

    ox, oy = origin
    vx, vy = vec
    tip = (ox + vx, oy + vy)

    ax.annotate(
        "", xy=tip, xytext=(ox, oy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            mutation_scale=18,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder,
    )

    if label is not None:
        norm = math.hypot(vx, vy)
        if norm > 1e-9:
            lx = tip[0] + (vx / norm) * label_offset
            ly = tip[1] + (vy / norm) * label_offset
        else:
            lx, ly = tip
        ax.text(lx, ly, label, color=color, alpha=alpha,
                fontsize=style.VECTOR_LABEL_FONT_SIZE,
                fontweight="bold",
                ha="center", va="center", zorder=zorder + 1)


def plot_projection(ax, a, b, *,
                    a_color=None, b_color=None, proj_color=None,
                    show_dropline=True):
    """Draw vectors a, b, and the scalar projection of b onto a.

    The projection vector is `((a·b)/(a·a)) * a` — a vector along a whose tip
    is the foot of the perpendicular from b. A dashed dropline from b's tip to
    the projection foot makes the right-angle decomposition visible.
    """
    if a_color is None:
        a_color = style.VECTOR_A_COLOR
    if b_color is None:
        b_color = style.VECTOR_B_COLOR
    if proj_color is None:
        proj_color = style.VECTOR_PROJ_COLOR

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_dot_a = float(np.dot(a_arr, a_arr))
    if a_dot_a < 1e-12:
        plot_vector(ax, b, color=b_color, label="b")
        return
    scale = float(np.dot(a_arr, b_arr)) / a_dot_a
    foot = scale * a_arr  # tip of the projection vector

    plot_vector(ax, a, color=a_color, label="a", zorder=2)
    plot_vector(ax, b, color=b_color, label="b", zorder=3)
    # Projection drawn on top of a so the orange "shadow" stays visible even
    # when foot ≈ a; slight upward offset keeps the two arrowheads distinct.
    plot_vector(ax, tuple(foot), origin=(0.0, -0.04),
                color=proj_color,
                linewidth=style.VECTOR_LINEWIDTH,
                alpha=0.95, zorder=4)

    if show_dropline:
        ax.plot([foot[0], b_arr[0]], [foot[1], b_arr[1]],
                color=style.VECTOR_DROPLINE_COLOR,
                alpha=style.VECTOR_DROPLINE_ALPHA,
                linewidth=1.2, linestyle="--", zorder=1)


# =============================================================================
# CWT WRAPPER — single-shot full-signal computation
# =============================================================================

def compute_full_cwt(signal, sr, *,
                     chunk_size=None, overlap_factor=None,
                     root_note_hz=None, num_octaves=None,
                     num_cycles=None):
    """Run a full audio signal through GpuCWT, return stitched 2D spectrogram + freqs.

    Wraps the chunking pattern from comparison.py. Requires CUDA-capable GPU.
    Returns (cwt_data, cwt_freqs, start_sample, end_sample) where cwt_data
    is (n_freqs, n_time_bins) and [start_sample, end_sample) is the half-open
    range of input samples actually represented by the CWT. Callers should
    slice companion arrays (waveform, inst_freq) to this window so all four
    plots share an aligned x-axis.

    Short signals: pass `root_note_hz` matched to the signal's lowest frequency
    of interest. The wavelet at the lowest frequency has time support
    ~num_cycles/root_note_hz seconds; the signal must be at least that long or
    a ValueError is raised so callers can shorten the freq range or lengthen
    the signal. Lower `num_cycles` to trade frequency resolution for time
    resolution at the low end.
    """
    from subshader.config import CWTConfig
    from subshader.dsp.cwt import CpuCWT, GpuCWT
    from subshader.renderer.frame_buffer import CircularFrameBuffer
    from subshader.utils import gpu_available

    # Bypass get_default_config() to skip its audio-file existence check —
    # figure generation doesn't load any audio file. The dataclass default
    # field values are all we need for the CWT parameters (chunk_size,
    # root_note_hz, num_octaves, num_cycles, overlap_factor), and the caller
    # overrides each one below where it's been explicitly passed.
    config = CWTConfig()
    if root_note_hz is not None:
        config.root_note_hz = root_note_hz
    if num_octaves is not None:
        config.num_octaves = num_octaves
    if num_cycles is not None:
        config.num_cycles = num_cycles

    # Default to high overlap for figure use: each frame's CWT output only
    # represents the trailing hop_size samples of its chunk (extract_hop_center),
    # so smaller hop = more time bins per second of signal.
    if overlap_factor is None:
        overlap_factor = 0.875
    config.overlap_factor = overlap_factor

    # Lowest-frequency wavelet's time support sets the floor on chunk_size.
    min_chunk_for_wavelet = int(math.ceil(config.num_cycles * sr / config.root_note_hz))

    if len(signal) < min_chunk_for_wavelet:
        raise ValueError(
            f"Signal too short for requested frequency range: "
            f"len(signal)={len(signal)} samples ({len(signal)/sr*1000:.1f} ms), "
            f"but root_note_hz={config.root_note_hz} Hz with num_cycles={config.num_cycles} "
            f"requires at least {min_chunk_for_wavelet} samples "
            f"({min_chunk_for_wavelet/sr*1000:.1f} ms). "
            f"Either raise root_note_hz, lower num_cycles, or lengthen the signal."
        )

    if chunk_size is None:
        # Aim for ~2x the widest wavelet's time support so the lowest-freq
        # kernel has fully-overlapped headroom inside the chunk (avoids
        # chunk-boundary mosaic in the spectrogram). Cap by signal length and
        # the default real-time chunk size; never go below min_chunk_for_wavelet.
        desired = 2 * min_chunk_for_wavelet
        chunk_size = min(config.chunk_size, desired, len(signal))
        chunk_size = max(chunk_size, min_chunk_for_wavelet)
    config.chunk_size = chunk_size

    cs = config.chunk_size
    of = config.overlap_factor
    hop = max(1, int(cs * (1 - of)))
    n_frames = max(1, (len(signal) - cs) // hop + 1)

    chunks = [signal[i * hop : i * hop + cs] for i in range(n_frames)]
    chunks = [c for c in chunks if len(c) == cs]
    if not chunks:
        raise ValueError(
            f"Signal too short for chunk_size={cs}, hop={hop}: "
            f"len(signal)={len(signal)}"
        )

    cwt = GpuCWT(config) if gpu_available() else CpuCWT(config)
    buf = CircularFrameBuffer(frame_shape=cwt.get_output_shape(),
                              num_frames=len(chunks))
    for chunk in chunks:
        buf.push_frame(cwt.process(chunk))

    # Each frame's CWT output represents the trailing `new_width` samples of
    # the *reliable region* (a centered slice of the chunk), not the trailing
    # hop_size samples of the full chunk. So map from the reliable_slice that
    # the CWT actually used:
    slice_end = cwt.reliable_slice.stop
    keep = cwt.reliable_slice.stop - cwt.reliable_slice.start
    new_width = min(hop, keep)
    start_sample = slice_end - new_width
    end_sample = (len(chunks) - 1) * hop + slice_end
    return buf.get_flattened_buffer(), cwt.freqs, start_sample, end_sample
