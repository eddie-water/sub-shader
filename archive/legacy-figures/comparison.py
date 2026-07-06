"""
Method-vs-method comparison figures and timing tables.

To add a new comparison method, append an entry to COMPARISON_METHODS:
    {"name": "my_method", "fn": my_compute_function, "label": "Display Label"}
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from subshader.config import get_default_config, CWTConfig, ColorNormalizationConfig
from subshader.audio.reader import AudioReader
from subshader.dsp.cwt import CpuCWT, GpuCWT
from subshader.dsp.pywavelet import PywaveletCWT
from subshader.dsp.stft import STFT
from subshader.renderer.frame_buffer import CircularFrameBuffer

from utilities import (
    IMAGES_GENERATED_DIR,
    AUDIO_COMPARISON_1,
    AUDIO_COMPARISON_2,
    AUDIO_COMPARISON_3,
    DAW_IMAGE_COMPARISON_1,
    DAW_IMAGE_COMPARISON_2,
    DAW_IMAGE_COMPARISON_3,
    NUM_FRAMES,
    gpu_available,
    time_call,
    TimingAccumulator,
    live_progress,
    clear_progress,
    print_section_start,
    print_section_end,
    print_results_header,
    print_results_row,
    compute_freq_yticks,
    build_bouncing_chirp_chunks,
)
from utilities import style
from utilities.wav_export import export_signal_to_wav


# =============================================================================
# COMPARISON METHODS — extensible method config list
# =============================================================================

def _run_numpy_cwt(chunk, npwt):
    return npwt.process(chunk)


def _run_pywavelet(chunk, pywt):
    return pywt.process(chunk)


def _run_gpu_cwt(chunk, cpwt):
    return cpwt.process(chunk)


COMPARISON_METHODS = [
    {"name": "stft",      "label": "STFT"},
    {"name": "pywavelet", "label": "PyWavelet"},
    {"name": "numpy_cwt", "label": "SubShader (NumPy)"},
    {"name": "gpu_cwt",   "label": "SubShader (GPU)"},
]


# =============================================================================
# COMPARISON GRID (--comparison-grid)
# =============================================================================

def generate_comparison_grid(stub_pywt: bool = False, dpi: int = 0, comparison: bool = False):
    """
    Generate a 5x3 comparison grid figure.

    Columns: Bouncing Chirp (synthetic), Polyphonic (file), Musical (beltran first 20s)
    Rows:    Reference (waveform/inst-freq), DAW (placeholder), STFT, PyWavelet CWT, SubShader CWT

    Args:
        stub_pywt: Skip PyWavelet computation, use random stub spectrograms (faster)
        dpi: Output DPI. When > 0, file is named comparison_grid_{dpi}dpi.png.
             When 0 (default), file is named comparison_grid.png or comparison_grid_STUB_PYWT.png.
        comparison: Collect per-method timing stats and print after processing all columns.

    Output: assets/images/benchmarks/comparison_grid[_Ndpi].png
    """
    # GpuCWT import deferred to avoid unconditional GPU import at module load
    GPU_COMP_AVAILABLE = gpu_available()

    os.makedirs(IMAGES_GENERATED_DIR, exist_ok=True)

    config = get_default_config()
    chunk_size = config.chunk_size
    overlap_factor = config.overlap_factor

    # ── Signal definitions ────────────────────────────────────────────────────
    CHIRP_DURATION_S = 6.0
    chirp_n_frames = int(
        CHIRP_DURATION_S * config.sample_rate
        / (chunk_size * (1 - overlap_factor))
    )
    chirp_n_frames = max(chirp_n_frames, NUM_FRAMES)

    # DAW reference images (row 1) — placeholders until user generates Edison screenshots
    signal_specs = [
        {"label": "Bouncing Chirp", "type": "chirp", "daw_image": DAW_IMAGE_COMPARISON_1},
        {"label": "MIDI Sine Waves", "type": "file", "path": AUDIO_COMPARISON_2, "daw_image": DAW_IMAGE_COMPARISON_2},
        {"label": "Beltran SoundCloud Rip (4 Bars)", "type": "file", "path": AUDIO_COMPARISON_3, "daw_image": DAW_IMAGE_COMPARISON_3},
    ]

    # ── DSP result containers (per-column) ───────────────────────────────────
    column_data = []
    if comparison:
        all_timings = {}  # {signal_label: TimingAccumulator}

    for col_idx, spec in enumerate(signal_specs):
        print_section_start(f"Processing column {col_idx + 1}/3: {spec['label']}")

        if spec["type"] == "chirp":
            sr = config.sample_rate
            chunks, raw_waveform, chirp_inst_freq, _ = build_bouncing_chirp_chunks(
                sr=int(sr), chunk_size=chunk_size, overlap_factor=overlap_factor,
                n_frames=chirp_n_frames,
            )
            n_frames = len(chunks)
            chunk_iter = iter(chunks)
            raw_sr = sr
            # Export bouncing chirp as WAV so it can be dragged into FL Studio / Edison
            chirp_wav_path = AUDIO_COMPARISON_1
            export_signal_to_wav(raw_waveform, int(sr), chirp_wav_path)
            col_config = config
        else:
            audio_path = spec["path"]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            col_config = CWTConfig(
                file_path=audio_path,
                chunk_size=chunk_size,
                overlap_factor=overlap_factor,
            )
            reader = AudioReader(col_config)
            sr = col_config.sample_rate
            raw_waveform = reader.get_entire_audio()
            hop = int(chunk_size * (1 - overlap_factor))
            file_n_frames = int(len(raw_waveform) / hop)
            n_frames = spec.get("max_frames", file_n_frames)
            chunk_iter = None
            raw_sr = sr
            max_samples = int(n_frames * chunk_size * (1 - overlap_factor) + chunk_size)
            if len(raw_waveform) > max_samples:
                raw_waveform = raw_waveform[:max_samples]

        npwt = CpuCWT(col_config)
        stft = STFT(col_config)
        cpwt = None
        if GPU_COMP_AVAILABLE and comparison:
            cpwt = GpuCWT(col_config)
        if not stub_pywt:
            pywt = PywaveletCWT(col_config)
            cwt_freqs = pywt.freqs
        else:
            cwt_freqs = npwt.freqs

        n_cwt_freqs = len(cwt_freqs)
        stft_target_w = npwt.output_n

        color_norm = ColorNormalizationConfig()
        stft_buf = CircularFrameBuffer(
            frame_shape=(n_cwt_freqs, stft_target_w),
            num_frames=n_frames,
            color_norm_config=color_norm,
        )
        if not stub_pywt:
            pywt_buf = CircularFrameBuffer(
                frame_shape=pywt.get_output_shape(),
                num_frames=n_frames,
                color_norm_config=color_norm,
            )
        npwt_buf = CircularFrameBuffer(
            frame_shape=npwt.get_output_shape(),
            num_frames=n_frames,
            color_norm_config=color_norm,
        )

        if comparison:
            methods = ["STFT", "SubShader (CPU)"]
            if not stub_pywt:
                methods.insert(1, "PyWavelet")
            if GPU_COMP_AVAILABLE:
                methods.append("SubShader (GPU)")
            acc = TimingAccumulator(n_frames, methods)

        frames_processed = 0
        for _ in range(n_frames):
            chunk = next(chunk_iter, None) if chunk_iter is not None else reader.get_chunk()
            if chunk is None:
                break

            if comparison:
                stft_log, acc["STFT"][acc.current_idx] = time_call(stft.process, chunk)
            else:
                stft_log = stft.process(chunk)
            stft_buf.push_frame(stft_log)

            if not stub_pywt:
                if comparison:
                    pywt_out, acc["PyWavelet"][acc.current_idx] = time_call(pywt.process, chunk)
                else:
                    pywt_out, _ = time_call(pywt.process, chunk)
                pywt_buf.push_frame(pywt_out)

            if comparison:
                npwt_out, acc["SubShader (CPU)"][acc.current_idx] = time_call(npwt.process, chunk)
            else:
                npwt_out, _ = time_call(npwt.process, chunk)
            npwt_buf.push_frame(npwt_out)

            if cpwt is not None and comparison:
                _, acc["SubShader (GPU)"][acc.current_idx] = time_call(cpwt.process, chunk)

            if comparison:
                acc.advance()

            frames_processed += 1
            live_progress(frames_processed, n_frames)

        clear_progress()
        print_section_end()

        frame_w = npwt_buf.width
        actual_w = frames_processed * frame_w
        stft_spec = stft_buf.get_flattened_buffer()[:, -actual_w:]
        npwt_spec = npwt_buf.get_flattened_buffer()[:, -actual_w:]

        if not stub_pywt:
            pywt_spec = pywt_buf.get_flattened_buffer()[:, -actual_w:]
            pywt_vmax = pywt_buf.get_intensity_max()
        else:
            pywt_spec = np.random.rand(n_cwt_freqs, actual_w).astype(np.float32)
            pywt_vmax = pywt_spec.max()

        hop_size = int(chunk_size * (1 - overlap_factor))
        duration_s = ((frames_processed - 1) * hop_size + chunk_size) / sr
        ytick_bins, ytick_labels = compute_freq_yticks(cwt_freqs)

        # Trim waveform to match DSP duration
        waveform_samples = int(duration_s * raw_sr)
        waveform = raw_waveform[:waveform_samples]
        waveform_time = np.linspace(0, duration_s, len(waveform), endpoint=False)

        col_entry = {
            "label":        spec["label"],
            "stft":         stft_spec,
            "pywt":         pywt_spec,
            "npwt":         npwt_spec,
            "stft_vmax":    stft_buf.get_intensity_max(),
            "pywt_vmax":    pywt_vmax,
            "npwt_vmax":    npwt_buf.get_intensity_max(),
            "n_cwt_freqs":  n_cwt_freqs,
            "cwt_freqs":    cwt_freqs,
            "duration_s":   duration_s,
            "ytick_bins":   ytick_bins,
            "ytick_labels": ytick_labels,
            "waveform":     waveform,
            "waveform_time": waveform_time,
        }
        if spec["type"] == "chirp":
            # Use same time mapping as waveform (spans full duration_s)
            col_entry["chirp_inst_freq"] = chirp_inst_freq[:len(waveform)]
        column_data.append(col_entry)
        if comparison:
            acc.trim()
            all_timings[spec["label"]] = acc

    if comparison:
        print_section_start("Per-Method Timing Comparison")
        for signal_label, acc in all_timings.items():
            print(f"\n  {signal_label}:")
            print_results_header()
            for method in acc.methods:
                print_results_row(method, acc.arrays[method])
        print_section_end()

    # ── Build 5x4 figure (label col + 3 data cols) ─────────────────────────
    print_section_start("Rendering 5x3 comparison grid")

    GRID_ROWS = 5  # Reference + DAW + STFT + PyWavelet + SubShader
    GRID_DATA_COLS = 3
    pywt_label = "PyWavelet Stub" if stub_pywt else "PyWavelet"
    row_labels = ["Reference", "DAW", "STFT", pywt_label, "SubShader"]
    LABEL_RATIO = len(max(row_labels, key=len)) * style.LABEL_CHAR_WIDTH + style.LABEL_PAD
    fig = plt.figure(figsize=(style.GRID_FIGSIZE_W, style.GRID_FIGSIZE_H))
    gs = fig.add_gridspec(
        GRID_ROWS, GRID_DATA_COLS + 1,
        width_ratios=[LABEL_RATIO, 1, 1, 1],
        hspace=style.GRID_HSPACE, wspace=style.GRID_WSPACE,
    )

    # Build axes array: axes[row][col] where col 0 = label, 1-3 = data
    axes = []
    for r in range(GRID_ROWS):
        row_axes = []
        for c in range(GRID_DATA_COLS + 1):
            row_axes.append(fig.add_subplot(gs[r, c]))
        axes.append(row_axes)
    # Hide label column axes — labels placed after layout is finalized (below)
    for r in range(GRID_ROWS):
        axes[r][0].axis("off")

    for col_idx, (col, spec) in enumerate(zip(column_data, signal_specs)):
        dc = col_idx + 1  # data column (0 is label column)
        # ── Row 0: Reference — waveform time series / instantaneous freq ──
        ax_ref = axes[0][dc]

        if spec["type"] == "chirp":
            # Map inst_freq (Hz) → bin indices matching spectrogram y-axis
            cwt_freqs = col["cwt_freqs"]
            inst_freq_hz = col["chirp_inst_freq"]
            inst_freq_bins = np.interp(inst_freq_hz, cwt_freqs,
                                       np.arange(len(cwt_freqs)))
            ax_ref.plot(col["waveform_time"], inst_freq_bins,
                        color=style.GRID_WAVEFORM_COLOR, linewidth=1.5, alpha=0.9)
            ax_ref.set_ylim(0, col["n_cwt_freqs"])
            ax_ref.set_facecolor(style.BG_COLOR)
            ax_ref.grid(True, color="white", alpha=0.08, linewidth=0.5)
            # Use same freq ticks as spectrogram rows
            if dc == 1:
                ax_ref.set_yticks(col["ytick_bins"])
                ax_ref.set_yticklabels(col["ytick_labels"], fontsize=style.TICK_LABEL_SIZE)
            else:
                ax_ref.set_yticks([])
        else:
            # Audio files: waveform time series
            ax_ref.plot(col["waveform_time"], col["waveform"],
                        color=style.GRID_WAVEFORM_COLOR, linewidth=0.2, alpha=0.8)
            peak = max(np.abs(col["waveform"]).max(), 1e-6)
            ax_ref.set_ylim(-peak * 1.10, peak * 1.10)
            ax_ref.set_facecolor(style.BG_COLOR)
            ax_ref.grid(True, color="gray", alpha=0.15, linewidth=0.5)
            if dc != 1:
                ax_ref.set_yticks([])

        ax_ref.set_xlim(0, col["duration_s"])
        for spine in ax_ref.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.8)
        plt.setp(ax_ref.get_xticklabels(), visible=False)
        ax_ref.tick_params(axis="x", length=0)

        # ── Row 1: DAW — placeholder / reference image ───────────────────
        ax_daw = axes[1][dc]

        daw_path = spec.get("daw_image")
        if daw_path and os.path.exists(daw_path):
            daw_img = plt.imread(daw_path)
            ax_daw.imshow(daw_img, aspect="auto",
                          extent=[0, col["duration_s"], 0, col["n_cwt_freqs"]])
        else:
            ax_daw.set_facecolor("#2a2a2a")
            ax_daw.text(0.5, 0.5, "placeholder — generate in FL Studio",
                        transform=ax_daw.transAxes, ha="center", va="center",
                        fontsize=18, color="#666666", style="italic")
            ax_daw.set_ylim(0, col["n_cwt_freqs"])
        # Y-axis: freq ticks on first data column to match spectrogram rows
        if dc == 1:
            ax_daw.set_yticks(col["ytick_bins"])
            ax_daw.set_yticklabels(col["ytick_labels"], fontsize=style.TICK_LABEL_SIZE)
        else:
            ax_daw.set_yticks([])

        ax_daw.set_xlim(0, col["duration_s"])
        for spine in ax_daw.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.8)
        plt.setp(ax_daw.get_xticklabels(), visible=False)
        ax_daw.tick_params(axis="x", length=0)

        # ── Rows 2-4: STFT, PyWavelet, SubShader ─────────────────────────
        spec_rows = [
            (col["stft"], col["stft_vmax"]),
            (col["pywt"], col["pywt_vmax"]),
            (col["npwt"], col["npwt_vmax"]),
        ]
        for row_idx, (spec_data, vmax) in enumerate(spec_rows):
            ax = axes[row_idx + 2][dc]
            extent = [0, col["duration_s"], 0, col["n_cwt_freqs"]]
            ax.imshow(
                spec_data, cmap=style.GRID_CMAP, aspect="auto",
                origin="lower", extent=extent, vmin=0, vmax=vmax,
            )
            ax.grid(True, color="white", alpha=0.08, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.8)

            # Y-axis: "Freq (Hz)" on first data column only
            if dc == 1:
                ax.set_yticks(col["ytick_bins"])
                ax.set_yticklabels(col["ytick_labels"], fontsize=style.TICK_LABEL_SIZE)
                if row_idx == 0:  # middle row of grid (STFT = row 2 of 5)
                    ax.set_ylabel("Freq (Hz)", fontsize=style.AXIS_LABEL_FONT_SIZE, labelpad=4)
            else:
                ax.set_yticks([])

            # X-axis: "Time (s)" on bottom row only
            if row_idx == 2:  # bottom spectrogram row (SubShader)
                ax.tick_params(axis="x", labelsize=style.TICK_LABEL_SIZE)
                if dc == 2:  # center data column
                    ax.set_xlabel("Time (s)", fontsize=style.AXIS_LABEL_FONT_SIZE, labelpad=4)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis="x", length=0)

    # Column labels across the top (data columns only) — use GRID_TITLE_PAD for breathing room
    for col_idx, col in enumerate(column_data):
        axes[0][col_idx + 1].set_title(
            col["label"], fontsize=style.LABEL_FONT_SIZE, fontweight="bold",
            pad=style.GRID_TITLE_PAD,
        )

    # Scale horizontal margins by aspect ratio so they match vertical margins in inches
    h_margin = style.GRID_MARGIN * style.GRID_FIGSIZE_H / style.GRID_FIGSIZE_W
    plt.subplots_adjust(left=h_margin, right=1 - h_margin,
                        top=1 - style.GRID_MARGIN, bottom=style.GRID_MARGIN)

    # Place row labels after layout is finalized so we can query rendered positions.
    # Draw once to resolve tick label geometry, then center labels between the
    # figure left edge and the y-axis decorations' left extent.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    # Use the STFT row (has yticks + ylabel) to find where y-axis labels start
    yaxis_bbox = axes[2][1].yaxis.get_tightbbox(rend)
    yaxis_left_frac = yaxis_bbox.x0 / fig_width_px
    label_x = (h_margin + yaxis_left_frac) / 2

    for r, label in enumerate(row_labels):
        row_pos = axes[r][1].get_position()
        label_y = row_pos.y0 + row_pos.height / 2
        fig.text(label_x, label_y, label, ha="center", va="center",
                 fontweight="bold", fontsize=style.LABEL_FONT_SIZE, color="black")

    # When --dpi is passed explicitly, the output is always named by DPI (no stub suffix).
    # The stub suffix only applies to the default 200 DPI path without an explicit DPI request.
    # Caller uses dpi=0 as sentinel to mean "use default naming".
    if dpi > 0:
        base_name = f"comparison_grid_{dpi}dpi.png"
    elif stub_pywt:
        base_name = "comparison_grid_STUB_PYWT.png"
    else:
        base_name = "comparison_grid.png"
    out_path = os.path.join(IMAGES_GENERATED_DIR, base_name)
    save_dpi = dpi if dpi > 0 else 200
    fig.savefig(out_path, dpi=save_dpi)
    plt.close(fig)
    print(f"Saved -> {out_path}")
    print_section_end()


# =============================================================================
# TIMING BAR CHART (--timing-chart)
# =============================================================================

def generate_timing_bar_chart(dpi: int = 200, num_frames: int = NUM_FRAMES):
    """
    Run STFT, PyWavelet, and CpuCWT timing over num_frames frames, then
    save a bar chart to assets/images/benchmarks/timing_bar_chart.png.

    Shows average time with min/max error bars for each pipeline component.
    """
    print_section_start("Timing Bar Chart")

    config = CWTConfig(file_path=AUDIO_COMPARISON_2)
    reader = AudioReader(config)

    pywt = PywaveletCWT(config)
    npwt = CpuCWT(config)
    stft = STFT(config)

    methods = ["AudioReader.get_chunk()", "STFT.process()", "PywaveletCWT.process()", "CpuCWT.process()"]
    acc = TimingAccumulator(num_frames, methods)

    for i in range(num_frames):
        chunk, acc["AudioReader.get_chunk()"][i] = time_call(reader.get_chunk)
        if chunk is None:
            acc.current_idx = i
            break

        _, acc["STFT.process()"][i] = time_call(stft.process, chunk)
        _, acc["PywaveletCWT.process()"][i] = time_call(pywt.process, chunk)
        _, acc["CpuCWT.process()"][i] = time_call(npwt.process, chunk)
        acc.current_idx = i + 1
        live_progress(acc.current_idx, num_frames)

    acc.trim()
    clear_progress()

    stats = acc.compute_stats()

    avgs   = [stats[m]["avg_ms"] for m in methods]
    mins   = [stats[m]["min_ms"] for m in methods]
    maxs   = [stats[m]["max_ms"] for m in methods]
    yerr   = [
        [avgs[j] - mins[j] for j in range(len(methods))],
        [maxs[j] - avgs[j] for j in range(len(methods))],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    ax.bar(x, avgs, yerr=yerr, capsize=4, color="#5588bb", error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("SubShader Component Timing (avg \u00b1 min/max)", fontsize=13)
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out_path = os.path.join(IMAGES_GENERATED_DIR, "timing_bar_chart.png")
    os.makedirs(IMAGES_GENERATED_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    print_section_end()
