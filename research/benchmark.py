"""
SubShader Benchmark Suite.

Modes:
  (default)      Run SubShader with default config
  --timing       Run SubShader with live per-stage timing instrumentation
  --figures      Generate the 3 README comparison PNGs
  --unit-tests   Run unit tests (NumPy vs CuPy verification, etc.)
  --all          All of the above
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft as scipy_stft, resample as scipy_resample

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("[benchmark] seaborn not installed — --seaborn flag will be ignored.\n")

from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer

from benchmark_utilities import (
    live_row, live_progress,
    print_figure_header, print_figure_results,
    print_results_table, print_header, print_total,
    compute_timing_stats,
)

# =============================================================================
# GPU DETECTION
# =============================================================================

def _gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False

GPU_AVAILABLE = _gpu_available()

if not GPU_AVAILABLE:
    print("[benchmark] No GPU detected — CuPy benchmarks will be skipped, "
          "figures will use NumPyWavelet as fallback.\n")

# =============================================================================
# CONSTANTS
# =============================================================================

BENCHMARKS_DIR   = "assets/images/benchmarks"
BENCHMARKS_SEABORN_DIR = "assets/images/benchmarks/seaborn"
HEATMAP_MAX_ROWS = 128
HEATMAP_MAX_COLS = 512

AUDIO_DEFAULT    = "assets/audio/daw/a2a3_a4_minor_scale.wav"
AUDIO_CHIRP      = "assets/audio/daw/chirp_beat.wav"
AUDIO_POLYPHONIC = "assets/audio/daw/polyphonic_audio_example.wav"
AUDIO_MUSICAL    = "assets/audio/daw/musical_audio_example.wav"

MIDI_POLYPHONIC  = "assets/images/polyphonic-signal-example-midi-notes.png"
DAW_POLYPHONIC   = "assets/images/polyphonic-signal-example-edison-spectrogram.png"
DAW_MUSICAL      = "assets/images/musical-signal-example-edison-spectrogram.png"

STFT_NPERSEG     = 1024
NUM_FRAMES       = 128   # frames to accumulate for figure snapshots

CHIRP_F0         = 200    # Hz — chirp start frequency
CHIRP_F1         = 20_000  # Hz — chirp end frequency


# =============================================================================
# TIMED SUBSHADER (--timing)
# =============================================================================

class TimedSubShader:
    """
    Run the SubShader pipeline with live timing instrumentation.

    Mirrors the real SubShader pipeline (AudioInput → CWT → buffer) but
    wraps each stage with perf_counter and displays a live-updating table.
    """

    def __init__(self, audio_path: str = AUDIO_DEFAULT, num_frames: int = NUM_FRAMES):
        self.audio_path = audio_path
        self.num_frames = num_frames

    def run(self):
        """Run the timed pipeline: init timing, then runtime loop timing."""
        print("\n=== SubShader Timed Pipeline ===\n")

        # ----- Init timing -----
        print("Init stage:")
        print_header()

        config = get_default_config()
        config.audio.file_path = self.audio_path

        init_times = np.empty(1)

        t0 = time.perf_counter()
        audio_input = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )
        init_times[0] = (time.perf_counter() - t0) * 1000.0
        live_row("AudioInput", 1, 1, init_times)

        sr = audio_input.get_sample_rate()

        init_times = np.empty(1)
        t0 = time.perf_counter()
        if GPU_AVAILABLE:
            from subshader.dsp.wavelet import CuWavelet
            wavelet = CuWavelet(
                sample_rate=sr,
                input_n=config.audio.chunk_size,
                config=config.wavelet,
            )
        else:
            wavelet = NumPyWavelet(
                sample_rate=sr,
                input_n=config.audio.chunk_size,
                config=config.wavelet,
            )
        init_times[0] = (time.perf_counter() - t0) * 1000.0
        backend = "CuWavelet (GPU)" if GPU_AVAILABLE else "NumPyWavelet (CPU)"
        live_row(backend, 1, 1, init_times)

        # ----- Runtime loop timing -----
        print(f"\nRuntime loop ({self.num_frames} frames):")
        print_header()

        get_chunk_times = np.empty(self.num_frames)
        cwt_times       = np.empty(self.num_frames)
        total_times     = np.empty(self.num_frames)

        live_labels = ["get_chunk()", "cwt()", "Total frame"]
        live_time_arrays = [get_chunk_times, cwt_times, total_times]

        frames_processed = 0
        t_total_start = time.perf_counter()

        for i in range(self.num_frames):
            t_frame = time.perf_counter()

            t0 = time.perf_counter()
            audio_data = audio_input.get_chunk()
            get_chunk_times[i] = (time.perf_counter() - t0) * 1000.0

            if audio_data is None:
                break

            t0 = time.perf_counter()
            wavelet.cwt(audio_data)
            cwt_times[i] = (time.perf_counter() - t0) * 1000.0

            total_times[i] = (time.perf_counter() - t_frame) * 1000.0

            frames_processed = i + 1
            live_progress(frames_processed, self.num_frames)

        total_s = time.perf_counter() - t_total_start
        sys.stdout.write('\n')
        sys.stdout.flush()
        get_chunk_times = get_chunk_times[:frames_processed]
        cwt_times = cwt_times[:frames_processed]
        total_times = total_times[:frames_processed]
        print_results_table(live_labels, live_time_arrays)
        print_total(total_s)


# =============================================================================
# README FIGURES (--figures)
# =============================================================================

class ReadmeFigures:
    """Generate the 3 README comparison PNGs with integrated timing."""

    def __init__(self, num_frames: int = NUM_FRAMES, seaborn: bool = False):
        self.num_frames = num_frames
        self.seaborn    = seaborn
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)
        if seaborn:
            os.makedirs(BENCHMARKS_SEABORN_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public figure generators
    # -------------------------------------------------------------------------

    def chirp_signal_comparison(self):
        """STFT | PyWt | SubShader CWT on a synthetic linear chirp (100 Hz → 10 kHz).

        Layout (4 rows):
          Row 0: Instantaneous Frequency curve
          Row 1: STFT
          Row 2: PyWavelet CWT
          Row 3: SubShader CWT
        """
        from scipy.signal import chirp as scipy_chirp

        config    = get_default_config()
        sr        = int(config.wavelet.typical_sampling_freq)
        chunk_size = config.audio.chunk_size
        hop_size  = int(chunk_size * (1 - config.audio.overlap_factor))

        total_samples = hop_size * self.num_frames + chunk_size
        t      = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
        signal = scipy_chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=t[-1], method='linear').astype(np.float64)

        chunks = [signal[i * hop_size: i * hop_size + chunk_size]
                  for i in range(self.num_frames)]

        return self._generate_comparison_figure(
            title=f"Chirp Signal ({CHIRP_F0} Hz \u2192 {CHIRP_F1 // 1000} kHz) \u2014 STFT vs PyWavelet vs SubShader CWT",
            display_title="Chirp Signal Comparison",
            filename="chirp_signal_comparison.png",
            top_rows=[
                {"type": "freq_line", "f0": CHIRP_F0, "f1": CHIRP_F1, "title": "Instantaneous Frequency"},
            ],
            audio_chunks=chunks,
            sample_rate=float(sr),
        )

    def polyphonic_signal_comparison(self):
        """MIDI | Edison | STFT | PyWt | SubShader CWT on polyphonic audio.

        Layout (5 rows):
          Row 0: MIDI piano roll screenshot
          Row 1: DAW spectrogram (Edison)
          Row 2: STFT
          Row 3: PyWavelet CWT
          Row 4: SubShader CWT
        """
        return self._generate_comparison_figure(
            audio_path=AUDIO_POLYPHONIC,
            title="Polyphonic Signal \u2014 STFT vs PyWavelet vs SubShader CWT",
            display_title="Polyphonic Signal Comparison",
            filename="polyphonic_signal_comparison.png",
            top_rows=[
                {"type": "image", "path": MIDI_POLYPHONIC, "title": "MIDI Piano Roll"},
                {"type": "image", "path": DAW_POLYPHONIC, "title": "DAW Spectrogram (Edison)"},
            ],
        )

    def musical_signal_comparison(self):
        """Waveform | Edison | STFT | PyWt | SubShader CWT on a full musical track.

        Layout (5 rows):
          Row 0: Audio waveform
          Row 1: DAW spectrogram (Edison)
          Row 2: STFT
          Row 3: PyWavelet CWT
          Row 4: SubShader CWT
        """
        return self._generate_comparison_figure(
            audio_path=AUDIO_MUSICAL,
            title="Musical Signal \u2014 STFT vs PyWavelet vs SubShader CWT",
            display_title="Musical Signal Comparison",
            filename="musical_signal_comparison.png",
            top_rows=[
                {"type": "waveform", "title": "Audio Waveform"},
                {"type": "image", "path": DAW_MUSICAL, "title": "DAW Spectrogram (Edison)"},
            ],
        )

    def run_all(self):
        """Generate all 3 comparison figures."""
        required_files = [
            AUDIO_POLYPHONIC,
            AUDIO_MUSICAL,
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f"Missing required audio files:\n"
                + "\n".join(f"  - {f}" for f in missing)
            )

        print(f"\n=== Generating README Figures \u2192 {BENCHMARKS_DIR}/ ===\n")

        chirp_timing = self.chirp_signal_comparison()
        poly_timing = self.polyphonic_signal_comparison()
        music_timing = self.musical_signal_comparison()

        print("\nAll figures saved.\n")

        return {
            "chirp": chirp_timing,
            "polyphonic": poly_timing,
            "musical": music_timing,
        }

    # -------------------------------------------------------------------------
    # Stub layout — instant render with placeholder data
    # -------------------------------------------------------------------------

    def stub_layouts(self):
        """Render all 3 figure layouts with random noise — no DSP, instant."""
        stub_dir = os.path.join(BENCHMARKS_DIR, "stubs")
        os.makedirs(stub_dir, exist_ok=True)
        print(f"\n=== Stub Layouts → {stub_dir}/ ===\n")

        configs = [
            {
                "title": f"Chirp Signal ({CHIRP_F0} Hz \u2192 {CHIRP_F1 // 1000} kHz)",
                "filename": "chirp_signal_comparison_STUB.png",
                "top_rows": [
                    {"type": "freq_line", "f0": CHIRP_F0, "f1": CHIRP_F1, "title": "Instantaneous Frequency"},
                ],
            },
            {
                "title": "Polyphonic Signal",
                "filename": "polyphonic_signal_comparison_STUB.png",
                "top_rows": [
                    {"type": "image", "path": MIDI_POLYPHONIC, "title": "MIDI Piano Roll"},
                    {"type": "image", "path": DAW_POLYPHONIC, "title": "DAW Spectrogram (Edison)"},
                ],
            },
            {
                "title": "Beltran Audio Clip",
                "filename": "musical_signal_comparison_STUB.png",
                "top_rows": [
                    {"type": "waveform", "title": "Audio Waveform"},
                    {"type": "image", "path": DAW_MUSICAL, "title": "DAW Spectrogram (Edison)"},
                ],
            },
        ]

        duration_s   = 30.0
        n_cwt_freqs  = 116
        n_time_bins  = 512

        # Fake frequency axis and y-ticks
        cwt_freqs = np.logspace(np.log10(27.5), np.log10(21500), n_cwt_freqs)
        freq_min, freq_max = cwt_freqs[0], cwt_freqs[-1]
        freq_ticks = [20, 200, 2000, 20000]
        spec_ytick_bins, spec_ytick_labels = [], []
        for f in freq_ticks:
            if freq_min <= f <= freq_max:
                spec_ytick_bins.append(float(np.interp(f, cwt_freqs, np.arange(n_cwt_freqs))))
                spec_ytick_labels.append(f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}')

        extent_spec = [0, duration_s, 0, n_cwt_freqs]
        t_audio = np.linspace(0, duration_s, n_time_bins)

        for cfg in configs:
            top_rows = cfg["top_rows"]
            n_top    = len(top_rows)
            n_total  = n_top + 3
            cmap     = "inferno"

            fig = plt.figure(figsize=(20, 4 * n_total))
            fig.suptitle(cfg["title"], fontsize=32, y=0.975)
            fig.text(0.5, 0.925, "STFT  ·  PyWavelet CWT  ·  SubShader CWT",
                     ha='center', fontsize=24, color='black')
            gs = gridspec.GridSpec(n_total, 1, figure=fig,
                                  height_ratios=[1] * n_total, hspace=0.22)
            fig.subplots_adjust(left=0.06, right=0.94, bottom=0.04, top=0.90)

            # Spectrogram axes
            ax_stft = fig.add_subplot(gs[n_top])
            ax_pywt = fig.add_subplot(gs[n_top + 1], sharex=ax_stft, sharey=ax_stft)
            ax_npwt = fig.add_subplot(gs[n_top + 2], sharex=ax_stft, sharey=ax_stft)

            # Top rows
            for idx, row in enumerate(top_rows):
                rtype = row["type"]
                if rtype == "waveform":
                    ax = fig.add_subplot(gs[idx], sharex=ax_stft)
                    y_stub = np.random.randn(n_time_bins) * 0.3
                    ax.fill_between(t_audio, -np.abs(y_stub), np.abs(y_stub),
                                    color="#606060", alpha=0.75)
                    ax.set_ylabel("Amplitude", fontsize=9)
                    ax.set_title(row["title"], fontsize=24, loc="left")
                    ax.tick_params(labelsize=16)
                    plt.setp(ax.get_xticklabels(), visible=False)
                elif rtype == "freq_line":
                    ax = fig.add_subplot(gs[idx], sharex=ax_stft, sharey=ax_stft)
                    t_curve = np.linspace(0, duration_s, 500)
                    f_curve = row['f0'] + (row['f1'] - row['f0']) * t_curve / duration_s
                    bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
                    ax.plot(t_curve, bin_curve, color='#AAAAAA', linewidth=2)
                    ax.set_facecolor('#1A1A1A')
                    ax.set_title(row["title"], fontsize=24, loc="left")
                    ax.set_yticks(spec_ytick_bins)
                    ax.set_yticklabels(spec_ytick_labels)
                    ax.tick_params(labelsize=16)
                    plt.setp(ax.get_xticklabels(), visible=False)
                elif rtype == "image":
                    ax = fig.add_subplot(gs[idx])
                    img_path = row.get("path")
                    if img_path and os.path.exists(img_path):
                        img = plt.imread(img_path)
                        ax.imshow(img, aspect="auto", origin="upper", cmap='gray')
                        ax.set_title(row["title"], fontsize=24, loc="left")
                        ax.axis("off")
                    else:
                        self._placeholder_ax(ax)
                        ax.set_title(row["title"], fontsize=24, loc="left")

            # Stub spectrograms — random noise
            noise = np.random.rand(n_cwt_freqs, n_time_bins)
            for ax, label in [(ax_stft, "STFT"), (ax_pywt, "PyWavelet CWT"),
                               (ax_npwt, "SubShader CWT")]:
                ax.imshow(noise, cmap=cmap, aspect="auto", origin="lower",
                          extent=extent_spec)
                ax.set_title(f"{label} \u2014 avg XX.XX ms/frame", fontsize=24, loc="left")
                ax.set_yticks(spec_ytick_bins)
                ax.set_yticklabels(spec_ytick_labels)
                ax.tick_params(labelsize=16)
                if ax is not ax_npwt:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    ax.set_xlabel("Time (s)", fontsize=9)

            path = os.path.join(stub_dir, cfg["filename"])
            fig.savefig(path, dpi=100)
            plt.close(fig)
            print(f"Saved \u2192 {path}")

        print("\nDone.\n")

    # -------------------------------------------------------------------------
    # Internal: comparison figure pipeline
    # -------------------------------------------------------------------------

    @staticmethod
    def _placeholder_ax(ax, label="reference not available"):
        """Render a polished placeholder for missing reference images."""
        ax.set_facecolor("#1A1A1A")
        for x in np.linspace(0, 1, 20):
            ax.axline((x, 0), slope=1, color="#333333", linewidth=0.5,
                     transform=ax.transAxes)
        ax.text(0.5, 0.5, f"[ {label} ]",
                ha="center", va="center", color="#555555",
                fontsize=11, transform=ax.transAxes)
        ax.axis("off")

    @staticmethod
    def _downsample_spec(arr, max_rows=HEATMAP_MAX_ROWS, max_cols=HEATMAP_MAX_COLS):
        """Downsample spectrogram to fit heatmap constraints."""
        h, w = arr.shape
        target_w = min(w, max_cols)
        target_h = min(h, max_rows)
        if w != target_w:
            arr = scipy_resample(arr, target_w, axis=1)
        if h != target_h:
            arr = scipy_resample(arr, target_h, axis=0)
        return np.clip(arr, 0, None).astype(np.float32)

    def _generate_comparison_figure(self, title, display_title, filename,
                                     top_rows, audio_path=None,
                                     audio_chunks=None, sample_rate=None):
        """
        Generate a stacked comparison figure and save to disk.

        The figure has variable top rows (defined by top_rows descriptors)
        followed by 3 fixed spectrogram rows (STFT, PyWavelet CWT, SubShader CWT).

        top_rows: list of dicts, each with "type" and type-specific keys:
          - {"type": "freq_line", "f0": ..., "f1": ..., "title": ...}
          - {"type": "image", "path": ..., "title": ...}
          - {"type": "waveform", "title": ...}
        """
        config = get_default_config()

        if audio_chunks is not None:
            sr = sample_rate
            chunk_iter = iter(audio_chunks)
            num_frames = len(audio_chunks)
        else:
            config.audio.file_path = audio_path
            ai = AudioInput(
                path=config.audio.file_path,
                chunk_size=config.audio.chunk_size,
                overlap_factor=config.audio.overlap_factor,
            )
            sr = ai.get_sample_rate()
            chunk_iter = None
            # Process entire audio file
            num_frames = (ai.total_samples - config.audio.chunk_size) // ai.hop_size + 1

        pywt = PyWavelet(  sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
        npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

        # STFT setup — crop to chromatic scale frequency range
        stft_freqs        = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / sr)
        freq_min, freq_max = pywt.freqs[0], pywt.freqs[-1]
        stft_freq_mask    = (stft_freqs >= freq_min) & (stft_freqs <= freq_max)
        stft_cropped_freqs = stft_freqs[stft_freq_mask]
        stft_target_w     = pywt.output_n

        cwt_freqs   = pywt.freqs
        n_cwt_freqs = len(cwt_freqs)

        # Circular buffers
        color_norm = ColorNormalizationConfig()
        audio_buf  = AudioFrameBuffer(chunk_size=config.audio.chunk_size, num_chunks=num_frames)
        stft_buf   = CircularFrameBuffer(frame_shape=(n_cwt_freqs, stft_target_w), num_frames=num_frames, color_norm_config=color_norm)
        pywt_buf   = CircularFrameBuffer(frame_shape=pywt.get_output_shape(),       num_frames=num_frames, color_norm_config=color_norm)
        npwt_buf   = CircularFrameBuffer(frame_shape=npwt.get_output_shape(),       num_frames=num_frames, color_norm_config=color_norm)

        # Process frames — with per-method timing and live progress
        stft_times = np.empty(num_frames)
        pywt_times = np.empty(num_frames)
        npwt_times = np.empty(num_frames)
        frames_processed = 0
        save_path = os.path.join(BENCHMARKS_DIR, filename)

        print_figure_header(display_title)

        t_total_start = time.perf_counter()

        for i in range(num_frames):
            chunk = next(chunk_iter, None) if chunk_iter is not None else ai.get_chunk()
            if chunk is None:
                break

            audio_buf.push_chunk(chunk)

            t0 = time.perf_counter()
            _, _, Zxx      = scipy_stft(chunk, fs=sr, nperseg=STFT_NPERSEG)
            stft_mag       = np.abs(Zxx)[stft_freq_mask, :]
            stft_resampled = scipy_resample(stft_mag, stft_target_w, axis=1)
            stft_resampled = np.clip(stft_resampled, 0, None)

            stft_log = np.zeros((n_cwt_freqs, stft_target_w))
            for col in range(stft_target_w):
                stft_log[:, col] = np.interp(cwt_freqs, stft_cropped_freqs,
                                             stft_resampled[:, col], left=0.0, right=0.0)
            stft_times[i] = (time.perf_counter() - t0) * 1000.0
            stft_buf.push_frame(stft_log)

            t0 = time.perf_counter()
            pywt_buf.push_frame(pywt.cwt(chunk))
            pywt_times[i] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            npwt_buf.push_frame(npwt.cwt(chunk))
            npwt_times[i] = (time.perf_counter() - t0) * 1000.0

            frames_processed = i + 1
            live_progress(frames_processed, num_frames)

        total_s = time.perf_counter() - t_total_start

        # Clear the live progress line before printing results block
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        sys.stdout.flush()

        # Trim timing arrays to actual frame count
        stft_times = stft_times[:frames_processed]
        pywt_times = pywt_times[:frames_processed]
        npwt_times = npwt_times[:frames_processed]

        labels = ["STFT", "PyWavelet CWT", "SubShader CWT"]
        timing = {
            "STFT":           compute_timing_stats(stft_times),
            "PyWavelet CWT":  compute_timing_stats(pywt_times),
            "SubShader CWT":  compute_timing_stats(npwt_times),
        }

        print_figure_results(frames_processed, labels,
                             [stft_times, pywt_times, npwt_times], total_s, save_path)

        # Flatten buffers
        stft_spec = stft_buf.get_flattened_buffer()
        pywt_spec = pywt_buf.get_flattened_buffer()
        npwt_spec = npwt_buf.get_flattened_buffer()
        spec_w    = pywt_spec.shape[1]
        x, y_min, y_max = audio_buf.get_downsampled(spec_w)

        duration_s  = audio_buf.total_samples / sr
        extent_spec = [0, duration_s, 0, n_cwt_freqs]
        t_audio     = np.linspace(0, duration_s, len(x))

        # Y-tick positions at log-spaced frequencies
        freq_ticks = [20, 200, 2000, 20000]
        spec_ytick_bins   = []
        spec_ytick_labels = []
        for f in freq_ticks:
            if freq_min <= f <= freq_max:
                spec_ytick_bins.append(float(np.interp(f, cwt_freqs, np.arange(n_cwt_freqs))))
                spec_ytick_labels.append(f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}')

        # ── Figure layout ─────────────────────────────────────────────────────
        n_top   = len(top_rows)
        n_total = n_top + 3   # 3 spectrogram rows always
        cmap    = "inferno"

        fig = plt.figure(figsize=(20, 4 * n_total))
        fig.suptitle(title, fontsize=32, y=0.975)
        fig.text(0.5, 0.925, "STFT  ·  PyWavelet CWT  ·  SubShader CWT",
                 ha='center', fontsize=24, color='black')
        gs  = gridspec.GridSpec(n_total, 1, figure=fig,
                                height_ratios=[1] * n_total,
                                hspace=0.22)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.04, top=0.90)

        # Create spectrogram axes first (so top rows can share axes)
        ax_stft = fig.add_subplot(gs[n_top])
        ax_pywt = fig.add_subplot(gs[n_top + 1], sharex=ax_stft, sharey=ax_stft)
        ax_npwt = fig.add_subplot(gs[n_top + 2], sharex=ax_stft, sharey=ax_stft)

        # ── Top rows (variable per figure) ────────────────────────────────────
        for idx, row in enumerate(top_rows):
            rtype = row["type"]

            if rtype == "waveform":
                ax = fig.add_subplot(gs[idx], sharex=ax_stft)
                ax.fill_between(t_audio, y_min, y_max, color="#606060", alpha=0.75)
                ax.set_ylabel("Amplitude", fontsize=9)
                ax.set_title(row["title"], fontsize=24, loc="left")
                ax.tick_params(labelsize=16)
                plt.setp(ax.get_xticklabels(), visible=False)

            elif rtype == "freq_line":
                ax = fig.add_subplot(gs[idx], sharex=ax_stft, sharey=ax_stft)
                t_curve   = np.linspace(0, duration_s, 500)
                f_curve   = row['f0'] + (row['f1'] - row['f0']) * t_curve / duration_s
                bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
                ax.plot(t_curve, bin_curve, color='#AAAAAA', linewidth=2)
                ax.set_facecolor('#1A1A1A')
                ax.set_title(row["title"], fontsize=24, loc="left")
                ax.set_yticks(spec_ytick_bins)
                ax.set_yticklabels(spec_ytick_labels)
                ax.tick_params(labelsize=16)
                plt.setp(ax.get_xticklabels(), visible=False)

            elif rtype == "image":
                ax = fig.add_subplot(gs[idx])
                img_path = row.get("path")
                if img_path and os.path.exists(img_path):
                    img = plt.imread(img_path)
                    ax.imshow(img, aspect="auto", origin="upper", cmap='gray')
                    ax.set_title(row["title"], fontsize=24, loc="left")
                    ax.axis("off")
                else:
                    self._placeholder_ax(ax)
                    ax.set_title(row["title"], fontsize=24, loc="left")

        # ── Spectrogram rows ──────────────────────────────────────────────────

        # STFT
        ax_stft.imshow(stft_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=stft_buf.get_intensity_max())
        ax_stft.set_title(f"STFT \u2014 avg {timing['STFT']['avg_ms']:.2f} ms/frame", fontsize=24, loc="left")
        ax_stft.set_ylabel("Freq", fontsize=9)
        ax_stft.set_yticks(spec_ytick_bins)
        ax_stft.set_yticklabels(spec_ytick_labels)
        ax_stft.tick_params(labelsize=8)
        plt.setp(ax_stft.get_xticklabels(), visible=False)

        # PyWavelet CWT
        ax_pywt.imshow(pywt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=pywt_buf.get_intensity_max())
        ax_pywt.set_title(f"PyWavelet CWT \u2014 avg {timing['PyWavelet CWT']['avg_ms']:.2f} ms/frame", fontsize=24, loc="left")
        ax_pywt.set_ylabel("Freq", fontsize=9)
        ax_pywt.set_yticks(spec_ytick_bins)
        ax_pywt.set_yticklabels(spec_ytick_labels)
        ax_pywt.tick_params(labelsize=8)
        plt.setp(ax_pywt.get_xticklabels(), visible=False)

        # SubShader CWT
        subshader_base  = "SubShader CWT" if GPU_AVAILABLE else "SubShader CWT (NumPy)"
        subshader_label = f"{subshader_base} \u2014 avg {timing['SubShader CWT']['avg_ms']:.2f} ms/frame"
        ax_npwt.imshow(npwt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=npwt_buf.get_intensity_max())
        ax_npwt.set_title(subshader_label, fontsize=24, loc="left")
        ax_npwt.set_ylabel("Freq", fontsize=9)
        ax_npwt.set_yticks(spec_ytick_bins)
        ax_npwt.set_yticklabels(spec_ytick_labels)
        ax_npwt.set_xlabel("Time (s)", fontsize=9)
        ax_npwt.tick_params(labelsize=8)

        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        # ── Seaborn variant ───────────────────────────────────────────────────
        if self.seaborn and SEABORN_AVAILABLE:
            seaborn_filename = filename.replace(".png", "_seaborn.png")
            self._generate_seaborn_figure(
                stft_spec=stft_spec, pywt_spec=pywt_spec, npwt_spec=npwt_spec,
                audio_x=t_audio, audio_ymin=y_min, audio_ymax=y_max,
                top_rows=top_rows,
                cwt_freqs=cwt_freqs, duration_s=duration_s, n_cwt_freqs=n_cwt_freqs,
                timing=timing, title=title, filename=seaborn_filename,
                stft_vmax=stft_buf.get_intensity_max(),
                pywt_vmax=pywt_buf.get_intensity_max(),
                npwt_vmax=npwt_buf.get_intensity_max(),
                spec_ytick_bins=spec_ytick_bins, spec_ytick_labels=spec_ytick_labels,
            )

        return timing

    def _generate_seaborn_figure(self, stft_spec, pywt_spec, npwt_spec,
                                  audio_x, audio_ymin, audio_ymax,
                                  top_rows, cwt_freqs, duration_s, n_cwt_freqs,
                                  timing, title, filename,
                                  stft_vmax, pywt_vmax, npwt_vmax,
                                  spec_ytick_bins, spec_ytick_labels):
        """
        Generate seaborn-styled heatmap figures with the same row layout
        as the matplotlib figure.
        """
        sns.set_theme(style="dark", rc={
            "axes.facecolor": "#0D1117",
            "figure.facecolor": "#0D1117",
            "text.color": "#C9D1D9",
            "axes.labelcolor": "#C9D1D9",
            "xtick.color": "#8B949E",
            "ytick.color": "#8B949E",
        })

        timing_subtitle = (
            f"Avg per frame:  STFT {timing['STFT']['avg_ms']:.2f} ms  |  "
            f"PyWavelet {timing['PyWavelet CWT']['avg_ms']:.2f} ms  |  "
            f"SubShader {timing['SubShader CWT']['avg_ms']:.2f} ms"
        )

        n_top   = len(top_rows)
        n_total = n_top + 3

        fig = plt.figure(figsize=(22, 4.5 * n_total))
        fig.suptitle(f"{title}\n{timing_subtitle}", fontsize=13, y=0.99)
        gs  = gridspec.GridSpec(n_total, 1, figure=fig,
                                height_ratios=[1] * n_total, hspace=0.15)
        fig.subplots_adjust(left=0.07, right=0.92, bottom=0.04, top=0.93)

        # ── Top rows ─────────────────────────────────────────────────────────
        for idx, row in enumerate(top_rows):
            ax = fig.add_subplot(gs[idx])
            rtype = row["type"]

            if rtype == "waveform":
                ax.set_facecolor("#0D1117")
                ax.fill_between(audio_x, audio_ymin, audio_ymax,
                                color="#FF8C42", alpha=0.85)
                ax.set_ylim([np.min([audio_ymin, audio_ymax]),
                             np.max([audio_ymin, audio_ymax])])
                ax.set_xlim([0, duration_s])
                ax.margins(x=0, y=0)
                ax.set_ylabel("Amplitude", fontsize=9)
                ax.set_title(row["title"], fontsize=24, loc="left")
                ax.tick_params(labelsize=16)
                plt.setp(ax.get_xticklabels(), visible=False)

            elif rtype == "freq_line":
                ax.set_facecolor("#2D1B4E")
                t_curve   = np.linspace(0, duration_s, 500)
                f_curve   = row['f0'] + (row['f1'] - row['f0']) * t_curve / duration_s
                bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
                ax.plot(t_curve, bin_curve, color='#FFD700', linewidth=2.5)
                ax.set_title(row["title"], fontsize=24, loc="left")
                ax.set_yticks(spec_ytick_bins)
                ax.set_yticklabels(spec_ytick_labels)
                ax.tick_params(labelsize=8, colors="#C9D1D9")
                plt.setp(ax.get_xticklabels(), visible=False)

            elif rtype == "image":
                img_path = row.get("path")
                if img_path and os.path.exists(img_path):
                    img = plt.imread(img_path)
                    ax.imshow(img, aspect="auto", origin="upper")
                    ax.set_title(row["title"], fontsize=24, loc="left")
                    ax.axis("off")
                else:
                    self._placeholder_ax(ax)
                    ax.set_title(row["title"], fontsize=24, loc="left")

        # ── Spectrogram heatmaps ──────────────────────────────────────────────
        specs = [
            (stft_spec, stft_vmax, f"STFT \u2014 avg {timing['STFT']['avg_ms']:.2f} ms/frame"),
            (pywt_spec, pywt_vmax, f"PyWavelet CWT \u2014 avg {timing['PyWavelet CWT']['avg_ms']:.2f} ms/frame"),
        ]
        subshader_base = "SubShader CWT" if GPU_AVAILABLE else "SubShader CWT (NumPy)"
        specs.append(
            (npwt_spec, npwt_vmax, f"{subshader_base} \u2014 avg {timing['SubShader CWT']['avg_ms']:.2f} ms/frame")
        )

        for j, (spec, vmax, spec_title) in enumerate(specs):
            ax = fig.add_subplot(gs[n_top + j])
            ds = self._downsample_spec(spec)
            sns.heatmap(ds, ax=ax, cmap="inferno", vmin=0, vmax=vmax,
                        xticklabels=False, yticklabels=False,
                        cbar=True, cbar_kws={"shrink": 0.8, "pad": 0.01, "label": "Intensity"})
            ax.invert_yaxis()
            ax.set_xlim([0, duration_s])
            scale = ds.shape[0] / n_cwt_freqs
            ax.set_yticks([b * scale for b in spec_ytick_bins])
            ax.set_yticklabels(spec_ytick_labels, fontsize=16)
            ax.set_title(spec_title, fontsize=24, loc="left")
            
            if j < len(specs) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                # Bottom row time ticks
                desired_s = np.linspace(0, duration_s, 6)
                ax.set_xticks(desired_s)
                ax.set_xticklabels([f"{t:.1f}" for t in desired_s], fontsize=8)
                ax.set_xlabel("Time (s)", fontsize=9)

        path = os.path.join(BENCHMARKS_SEABORN_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        sns.reset_orig()
        print(f"Saved seaborn figure \u2192 {path}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_default():
    """Run SubShader with default configuration."""
    from subshader.__main__ import main
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SubShader benchmark suite"
    )
    parser.add_argument("--timing",     action="store_true", help="Run pipeline with live timing instrumentation")
    parser.add_argument("--figures",    action="store_true", help="Generate the 3 README comparison PNGs")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests (NumPy vs CuPy, etc.)")
    parser.add_argument("--all",        action="store_true", help="Run all modes")
    parser.add_argument("--seaborn",    action="store_true",
                        help="With --figures: generate seaborn heatmap variants in benchmarks/seaborn/")
    parser.add_argument("--stub",      action="store_true",
                        help="Generate stub layout figures with placeholder data (instant)")
    args = parser.parse_args()

    any_flag = args.timing or args.figures or args.unit_tests or args.all or args.seaborn or args.stub

    if args.seaborn and not args.figures and not args.all:
        print("[benchmark] --seaborn has no effect without --figures or --all.\n")

    if not any_flag:
        run_default()
    else:
        if args.timing or args.all:
            TimedSubShader().run()

        if args.stub:
            ReadmeFigures().stub_layouts()

        if args.figures or args.all:
            use_seaborn = args.seaborn
            if use_seaborn and not SEABORN_AVAILABLE:
                print("[benchmark] --seaborn requested but seaborn is not installed. "
                      "pip install seaborn\n")
            ReadmeFigures(seaborn=use_seaborn).run_all()

        if args.unit_tests or args.all:
            from unit_tests import run_all as run_unit_tests
            run_unit_tests()
