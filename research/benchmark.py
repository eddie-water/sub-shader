"""
SubShader Benchmark Suite.

Modes:
  (default)                   Run all modes (same as --all)
  --timing                    STFT vs PyWavelet vs NumPy CWT vs CuPy CWT timing comparison
  --figures                   Generate 3 README comparison PNGs (matplotlib)
  --figures --stub            Generate stub layouts instead of real DSP (fast iteration)
  --figures --stub-pywt       Skip PyWavelet, use random stubs, save to stub folder (faster)
  --seaborn                   Generate 3 comparison PNGs (seaborn heatmap style)
  --seaborn --stub-pywt       Seaborn with stubbed PyWavelet, save to stub folder
  --comparison-grid           Generate 3x3 comparison grid (signals x representations)
  --unit-tests                Run unit tests (NumPy vs CuPy verification, etc.)
  --all                       Run timing, figures (matplotlib), figures (seaborn), unit tests
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import argparse

# Configure matplotlib BEFORE any pyplot imports
import matplotlib
matplotlib.use('Agg')

# Use matplotlib's default font configuration
# (no custom font override)

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn  # noqa: F401
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("[benchmark] seaborn not installed -- --seaborn flag will be ignored.\n")

from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer

from utilities import (
    BENCHMARKS_DIR,
    BENCHMARKS_SEABORN_DIR,
    BENCHMARKS_STUBS_DIR,
    AUDIO_DEFAULT,
    AUDIO_BOUNCING_CHIRP,
    AUDIO_POLYPHONIC,
    AUDIO_MUSICAL,
    AUDIO_BELTRAN,
    AUDIO_BELTRAN_16BAR,
    AUDIO_BELTRAN_8BAR,
    MIDI_POLYPHONIC,
    DAW_POLYPHONIC,
    DAW_MUSICAL,
    DAW_BELTRAN_16BAR,
    DAW_BELTRAN_8BAR,
    STFT_NPERSEG,
    NUM_FRAMES,
    CHIRP_F0,
    CHIRP_F1,
    gpu_available,
    time_call,
    TimingAccumulator,
    live_progress,
    clear_progress,
    print_section_start,
    print_section_end,
    print_separator,
    print_init_header,
    print_init_row,
    print_init_total,
    print_results_header,
    print_results_row,
    print_loop_summary,
    print_total_time,
    compute_timing_stats,
    run_modes,
    compute_freq_yticks,
    create_figure_scaffold,
    render_top_row,
    render_spectrogram_row,
    set_backend,
    compute_stft_frame,
    build_chirp_chunks,
    build_wandering_chirp_chunks,
    build_fm_chirp_chunks,
    build_bouncing_chirp_chunks,
)

# =============================================================================
# GPU DETECTION
# =============================================================================

GPU_AVAILABLE = gpu_available()

if not GPU_AVAILABLE:
    print("[benchmark] No GPU detected -- CuPy benchmarks will be skipped, "
          "figures will use NumPyWavelet as fallback.\n")


# =============================================================================
# TIMED SUBSHADER (--timing)
# =============================================================================

class TimedSubShader:
    """
    Run the SubShader pipeline with live timing instrumentation.

    Mirrors the real SubShader pipeline (AudioInput -> CWT -> buffer) but
    wraps each stage with perf_counter and displays a live-updating table.
    """

    def __init__(self, audio_path: str = AUDIO_DEFAULT, num_frames: int = NUM_FRAMES):
        self.audio_path = audio_path
        self.num_frames = num_frames

    def run(self):
        """Run the timed pipeline: init timing, then runtime loop timing."""
        print("\nSubShader Timing Benchmark\n")

        # ── Init timing ──────────────────────────────────────────────────────
        print_section_start("Init Timing")
        print_init_header()

        config = get_default_config()
        config.audio.file_path = self.audio_path
        total_init_ms = 0.0

        audio_input, ms = time_call(
            AudioInput,
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )
        print_init_row("AudioInput", ms)
        total_init_ms += ms

        sr = audio_input.get_sample_rate()

        if GPU_AVAILABLE:
            from subshader.dsp.wavelet import CuWavelet
            wavelet, ms = time_call(
                CuWavelet,
                sample_rate=sr,
                input_n=config.audio.chunk_size,
                config=config.wavelet,
            )
            backend_name = "CuWavelet (GPU)"
        else:
            wavelet, ms = time_call(
                NumPyWavelet,
                sample_rate=sr,
                input_n=config.audio.chunk_size,
                config=config.wavelet,
            )
            backend_name = "NumPyWavelet (CPU)"

        print_init_row(backend_name, ms)
        total_init_ms += ms

        print_init_total(total_init_ms)
        print_section_end()

        # ── Runtime loop ─────────────────────────────────────────────────────
        print_section_start(f"Runtime Loop ({self.num_frames} frames)")

        acc = TimingAccumulator(self.num_frames, ["get_chunk()", "cwt()"])

        for i in range(self.num_frames):
            audio_data, acc["get_chunk()"][i] = time_call(audio_input.get_chunk)

            if audio_data is None:
                acc.current_idx = i
                break

            _, acc["cwt()"][i] = time_call(wavelet.cwt, audio_data)
            acc.current_idx = i + 1
            live_progress(i + 1, self.num_frames)

        acc.trim()

        clear_progress()
        print_separator()
        print("Results:")
        print_results_header()
        print_results_row("get_chunk()", acc["get_chunk()"])
        print_results_row("cwt()", acc["cwt()"])

        total_times = acc["get_chunk()"] + acc["cwt()"]
        print_loop_summary(len(total_times), total_times)
        print_section_end()


# =============================================================================
# README FIGURES (--figures)
# =============================================================================

class ReadmeFigures:
    """Generate the 3 README comparison PNGs with integrated timing."""

    def __init__(self, num_frames: int = NUM_FRAMES, backends=None, stub_pywt: bool = False):
        self.num_frames = num_frames
        self.backends   = backends or ["matplotlib"]
        self.stub_pywt  = stub_pywt
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)
        if "seaborn" in self.backends:
            os.makedirs(BENCHMARKS_SEABORN_DIR, exist_ok=True)
        # When stub_pywt is enabled, ensure stub dir exists for output
        if stub_pywt:
            os.makedirs(BENCHMARKS_STUBS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public figure generators
    # -------------------------------------------------------------------------

    def chirp_signal_comparison(self):
        """STFT | PyWt | SubShader CWT on a synthetic linear chirp (100 Hz -> 10 kHz).

        Layout (4 rows):
          Row 0: Instantaneous Frequency curve
          Row 1: STFT
          Row 2: PyWavelet CWT
          Row 3: SubShader CWT
        """
        config    = get_default_config()
        sr        = int(config.wavelet.typical_sampling_freq)
        chunk_size = config.audio.chunk_size

        chunks = build_chirp_chunks(
            CHIRP_F0,
            CHIRP_F1,
            sr,
            chunk_size,
            config.audio.overlap_factor,
            self.num_frames,
        )

        return self._generate_comparison_figure(
            title=f"Chirp Signal ({CHIRP_F0} Hz to {CHIRP_F1 // 1000} kHz)",
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
            title="Polyphonic Signal",
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
            title="Beltran Audio Clip",
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

        print(f"\nGenerating README Figures -> {BENCHMARKS_DIR}/\n")

        chirp_timing = self.chirp_signal_comparison()
        poly_timing = self.polyphonic_signal_comparison()
        music_timing = self.musical_signal_comparison()

        return {
            "chirp": chirp_timing,
            "polyphonic": poly_timing,
            "musical": music_timing,
        }

    # -------------------------------------------------------------------------
    # Stub layout -- instant render with placeholder data
    # -------------------------------------------------------------------------

    def stub_layouts(self):
        """Render all 3 figure layouts with random noise -- no DSP, instant."""
        stub_dir = os.path.join(BENCHMARKS_DIR, "stubs")
        os.makedirs(stub_dir, exist_ok=True)
        print_section_start(f"Stub Layouts -> {stub_dir}/")

        configs = [
            {
                "title": f"Chirp Signal ({CHIRP_F0} Hz to {CHIRP_F1 // 1000} kHz)",
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
        subtitle     = "STFT  |  PyWavelet CWT  |  SubShader CWT"

        cwt_freqs = np.logspace(np.log10(27.5), np.log10(21500), n_cwt_freqs)
        ytick_bins, ytick_labels = compute_freq_yticks(cwt_freqs)
        extent_spec = [0, duration_s, 0, n_cwt_freqs]
        t_audio = np.linspace(0, duration_s, n_time_bins)
        y_stub = np.random.randn(n_time_bins) * 0.3

        for cfg in configs:
            top_rows = cfg["top_rows"]
            fig, gs, ax_stft, ax_pywt, ax_npwt = create_figure_scaffold(
                cfg["title"], subtitle, len(top_rows))

            for idx, row in enumerate(top_rows):
                render_top_row(fig, gs, idx, row, ax_stft,
                               t_audio=t_audio,
                               y_min=-np.abs(y_stub), y_max=np.abs(y_stub),
                               cwt_freqs=cwt_freqs, duration_s=duration_s,
                               ytick_bins=ytick_bins, ytick_labels=ytick_labels)

            noise = np.random.rand(n_cwt_freqs, n_time_bins)
            stub_vmax = noise.max()
            for ax, label, bottom in [(ax_stft, "STFT", False),
                                      (ax_pywt, "PyWavelet CWT", False),
                                      (ax_npwt, "SubShader CWT", True)]:
                render_spectrogram_row(
                    ax, noise,
                    title=label,
                    extent=extent_spec, vmax=stub_vmax,
                    ytick_bins=ytick_bins, ytick_labels=ytick_labels,
                    is_bottom=bottom)

            path = os.path.join(stub_dir, cfg["filename"])
            fig.savefig(path, dpi=100)
            plt.close(fig)
            print(f"Saved -> {path}")

        print_section_end()

    # -------------------------------------------------------------------------
    # Internal: comparison figure pipeline
    # -------------------------------------------------------------------------

    def _generate_comparison_figure(self, title, display_title, filename,
                                     top_rows, audio_path=None,
                                     audio_chunks=None, sample_rate=None,
                                     wavelet_config=None):
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
            # Use NUM_FRAMES as the limit for file-based audio
            num_frames = NUM_FRAMES

        wc = wavelet_config if wavelet_config is not None else config.wavelet
        if not self.stub_pywt:
            pywt = PyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=wc)
        else:
            pywt = None
        npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=wc)

        # STFT setup -- crop to chromatic scale frequency range
        stft_freqs        = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / sr)
        # Use pywt freqs if available, otherwise use npwt freqs
        ref_freqs = pywt.freqs if pywt is not None else npwt.freqs
        freq_min, freq_max = ref_freqs[0], ref_freqs[-1]
        stft_freq_mask    = (stft_freqs >= freq_min) & (stft_freqs <= freq_max)
        stft_cropped_freqs = stft_freqs[stft_freq_mask]
        stft_target_w     = (pywt.output_n if pywt is not None else npwt.output_n)

        cwt_freqs   = ref_freqs
        n_cwt_freqs = len(cwt_freqs)

        # Circular buffers
        color_norm = ColorNormalizationConfig()
        audio_buf  = AudioFrameBuffer(chunk_size=config.audio.chunk_size, num_chunks=num_frames)
        stft_buf   = CircularFrameBuffer(frame_shape=(n_cwt_freqs, stft_target_w), num_frames=num_frames, color_norm_config=color_norm)
        # Use npwt shape for pywt_buf (they have the same output shape)
        pywt_buf   = CircularFrameBuffer(frame_shape=npwt.get_output_shape(),       num_frames=num_frames, color_norm_config=color_norm)
        npwt_buf   = CircularFrameBuffer(frame_shape=npwt.get_output_shape(),       num_frames=num_frames, color_norm_config=color_norm)

        # Process frames -- with per-method timing and live progress
        stft_times = np.empty(num_frames)
        pywt_times = np.empty(num_frames)
        npwt_times = np.empty(num_frames)
        frames_processed = 0

        print_section_start(display_title)

        t_total_start = time.perf_counter()

        for i in range(num_frames):
            chunk = next(chunk_iter, None) if chunk_iter is not None else ai.get_chunk()
            if chunk is None:
                break

            audio_buf.push_chunk(chunk)

            stft_log, stft_times[i] = time_call(
                compute_stft_frame,
                chunk,
                sr,
                STFT_NPERSEG,
                stft_freq_mask,
                stft_cropped_freqs,
                cwt_freqs,
                stft_target_w,
            )
            stft_buf.push_frame(stft_log)

            if self.stub_pywt:
                # Generate random stub spectrogram instead of computing PyWavelet
                pywt_times[i] = 0.0
                pywt_stub = np.random.rand(n_cwt_freqs, stft_target_w).astype(np.float32)
                pywt_buf.push_frame(pywt_stub)
            else:
                _, pywt_times[i] = time_call(pywt.cwt, chunk)
                pywt_buf.push_frame(_)

            _, npwt_times[i] = time_call(npwt.cwt, chunk)
            npwt_buf.push_frame(_)

            frames_processed = i + 1
            live_progress(frames_processed, num_frames)

        total_s = time.perf_counter() - t_total_start

        # Trim timing arrays to actual frame count
        stft_times = stft_times[:frames_processed]
        pywt_times = pywt_times[:frames_processed]
        npwt_times = npwt_times[:frames_processed]

        # Guard against empty timing arrays (no frames processed)
        if frames_processed == 0:
            clear_progress()
            print_separator()
            print("Results:")
            print("No frames processed - audio file may be missing or invalid.")
            print_section_end()
            return {}

        pywt_label = "PyWavelet CWT (stub)" if self.stub_pywt else "PyWavelet CWT"
        labels = ["STFT", pywt_label, "SubShader CWT"]
        timing = {
            "STFT":           compute_timing_stats(stft_times),
            pywt_label:       compute_timing_stats(pywt_times),
            "SubShader CWT":  compute_timing_stats(npwt_times),
        }

        clear_progress()
        print_separator()
        print("Results:")
        print_results_header()
        for label, times in zip(labels, [stft_times, pywt_times, npwt_times]):
            print_results_row(label, times)
        print_total_time(total_s * 1000)

        # Flatten buffers and trim to actual frames processed
        frame_w = npwt_buf.width
        actual_w = frames_processed * frame_w
        stft_spec = stft_buf.get_flattened_buffer()[:, -actual_w:]
        pywt_spec = pywt_buf.get_flattened_buffer()[:, -actual_w:]
        npwt_spec = npwt_buf.get_flattened_buffer()[:, -actual_w:]
        spec_w    = pywt_spec.shape[1]

        # Trim audio buffer to actual samples processed and downsample for waveform
        actual_samples = frames_processed * config.audio.chunk_size
        audio_trimmed = audio_buf.get_flattened_buffer()[-actual_samples:]
        window_size = max(1, actual_samples // spec_w)
        trimmed_len = window_size * spec_w
        windowed = audio_trimmed[:trimmed_len].reshape(spec_w, window_size)
        y_min = windowed.min(axis=1)
        y_max = windowed.max(axis=1)

        duration_s  = actual_samples / sr
        extent_spec = [0, duration_s, 0, n_cwt_freqs]
        t_audio     = np.linspace(0, duration_s, spec_w)

        spec_ytick_bins, spec_ytick_labels = compute_freq_yticks(cwt_freqs)

        # ── Render for each backend ───────────────────────────────────────────
        subshader_base = "SubShader CWT" if GPU_AVAILABLE else "SubShader CWT (NumPy)"
        n_top = len(top_rows)
        print_separator()

        for backend in self.backends:
            set_backend(backend)

            if backend == "seaborn":
                output_dir = BENCHMARKS_SEABORN_DIR
                out_filename = filename.replace(".png", "_seaborn.png")
                # Seaborn uses per-row vmax for richer contrast
                stft_vmax = stft_buf.get_intensity_max()
                pywt_vmax = pywt_buf.get_intensity_max()
                npwt_vmax = npwt_buf.get_intensity_max()
                suptitle = title
                subtitle = None
            else:
                output_dir = BENCHMARKS_STUBS_DIR if self.stub_pywt else BENCHMARKS_DIR
                # Add _STUB_PYWT suffix to filename when using stub_pywt
                if self.stub_pywt:
                    out_filename = filename.replace(".png", "_STUB_PYWT.png")
                else:
                    out_filename = filename
                # Per-row vmax so each method's detail is visible
                stft_vmax = stft_buf.get_intensity_max()
                pywt_vmax = pywt_buf.get_intensity_max()
                npwt_vmax = npwt_buf.get_intensity_max()
                suptitle = title
                pywt_subtitle = "PyWavelet CWT (stub)" if self.stub_pywt else "PyWavelet CWT"
                subtitle = f"STFT  |  {pywt_subtitle}  |  SubShader CWT"

            fig, gs, ax_stft, ax_pywt, ax_npwt = create_figure_scaffold(
                suptitle, subtitle, n_top)

            for idx, row in enumerate(top_rows):
                render_top_row(fig, gs, idx, row, ax_stft,
                               t_audio=t_audio, y_min=y_min, y_max=y_max,
                               cwt_freqs=cwt_freqs, duration_s=duration_s,
                               ytick_bins=spec_ytick_bins,
                               ytick_labels=spec_ytick_labels)

            pywt_plot_label = "PyWavelet CWT (stub)" if self.stub_pywt else "PyWavelet CWT"
            spec_rows = [
                (ax_stft, stft_spec, "STFT", stft_vmax, False),
                (ax_pywt, pywt_spec, pywt_plot_label, pywt_vmax, False),
                (ax_npwt, npwt_spec, subshader_base, npwt_vmax, True),
            ]
            for ax, data, label, vmax, bottom in spec_rows:
                render_spectrogram_row(
                    ax, data,
                    title=label,
                    extent=extent_spec, vmax=vmax,
                    ytick_bins=spec_ytick_bins, ytick_labels=spec_ytick_labels,
                    is_bottom=bottom,
                    n_cwt_freqs=n_cwt_freqs, duration_s=duration_s)

            save_path_backend = os.path.join(output_dir, out_filename)
            if backend == "seaborn":
                fig.savefig(save_path_backend, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            else:
                fig.savefig(save_path_backend, dpi=150)
            plt.close(fig)
            print(f"Saved {backend} -> {save_path_backend}")

        # Reset to matplotlib when done
        set_backend("matplotlib")
        print_section_end()

        return timing


# =============================================================================
# WAV EXPORT HELPER
# =============================================================================

def export_signal_to_wav(signal: np.ndarray, sample_rate: int, path: str) -> None:
    """Write a 1-D float signal to a 16-bit WAV file."""
    import soundfile as sf
    # Normalize to [-1, 1] if needed
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    print(f"Exported WAV -> {path}  ({len(signal)} samples, {sample_rate} Hz, "
          f"{len(signal) / sample_rate:.2f}s)")


# =============================================================================
# COMPARISON GRID (--comparison-grid)
# =============================================================================

def generate_comparison_grid(stub_pywt: bool = False, dpi: int = 0):
    """
    Generate a 5x3 comparison grid figure.

    Columns: Bouncing Chirp (synthetic), Polyphonic (file), Musical (beltran first 20s)
    Rows:    Reference (waveform/inst-freq), DAW (placeholder), STFT, PyWavelet CWT, SubShader CWT

    Args:
        stub_pywt: Skip PyWavelet computation, use random stub spectrograms (faster)
        dpi: Output DPI. When > 0, file is named comparison_grid_{dpi}dpi.png.
             When 0 (default), file is named comparison_grid.png or comparison_grid_STUB.png.

    Output: assets/images/benchmarks/comparison_grid[_Ndpi].png
    """

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    config = get_default_config()
    sr_default = int(config.wavelet.typical_sampling_freq)
    chunk_size = config.audio.chunk_size
    overlap_factor = config.audio.overlap_factor
    wc = config.wavelet

    # ── Visual config (single source for all grid plots) ─────────────────────
    GRID_CMAP = "inferno"
    GRID_WAVEFORM_COLOR = "#FFDB4A"

    # ── Signal definitions ────────────────────────────────────────────────────
    CHIRP_DURATION_S = 6.0
    chirp_n_frames = int(
        CHIRP_DURATION_S * sr_default
        / (chunk_size * (1 - overlap_factor))
    )
    chirp_n_frames = max(chirp_n_frames, NUM_FRAMES)

    # DAW reference images (row 1) — placeholders until user generates Edison screenshots
    signal_specs = [
        {"label": "Bouncing Chirp", "type": "chirp", "daw_image": os.path.join(BENCHMARKS_DIR, "bouncing_chirp.png")},
        {"label": "Polyphonic", "type": "file",  "path": AUDIO_POLYPHONIC, "daw_image": DAW_POLYPHONIC},
        {"label": "Beltran SoundCloud Rip (8 Bars)", "type": "file", "path": AUDIO_BELTRAN_8BAR, "daw_image": DAW_BELTRAN_8BAR},
    ]

    # ── DSP result containers (per-column) ───────────────────────────────────
    column_data = []

    for col_idx, spec in enumerate(signal_specs):
        print_section_start(f"Processing column {col_idx + 1}/3: {spec['label']}")

        if spec["type"] == "chirp":
            sr = sr_default
            chunks, raw_waveform, chirp_inst_freq, _ = build_bouncing_chirp_chunks(
                sr=sr, chunk_size=chunk_size, overlap_factor=overlap_factor,
                n_frames=chirp_n_frames,
            )
            n_frames = len(chunks)
            chunk_iter = iter(chunks)
            raw_sr = sr
            # Export bouncing chirp as WAV so it can be dragged into FL Studio / Edison
            chirp_wav_path = AUDIO_BOUNCING_CHIRP
            export_signal_to_wav(raw_waveform, sr, chirp_wav_path)
        else:
            audio_path = spec["path"]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            ai = AudioInput(
                path=audio_path,
                chunk_size=chunk_size,
                overlap_factor=overlap_factor,
            )
            sr = ai.get_sample_rate()
            raw_waveform = ai.get_entire_audio()
            hop = int(chunk_size * (1 - overlap_factor))
            file_n_frames = int(len(raw_waveform) / hop)
            n_frames = spec.get("max_frames", file_n_frames)
            chunk_iter = None
            raw_sr = sr
            max_samples = int(n_frames * chunk_size * (1 - overlap_factor) + chunk_size)
            if len(raw_waveform) > max_samples:
                raw_waveform = raw_waveform[:max_samples]

        npwt = NumPyWavelet(sample_rate=sr, input_n=chunk_size, config=wc)
        if not stub_pywt:
            pywt = PyWavelet(sample_rate=sr, input_n=chunk_size, config=wc)
            cwt_freqs = pywt.freqs
        else:
            cwt_freqs = npwt.freqs

        n_cwt_freqs = len(cwt_freqs)

        stft_freqs = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / sr)
        freq_min, freq_max = cwt_freqs[0], cwt_freqs[-1]
        stft_freq_mask = (stft_freqs >= freq_min) & (stft_freqs <= freq_max)
        stft_cropped_freqs = stft_freqs[stft_freq_mask]
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

        frames_processed = 0
        for _ in range(n_frames):
            chunk = next(chunk_iter, None) if chunk_iter is not None else ai.get_chunk()
            if chunk is None:
                break

            stft_log = compute_stft_frame(
                chunk, sr, STFT_NPERSEG, stft_freq_mask,
                stft_cropped_freqs, cwt_freqs, stft_target_w,
            )
            stft_buf.push_frame(stft_log)

            if not stub_pywt:
                pywt_out, _ = time_call(pywt.cwt, chunk)
                pywt_buf.push_frame(pywt_out)

            npwt_out, _ = time_call(npwt.cwt, chunk)
            npwt_buf.push_frame(npwt_out)

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

        duration_s = frames_processed * chunk_size / sr
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

    # ── Build 5x3 figure ─────────────────────────────────────────────────────
    print_section_start("Rendering 5x3 comparison grid")

    GRID_ROWS = 5  # Reference + DAW + STFT + PyWavelet + SubShader
    GRID_COLS = 3
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(24, 16),
        gridspec_kw={
            "hspace": 0.08,
            "wspace": 0.04,
        },
    )

    pywt_label = "PyWavelet (stub)" if stub_pywt else "PyWavelet"

    for col_idx, (col, spec) in enumerate(zip(column_data, signal_specs)):
        # ── Row 0: Reference — waveform time series / instantaneous freq ──
        ax_ref = axes[0][col_idx]

        if spec["type"] == "chirp":
            # Map inst_freq (Hz) → bin indices matching spectrogram y-axis
            cwt_freqs = col["cwt_freqs"]
            inst_freq_hz = col["chirp_inst_freq"]
            inst_freq_bins = np.interp(inst_freq_hz, cwt_freqs,
                                       np.arange(len(cwt_freqs)))
            ax_ref.plot(col["waveform_time"], inst_freq_bins,
                        color=GRID_WAVEFORM_COLOR, linewidth=1.5, alpha=0.9)
            ax_ref.set_ylim(0, col["n_cwt_freqs"])
            ax_ref.set_facecolor("#1a1a2e")
            ax_ref.grid(True, color="white", alpha=0.08, linewidth=0.5)
            # Use same freq ticks as spectrogram rows
            if col_idx == 0:
                ax_ref.set_yticks(col["ytick_bins"])
                ax_ref.set_yticklabels(col["ytick_labels"], fontsize=14)
            else:
                ax_ref.set_yticks([])
        else:
            # Audio files: waveform time series
            ax_ref.plot(col["waveform_time"], col["waveform"],
                        color=GRID_WAVEFORM_COLOR, linewidth=0.2, alpha=0.8)
            peak = max(np.abs(col["waveform"]).max(), 1e-6)
            ax_ref.set_ylim(-peak * 1.10, peak * 1.10)
            ax_ref.set_facecolor("#1a1a2e")
            ax_ref.grid(True, color="gray", alpha=0.15, linewidth=0.5)
            if col_idx != 0:
                ax_ref.set_yticks([])

        ax_ref.set_xlim(0, col["duration_s"])
        for spine in ax_ref.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.8)
        plt.setp(ax_ref.get_xticklabels(), visible=False)
        ax_ref.tick_params(axis="x", length=0)

        # ── Row 1: DAW — placeholder / reference image ───────────────────
        ax_daw = axes[1][col_idx]

        daw_path = spec.get("daw_image")
        if daw_path and os.path.exists(daw_path):
            daw_img = plt.imread(daw_path)
            ax_daw.imshow(daw_img, aspect="auto",
                          extent=[0, col["duration_s"], 0, 1])
        else:
            ax_daw.set_facecolor("#2a2a2a")
            ax_daw.text(0.5, 0.5, "placeholder — generate in FL Studio",
                        transform=ax_daw.transAxes, ha="center", va="center",
                        fontsize=18, color="#666666", style="italic")
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
            ax = axes[row_idx + 2][col_idx]
            extent = [0, col["duration_s"], 0, col["n_cwt_freqs"]]
            ax.imshow(
                spec_data, cmap=GRID_CMAP, aspect="auto",
                origin="lower", extent=extent, vmin=0, vmax=vmax,
            )
            ax.grid(True, color="white", alpha=0.08, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.8)

            # Y-axis: "Freq (Hz)" on left column only
            if col_idx == 0:
                ax.set_yticks(col["ytick_bins"])
                ax.set_yticklabels(col["ytick_labels"], fontsize=14)
                if row_idx == 1:  # middle spectrogram row (PyWavelet)
                    ax.set_ylabel("Freq (Hz)", fontsize=18, labelpad=4)
            else:
                ax.set_yticks([])

            # X-axis: "Time (s)" on bottom row only
            if row_idx == 2:  # bottom spectrogram row (SubShader)
                ax.tick_params(axis="x", labelsize=14)
                if col_idx == 1:  # center column
                    ax.set_xlabel("Time (s)", fontsize=18, labelpad=4)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis="x", length=0)

    # Column labels across the top
    for col_idx, col in enumerate(column_data):
        axes[0][col_idx].set_title(col["label"], fontsize=24, fontweight="bold", pad=8)

    # Row labels on the RIGHT side (rotated 90° clockwise)
    right_labels = ["Reference", "DAW", "STFT", pywt_label, "SubShader"]
    for row_idx, label in enumerate(right_labels):
        ax_right = axes[row_idx][GRID_COLS - 1]
        ax_right.annotate(
            label, xy=(1.02, 0.5), xycoords="axes fraction",
            fontsize=20, fontweight="bold", va="center", ha="left",
            rotation=-90,
        )

    GRID_MARGIN = 0.05
    plt.subplots_adjust(left=GRID_MARGIN, right=1 - GRID_MARGIN,
                        top=1 - GRID_MARGIN, bottom=GRID_MARGIN,
                        wspace=0.04, hspace=0.08)

    # When --dpi is passed explicitly, the output is always named by DPI (no stub suffix).
    # The stub suffix only applies to the default 200 DPI path without an explicit DPI request.
    # Caller uses dpi=0 as sentinel to mean "use default naming".
    if dpi > 0:
        base_name = f"comparison_grid_{dpi}dpi.png"
    elif stub_pywt:
        base_name = "comparison_grid_STUB.png"
    else:
        base_name = "comparison_grid.png"
    out_path = os.path.join(BENCHMARKS_DIR, base_name)
    save_dpi = dpi if dpi > 0 else 200
    fig.savefig(out_path, dpi=save_dpi)
    plt.close(fig)
    print(f"Saved -> {out_path}")
    print_section_end()


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
    parser.add_argument("--timing",          action="store_true", help="STFT vs PyWavelet vs CWT timing comparison")
    parser.add_argument("--figures",         action="store_true", help="Generate all 3 README comparison PNGs (matplotlib)")
    parser.add_argument("--figures-chirp",      action="store_true", help="Generate chirp signal comparison figure only")
    parser.add_argument("--figures-polyphonic",  action="store_true", help="Generate polyphonic signal comparison figure only")
    parser.add_argument("--figures-musical",     action="store_true", help="Generate musical signal comparison figure only")
    parser.add_argument("--seaborn",         action="store_true", help="Generate 3 comparison PNGs (seaborn heatmap style)")
    parser.add_argument("--comparison-grid", action="store_true", help="Generate 3x3 comparison grid (signals x representations)")
    parser.add_argument("--unit-tests",      action="store_true", help="Run unit tests (NumPy vs CuPy, etc.)")
    parser.add_argument("--all",             action="store_true", help="Run all modes")
    parser.add_argument("--stub",            action="store_true",
                        help="With --figures: generate stub layouts instead of real DSP (fast iteration)")
    parser.add_argument("--stub-pywt",       action="store_true",
                        help="With --figures: skip PyWavelet computation, use random stub spectrograms (faster, saves to stub folder)")
    parser.add_argument("--dpi",             type=int, default=0,
                        help="Output DPI for --comparison-grid. When set, output is named comparison_grid_{dpi}dpi.png. Default: 200 DPI with standard filename.")
    args = parser.parse_args()

    any_figure_individual = args.figures_chirp or args.figures_polyphonic or args.figures_musical
    any_flag = (args.timing or args.figures or any_figure_individual or args.seaborn
                or args.unit_tests or args.stub or args.comparison_grid)
    if args.all or not any_flag:
        args.timing = args.figures = args.seaborn = args.unit_tests = True

    if args.seaborn and not SEABORN_AVAILABLE:
        print("[benchmark] --seaborn requested but seaborn is not installed. "
              "pip install seaborn\n")
        args.seaborn = False

    modes = []
    if args.timing:
        modes.append(("Timing Comparison", lambda: TimedSubShader().run()))

    if args.figures or any_figure_individual:
        backends = ["matplotlib"]
        if args.seaborn:
            backends.append("seaborn")
        if args.stub:
            modes.append(("Stub Layouts", lambda: ReadmeFigures().stub_layouts()))
        elif any_figure_individual and not args.figures:
            rf = ReadmeFigures(backends=backends, stub_pywt=args.stub_pywt)
            if args.figures_chirp:
                modes.append(("Chirp Figure", lambda r=rf: r.chirp_signal_comparison()))
            if args.figures_polyphonic:
                modes.append(("Polyphonic Figure", lambda r=rf: r.polyphonic_signal_comparison()))
            if args.figures_musical:
                modes.append(("Musical Figure", lambda r=rf: r.musical_signal_comparison()))
        else:
            modes.append(("Figures", lambda b=backends, sp=args.stub_pywt: ReadmeFigures(backends=b, stub_pywt=sp).run_all()))
    elif args.seaborn:
        modes.append(("Seaborn Figures", lambda sp=args.stub_pywt: ReadmeFigures(backends=["seaborn"], stub_pywt=sp).run_all()))

    if args.comparison_grid:
        modes.append(("Comparison Grid", lambda sp=args.stub_pywt, d=args.dpi: generate_comparison_grid(stub_pywt=sp, dpi=d)))

    if args.unit_tests:
        from utilities.unit_tests import run_all as run_unit_tests
        modes.append(("Unit Tests", run_unit_tests))

    if modes:
        run_modes(modes)
    else:
        run_default()
