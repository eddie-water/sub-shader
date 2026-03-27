import os
import time

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer

from utilities import style
from utilities import (
    BENCHMARKS_DIR,
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
    compute_freq_yticks,
    create_figure_scaffold,
    render_top_row,
    render_spectrogram_row,
    compute_stft_frame,
    build_chirp_chunks,
    build_wandering_chirp_chunks,
    build_fm_chirp_chunks,
    build_bouncing_chirp_chunks,
)

GPU_AVAILABLE = gpu_available()


# =============================================================================
# README FIGURES (--figures)
# =============================================================================

class ReadmeFigures:
    """Generate the 3 README comparison PNGs with integrated timing."""

    def __init__(self, num_frames: int = NUM_FRAMES, stub_pywt: bool = False):
        self.num_frames = num_frames
        self.stub_pywt  = stub_pywt
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)
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
                    is_bottom=bottom, cmap=style.GRID_CMAP)

            path = os.path.join(stub_dir, cfg["filename"])
            fig.savefig(path, dpi=style.STUB_DPI)
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
        pywt_subtitle = "PyWavelet CWT (stub)" if self.stub_pywt else "PyWavelet CWT"
        subtitle = f"STFT  |  {pywt_subtitle}  |  SubShader CWT"

        fig, gs, ax_stft, ax_pywt, ax_npwt = create_figure_scaffold(
            title, subtitle, n_top)

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
                is_bottom=bottom, cmap=style.GRID_CMAP,
                n_cwt_freqs=n_cwt_freqs, duration_s=duration_s)

        save_path = os.path.join(output_dir, out_filename)
        fig.savefig(save_path, dpi=style.DEFAULT_DPI)
        plt.close(fig)
        print(f"Saved -> {save_path}")
        print_section_end()

        return timing
