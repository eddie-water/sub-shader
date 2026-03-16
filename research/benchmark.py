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
import time
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft as scipy_stft, resample as scipy_resample

from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet
from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer

from benchmark_utilities import (
    NEW_LINE,
    live_row, live_rows,
    print_header, print_total,
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

AUDIO_DEFAULT    = "assets/audio/daw/a2a3_a4_minor_scale.wav"
AUDIO_CHIRP      = "assets/audio/daw/chirp_beat.wav"
AUDIO_POLYPHONIC = "assets/audio/daw/c4_and_c7_4_arps.wav"
AUDIO_MUSICAL    = "assets/audio/songs/beltran_sc_rip.wav"

FL_STUDIO_CHIRP      = None
FL_STUDIO_POLYPHONIC = None
FL_STUDIO_MUSICAL    = "assets/images/beltran_souncloud_wav_0m_8s_to_0m_25s.png"

STFT_NPERSEG     = 1024
NUM_FRAMES       = 128   # frames to accumulate for figure snapshots


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
            live_rows(live_labels, frames_processed, self.num_frames,
                      live_time_arrays, num_rows=3)

        total_s = time.perf_counter() - t_total_start
        print_total(total_s)


# =============================================================================
# README FIGURES (--figures)
# =============================================================================

class ReadmeFigures:
    """Generate the 3 README comparison PNGs with integrated timing."""

    def __init__(self, num_frames: int = NUM_FRAMES):
        self.num_frames = num_frames
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public figure generators
    # -------------------------------------------------------------------------

    def chirp_signal_comparison(self):
        """STFT | PyWt | SubShader CWT on a synthetic linear chirp (100 Hz → 10 kHz)."""
        print("Chirp Signal Comparison\n",      flush=True)

        from scipy.signal import chirp as scipy_chirp

        config    = get_default_config()
        sr        = int(config.wavelet.typical_sampling_freq)
        chunk_size = config.audio.chunk_size
        hop_size  = int(chunk_size * (1 - config.audio.overlap_factor))

        total_samples = hop_size * self.num_frames + chunk_size
        t      = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
        signal = scipy_chirp(t, f0=100, f1=10000, t1=t[-1], method='linear').astype(np.float64)

        chunks = [signal[i * hop_size: i * hop_size + chunk_size]
                  for i in range(self.num_frames)]

        return self._generate_comparison_figure(
            audio_path=None,
            title="Chirp Signal (100 Hz → 10 kHz) — STFT vs PyWavelet vs SubShader CWT",
            filename="chirp_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_CHIRP,
            audio_chunks=chunks,
            sample_rate=float(sr),
            freq_line={'f0': 100, 'f1': 10000},
        )

    def polyphonic_signal_comparison(self):
        print("Polyphonic Signal Comparison\n",  flush=True)

        """FL Studio | STFT | PyWt | SubShader CWT on polyphonic audio."""
        return self._generate_comparison_figure(
            audio_path=AUDIO_POLYPHONIC,
            title="Polyphonic Signal — FL Studio vs STFT vs PyWavelet vs SubShader CWT",
            filename="polyphonic_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_POLYPHONIC,
        )

    def musical_signal_comparison(self):
        print("Musical Signal Comparison\n",     flush=True)

        """FL Studio | STFT | PyWt | SubShader CWT on a full musical track."""
        return self._generate_comparison_figure(
            audio_path=AUDIO_MUSICAL,
            title="Musical Signal — FL Studio vs STFT vs PyWavelet vs SubShader CWT",
            filename="musical_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_MUSICAL,
        )

    def run_all(self):
        """Generate all 3 comparison figures."""
        print(f"\n=== Generating README Figures → {BENCHMARKS_DIR}/ ===\n")

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
    # Internal: comparison figure pipeline
    # -------------------------------------------------------------------------

    def _generate_comparison_figure(self, audio_path, title, filename,
                                     fl_studio_img_path=None,
                                     audio_chunks=None, sample_rate=None,
                                     freq_line=None):
        """
        Generate a 5-row stacked comparison figure and save to disk.

        Layout:
          Row 0 : Audio waveform
          Row 1 : FL Studio VST reference (or placeholder)
          Row 2 : STFT
          Row 3 : PyWavelet CWT
          Row 4 : SubShader CWT
        """
        config = get_default_config()

        if audio_chunks is not None:
            sr = sample_rate
            chunk_iter = iter(audio_chunks)
        else:
            config.audio.file_path = audio_path
            ai = AudioInput(
                path=config.audio.file_path,
                chunk_size=config.audio.chunk_size,
                overlap_factor=config.audio.overlap_factor,
            )
            sr = ai.get_sample_rate()
            chunk_iter = None

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
        audio_buf  = AudioFrameBuffer(chunk_size=config.audio.chunk_size, num_chunks=self.num_frames)
        stft_buf   = CircularFrameBuffer(frame_shape=(n_cwt_freqs, stft_target_w), num_frames=self.num_frames, color_norm_config=color_norm)
        pywt_buf   = CircularFrameBuffer(frame_shape=pywt.get_output_shape(),       num_frames=self.num_frames, color_norm_config=color_norm)
        npwt_buf   = CircularFrameBuffer(frame_shape=npwt.get_output_shape(),       num_frames=self.num_frames, color_norm_config=color_norm)

        # Process frames — with per-method timing and live table
        stft_times = np.empty(self.num_frames)
        pywt_times = np.empty(self.num_frames)
        npwt_times = np.empty(self.num_frames)
        frames_processed = 0

        live_labels = ["STFT", "PyWavelet CWT", "SubShader CWT"]
        live_times  = [stft_times, pywt_times, npwt_times]
        print_header()

        for i in range(self.num_frames):
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
            live_rows(live_labels, frames_processed, self.num_frames,
                      live_times, num_rows=3)
        
        print(NEW_LINE)

        # Trim timing arrays to actual frame count
        stft_times = stft_times[:frames_processed]
        pywt_times = pywt_times[:frames_processed]
        npwt_times = npwt_times[:frames_processed]

        timing = {
            "STFT":           compute_timing_stats(stft_times),
            "PyWavelet CWT":  compute_timing_stats(pywt_times),
            "SubShader CWT":  compute_timing_stats(npwt_times),
        }

        # Flatten buffers
        stft_spec = stft_buf.get_flattened_buffer()
        pywt_spec = pywt_buf.get_flattened_buffer()
        npwt_spec = npwt_buf.get_flattened_buffer()
        spec_w    = pywt_spec.shape[1]
        x, y_min, y_max = audio_buf.get_downsampled(spec_w)

        duration_s  = audio_buf.total_samples / sr
        extent_spec = [0, duration_s, 0, n_cwt_freqs]
        t_audio     = np.linspace(0, duration_s, len(x))

        # Y-tick positions at octave (A-note) boundaries
        a_note_freqs  = [27.5 * (2 ** i) for i in range(11)]
        spec_ytick_bins   = []
        spec_ytick_labels = []
        for f in a_note_freqs:
            if freq_min <= f <= freq_max:
                spec_ytick_bins.append(float(np.interp(f, cwt_freqs, np.arange(n_cwt_freqs))))
                spec_ytick_labels.append(f'{f/1000:.1f}k' if f >= 1000 else f'{int(round(f))}')

        # ── Figure layout ─────────────────────────────────────────────────────
        timing_subtitle = (
            f"Avg per frame:  STFT {timing['STFT']['avg_ms']:.2f} ms  |  "
            f"PyWavelet {timing['PyWavelet CWT']['avg_ms']:.2f} ms  |  "
            f"SubShader {timing['SubShader CWT']['avg_ms']:.2f} ms"
        )
        cmap = "magma"
        fig = plt.figure(figsize=(14, 18))
        fig.suptitle(f"{title}\n{timing_subtitle}", fontsize=13, y=0.995)
        gs  = gridspec.GridSpec(5, 1, figure=fig,
                                height_ratios=[0.5, 1.5, 1.5, 1.5, 1.5],
                                hspace=0.35)
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.05, top=0.96)

        ax_stft  = fig.add_subplot(gs[2])
        ax_pywt  = fig.add_subplot(gs[3], sharex=ax_stft, sharey=ax_stft)
        ax_npwt  = fig.add_subplot(gs[4], sharex=ax_stft, sharey=ax_stft)
        ax_audio = fig.add_subplot(gs[0], sharex=ax_stft)

        if freq_line is not None:
            ax_fl = fig.add_subplot(gs[1], sharex=ax_stft, sharey=ax_stft)
        else:
            ax_fl = fig.add_subplot(gs[1])

        # Row 0: Audio waveform
        ax_audio.fill_between(t_audio, y_min, y_max, color="#1A1A1A", alpha=0.75)
        ax_audio.set_ylabel("Amplitude", fontsize=9)
        ax_audio.set_title("Audio Waveform", fontsize=10, loc="left")
        ax_audio.tick_params(labelsize=8)
        plt.setp(ax_audio.get_xticklabels(), visible=False)

        # Row 1: Frequency line or FL Studio reference
        if freq_line is not None:
            t_curve   = np.linspace(0, duration_s, 500)
            f_curve   = freq_line['f0'] + (freq_line['f1'] - freq_line['f0']) * t_curve / duration_s
            bin_curve = np.interp(f_curve, cwt_freqs, np.arange(n_cwt_freqs))
            ax_fl.plot(t_curve, bin_curve, color='white', linewidth=2)
            ax_fl.set_facecolor('#1A1A1A')
            ax_fl.set_title("Instantaneous Frequency", fontsize=10, loc="left")
            ax_fl.set_ylabel("Freq", fontsize=9)
            ax_fl.set_yticks(spec_ytick_bins)
            ax_fl.set_yticklabels(spec_ytick_labels)
            ax_fl.tick_params(labelsize=8)
            plt.setp(ax_fl.get_xticklabels(), visible=False)
        else:
            if fl_studio_img_path and os.path.exists(fl_studio_img_path):
                img = plt.imread(fl_studio_img_path)
                ax_fl.imshow(img, aspect="auto", origin="upper")
            else:
                ax_fl.set_facecolor("#2A2A2A")
                ax_fl.text(0.5, 0.5, "FL Studio VST\n(reference not available)",
                           ha="center", va="center", color="#888888",
                           fontsize=10, transform=ax_fl.transAxes)
            ax_fl.set_title("FL Studio VST  (reference)", fontsize=10, loc="left")
            ax_fl.axis("off")

        # Row 2: STFT
        ax_stft.imshow(stft_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=stft_buf.get_intensity_max())
        ax_stft.set_title(f"STFT — avg {timing['STFT']['avg_ms']:.2f} ms/frame", fontsize=10, loc="left")
        ax_stft.set_ylabel("Freq", fontsize=9)
        ax_stft.set_yticks(spec_ytick_bins)
        ax_stft.set_yticklabels(spec_ytick_labels)
        ax_stft.tick_params(labelsize=8)
        plt.setp(ax_stft.get_xticklabels(), visible=False)

        # Row 3: PyWavelet CWT
        ax_pywt.imshow(pywt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=pywt_buf.get_intensity_max())
        ax_pywt.set_title(f"PyWavelet CWT — avg {timing['PyWavelet CWT']['avg_ms']:.2f} ms/frame", fontsize=10, loc="left")
        ax_pywt.set_ylabel("Freq", fontsize=9)
        ax_pywt.set_yticks(spec_ytick_bins)
        ax_pywt.set_yticklabels(spec_ytick_labels)
        ax_pywt.tick_params(labelsize=8)
        plt.setp(ax_pywt.get_xticklabels(), visible=False)

        # Row 4: SubShader CWT
        subshader_base  = "SubShader CWT" if GPU_AVAILABLE else "SubShader CWT (NumPy)"
        subshader_label = f"{subshader_base} — avg {timing['SubShader CWT']['avg_ms']:.2f} ms/frame"
        ax_npwt.imshow(npwt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=npwt_buf.get_intensity_max())
        ax_npwt.set_title(subshader_label, fontsize=10, loc="left")
        ax_npwt.set_ylabel("Freq", fontsize=9)
        ax_npwt.set_yticks(spec_ytick_bins)
        ax_npwt.set_yticklabels(spec_ytick_labels)
        ax_npwt.set_xlabel("Time (s)", fontsize=9)
        ax_npwt.tick_params(labelsize=8)

        path = os.path.join(BENCHMARKS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}\n")

        return timing


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
    args = parser.parse_args()

    any_flag = args.timing or args.figures or args.unit_tests or args.all

    if not any_flag:
        run_default()
    else:
        if args.timing or args.all:
            TimedSubShader().run()

        if args.figures or args.all:
            ReadmeFigures().run_all()

        if args.unit_tests or args.all:
            from unit_tests import run_all as run_unit_tests
            run_unit_tests()
