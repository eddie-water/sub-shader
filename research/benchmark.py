"""
Benchmark Module for SubShader Performance Analysis.

Two top-level classes:
  - ModuleBenchmarks : time individual pipeline components, print table, save CSV
  - ReadmeFigures    : generate comparison PNGs for README image placeholders
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import csv
import time
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless — no display required for figure saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft as scipy_stft, resample as scipy_resample

from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet, CuPyWavelet
from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer

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

NUM_ITERATIONS        = 500   # fast components (audio read, STFT, NumPy/CuPy CWT)
NUM_ITERATIONS_SLOW   = 20    # slow components (PyWavelet ~180 ms/call → ~4 s total)
BENCHMARKS_DIR   = "assets/images/benchmarks"
RESULTS_CSV      = "research/benchmark_results.csv"

AUDIO_DEFAULT    = "assets/audio/daw/a2a3_a4_minor_scale.wav"
AUDIO_CHIRP      = "assets/audio/daw/chirp_beat.wav"
AUDIO_POLYPHONIC = "assets/audio/daw/c4_and_c7_4_arps.wav"
AUDIO_MUSICAL    = "assets/audio/songs/beltran_sc_rip.wav"

# FL Studio VST reference screenshots — used as the ground-truth panel in comparison figures.
# Set to None for signals where no screenshot is available (a grey placeholder is shown instead).
FL_STUDIO_CHIRP      = None
FL_STUDIO_POLYPHONIC = None
FL_STUDIO_MUSICAL    = "assets/images/beltran_souncloud_wav_0m_8s_to_0m_25s.png"

STFT_NPERSEG     = 1024
NUM_FRAMES       = 128   # frames to accumulate for figure snapshots

# =============================================================================
# MODULE BENCHMARKS
# =============================================================================

class ModuleBenchmarks:
    """
    Time individual SubShader pipeline components in isolation.

    Usage::

        mb = ModuleBenchmarks()
        results = mb.run_all()      # prints table, saves CSV, returns list of dicts
    """

    def __init__(self, audio_path: str = AUDIO_DEFAULT, n_iter: int = NUM_ITERATIONS):
        self.n_iter = n_iter

        config = get_default_config()
        config.audio.file_path = audio_path

        self.audio_input = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )
        self.sample_rate = self.audio_input.get_sample_rate()
        self.audio_data  = self.audio_input.get_chunk()

        self.pywt = PyWavelet(
            sample_rate=self.sample_rate,
            input_n=config.audio.chunk_size,
            config=config.wavelet,
        )
        self.npwt = NumPyWavelet(
            sample_rate=self.sample_rate,
            input_n=config.audio.chunk_size,
            config=config.wavelet,
        )
        if GPU_AVAILABLE:
            self.cpwt = CuPyWavelet(
                sample_rate=self.sample_rate,
                input_n=config.audio.chunk_size,
                config=config.wavelet,
            )
        else:
            self.cpwt = None

    # -------------------------------------------------------------------------
    # Timers
    # -------------------------------------------------------------------------

    def _time_fn(self, label: str, fn, *args, n_iter: int = None) -> dict:
        """Run fn(*args) for n_iter iterations, return timing stats (ms)."""
        n = n_iter if n_iter is not None else self.n_iter
        times = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter()
            fn(*args)
            times[i] = (time.perf_counter() - t0) * 1000.0
        return {
            "component":  label,
            "avg_ms":     float(np.mean(times)),
            "min_ms":     float(np.min(times)),
            "max_ms":     float(np.max(times)),
            "iterations": n,
        }

    def time_audio_input(self) -> dict:
        """Benchmark AudioInput.get_chunk() — resets file position each iteration."""
        def _read():
            self.audio_input.file_pos = 0
            self.audio_input.get_chunk()

        return self._time_fn("AudioInput.get_chunk()", _read)

    def time_stft(self) -> dict:
        """Benchmark scipy.signal.stft on a single audio chunk."""
        def _stft():
            scipy_stft(self.audio_data, fs=self.sample_rate, nperseg=STFT_NPERSEG)

        return self._time_fn("scipy STFT", _stft)

    def time_pywt_cwt(self) -> dict:
        """Benchmark PyWavelet.cwt() — uses fewer iterations (slow ~180 ms/call)."""
        return self._time_fn("PyWavelet.cwt()", self.pywt.cwt, self.audio_data,
                             n_iter=NUM_ITERATIONS_SLOW)

    def time_numpy_cwt(self) -> dict:
        """Benchmark NumPyWavelet.cwt()."""
        return self._time_fn("NumPyWavelet.cwt()", self.npwt.cwt, self.audio_data)

    def time_cupy_cwt(self) -> dict:
        """Benchmark CuPyWavelet.cwt() — returns None if no GPU is available."""
        if self.cpwt is None:
            print("  [skipped] CuPyWavelet.cwt() — no GPU available")
            return None
        return self._time_fn("CuPyWavelet.cwt()", self.cpwt.cwt, self.audio_data)

    # -------------------------------------------------------------------------
    # NumPy vs CuPy correctness verification
    # -------------------------------------------------------------------------

    def verify_numpy_vs_cupy(self, save_figure: bool = True):
        """
        Verify that NumPyWavelet and CuPyWavelet produce numerically equivalent outputs.

        Checks:
          1. Global absolute / relative error statistics
          2. Per-frequency-bin Pearson correlation (all bins should be > 0.999)
          3. Saves a 3-panel diff figure: NumPy | CuPy | abs(diff)

        Requires a GPU — exits early with a message if none is available.
        """
        if self.cpwt is None:
            print("[verify] Skipped — no GPU available. Reconnect GPU and re-run.")
            return

        print("\nVerifying NumPy vs CuPy CWT outputs …\n")

        np_coefs = self.npwt.cwt(self.audio_data)   # (freq_bins, time_bins)
        cp_coefs = self.cpwt.cwt(self.audio_data)   # same shape

        diff       = np.abs(np_coefs - cp_coefs)
        max_val    = np_coefs.max() + 1e-12
        rel_error  = diff.max() / max_val

        print(f"  shape          : {np_coefs.shape}")
        print(f"  max abs diff   : {diff.max():.2e}")
        print(f"  mean abs diff  : {diff.mean():.2e}")
        print(f"  rel error      : {rel_error:.2e}")

        # Per-frequency-bin Pearson correlation
        n_freq    = np_coefs.shape[0]
        corrs     = np.array([
            np.corrcoef(np_coefs[i], cp_coefs[i])[0, 1]
            for i in range(n_freq)
        ])
        low_corr  = np.where(corrs < 0.999)[0]
        print(f"  bins < 0.999 r : {len(low_corr)} / {n_freq}", end="")
        if len(low_corr):
            print(f"  ← bins: {low_corr[:10]}")
        else:
            print("  ✓")

        passed = rel_error < 1e-3 and len(low_corr) == 0
        print(f"\n  {'PASS ✓' if passed else 'FAIL ✗ — inspect diff figure'}")

        if save_figure:
            self._save_diff_figure(np_coefs, cp_coefs, diff)

    def _save_diff_figure(self, np_coefs, cp_coefs, diff):
        """Save 3-panel figure: NumPy | CuPy | abs(diff)."""
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("NumPy vs CuPy CWT — correctness check", fontsize=12)

        vmax = max(np_coefs.max(), cp_coefs.max())

        axes[0].imshow(np_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
        axes[0].set_title("NumPyWavelet.cwt()")
        axes[0].set_ylabel("Frequency bin")
        axes[0].set_xlabel("Time bin")

        axes[1].imshow(cp_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
        axes[1].set_title("CuPyWavelet.cwt()")
        axes[1].set_xlabel("Time bin")

        im = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto", origin="lower")
        axes[2].set_title("abs(NumPy − CuPy)")
        axes[2].set_xlabel("Time bin")
        fig.colorbar(im, ax=axes[2], shrink=0.8, label="abs diff")

        fig.tight_layout()
        path = os.path.join(BENCHMARKS_DIR, "numpy_vs_cupy_diff.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Diff figure saved → {path}\n")

    # -------------------------------------------------------------------------
    # run_all
    # -------------------------------------------------------------------------

    def run_all(self, save_csv: bool = True) -> list:
        """
        Run all timers, print a results table, optionally save CSV.

        Returns:
            list[dict]: Each dict has keys component, avg_ms, min_ms, max_ms, iterations.
        """
        print("\nRunning module benchmarks …\n")

        timers = [
            self.time_audio_input,
            self.time_stft,
            self.time_pywt_cwt,
            self.time_numpy_cwt,
            self.time_cupy_cwt,
        ]

        results = []
        t_total_start = time.perf_counter()
        for fn in timers:
            print(f"  timing {fn.__name__} …", flush=True)
            result = fn()
            if result is not None:
                results.append(result)
        total_s = time.perf_counter() - t_total_start

        self._print_table(results, total_s)

        if save_csv:
            self._save_csv(results)

        return results

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _print_table(self, results: list, total_s: float = None):
        header = f"{'Component':<32} {'Iterations':>12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}"
        sep    = "-" * len(header)
        print(f"\n{header}")
        print(sep)
        for r in results:
            print(
                f"{r['component']:<32} "
                f"{r['iterations']:>12} "
                f"{r['avg_ms']:>10.2f} "
                f"{r['min_ms']:>10.2f} "
                f"{r['max_ms']:>10.2f}"
            )
        if total_s is not None:
            print(sep)
            print(f"{'Total test time':<32} {'':>12} {total_s * 1000:>10.2f} ms   ({total_s:.2f} s)")
        print()

    def _save_csv(self, results: list):
        os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["component", "avg_ms", "min_ms", "max_ms", "iterations"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved → {RESULTS_CSV}")


# =============================================================================
# README FIGURES
# =============================================================================

class ReadmeFigures:
    """
    Generate comparison PNGs for README image placeholders.

    Saves all figures to assets/images/benchmarks/.

    Usage::

        ReadmeFigures().run_all()
    """

    def __init__(self, num_frames: int = NUM_FRAMES):
        self.num_frames = num_frames
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public figure generators
    # -------------------------------------------------------------------------

    def chirp_signal_comparison(self):
        """FL Studio | STFT | PyWt | SubShader CWT on a chirp signal."""
        self._generate_comparison_figure(
            audio_path=AUDIO_CHIRP,
            title="Chirp Signal — FL Studio vs STFT vs PyWavelet vs SubShader CWT",
            filename="chirp_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_CHIRP,
        )

    def polyphonic_signal_comparison(self):
        """FL Studio | STFT | PyWt | SubShader CWT on polyphonic audio."""
        self._generate_comparison_figure(
            audio_path=AUDIO_POLYPHONIC,
            title="Polyphonic Signal — FL Studio vs STFT vs PyWavelet vs SubShader CWT",
            filename="polyphonic_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_POLYPHONIC,
        )

    def musical_signal_comparison(self):
        """FL Studio | STFT | PyWt | SubShader CWT on a full musical track."""
        self._generate_comparison_figure(
            audio_path=AUDIO_MUSICAL,
            title="Musical Signal — FL Studio vs STFT vs PyWavelet vs SubShader CWT",
            filename="musical_signal_comparison.png",
            fl_studio_img_path=FL_STUDIO_MUSICAL,
        )

    def timing_bar_chart(self, results: list = None):
        """
        Bar chart of ModuleBenchmarks timing results.

        Args:
            results: Pre-computed results from ModuleBenchmarks.run_all().
                     If None, ModuleBenchmarks is run automatically.
        """
        if results is None:
            print("  Running ModuleBenchmarks to collect timing data …")
            results = ModuleBenchmarks().run_all(save_csv=False)

        labels  = [r["component"] for r in results]
        avgs    = [r["avg_ms"]    for r in results]
        mins    = [r["min_ms"]    for r in results]
        maxs    = [r["max_ms"]    for r in results]

        x      = np.arange(len(labels))
        err_lo = np.array(avgs) - np.array(mins)
        err_hi = np.array(maxs) - np.array(avgs)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(x, avgs, color="#5A7FBF", edgecolor="white", linewidth=0.5)
        ax.errorbar(x, avgs, yerr=[err_lo, err_hi], fmt="none", color="black",
                    capsize=4, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Time (ms)")
        ax.set_title("SubShader Component Timing (avg ± min/max)")
        ax.set_facecolor("#F8F8F8")
        fig.tight_layout()

        path = os.path.join(BENCHMARKS_DIR, "timing_bar_chart.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    def numpy_vs_cupy_figure(self, results: list = None):
        """
        Side-by-side NumPy CWT vs CuPy CWT output + timing comparison subtitle.

        Args:
            results: Pre-computed results from ModuleBenchmarks.run_all().
                     If None, ModuleBenchmarks is run automatically.
        """
        if not GPU_AVAILABLE:
            print("  [skipped] numpy_vs_cupy_figure — no GPU available")
            return

        if results is None:
            print("  Running ModuleBenchmarks for numpy_vs_cupy timing …")
            results = ModuleBenchmarks().run_all(save_csv=False)

        # Grab timing entries
        np_row  = next((r for r in results if "NumPy" in r["component"]), None)
        cp_row  = next((r for r in results if "CuPy"  in r["component"]), None)

        config = get_default_config()
        config.audio.file_path = AUDIO_DEFAULT
        ai = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )
        sr   = ai.get_sample_rate()
        data = ai.get_chunk()

        npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
        cpwt = CuPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

        np_coefs = npwt.cwt(data)
        cp_coefs = cpwt.cwt(data)

        subtitle = ""
        if np_row and cp_row:
            subtitle = (
                f"NumPy avg {np_row['avg_ms']:.2f} ms   |   "
                f"CuPy avg {cp_row['avg_ms']:.2f} ms   |   "
                f"Speedup {np_row['avg_ms'] / cp_row['avg_ms']:.1f}×"
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"NumPyWavelet vs CuPyWavelet CWT\n{subtitle}", fontsize=11)

        axes[0].imshow(np_coefs, cmap="magma", aspect="auto", origin="lower")
        axes[0].set_title("NumPyWavelet.cwt()")
        axes[0].set_xlabel("Time bins")
        axes[0].set_ylabel("Frequency bins")

        axes[1].imshow(cp_coefs, cmap="magma", aspect="auto", origin="lower")
        axes[1].set_title("CuPyWavelet.cwt()")
        axes[1].set_xlabel("Time bins")

        fig.tight_layout()
        path = os.path.join(BENCHMARKS_DIR, "numpy_vs_cupy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    def run_all(self):
        """Run all figure generators and save PNGs."""
        print(f"\nGenerating README figures → {BENCHMARKS_DIR}/\n")

        # Collect timing data once and reuse
        print("  Running ModuleBenchmarks for timing figures …")
        mb_results = ModuleBenchmarks().run_all(save_csv=False)

        print("  chirp_signal_comparison …",      flush=True)
        self.chirp_signal_comparison()

        print("  polyphonic_signal_comparison …",  flush=True)
        self.polyphonic_signal_comparison()

        print("  musical_signal_comparison …",     flush=True)
        self.musical_signal_comparison()

        print("  timing_bar_chart …",              flush=True)
        self.timing_bar_chart(results=mb_results)

        print("  numpy_vs_cupy_figure …",          flush=True)
        self.numpy_vs_cupy_figure(results=mb_results)

        print("\nAll figures saved.\n")

    # -------------------------------------------------------------------------
    # Internal: comparison figure pipeline
    # -------------------------------------------------------------------------

    def _generate_comparison_figure(self, audio_path: str, title: str, filename: str,
                                     fl_studio_img_path: str = None):
        """
        Generate a 4-panel spectrogram comparison and save to disk.

        Layout:
          Row 0 : Audio waveform (full width)
          Row 1 : FL Studio VST | STFT | PyWavelet CWT | SubShader CWT

        fl_studio_img_path : path to a FL Studio screenshot, or None for a placeholder.
        """
        config = get_default_config()
        config.audio.file_path = audio_path

        ai = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )
        sr = ai.get_sample_rate()

        pywt = PyWavelet( sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
        npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

        # STFT frequency crop — match chromatic scale range used by CWT
        stft_freqs     = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / sr)
        freq_min, freq_max = pywt.freqs[0], pywt.freqs[-1]
        stft_freq_mask = (stft_freqs >= freq_min) & (stft_freqs <= freq_max)
        stft_num_freqs = int(stft_freq_mask.sum())
        stft_target_w  = pywt.output_n

        # Circular buffers
        color_norm = ColorNormalizationConfig()
        audio_buf  = AudioFrameBuffer(chunk_size=config.audio.chunk_size, num_chunks=self.num_frames)
        stft_buf   = CircularFrameBuffer(frame_shape=(stft_num_freqs, stft_target_w), num_frames=self.num_frames, color_norm_config=color_norm)
        pywt_buf   = CircularFrameBuffer(frame_shape=pywt.get_output_shape(),         num_frames=self.num_frames, color_norm_config=color_norm)
        npwt_buf   = CircularFrameBuffer(frame_shape=npwt.get_output_shape(),         num_frames=self.num_frames, color_norm_config=color_norm)

        # Process frames
        for _ in range(self.num_frames):
            chunk = ai.get_chunk()
            if chunk is None:
                break

            audio_buf.push_chunk(chunk)

            _, _, Zxx      = scipy_stft(chunk, fs=sr, nperseg=STFT_NPERSEG)
            stft_mag       = np.abs(Zxx)[stft_freq_mask, :]
            stft_resampled = scipy_resample(stft_mag, stft_target_w, axis=1)
            stft_resampled = np.clip(stft_resampled, 0, None)
            stft_buf.push_frame(stft_resampled)

            pywt_buf.push_frame(pywt.cwt(chunk))
            npwt_buf.push_frame(npwt.cwt(chunk))

        # Flatten buffers
        stft_spec = stft_buf.get_flattened_buffer()
        pywt_spec = pywt_buf.get_flattened_buffer()
        npwt_spec = npwt_buf.get_flattened_buffer()
        spec_w    = pywt_spec.shape[1]
        x, y_min, y_max = audio_buf.get_downsampled(spec_w)

        # Shared axis coordinates
        duration_s   = audio_buf.total_samples / sr
        freq_min_khz = freq_min / 1000.0
        freq_max_khz = freq_max / 1000.0
        # extent = [x_left, x_right, y_bottom, y_top] in data units
        extent_spec  = [0, duration_s, freq_min_khz, freq_max_khz]
        # Audio x in seconds — linspace over the same time window
        t_audio      = np.linspace(0, duration_s, len(x))

        # ── Figure layout: 5 rows, 1 column ───────────────────────────────────
        cmap = "magma"
        fig = plt.figure(figsize=(14, 18))
        fig.suptitle(title, fontsize=13, y=0.995)
        gs  = gridspec.GridSpec(5, 1, figure=fig,
                                height_ratios=[0.5, 1.5, 1.5, 1.5, 1.5],
                                hspace=0.35)
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.05, top=0.96)

        # Spectrogram axes created first so sharex/sharey can reference ax_stft
        ax_stft = fig.add_subplot(gs[2])
        ax_pywt = fig.add_subplot(gs[3], sharex=ax_stft, sharey=ax_stft)
        ax_npwt = fig.add_subplot(gs[4], sharex=ax_stft, sharey=ax_stft)

        # Audio waveform shares x-axis (time in seconds), independent y
        ax_audio = fig.add_subplot(gs[0], sharex=ax_stft)

        # FL Studio reference — independent axes (screenshot, not time-aligned)
        ax_fl = fig.add_subplot(gs[1])

        # ── Row 0: Audio waveform ──────────────────────────────────────────────
        ax_audio.fill_between(t_audio, y_min, y_max, color="#1A1A1A", alpha=0.75)
        ax_audio.set_ylabel("Amplitude", fontsize=9)
        ax_audio.set_title("Audio Waveform", fontsize=10, loc="left")
        ax_audio.tick_params(labelsize=8)
        plt.setp(ax_audio.get_xticklabels(), visible=False)

        # ── Row 1: FL Studio VST reference ────────────────────────────────────
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

        # ── Row 2: STFT ───────────────────────────────────────────────────────
        ax_stft.imshow(stft_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=stft_buf.get_intensity_max())
        ax_stft.set_title("STFT", fontsize=10, loc="left")
        ax_stft.set_ylabel("Freq (kHz)", fontsize=9)
        ax_stft.tick_params(labelsize=8)
        plt.setp(ax_stft.get_xticklabels(), visible=False)

        # ── Row 3: PyWavelet CWT ──────────────────────────────────────────────
        ax_pywt.imshow(pywt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=pywt_buf.get_intensity_max())
        ax_pywt.set_title("PyWavelet CWT", fontsize=10, loc="left")
        ax_pywt.set_ylabel("Freq (kHz)", fontsize=9)
        ax_pywt.tick_params(labelsize=8)
        plt.setp(ax_pywt.get_xticklabels(), visible=False)

        # ── Row 4: SubShader CWT ──────────────────────────────────────────────
        subshader_label = "SubShader CWT" if GPU_AVAILABLE else "SubShader CWT (NumPy)"
        ax_npwt.imshow(npwt_spec, cmap=cmap, aspect="auto", origin="lower",
                       extent=extent_spec, vmin=0, vmax=npwt_buf.get_intensity_max())
        ax_npwt.set_title(subshader_label, fontsize=10, loc="left")
        ax_npwt.set_ylabel("Freq (kHz)", fontsize=9)
        ax_npwt.set_xlabel("Time (s)", fontsize=9)
        ax_npwt.tick_params(labelsize=8)

        path = os.path.join(BENCHMARKS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SubShader benchmarks and README figure generator"
    )
    parser.add_argument("--modules", action="store_true", help="Run ModuleBenchmarks")
    parser.add_argument("--figures", action="store_true", help="Run ReadmeFigures")
    parser.add_argument("--verify",  action="store_true", help="Verify NumPy vs CuPy correctness (requires GPU)")
    parser.add_argument("--all",     action="store_true", help="Run modules + figures (not verify)")
    args = parser.parse_args()

    any_flag = args.modules or args.figures or args.verify or args.all

    # Default (no flags): run modules + figures
    run_modules = args.modules or args.all or not any_flag
    run_figures = args.figures or args.all or not any_flag

    if run_modules:
        ModuleBenchmarks().run_all()

    if run_figures:
        ReadmeFigures().run_all()

    if args.verify:
        ModuleBenchmarks().verify_numpy_vs_cupy()
