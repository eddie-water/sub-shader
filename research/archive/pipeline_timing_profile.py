"""
Pipeline Latency Profiler for SubShader.

Instruments every stage of the audio-visual pipeline to measure wall-clock
latency per stage, identify the bottleneck, and compare against DAW STFT
baseline expectations.

Stages measured:
  - AudioInput.get_chunk()        — disk/memory seek + read
  - fft (CPU pre-processing)      — NumPy FFT before GPU upload
  - GPU upload                    — cp.asarray (host → device)
  - GPU multiply                  — frequency-domain convolution
  - GPU download                  — cp.asnumpy (device → host)
  - ifft (CPU post-processing)    — NumPy IFFT
  - CWT total                     — wavelet.cwt() end-to-end
  - normalize_by_scale            — scale normalization
  - compute_mag                   — |x| magnitude
  - discard_unreliable_coefs      — reliable region slice
  - extract_hop_center            — hop-center trim
  - downsample                    — time-axis downsampling
  - CircularFrameBuffer.push_frame — buffer update + flattening
  - Renderer.update_texture       — texture upload (CPU→OpenGL)
  - gl_context operations         — clear + render + swap

Usage:
    cd research/
    python pipeline_timing_profile.py [--frames N] [--headless]

    --frames N     Number of frames to profile (default: 64)
    --headless     Skip GL render stage (for CI or GPU-only environments)
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path so subshader package is importable from research/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput

# ── Timing helpers ────────────────────────────────────────────────────────────

def time_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def stats(arr):
    """Return dict of timing statistics (ms) from a list of floats."""
    a = np.array(arr, dtype=np.float64)
    return {
        "n":      len(a),
        "avg_ms": float(np.mean(a)),
        "med_ms": float(np.median(a)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "std_ms": float(np.std(a)),
    }


# ── GPU-stage instrumented CWT ────────────────────────────────────────────────

class InstrumentedCuWavelet:
    """
    Wraps CuWavelet and captures per-sub-stage timing on every cwt() call.

    Sub-stages tracked:
        fft_cpu      — NumPy FFT (input signal, CPU)
        gpu_upload   — cp.asarray (host → device)
        gpu_multiply — frequency-domain elementwise multiply
        gpu_download — cp.asnumpy (device → host)
        ifft_cpu     — NumPy IFFT + slice (CPU)
        normalize    — normalize_by_scale
        mag          — compute_mag
        discard      — discard_unreliable_coefs
        hop_center   — extract_hop_center
        downsample   — downsample
        cwt_total    — full cwt() wall time
    """

    SUBSTAGES = [
        "fft_cpu",
        "gpu_upload",
        "gpu_multiply",
        "gpu_download",
        "ifft_cpu",
        "normalize",
        "mag",
        "discard",
        "hop_center",
        "downsample",
        "cwt_total",
    ]

    def __init__(self, wavelet):
        self._w = wavelet
        self.timing = {s: [] for s in self.SUBSTAGES}

    def cwt(self, input_data: np.ndarray) -> np.ndarray:
        """Drop-in replacement for wavelet.cwt() with sub-stage timing."""
        import cupy as cp
        from numpy.fft import fft as np_fft, ifft as np_ifft

        t_total = time.perf_counter()

        # ── Validate shape (same as base class) ────────────────────────────
        if input_data.shape != self._w.input_shape:
            raise ValueError(
                f"Input shape {input_data.shape} != expected {self._w.input_shape}"
            )

        data = np.asarray(input_data, dtype=np.float64)

        # ── Sub-stage: CPU FFT ──────────────────────────────────────────────
        t0 = time.perf_counter()
        input_f = np_fft(data, self._w.max_conv_n)
        t_fft = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: GPU upload ───────────────────────────────────────────
        t0 = time.perf_counter()
        input_f_gpu = cp.asarray(input_f, dtype=cp.complex64, order="C")
        cp.cuda.Stream.null.synchronize()        # ensure upload is complete
        t_upload = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: GPU multiply (convolution) ───────────────────────────
        t0 = time.perf_counter()
        conv_f_gpu = input_f_gpu * self._w.kernel_f_bank_gpu
        cp.cuda.Stream.null.synchronize()
        t_multiply = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: GPU download ─────────────────────────────────────────
        t0 = time.perf_counter()
        conv_f_cpu = cp.asnumpy(conv_f_gpu)
        t_download = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: CPU iFFT + slice ─────────────────────────────────────
        t0 = time.perf_counter()
        conv_tf = np_ifft(conv_f_cpu, axis=1)
        cwt_coefs = conv_tf[:, : self._w.input_n]
        t_ifft = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: normalize_by_scale ───────────────────────────────────
        t0 = time.perf_counter()
        cwt_coefs = self._w.normalize_by_scale(cwt_coefs)
        t_norm = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: magnitude ────────────────────────────────────────────
        t0 = time.perf_counter()
        mag_coefs = self._w.compute_mag(cwt_coefs)
        t_mag = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: discard_unreliable_coefs ─────────────────────────────
        t0 = time.perf_counter()
        reliable = self._w.discard_unreliable_coefs(mag_coefs)
        t_discard = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: extract_hop_center ───────────────────────────────────
        t0 = time.perf_counter()
        hop_center = self._w.extract_hop_center(reliable)
        t_hop = (time.perf_counter() - t0) * 1000.0

        # ── Sub-stage: downsample ───────────────────────────────────────────
        t0 = time.perf_counter()
        output = self._w.downsample(hop_center, self._w.output_n)
        t_down = (time.perf_counter() - t0) * 1000.0

        t_total_ms = (time.perf_counter() - t_total) * 1000.0

        # ── Record ──────────────────────────────────────────────────────────
        self.timing["fft_cpu"].append(t_fft)
        self.timing["gpu_upload"].append(t_upload)
        self.timing["gpu_multiply"].append(t_multiply)
        self.timing["gpu_download"].append(t_download)
        self.timing["ifft_cpu"].append(t_ifft)
        self.timing["normalize"].append(t_norm)
        self.timing["mag"].append(t_mag)
        self.timing["discard"].append(t_discard)
        self.timing["hop_center"].append(t_hop)
        self.timing["downsample"].append(t_down)
        self.timing["cwt_total"].append(t_total_ms)

        return output


class InstrumentedNpWavelet:
    """
    Same as InstrumentedCuWavelet but for NpWavelet (no GPU stages).
    """

    SUBSTAGES = [
        "fft_cpu",
        "cpu_multiply",
        "ifft_cpu",
        "normalize",
        "mag",
        "discard",
        "hop_center",
        "downsample",
        "cwt_total",
    ]

    def __init__(self, wavelet):
        self._w = wavelet
        self.timing = {s: [] for s in self.SUBSTAGES}

    def cwt(self, input_data: np.ndarray) -> np.ndarray:
        from numpy.fft import fft as np_fft, ifft as np_ifft

        t_total = time.perf_counter()

        data = np.asarray(input_data, dtype=np.float64)

        t0 = time.perf_counter()
        input_f = np_fft(data, self._w.max_conv_n)
        t_fft = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        conv_f = input_f * self._w.kernel_f_bank
        t_mul = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        conv_tf = np_ifft(conv_f, axis=1)
        cwt_coefs = conv_tf[:, : self._w.input_n]
        t_ifft = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cwt_coefs = self._w.normalize_by_scale(cwt_coefs)
        t_norm = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        mag_coefs = self._w.compute_mag(cwt_coefs)
        t_mag = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        reliable = self._w.discard_unreliable_coefs(mag_coefs)
        t_discard = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        hop_center = self._w.extract_hop_center(reliable)
        t_hop = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        output = self._w.downsample(hop_center, self._w.output_n)
        t_down = (time.perf_counter() - t0) * 1000.0

        t_total_ms = (time.perf_counter() - t_total) * 1000.0

        self.timing["fft_cpu"].append(t_fft)
        self.timing["cpu_multiply"].append(t_mul)
        self.timing["ifft_cpu"].append(t_ifft)
        self.timing["normalize"].append(t_norm)
        self.timing["mag"].append(t_mag)
        self.timing["discard"].append(t_discard)
        self.timing["hop_center"].append(t_hop)
        self.timing["downsample"].append(t_down)
        self.timing["cwt_total"].append(t_total_ms)

        return output


# ── Structural latency calculator ─────────────────────────────────────────────

def compute_structural_latency(chunk_size, overlap_factor, sample_rate):
    """
    Return a dict of derived structural latency values.

    Structural latency is the latency baked into the configuration before
    any code runs. The CWT must receive chunk_size samples before it can
    produce a frame — the audio is already chunk_duration_ms old by the
    time processing starts.
    """
    hop_size = int(chunk_size * (1.0 - overlap_factor))
    return {
        "chunk_size":        chunk_size,
        "overlap_factor":    overlap_factor,
        "hop_size":          hop_size,
        "chunk_duration_ms": (chunk_size / sample_rate) * 1000.0,
        "hop_duration_ms":   (hop_size / sample_rate) * 1000.0,
        "frames_per_sec":    sample_rate / hop_size,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

W_LABEL = 30
W_AVG   = 9
W_MED   = 9
W_MIN   = 9
W_MAX   = 9
W_P95   = 9


def print_header():
    print(f"{'Stage':<{W_LABEL}}  {'avg':>{W_AVG}}  {'med':>{W_MED}}  {'min':>{W_MIN}}  {'max':>{W_MAX}}  {'p95':>{W_P95}}")
    print("-" * (W_LABEL + W_AVG + W_MED + W_MIN + W_MAX + W_P95 + 12))


def print_row(label, s, highlight=False):
    marker = " <<" if highlight else ""
    print(
        f"{label:<{W_LABEL}}  "
        f"{s['avg_ms']:>{W_AVG}.2f}  "
        f"{s['med_ms']:>{W_MED}.2f}  "
        f"{s['min_ms']:>{W_MIN}.2f}  "
        f"{s['max_ms']:>{W_MAX}.2f}  "
        f"{s['p95_ms']:>{W_P95}.2f}"
        f"{marker}"
    )


def section(title):
    print()
    print(f"{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Main profiler ─────────────────────────────────────────────────────────────

def run_profile(n_frames: int, headless: bool):
    print("\nSubShader Pipeline Latency Profiler")
    print("=" * 60)

    # ── Config ─────────────────────────────────────────────────────────────
    config = get_default_config()
    sample_rate = config.wavelet.typical_sampling_freq
    chunk_size = config.audio.chunk_size
    overlap_factor = config.audio.overlap_factor

    structural = compute_structural_latency(chunk_size, overlap_factor, sample_rate)

    section("Configuration")
    print(f"  Audio file:     {config.audio.file_path}")
    print(f"  Sample rate:    {sample_rate:.0f} Hz")
    print(f"  chunk_size:     {chunk_size} samples  ({structural['chunk_duration_ms']:.1f} ms of audio)")
    print(f"  overlap_factor: {overlap_factor}")
    print(f"  hop_size:       {structural['hop_size']} samples  ({structural['hop_duration_ms']:.1f} ms)")
    print(f"  Frame rate:     {structural['frames_per_sec']:.1f} fps (max achievable)")
    print(f"  Frames to time: {n_frames}")

    # ── Structural latency analysis ─────────────────────────────────────────
    section("Structural Latency Analysis")
    print(f"  The CWT MUST see {chunk_size} samples before producing a frame.")
    print(f"  This means the visualization is ALWAYS at least")
    print(f"  {structural['chunk_duration_ms']:.0f} ms behind real-time — before any processing.")
    print()
    print(f"  Visual updates fire every {structural['hop_duration_ms']:.1f} ms ({structural['frames_per_sec']:.1f} fps).")
    print(f"  Between updates, the visualization is frozen.")
    print()
    daw_window_ms = (1024 / sample_rate) * 1000
    daw_hop_ms    = (256  / sample_rate) * 1000
    daw_fps       = sample_rate / 256
    print(f"  DAW STFT baseline (typical): 1024 samples, 256-sample hop")
    print(f"    Window:  {daw_window_ms:.1f} ms  (vs {structural['chunk_duration_ms']:.0f} ms — {structural['chunk_duration_ms']/daw_window_ms:.0f}x longer)")
    print(f"    Hop:     {daw_hop_ms:.1f} ms  (vs {structural['hop_duration_ms']:.1f} ms — {structural['hop_duration_ms']/daw_hop_ms:.0f}x longer)")
    print(f"    Rate:    {daw_fps:.0f} fps  (vs {structural['frames_per_sec']:.1f} fps — {daw_fps/structural['frames_per_sec']:.0f}x more frequent)")

    # ── Init audio input ────────────────────────────────────────────────────
    section("Initialization")
    audio_input, t_audio_init = time_call(
        AudioInput,
        path=config.audio.file_path,
        chunk_size=chunk_size,
        overlap_factor=overlap_factor,
    )
    print(f"  AudioInput init:   {t_audio_init:.1f} ms")

    # ── GPU / CPU wavelet init ──────────────────────────────────────────────
    try:
        import cupy as cp
        from subshader.dsp.wavelet import CuWavelet
        wavelet_raw, t_wavelet_init = time_call(
            CuWavelet,
            sample_rate=audio_input.get_sample_rate(),
            input_n=chunk_size,
            config=config.wavelet,
            overlap_factor=overlap_factor,
        )
        backend = "CuWavelet (GPU)"
        instrumented_wavelet = InstrumentedCuWavelet(wavelet_raw)
        gpu_mode = True
        print(f"  CuWavelet init:    {t_wavelet_init:.1f} ms  [GPU mode]")
        print(f"  num_freqs:         {wavelet_raw.num_freqs}")
        print(f"  max_conv_n:        {wavelet_raw.max_conv_n}")
        kernel_bytes = wavelet_raw.kernel_f_bank.nbytes
        print(f"  kernel bank size:  {kernel_bytes / 1024:.0f} KB ({wavelet_raw.num_freqs} × {wavelet_raw.max_conv_n} complex64)")
    except Exception as e:
        from subshader.dsp.wavelet import NpWavelet
        wavelet_raw, t_wavelet_init = time_call(
            NpWavelet,
            sample_rate=audio_input.get_sample_rate(),
            input_n=chunk_size,
            config=config.wavelet,
            overlap_factor=overlap_factor,
        )
        backend = "NpWavelet (CPU)"
        instrumented_wavelet = InstrumentedNpWavelet(wavelet_raw)
        gpu_mode = False
        print(f"  NpWavelet init:    {t_wavelet_init:.1f} ms  [CPU mode — GPU unavailable: {e}]")
        print(f"  num_freqs:         {wavelet_raw.num_freqs}")
        print(f"  max_conv_n:        {wavelet_raw.max_conv_n}")

    # ── Buffer init ─────────────────────────────────────────────────────────
    from subshader.viz.plotter import CircularFrameBuffer
    from subshader.config import ColorNormalizationConfig
    color_norm_cfg = ColorNormalizationConfig()
    frame_shape = wavelet_raw.get_output_shape()
    buf = CircularFrameBuffer(
        frame_shape=frame_shape,
        num_frames=config.viz.num_frames,
        color_norm_config=color_norm_cfg,
    )
    print(f"  CircularFrameBuffer: {frame_shape} × {config.viz.num_frames} frames")

    # ── GL renderer init (optional) ─────────────────────────────────────────
    renderer = None
    gl_context = None
    if not headless:
        try:
            from subshader.viz.plotter import GLContext, Renderer
            gl_context, t_gl = time_call(GLContext, title="SubShader Latency Profiler")
            texture_shape = buf.get_shape()
            renderer, t_rend = time_call(
                Renderer,
                ctx=gl_context.ctx,
                texture_shape=texture_shape,
                gamma=config.viz.gamma,
            )
            print(f"  GLContext init:     {t_gl:.1f} ms")
            print(f"  Renderer init:      {t_rend:.1f} ms  (texture {texture_shape[1]}×{texture_shape[0]})")
        except Exception as e:
            print(f"  GL init skipped: {e}")
            headless = True

    # ── Runtime profiling loop ──────────────────────────────────────────────
    section(f"Runtime Loop ({n_frames} frames, backend={backend})")
    print_header()

    t_get_chunk  = []
    t_cwt_total  = []
    t_push_frame = []
    t_update_tex = []
    t_gl_ops     = []
    t_loop_total = []

    frames_done = 0
    for i in range(n_frames):
        t_loop_start = time.perf_counter()

        # Audio chunk
        audio_data, ms_chunk = time_call(audio_input.get_chunk)
        if audio_data is None:
            break
        t_get_chunk.append(ms_chunk)

        # CWT (instrumented)
        t0 = time.perf_counter()
        coefs = instrumented_wavelet.cwt(audio_data)
        t_cwt_total.append((time.perf_counter() - t0) * 1000.0)

        # Buffer push
        _, ms_push = time_call(buf.push_frame, coefs)
        t_push_frame.append(ms_push)

        # GL render (optional)
        if renderer is not None:
            flattened = buf.get_flattened_buffer()
            _, ms_tex = time_call(renderer.update_texture, flattened)
            t_update_tex.append(ms_tex)

            renderer.set_intensity_max(buf.get_intensity_max())
            t0 = time.perf_counter()
            gl_context.clear_graphic()
            renderer.render_graphic()
            gl_context.display_graphic()
            t_gl_ops.append((time.perf_counter() - t0) * 1000.0)

        t_loop_total.append((time.perf_counter() - t_loop_start) * 1000.0)
        frames_done += 1

    if frames_done == 0:
        print("No frames processed — check audio file path.")
        return

    # ── Print runtime results ────────────────────────────────────────────────
    print_row("get_chunk()",          stats(t_get_chunk))
    print_row("cwt() total",          stats(t_cwt_total),   highlight=True)
    print_row("push_frame()",         stats(t_push_frame))
    if t_update_tex:
        print_row("update_texture()",     stats(t_update_tex))
    if t_gl_ops:
        print_row("gl_ops (clr+render+swap)", stats(t_gl_ops))
    print("-" * (W_LABEL + W_AVG + W_MED + W_MIN + W_MAX + W_P95 + 12))
    print_row("loop total",           stats(t_loop_total),  highlight=True)

    # ── CWT sub-stage breakdown ──────────────────────────────────────────────
    section(f"CWT Sub-Stage Breakdown ({backend})")
    print_header()
    for stage in instrumented_wavelet.SUBSTAGES:
        times = instrumented_wavelet.timing[stage]
        if times:
            s = stats(times)
            is_bottleneck = (stage != "cwt_total" and s["avg_ms"] > 5.0)
            print_row(stage, s, highlight=is_bottleneck)

    # ── Latency budget analysis ──────────────────────────────────────────────
    section("Latency Budget Analysis")

    hop_ms   = structural["hop_duration_ms"]
    chunk_ms = structural["chunk_duration_ms"]

    avg_cwt  = stats(t_cwt_total)["avg_ms"]
    avg_loop = stats(t_loop_total)["avg_ms"]

    print(f"  Hop interval (max time budget per frame): {hop_ms:.1f} ms")
    print(f"  Loop wall time (avg):                     {avg_loop:.1f} ms")
    budget_remaining = hop_ms - avg_loop
    print(f"  Budget remaining:                         {budget_remaining:.1f} ms")
    print()
    print(f"  Minimum achievable latency (structural):")
    print(f"    Audio window:   {chunk_ms:.0f} ms  (CWT needs {chunk_size} samples)")
    print(f"    Frame interval: {hop_ms:.1f} ms  (hop_size = {structural['hop_size']})")
    print(f"    Total floor:    ~{chunk_ms + hop_ms:.0f} ms  (window + one hop)")
    print()
    if avg_cwt < hop_ms * 0.5:
        print(f"  CWT is NOT the bottleneck (avg {avg_cwt:.1f} ms < 50% of {hop_ms:.1f} ms budget).")
        print(f"  The bottleneck is STRUCTURAL: chunk_size={chunk_size} and overlap_factor={overlap_factor}.")
    else:
        print(f"  CWT IS a bottleneck (avg {avg_cwt:.1f} ms >= 50% of {hop_ms:.1f} ms budget).")

    # ── Recommended configs ─────────────────────────────────────────────────
    section("Recommended Configurations for Lower Latency")
    configs_to_test = [
        ("Current",        chunk_size, overlap_factor),
        ("Moderate (50ms)",   2048,       0.5),
        ("Aggressive (25ms)", 2048,       0.25),
        ("DAW-like (6ms)",    512,        0.5),
    ]
    print(f"  {'Config':<22}  {'chunk_size':>10}  {'overlap':>8}  {'window_ms':>10}  {'hop_ms':>8}  {'fps':>8}")
    print("  " + "-" * 74)
    for name, cs, ov in configs_to_test:
        sl = compute_structural_latency(cs, ov, sample_rate)
        print(
            f"  {name:<22}  {cs:>10}  {ov:>8.2f}  "
            f"{sl['chunk_duration_ms']:>10.1f}  {sl['hop_duration_ms']:>8.1f}  {sl['frames_per_sec']:>8.1f}"
        )
    print()
    print("  Note: Smaller chunk_size reduces time-frequency resolution for")
    print("  low-frequency notes (wide wavelets require long windows). The")
    print("  current chunk_size=16384 is set to capture A0=27.5 Hz accurately.")
    print("  Reducing it will improve latency but degrade low-note accuracy.")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if gl_context is not None:
        import glfw
        glfw.terminate()
    audio_input.cleanup()
    wavelet_raw.cleanup()

    return {
        "structural": structural,
        "stages": {
            "get_chunk":    stats(t_get_chunk),
            "cwt_total":    stats(t_cwt_total),
            "push_frame":   stats(t_push_frame),
            "update_tex":   stats(t_update_tex) if t_update_tex else None,
            "gl_ops":       stats(t_gl_ops)     if t_gl_ops     else None,
            "loop_total":   stats(t_loop_total),
        },
        "cwt_substages": {
            s: stats(instrumented_wavelet.timing[s])
            for s in instrumented_wavelet.SUBSTAGES
            if instrumented_wavelet.timing[s]
        },
        "backend": backend,
        "frames_done": frames_done,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SubShader pipeline latency profiler")
    parser.add_argument("--frames",   type=int, default=64,
                        help="Number of frames to profile (default: 64)")
    parser.add_argument("--headless", action="store_true",
                        help="Skip GL render stage")
    args = parser.parse_args()

    results = run_profile(n_frames=args.frames, headless=args.headless)
