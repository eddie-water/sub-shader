import os
import datetime

import numpy as np

from subshader.config import get_default_config, CWTConfig, ColorNormalizationConfig
from subshader.audio.reader import AudioReader
from subshader.dsp.cwt import CpuCWT, GpuCWT
from subshader.renderer.frame_buffer import CircularFrameBuffer

from utilities import (
    AUDIO_BELTRAN,
    TIMING_DIR,
    NUM_FRAMES,
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
)

GPU_AVAILABLE = gpu_available()

# Path to the timing output template (relative to project root)
_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "utilities", "timing_template.txt"
)


def _load_timing_template() -> str:
    """Load the timing report template from disk."""
    with open(_TEMPLATE_PATH) as f:
        return f.read()


def _format_timing_report(
    audio_file: str,
    num_frames: int,
    backend: str,
    acc: "TimingAccumulator",
    cwt_stages: list[str],
) -> str:
    """Fill the timing template with measured values and return as string."""
    template = _load_timing_template()

    def _mean(key: str) -> float:
        return float(np.mean(acc[key])) if len(acc[key]) > 0 else 0.0

    def _std(key: str) -> float:
        return float(np.std(acc[key])) if len(acc[key]) > 0 else 0.0

    get_chunk_mean = _mean("get_chunk()")
    raw_cwt_mean   = _mean("raw_cwt")
    norm_mean      = _mean("normalize")
    mag_mean       = _mean("magnitude")
    edge_mean      = _mean("edge_trim")
    hop_mean       = _mean("hop_center")
    ds_mean        = _mean("downsample")
    push_mean      = _mean("push_frame()")

    total_mean = (
        get_chunk_mean + raw_cwt_mean + norm_mean + mag_mean
        + edge_mean + hop_mean + ds_mean + push_mean
    )
    if total_mean == 0.0:
        total_mean = 1e-9  # guard against division by zero

    get_chunk_std = _std("get_chunk()")
    raw_cwt_std   = _std("raw_cwt")
    norm_std      = _std("normalize")
    mag_std       = _std("magnitude")
    edge_std      = _std("edge_trim")
    hop_std       = _std("hop_center")
    ds_std        = _std("downsample")
    push_std      = _std("push_frame()")
    total_std = (
        get_chunk_std + raw_cwt_std + norm_std + mag_std
        + edge_std + hop_std + ds_std + push_std
    )

    fps = 1000.0 / total_mean if total_mean > 0 else 0.0

    def pct(v: float) -> float:
        return 100.0 * v / total_mean

    return template.format(
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        audio_file=audio_file,
        num_frames=num_frames,
        backend=backend,
        audio_get_chunk_mean=get_chunk_mean,
        audio_get_chunk_std=get_chunk_std,
        audio_get_chunk_pct=pct(get_chunk_mean),
        dsp_raw_cwt_mean=raw_cwt_mean,
        dsp_raw_cwt_std=raw_cwt_std,
        dsp_raw_cwt_pct=pct(raw_cwt_mean),
        dsp_normalize_mean=norm_mean,
        dsp_normalize_std=norm_std,
        dsp_normalize_pct=pct(norm_mean),
        dsp_magnitude_mean=mag_mean,
        dsp_magnitude_std=mag_std,
        dsp_magnitude_pct=pct(mag_mean),
        dsp_edge_trim_mean=edge_mean,
        dsp_edge_trim_std=edge_std,
        dsp_edge_trim_pct=pct(edge_mean),
        dsp_hop_center_mean=hop_mean,
        dsp_hop_center_std=hop_std,
        dsp_hop_center_pct=pct(hop_mean),
        dsp_downsample_mean=ds_mean,
        dsp_downsample_std=ds_std,
        dsp_downsample_pct=pct(ds_mean),
        renderer_push_frame_mean=push_mean,
        renderer_push_frame_std=push_std,
        renderer_push_frame_pct=pct(push_mean),
        total_mean=total_mean,
        total_std=total_std,
        fps=fps,
    )


def _write_timing_file(report: str) -> str:
    """Write report to assets/timing/ with a timestamp filename. Returns the path."""
    os.makedirs(TIMING_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TIMING_DIR, f"{ts}_timing.txt")
    with open(out_path, "w") as f:
        f.write(report)
    return out_path


class TimedSubShader:
    """
    Run the SubShader pipeline with live timing instrumentation.

    Mirrors the real SubShader pipeline (AudioReader -> CWT -> buffer) but
    wraps each stage with perf_counter and displays a live-updating table.
    """

    def __init__(self, audio_path: str = AUDIO_BELTRAN, num_frames: int = NUM_FRAMES):
        self.audio_path = audio_path
        self.num_frames = num_frames

    def run(self):
        """Run the timed pipeline: init timing, then runtime loop timing."""
        print("\nSubShader Timing Benchmark\n")

        # ── Init timing ──────────────────────────────────────────────────────
        print_section_start("Init Timing")
        print_init_header()

        config = CWTConfig(file_path=self.audio_path)
        total_init_ms = 0.0

        audio_reader, ms = time_call(AudioReader, config)
        print_init_row("AudioReader", ms)
        total_init_ms += ms

        if GPU_AVAILABLE:
            wavelet, ms = time_call(GpuCWT, config)
            backend_name = "GpuCWT (GPU)"
        else:
            wavelet, ms = time_call(CpuCWT, config)
            backend_name = "CpuCWT (CPU)"

        print_init_row(backend_name, ms)
        total_init_ms += ms

        print_init_total(total_init_ms)
        print_section_end()

        # ── Runtime loop ─────────────────────────────────────────────────────
        print_section_start(f"Runtime Loop ({self.num_frames} frames)")

        cwt_stages = ["raw_cwt", "normalize", "magnitude", "edge_trim", "hop_center", "downsample"]
        all_methods = ["get_chunk()"] + cwt_stages + ["push_frame()"]
        acc = TimingAccumulator(self.num_frames, all_methods)

        buf = CircularFrameBuffer(
            frame_shape=wavelet.get_output_shape(),
            num_frames=self.num_frames,
            color_norm_config=ColorNormalizationConfig(),
        )

        for i in range(self.num_frames):
            audio_data, acc["get_chunk()"][i] = time_call(audio_reader.get_chunk)

            if audio_data is None:
                acc.current_idx = i
                break

            result = wavelet.process(audio_data)
            acc["raw_cwt"][i] = wavelet._timing_transform_ms
            acc["normalize"][i] = wavelet._timing__normalize_by_scale_ms
            acc["magnitude"][i] = wavelet._timing__compute_mag_ms
            acc["edge_trim"][i] = wavelet._timing_discard_unreliable_coefs_ms
            acc["hop_center"][i] = wavelet._timing_extract_hop_center_ms
            acc["downsample"][i] = wavelet._timing_downsample_ms
            _, acc["push_frame()"][i] = time_call(buf.push_frame, result)
            acc.current_idx = i + 1
            live_progress(i + 1, self.num_frames)

        acc.trim()

        clear_progress()
        print_separator()
        print("Results:")
        print_results_header()
        print_results_row("get_chunk()", acc["get_chunk()"])
        print_separator()
        for stage in cwt_stages:
            print_results_row(f"  {stage}", acc[stage])
        cwt_total_times = sum(acc[s] for s in cwt_stages)
        print_results_row("cwt() total", cwt_total_times)
        print_separator()
        print_results_row("push_frame()", acc["push_frame()"])

        total_times = acc["get_chunk()"] + cwt_total_times + acc["push_frame()"]
        print_loop_summary(len(total_times), total_times)
        print_section_end()

        # ── Write timing report to file ───────────────────────────────────────
        report = _format_timing_report(
            audio_file=self.audio_path,
            num_frames=len(total_times),
            backend=backend_name,
            acc=acc,
            cwt_stages=cwt_stages,
        )
        print(report)
        out_path = _write_timing_file(report)
        print(f"Timing report written -> {out_path}")

        return acc


def run_timing(audio_path: str = AUDIO_BELTRAN, num_frames: int = NUM_FRAMES, stub: bool = False) -> None:
    """Entry point for --timing mode in test_suite.py."""
    runner = TimedSubShader(audio_path=audio_path, num_frames=num_frames)
    runner.run()


def run_default():
    """Run SubShader with default configuration."""
    from subshader.__main__ import main
    main()
