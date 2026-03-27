from subshader.config import get_default_config, ColorNormalizationConfig
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import NumPyWavelet
from subshader.viz.plotter import CircularFrameBuffer

from utilities import (
    AUDIO_DEFAULT,
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

        cwt_stages = ["raw_cwt", "normalize", "magnitude", "edge_trim", "hop_center", "downsample"]
        all_methods = ["get_chunk()"] + cwt_stages + ["push_frame()"]
        acc = TimingAccumulator(self.num_frames, all_methods)

        buf = CircularFrameBuffer(
            frame_shape=wavelet.get_output_shape(),
            num_frames=self.num_frames,
            color_norm_config=ColorNormalizationConfig(),
        )

        for i in range(self.num_frames):
            audio_data, acc["get_chunk()"][i] = time_call(audio_input.get_chunk)

            if audio_data is None:
                acc.current_idx = i
                break

            result = wavelet.cwt(audio_data)
            acc["raw_cwt"][i] = wavelet._timing_class_specific_cwt_ms
            acc["normalize"][i] = wavelet._timing_normalize_by_scale_ms
            acc["magnitude"][i] = wavelet._timing_compute_mag_ms
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


def run_default():
    """Run SubShader with default configuration."""
    from subshader.__main__ import main
    main()
