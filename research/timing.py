from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import NumPyWavelet

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


def run_default():
    """Run SubShader with default configuration."""
    from subshader.__main__ import main
    main()
