"""Utilities module for the benchmark suite."""

# Export signal registry
from .signals import SIGNALS, get_signal

# Export constants
from .constants import (
    AUDIO_REFERENCE_DIR,
    AUDIO_GENERATED_DIR,
    IMAGES_REFERENCE_DIR,
    IMAGES_GENERATED_DIR,
    TIMING_DIR,
    AUDIO_BOUNCING_CHIRP,
    AUDIO_MIDI_SINE_WAVES,
    AUDIO_BELTRAN,
    AUDIO_BELTRAN_16BAR,
    AUDIO_BELTRAN_8BAR,
    AUDIO_BELTRAN_4BAR,
    AUDIO_COMPARISON_1,
    AUDIO_COMPARISON_2,
    AUDIO_COMPARISON_3,
    DAW_IMAGE_BOUNCING_CHIRP_EDISON,
    DAW_IMAGE_MIDI_SINE_WAVES_EDISON,
    DAW_IMAGE_BELTRAN_16BAR_EDISON,
    DAW_IMAGE_BELTRAN_8BAR_EDISON,
    DAW_IMAGE_BELTRAN_4BAR_EDISON,
    DAW_IMAGE_COMPARISON_1,
    DAW_IMAGE_COMPARISON_2,
    DAW_IMAGE_COMPARISON_3,
    STFT_NPERSEG,
    NUM_FRAMES,
    CHIRP_F0,
    CHIRP_F1,
    HEATMAP_MAX_ROWS,
    HEATMAP_MAX_COLS,
    gpu_available,
)

# Export timing utilities
from .timing import time_call, TimingAccumulator

# Export printing utilities
from .printing import (
    print_section_start,
    print_section_end,
    print_separator,
    print_init_header,
    print_init_row,
    print_init_total,
    live_progress,
    clear_progress,
    print_results_header,
    print_results_row,
    print_loop_summary,
    print_total_time,
    compute_timing_stats,
    run_modes,
)

# Export style module
from . import style

# Export plotting utilities
from .plotting import (
    compute_freq_yticks,
    create_figure_scaffold,
    render_top_row,
    render_spectrogram_row,
)

# Export DSP helpers
from .dsp_helpers import (
    build_chirp_chunks,
    build_wandering_chirp_chunks,
    build_fm_chirp_chunks,
    build_bouncing_chirp,
    build_bouncing_chirp_chunks,
)

# Export WAV utilities
from .wav_export import export_signal_to_wav

__all__ = [
    # Constants — directories
    "AUDIO_REFERENCE_DIR",
    "AUDIO_GENERATED_DIR",
    "IMAGES_REFERENCE_DIR",
    "IMAGES_GENERATED_DIR",
    "TIMING_DIR",
    # Constants — audio files
    "AUDIO_BOUNCING_CHIRP",
    "AUDIO_MIDI_SINE_WAVES",
    "AUDIO_BELTRAN",
    "AUDIO_BELTRAN_16BAR",
    "AUDIO_BELTRAN_8BAR",
    "AUDIO_BELTRAN_4BAR",
    "AUDIO_COMPARISON_1",
    "AUDIO_COMPARISON_2",
    "AUDIO_COMPARISON_3",
    # Constants — DAW reference images
    "DAW_IMAGE_BOUNCING_CHIRP_EDISON",
    "DAW_IMAGE_MIDI_SINE_WAVES_EDISON",
    "DAW_IMAGE_BELTRAN_16BAR_EDISON",
    "DAW_IMAGE_BELTRAN_8BAR_EDISON",
    "DAW_IMAGE_BELTRAN_4BAR_EDISON",
    "DAW_IMAGE_COMPARISON_1",
    "DAW_IMAGE_COMPARISON_2",
    "DAW_IMAGE_COMPARISON_3",
    # Constants — DSP parameters
    "STFT_NPERSEG",
    "NUM_FRAMES",
    "CHIRP_F0",
    "CHIRP_F1",
    "HEATMAP_MAX_ROWS",
    "HEATMAP_MAX_COLS",
    "gpu_available",
    # Timing
    "time_call",
    "TimingAccumulator",
    # Printing
    "print_section_start",
    "print_section_end",
    "print_separator",
    "print_init_header",
    "print_init_row",
    "print_init_total",
    "live_progress",
    "clear_progress",
    "print_results_header",
    "print_results_row",
    "print_loop_summary",
    "print_total_time",
    "compute_timing_stats",
    "run_modes",
    # Style
    "style",
    # Plotting
    "compute_freq_yticks",
    "create_figure_scaffold",
    "render_top_row",
    "render_spectrogram_row",
    # DSP
    "build_chirp_chunks",
    "build_wandering_chirp_chunks",
    "build_fm_chirp_chunks",
    "build_bouncing_chirp",
    "build_bouncing_chirp_chunks",
    # WAV export
    "export_signal_to_wav",
    # Signal registry
    "SIGNALS",
    "get_signal",
]
