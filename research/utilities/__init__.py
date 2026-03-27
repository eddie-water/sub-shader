"""Utilities module for the benchmark suite."""

# Export constants
from .constants import (
    BENCHMARKS_DIR,
    BENCHMARKS_SEABORN_DIR,
    BENCHMARKS_STUBS_DIR,
    AUDIO_DEFAULT,
    AUDIO_CHIRP,
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
    compute_stft_frame,
    build_chirp_chunks,
    build_wandering_chirp_chunks,
    build_fm_chirp_chunks,
    build_bouncing_chirp,
    build_bouncing_chirp_chunks,
)

# Export WAV utilities
from .wav_export import export_signal_to_wav

__all__ = [
    # Constants
    "BENCHMARKS_DIR",
    "BENCHMARKS_SEABORN_DIR",
    "BENCHMARKS_STUBS_DIR",
    "AUDIO_DEFAULT",
    "AUDIO_CHIRP",
    "AUDIO_BOUNCING_CHIRP",
    "AUDIO_POLYPHONIC",
    "AUDIO_MUSICAL",
    "AUDIO_BELTRAN",
    "AUDIO_BELTRAN_16BAR",
    "MIDI_POLYPHONIC",
    "DAW_POLYPHONIC",
    "DAW_MUSICAL",
    "DAW_BELTRAN_16BAR",
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
    "compute_stft_frame",
    "build_chirp_chunks",
    "build_wandering_chirp_chunks",
    "build_bouncing_chirp",
    "build_bouncing_chirp_chunks",
    # WAV export
    "export_signal_to_wav",
]
