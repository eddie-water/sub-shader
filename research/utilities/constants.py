"""Shared constants for the benchmark suite."""

# =============================================================================
# DIRECTORIES
# =============================================================================

BENCHMARKS_DIR = "assets/images/benchmarks"
BENCHMARKS_SEABORN_DIR = "assets/images/benchmarks/seaborn"
BENCHMARKS_STUBS_DIR = "assets/images/benchmarks/stubs"

# =============================================================================
# AUDIO FILES
# =============================================================================

AUDIO_DEFAULT = "assets/audio/daw/a2a3_a4_minor_scale.wav"
AUDIO_CHIRP = "assets/audio/daw/chirp_beat.wav"
AUDIO_BOUNCING_CHIRP = "assets/audio/daw/bouncing_chirp.wav"
AUDIO_POLYPHONIC = "assets/audio/daw/polyphonic_audio_example.wav"
AUDIO_MUSICAL = "assets/audio/daw/musical_audio_example.wav"
AUDIO_BELTRAN = "assets/audio/songs/beltran_sc_rip.wav"
AUDIO_BELTRAN_16BAR = "assets/audio/songs/beltran_sc_rip_16_bar.wav"
AUDIO_BELTRAN_8BAR = "assets/audio/songs/beltran_sc_rip_8_bar.wav"

# =============================================================================
# REFERENCE IMAGES
# =============================================================================

MIDI_POLYPHONIC = "assets/images/polyphonic-signal-example-midi-notes.png"
DAW_POLYPHONIC = "assets/audio/daw/polyphonic_audio_example_edison.png"
DAW_MUSICAL = "assets/images/musical-signal-example-edison-spectrogram.png"
DAW_BELTRAN_16BAR = "assets/images/benchmarks/beltran_sc_rip_16_bar.png"
DAW_BELTRAN_8BAR = "assets/images/beltran_sc_rip_8_bar.png"

# =============================================================================
# DSP PARAMETERS
# =============================================================================

STFT_NPERSEG = 1024
NUM_FRAMES = 128  # frames to accumulate for figure snapshots

CHIRP_F0 = 200  # Hz -- chirp start frequency
CHIRP_F1 = 20_000  # Hz -- chirp end frequency

# =============================================================================
# VISUALIZATION
# =============================================================================

HEATMAP_MAX_ROWS = 128
HEATMAP_MAX_COLS = 512

# =============================================================================
# GPU DETECTION
# =============================================================================


def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False
