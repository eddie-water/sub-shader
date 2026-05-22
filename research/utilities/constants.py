"""Shared constants for the benchmark suite."""

# =============================================================================
# DIRECTORIES
# =============================================================================

AUDIO_REFERENCE_DIR = "assets/audio/reference"
AUDIO_GENERATED_DIR = "assets/audio/generated"

IMAGES_REFERENCE_DIR = "assets/images/reference"
IMAGES_GENERATED_DIR = "assets/images/generated"
IMAGES_DSP_DIR = "assets/images/dsp"

TIMING_DIR = "assets/timing"

# =============================================================================
# AUDIO FILES
# =============================================================================

# Reference audio — committed input files (recorded or curated)
AUDIO_BOUNCING_CHIRP = "assets/audio/generated/bouncing_chirp.wav"
AUDIO_MIDI_SINE_WAVES = "assets/audio/reference/midi_sine_waves.wav"
AUDIO_BELTRAN = "assets/audio/reference/beltran_sc_rip.wav"
AUDIO_BELTRAN_16BAR = "assets/audio/reference/beltran_sc_rip_16_bar.wav"
AUDIO_BELTRAN_8BAR = "assets/audio/reference/beltran_sc_rip_8_bar.wav"
AUDIO_BELTRAN_4BAR = "assets/audio/reference/beltran_sc_rip_4_bar.wav"

# Comparison Figure Audio
AUDIO_COMPARISON_1 = AUDIO_BOUNCING_CHIRP
AUDIO_COMPARISON_2 = AUDIO_MIDI_SINE_WAVES
AUDIO_COMPARISON_3 = AUDIO_BELTRAN_4BAR

# =============================================================================
# REFERENCE IMAGES
# =============================================================================

# Reference images — curated comparisons, not test suite outputs
DAW_IMAGE_BOUNCING_CHIRP_EDISON = "assets/images/dsp/figures/bouncing_chirp_edison.png"
DAW_IMAGE_MIDI_SINE_WAVES_EDISON = "assets/images/dsp/figures/midi_sine_wave_edison.png"
DAW_IMAGE_BELTRAN_16BAR_EDISON = "assets/images/generated/beltran_sc_rip_16_bar.png"
DAW_IMAGE_BELTRAN_8BAR_EDISON = "assets/images/diagnostics/beltran_sc_rip_8_bar.png"
DAW_IMAGE_BELTRAN_4BAR_EDISON = "assets/images/dsp/figures/beltran_sc_rip_4_bar_edison.png"

# Comparison Figure Reference Images
DAW_IMAGE_COMPARISON_1 = DAW_IMAGE_BOUNCING_CHIRP_EDISON
DAW_IMAGE_COMPARISON_2 = DAW_IMAGE_MIDI_SINE_WAVES_EDISON
DAW_IMAGE_COMPARISON_3 = DAW_IMAGE_BELTRAN_4BAR_EDISON

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
