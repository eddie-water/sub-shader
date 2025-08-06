#!/usr/bin/env python3
"""
Main entry point for SubShader, a real-time audio visualizer.

This module orchestrates the audio processing pipeline:
1. Audio input from file
2. Time-Frequency Analysis by the Continuous Wavelet Transform (CWT)
3. GPU-accelerated visualization using OpenGL shaders

"""

from subshader.utils.os_env_setup import setup_all_environments
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import CuWavelet
from subshader.viz.plotter import Shader
from subshader.utils.fps_utility import FpsUtility

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_SIZE = 2 << 12  # 4k samples per frame
FILE_PATH = "assets/audio/songs/beltran_soundcloud.wav"
NUM_FRAMES = 128

# =============================================================================
# EXCEPTIONS
# =============================================================================

class EndOfAudioException(Exception):
    """Raised when the audio file has been completely processed."""
    pass

class WindowCloseException(Exception):
    """Raised when the window is closed."""
    pass

# Gracefully exit on these exceptions
GRACEFUL_EXIT_EXCEPTIONS = (
    KeyboardInterrupt,
    EndOfAudioException, 
    WindowCloseException
)

# =============================================================================
# INITIALIZATION
# =============================================================================

# Audio Input - handles file reading and audio frame getter 
audio_input = AudioInput(path=FILE_PATH, window_size=WINDOW_SIZE)
sample_rate = audio_input.get_sample_rate()  # 44.1 kHz

# Wavelet Object - performs Continuous Wavelet Transform (CWT) using CuPy
wavelet = CuWavelet(sample_rate=sample_rate, window_size=WINDOW_SIZE)

# Plotter Object - GPU-accelerated shader plot
plot_shape = wavelet.get_shape()
plotter = Shader(file_path=FILE_PATH, frame_shape=plot_shape, num_frames=NUM_FRAMES)

# FPS utility - performance monitoring
fps = FpsUtility()

# =============================================================================
# MAIN LOOP
# =============================================================================

def main_loop():
    """
    Main loop. Loops until audio ends or window is closed.

    Processes audio frames through the pipeline:
    - Gets frame of audio data
    - Compute CWT coefficients on the audio
    - Updates the plot with normalized CWT coefficients
    - Monitors FPS 

    """
    while not plotter.should_window_close():
        # Start frame timing
        frame_start = fps.start_frame()

        # Grab a frame of audio and check for end of file
        if (audio_data := audio_input.get_frame()) is None:
           raise EndOfAudioException("Audio file processing complete")

        # Compute CWT on that frame
        coefs = wavelet.compute_cwt(audio_data)

        # Update plot
        plotter.update_plot(coefs)

        # End frame timing and report FPS if needed
        fps.end_frame_and_report(frame_start)
    
    raise WindowCloseException("Window Closed")

# =============================================================================
# CLEANUP FUNCTION
# =============================================================================

def everybody_cleanup():
    """Clean up all resources in the correct order."""
    print("Cleaning up resources and shutting down gracefully...")
    audio_input.cleanup()  # Close audio file
    wavelet.cleanup()      # Clean up GPU memory
    plotter.cleanup()      # Terminate GLFW and OpenGL

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    setup_all_environments()
    try:
        main_loop()
    except GRACEFUL_EXIT_EXCEPTIONS as e:
        print()
        if isinstance(e, KeyboardInterrupt):
            print("Keyboard Interrupt received.")
        else:
            print(f"{e} received.")
    everybody_cleanup()