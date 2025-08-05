#!/usr/bin/env python3
"""
Main entry point for the SubShader audio visualization application.

This module orchestrates the audio processing pipeline:
1. Audio input from file
2. Time-Frequency Analysis by the Continuous Wavelet Transform (CWT)
3. GPU-accelerated visualization using OpenGL shaders

Real-time audio visualization with GPU acceleration
"""

import time

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import ShadeWavelet
from subshader.viz.plotter import Shader
from subshader.utils.fps_utility import FPSUtility

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_SIZE = 2 << 11  # 4k samples per frame
DOWNSAMLPLE_STRIDE = 1  # No downsampling
FILE_PATH = "assets/audio/daw/chirp_beat.wav"

# =============================================================================
# INITIALIZATION
# =============================================================================

# Audio Input - handles file reading and frame extraction
audio_input = AudioInput(path=FILE_PATH, window_size=WINDOW_SIZE)
sample_rate = audio_input.get_sample_rate()  # 44.1 kHz

# Wavelet Object - performs Continuous Wavelet Transform
wavelet = ShadeWavelet(
    sample_rate=sample_rate,
    window_size=WINDOW_SIZE,
    ds_stride=DOWNSAMLPLE_STRIDE
)

# Plotter Object - GPU-accelerated visualization
plot_shape = wavelet.get_shape()
plotter = Shader(
    file_path=FILE_PATH,
    frame_shape=plot_shape,
    num_frames=128
)

# FPS utility - performance monitoring
fps = FPSUtility()

# =============================================================================
# MAIN LOOP
# =============================================================================

def main_loop():
    """
    Main application loop.
    
    Processes audio frames through the pipeline:
    1. Extract audio frame
    2. Compute CWT coefficients
    3. Update GPU visualization
    4. Monitors FPS 
    
    Loops until audio ends or window is closed.
    """
    while not plotter.should_window_close():
        # Start frame timing
        frame_start = fps.start_frame()

        # Grab a frame of audio
        audio_data = audio_input.get_frame()

        if audio_data is None:
            print("End of audio reached")
            break

        # Compute CWT on that frame
        coefs = wavelet.compute_cwt(audio_data)

        # Update plot
        plotter.update_plot(coefs)

        # End frame timing and report FPS if needed
        fps.end_frame_and_report(frame_start)

    # Clean shutdown
    plotter.cleanup()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main_loop()