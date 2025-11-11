#!/usr/bin/env python3

"""
SubShader is a real-time audio visualizer.

This module orchestrates the audio processing pipeline:
 - Retrieves audio data from a local file
 - Performs Time-Frequency Analysis on the audio via the Continuous Wavelet 
   Transform implemented with CuPy
 - Visualizes the time-frequency results with a 2D shader (OpenGL)
"""

# =============================================================================
# IMPORTS
# =============================================================================

from subshader.utils.logging import logger_init, get_logger
from subshader.utils.loop_timer import LoopTimer

from subshader.config import get_default_config

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import CuWavelet
from subshader.viz.plotter import ShaderPlot

from subshader import exceptions

# =============================================================================
# LOGGING
# =============================================================================

logger_init(log_level="INFO", console_output=False, file_output=True)
log = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load default configuration
config = get_default_config()

# Override default configs
config.audio.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav"

# =============================================================================
# MAIN APP
# =============================================================================

class SubShader:
    """Main class that orchestrates the audio visualization pipeline."""

    def __init__(self):
        """Initialize all high level modules."""
        log.info("Initializing modules...")

        # Audio Input - handles file reading and audio getter 
        self.audio_input = AudioInput(path=config.audio.file_path, 
                                      chunk_size=config.audio.chunk_size,
                                      overlap_factor=config.audio.overlap_factor)

        # Wavelet Object - performs the Continuous Wavelet Transform using CuPy
        self.wavelet = CuWavelet(sample_rate=self.audio_input.get_sample_rate(), 
                                 input_n=self.audio_input.get_chunk_size(),
                                 config=config.wavelet)

        # Plotter Object - GPU-accelerated shader plot of output results
        self.plotter = ShaderPlot(file_path=config.audio.file_path, 
                                  frame_shape=self.wavelet.get_output_shape(),
                                  config=config.viz)

        # Loop timer - performance monitoring
        self.loop_timer = LoopTimer()
        
        log.info("Initialization complete")
    
    def loop(self):
        """
        Main loop. Runs until audio ends or window is closed.

        Processes audio frames through the pipeline:
        - Retrieves a chunk of audio data with an overlap scheme
        - Compute CWT on the audio, then normalizes and downsamples the 
          resulting coefficients
        - Updates the plot with the results
        - FPS monitoring
        """
        log.info("Starting main loop")

        while not self.plotter.should_window_close():
            # Start loop timing
            loop_start = self.loop_timer.start_loop()

            # Retrieve audio chunk and check for end of audio
            if (audio_data := self.audio_input.get_chunk()) is None:
               raise exceptions.EndOfAudioException("Audio file processing complete - reached end of file.")

            # Perform CWT on audio
            coefs = self.wavelet.cwt(audio_data)

            # Update plot with CWT results
            self.plotter.update_plot(coefs)

            # End loop timing 
            self.loop_timer.end_loop_and_report(loop_start)

        raise exceptions.WindowCloseException("Window closed by user")

    def cleanup(self):
        """Idempotent cleanup: safe to call any time, even after partial init."""
        log.info("Cleaning up module resources")

        if self.plotter:
            try:
                self.plotter.cleanup()
            finally:
                self.plotter = None
        if self.wavelet:
            try:
                self.wavelet.cleanup()
            finally:
                self.wavelet = None
        if self.audio_input:
            try:
                self.audio_input.cleanup()
            finally:
                self.audio_input = None

        self.loop_timer = None

        log.info("Cleanup complete")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the SubShader application."""
    subshader = SubShader()

    try:
        subshader.loop()
    except exceptions.GRACEFUL_EXCEPTIONS as e:
        exceptions.reporter.report(e)
    finally:
        subshader.cleanup()
        
    log.info("Application shutdown complete")

if __name__ == '__main__':
    main()