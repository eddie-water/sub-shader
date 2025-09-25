#!/usr/bin/env python3

"""
SubShader is a real-time audio visualizer.

This module orchestrates the audio processing pipeline:
 - Retrieves audio data from a local file
 - Performs Time-Frequency Analysis on the audio via the Continuous Wavelet 
   Transform implemented with CuPy
 - Visualizes the results using a GPU-accelerated shader plot with OpenGL
"""

# =============================================================================
# IMPORTS
# =============================================================================

from subshader.utils.logging import logger_init, get_logger
from subshader.utils.os_env_setup import env_init
from subshader.utils.loop_timer import LoopTimer

from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput, AudioFileNotFoundError, EndOfAudioException
from subshader.dsp.wavelet import CuWavelet
from subshader.viz.plotter import ShaderPlot, WindowCloseException

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
config.audio.file_path = "assets/audio/songs/beltran_sc_rip_8bar.wav"
# TODO 36 this breaks when I do != 1.0 - maybe that's why I wasn't seeing much edge effects go away
config.wavelet.reliable_mid_section_p = 1.0

# =============================================================================
# EXCEPTIONS
# =============================================================================

GRACEFUL_EXCEPTIONS = (
    KeyboardInterrupt,
    RuntimeError,
    EndOfAudioException, 
    WindowCloseException,
    AudioFileNotFoundError
)

# =============================================================================
# MAIN APP
# =============================================================================

class SubShader:
    """Main class that orchestrates the audio visualization pipeline."""

    def __init__(self):
        """Initialize the SubShader application and all high levelmodules."""
        log.info("Init modules...")

        # Audio Input - handles file reading and audio getter 
        self.audio_input = AudioInput(
            path=config.audio.file_path, 
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor 
        )
        self.sample_rate = self.audio_input.get_sample_rate()

        # Wavelet Object - performs the Continuous Wavelet Transform using CuPy
        self.wavelet = CuWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size, 
            config=config.wavelet)
        self.result_shape = self.wavelet.get_output_shape()

        # Plotter Object - GPU-accelerated shader plot of output results
        self.plotter = ShaderPlot(
            file_path=config.audio.file_path, 
            frame_shape=self.result_shape, 
            num_frames=config.viz.num_frames,
            gamma=config.viz.gamma
        )

        # Loop timer - performance monitoring
        self.loop_timer = LoopTimer()
        
        log.info("Init complete")
    
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

            # Grab a chunk of audio and check for end of file
            if (audio_data := self.audio_input.get_chunk()) is None:
               raise EndOfAudioException("Audio file processing complete - reached EOF")

            # Compute CWT on audio
            coefs = self.wavelet.compute_cwt(audio_data)

            # Update plot with coefficient results
            self.plotter.update_plot(coefs)

            # End loop timing 
            self.loop_timer.end_loop_and_report(loop_start)

        raise WindowCloseException("Window Closed")

    def cleanup(self):
        """Idempotent cleanup: safe to call any time, even after partial init."""
        log.info("Cleaning up resources")

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
    except GRACEFUL_EXCEPTIONS as e:
        if hasattr(e, 'log_level') and hasattr(e, 'log_message'):
            getattr(log, e.log_level)(e.log_message)
        elif isinstance(e, KeyboardInterrupt):
            log.warning("Keyboard Interrupt received.")
        else:
            log.error(f"Unexpected error: {e}")
    finally:
        subshader.cleanup()
        
    log.info("Application shutdown complete")

if __name__ == '__main__':
    # TODO What is the value of the env_init? Can it be tucked in somewhere else?
    env_init()
    main()