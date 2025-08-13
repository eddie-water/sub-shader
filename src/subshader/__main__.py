#!/usr/bin/env python3
"""
SubShader is a real-time audio visualizer.

This module orchestrates the audio processing pipeline:
 - Retrieves audio data from a local file
 - Performs Time-Frequency Analysis on the audio via the Continuous Wavelet 
   Transform (CWT) implemented with CuPy
 - Visualizes the results using a GPU-accelerated shader plot with OpenGL

"""

from subshader.utils.logging import logger_init, get_logger
from subshader.utils.os_env_setup import env_init
from subshader.utils.loop_timer import LoopTimer

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import CuWavelet
from subshader.viz.plotter import ShaderPlot

# Init logging at the module level, not every time a class is instantiated
logger_init(log_level="INFO", console_output=False, file_output=True)
log = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_SIZE = 4096  # 4k samples per frame (was incorrectly 2 << 12 = 8192)
FILE_PATH = "assets/audio/songs/beltran_sc_rip.wav"
NUM_FRAMES = 128  # Increased from 32 for smooth visualization with 75% overlap

# Display settings
FULLSCREEN = False  # Set to True for fullscreen mode

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
    RuntimeError,
    EndOfAudioException, 
    WindowCloseException
)

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class SubShader:
    """
    Main application class that orchestrates the audio processing pipeline.
    
    Manages the lifecycle of all components:
    - Audio input processing
    - Wavelet transform computation
    - GPU-accelerated visualization
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the SubShader application."""
        self.audio_input = None
        self.wavelet = None
        self.plotter = None
        self.loop_timer = None
        self._initialized = False
    
    def init(self):
        """
        Initialize all high level components.
        
        Sets up the audio processing pipeline:
        - Audio input from file
        - Wavelet transform processor
        - GPU visualization
        - FPS monitor
        """
        log.info("Initializing components...")
        
        # Audio Input - handles file reading and audio frame getter 
        self.audio_input = AudioInput(path=FILE_PATH, window_size=WINDOW_SIZE)
        sample_rate = self.audio_input.get_sample_rate()  # 44.1 kHz

        # Wavelet Object - performs Continuous Wavelet Transform (CWT) using CuPy
        self.wavelet = CuWavelet(sample_rate=sample_rate, window_size=WINDOW_SIZE)

        # Plotter Object - GPU-accelerated shader plot of downsampled cwt results
        result_shape = self.wavelet.get_downsampled_result_shape()
        self.plotter = ShaderPlot(file_path=FILE_PATH, frame_shape=result_shape, num_frames=NUM_FRAMES, fullscreen=FULLSCREEN)

        # Loop timer - performance monitoring
        self.loop_timer = LoopTimer()
        
        self._initialized = True
        log.info("Init complete")

    def main_loop(self):
        """
        Main loop. Runs until audio ends or window is closed.

        Processes audio frames through the pipeline:
        - Gets frame of audio data
        - Compute CWT coefficients on the audio
        - Updates the plot with normalized CWT coefficients
        - Monitors FPS 
        """
        
        log.info("Starting main loop")
        
        while not self.plotter.should_window_close():
            # Start loop timing
            loop_start = self.loop_timer.start_loop()

            # Grab a frame of audio and check for end of file
            if (audio_data := self.audio_input.get_frame()) is None:
               raise EndOfAudioException("Audio file processing complete")

            # Compute CWT on audio
            coefs = self.wavelet.compute_cwt(audio_data)

            # Update plot with coefficient results
            self.plotter.update_plot(coefs)

            # End loop timing 
            self.loop_timer.end_loop_and_report(loop_start)

        raise WindowCloseException("Window Closed")

    def cleanup(self):
        """Clean up all resources in the correct order."""
        if not self._initialized:
            return

        log.info("Cleaning up resources")

        if self.audio_input:
            self.audio_input.cleanup()  # Close audio file

        if self.wavelet:
            self.wavelet.cleanup()      # Clean up GPU memory

        if self.plotter:
            self.plotter.cleanup()      # Terminate GLFW and OpenGL

        self._initialized = False
        log.info("Cleanup complete")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the SubShader application."""
    env_init()
    subshader = SubShader()
    subshader.init()

    try:
        subshader.main_loop()
    except GRACEFUL_EXIT_EXCEPTIONS as e:
        if isinstance(e, KeyboardInterrupt):
            log.warning("Keyboard Interrupt received.")
        elif isinstance(e, (EndOfAudioException, WindowCloseException)):
            log.warning(f"Graceful exit: {e}")
        else:
            log.error(f"Unexpected error: {e}")
    finally:
        subshader.cleanup()
        
    log.info("Application shutdown complete")

if __name__ == '__main__':
    main()