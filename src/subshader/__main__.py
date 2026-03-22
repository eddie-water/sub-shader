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

import argparse
import time

from subshader.utils.logging import logger_init, get_logger
from subshader.utils.loop_timer import LoopTimer

from subshader.config import get_default_config, ProcessingConfig

from subshader.audio.audio_input import AudioInput
from subshader.audio.audio_player import AudioPlayer
from subshader.dsp.wavelet import CuWavelet, NpWavelet
from subshader.utils.gpu import gpu_available
from subshader.viz.plotter import ShaderPlot

from subshader import exceptions

# =============================================================================
# LOGGING
# =============================================================================

logger_init(log_level="INFO", console_output=True, file_output=True)
log = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load default configuration
config = get_default_config()

# =============================================================================
# MAIN APP
# =============================================================================

class SubShader:
    """Main class that orchestrates the audio visualization pipeline."""

    def __init__(self, config: ProcessingConfig):
        """Initialize all high level modules."""
        log.info("Initializing modules...")

        # Audio Input - handles file reading and audio getter
        self.audio_input = AudioInput(
            path=config.audio.file_path,
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor,
        )

        # Audio Player - handles real-time playback via sounddevice (D-04)
        # Load entire audio file into memory as float32 for playback (D-02)
        audio_data = self.audio_input.get_entire_audio()
        self.audio_player = AudioPlayer(
            audio_data=audio_data,
            sample_rate=float(self.audio_input.get_sample_rate()),
        )

        # GPU Detection - select wavelet implementation (per D-07, D-08, D-09)
        if gpu_available():
            wavelet = CuWavelet
        else:
            log.warning("GPU unavailable, running on NumPy — expect slower performance")
            wavelet = NpWavelet

        # Wavelet Object - performs the Continuous Wavelet Transform
        self.wavelet = wavelet(
            sample_rate=self.audio_input.get_sample_rate(),
            input_n=self.audio_input.get_chunk_size(),
            config=config.wavelet,
            overlap_factor=config.audio.overlap_factor,
        )

        # Plotter Object - GPU-accelerated shader plot of output results
        self.plotter = ShaderPlot(
            file_path=config.audio.file_path,
            frame_shape=self.wavelet.get_output_shape(),
            config=config.viz,
        )

        # Loop timer - performance monitoring
        self.loop_timer = LoopTimer()

        log.info("Initialization complete")

    def loop(self):
        """
        Audio-clock-driven render loop.

        The audio device clock is the single source of truth (D-06).
        Each iteration checks the audio playback position. When the audio
        has advanced past the next chunk boundary, compute CWT and render.
        When no new chunk is ready, yield briefly to avoid busy-wait (D-08).
        If render falls behind, skip to current position (D-09).
        """
        log.info("Starting audio-synced render loop")

        # Start audio playback — audio and visualization begin simultaneously (D-10)
        self.audio_player.start()

        hop_size = self.audio_input.hop_size
        next_expected_sample = 0
        previous_playback_pos = 0

        while not self.plotter.should_window_close():
            loop_start = self.loop_timer.start_loop()

            playback_pos = self.audio_player.get_playback_sample()

            # Detect loop wrap: playback position jumped backward (Pitfall 4)
            if self.audio_player.has_looped():
                self.audio_player.clear_loop_event()
                next_expected_sample = 0
                self.audio_input.file_pos = 0
                previous_playback_pos = 0
                log.info("Audio looped — resetting visualization")

            if playback_pos < next_expected_sample:
                # Audio has not advanced to next chunk boundary — yield (D-08)
                time.sleep(0.001)
                continue

            # Audio has advanced: seek AudioInput to match current audio position (D-06)
            # If multiple chunks were skipped, render the most recent one (D-09)
            target_sample = (playback_pos // hop_size) * hop_size
            self.audio_input.file_pos = target_sample

            audio_data = self.audio_input.get_chunk()
            if audio_data is None:
                # Near end of file, wait for loop wrap
                time.sleep(0.001)
                continue

            coefs = self.wavelet.cwt(audio_data)
            self.plotter.update_plot(coefs)
            self.loop_timer.end_loop_and_report(loop_start)

            next_expected_sample = target_sample + hop_size
            previous_playback_pos = playback_pos

        raise exceptions.WindowCloseException("Window closed by user")

    def cleanup(self):
        """Idempotent cleanup: safe to call any time, even after partial init."""
        log.info("Cleaning up module resources")

        if hasattr(self, 'audio_player') and self.audio_player:
            try:
                self.audio_player.stop()
            finally:
                self.audio_player = None

        if hasattr(self, 'plotter') and self.plotter:
            try:
                self.plotter.cleanup()
            finally:
                self.plotter = None

        if hasattr(self, 'wavelet') and self.wavelet:
            try:
                self.wavelet.cleanup()
            finally:
                self.wavelet = None

        if hasattr(self, 'audio_input') and self.audio_input:
            try:
                self.audio_input.cleanup()
            finally:
                self.audio_input = None

        if hasattr(self, 'loop_timer'):
            self.loop_timer = None

        log.info("Cleanup complete")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the SubShader application."""
    parser = argparse.ArgumentParser(
        prog="subshader",
        description="SubShader real-time audio visualizer",
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to WAV audio file (uses default if not provided)",
    )
    args = parser.parse_args()

    if args.audio_file:
        config.audio.file_path = args.audio_file

    subshader = SubShader(config)
    try:
        subshader.loop()
    except exceptions.GRACEFUL_EXCEPTIONS as e:
        exceptions.reporter.report(e)
    finally:
        subshader.cleanup()

    log.info("Application shutdown complete")

if __name__ == '__main__':
    main()
