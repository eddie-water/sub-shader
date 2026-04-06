"""
Configuration Module for SubShader.

Centralized configuration management for SubShader module components using
a flat inheritance hierarchy:

  PipelineConfig       — base: file_path, chunk_size, overlap, sample_rate, total_samples
    CWTConfig          — adds wavelet-specific params (notes, octaves, root freq, etc.)
    RendererConfig     — adds rendering params (num_frames, color normalization)

All config objects flow through the pipeline. AudioStream discovers runtime
values (sample_rate, total_samples) and writes them back to the shared config.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from dataclasses import dataclass, field
from typing import Tuple, List

import tkinter as tk

from .utils.logging import get_logger

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

def _get_system_display_size() -> Tuple[int, int]:
    """
    Get system display dimensions with fallback.

    Returns:
        Tuple[int, int]: Screen width and height in pixels
    """
    try:
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        # Fallback to Full HD if system detection fails
        return 1920, 1080

# =============================================================================
# COLOR NORMALIZATION CONFIG
# =============================================================================

@dataclass
class ColorNormalizationConfig:
    """Configuration for color normalization across all data frames."""

    # Gamma correction factor for perceptual enhancement (gamma = 1 is no correction)
    gamma: float = 0.5

    # Exponential smoothing weight for global intensity tracking (lower = slower adaptation)
    global_intensity_smoothing_weight: float = 0.1

    # Percentile of the global intensity distribution used for normalization reference
    global_intensity_percentile: float = 99.0

    # Percentile used to normalize each individual frame
    frame_intensity_percentile: float = 95.0

    # Exponential decay rate applied to global intensity per frame
    decay_rate: float = 0.95

    # Initial global intensity estimate (prevents division by zero at startup)
    initial_intensity: float = 0.1

    # Percentile used to set per-frame brightness ceiling
    frame_brightness_percentile: float = 99.0

    def validate(self) -> List[str]:
        """Validate color normalization configuration parameters."""
        errors = []

        if not (0.0 < self.global_intensity_percentile <= 100.0):
            errors.append(
                f"global_intensity_percentile ({self.global_intensity_percentile}) "
                f"must be between 0 and 100"
            )

        if not (0.0 < self.frame_intensity_percentile <= 100.0):
            errors.append(
                f"frame_intensity_percentile ({self.frame_intensity_percentile}) "
                f"must be between 0 and 100"
            )

        if not (0.0 < self.decay_rate < 1.0):
            errors.append(f"decay_rate ({self.decay_rate}) must be strictly between 0 and 1")

        if self.initial_intensity <= 0:
            errors.append(f"initial_intensity ({self.initial_intensity}) must be positive")

        return errors

# =============================================================================
# BASE PIPELINE CONFIG
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Base configuration shared across all pipeline stages.

    Fields are grouped into two categories:
      - User-settable: set before pipeline construction
      - Runtime-discovered: written by AudioStream after file open

    Derived properties (hop_size, nyquist_freq) are computed on demand.
    """

    # --- User-settable ---

    # Path to the audio file for processing (per D-04: reference dir default)
    file_path: str = "assets/audio/reference/beltran_sc_rip.wav"

    # Number of samples per chunk. 1<<14 = 16384 samples at 44100 Hz covers
    # the full 10-octave chromatic range needed for low-frequency wavelet support.
    chunk_size: int = 1 << 14

    # Fraction of overlap between consecutive chunks (0.5 = 50% overlap)
    # TODO-45 Fix the overlap and plot overlap relationship
    overlap_factor: float = 0.5

    # --- Runtime-discovered by AudioStream ---

    # Sample rate discovered from audio file; default matches standard CD quality
    sample_rate: float = 44100.0

    # Total samples in the audio file; 0 until AudioStream opens the file
    total_samples: int = 0

    # --- Derived properties ---

    @property
    def hop_size(self) -> int:
        """Number of samples advanced per chunk (chunk_size * (1 - overlap_factor))."""
        return int(self.chunk_size * (1.0 - self.overlap_factor))

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency in Hz (sample_rate / 2)."""
        return self.sample_rate / 2.0

    def validate(self) -> List[str]:
        """
        Validate critical pipeline configuration parameters.

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []

        if not os.path.exists(self.file_path):
            errors.append(f"Audio file not found: {self.file_path}")

        if self.chunk_size <= 0:
            errors.append(f"chunk_size ({self.chunk_size}) must be positive")

        if not (0.0 < self.overlap_factor < 1.0):
            errors.append(
                f"overlap_factor ({self.overlap_factor}) must be strictly between 0.0 and 1.0"
            )

        return errors

# =============================================================================
# CWT CONFIG
# =============================================================================

@dataclass
class CWTConfig(PipelineConfig):
    """
    Configuration for the Continuous Wavelet Transform stage.

    Extends PipelineConfig with wavelet-specific parameters for chromatic
    scale construction and wavelet kernel shape.
    """

    # Number of frequency bins per octave (12 = semitone resolution)
    notes_per_octave: int = 12

    # Number of octaves to cover in the chromatic scale
    num_octaves: int = 10

    # Root note frequency in Hz (A0 = 27.5 Hz, the lowest piano key)
    root_note_hz: float = 27.5

    # Output width in time bins after downsampling CWT coefficients
    target_width: int = 64

    # Number of full oscillation cycles in the wavelet kernel (controls time/freq tradeoff)
    num_cycles: int = 6

    # Number of FWHM cycles used to define the reliable coefficient region
    num_fwhm_cycles: int = 3

    def validate(self) -> List[str]:
        """
        Validate CWT configuration.

        Extends PipelineConfig.validate() with wavelet-specific checks.

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if self.target_width <= 0:
            errors.append(f"target_width ({self.target_width}) must be positive")

        if self.notes_per_octave <= 0:
            errors.append(f"notes_per_octave ({self.notes_per_octave}) must be positive")

        if self.num_octaves <= 0:
            errors.append(f"num_octaves ({self.num_octaves}) must be positive")

        return errors

# =============================================================================
# RENDERER CONFIG
# =============================================================================

@dataclass
class RendererConfig(PipelineConfig):
    """
    Configuration for the shader-based renderer.

    Extends PipelineConfig with display and color normalization parameters.
    """

    # Number of time frames stored in the circular frame buffer
    num_frames: int = 256

    # Color normalization parameters (gamma, percentiles, decay)
    color_norm: ColorNormalizationConfig = field(default_factory=ColorNormalizationConfig)

    # Window dimensions — auto-detected from system display, fallback to Full HD
    window_width: int = field(default_factory=lambda: _get_system_display_size()[0])
    window_height: int = field(default_factory=lambda: _get_system_display_size()[1])

    def validate(self) -> List[str]:
        """
        Validate renderer configuration.

        Extends PipelineConfig.validate() with rendering-specific checks.

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if self.num_frames <= 0:
            errors.append(f"num_frames ({self.num_frames}) must be positive")

        if self.color_norm:
            errors.extend(self.color_norm.validate())

        return errors

# =============================================================================
# DEPRECATED COMPATIBILITY SHIM
# =============================================================================

# DEPRECATED: Use CWTConfig or PipelineConfig directly.
# Will be removed in next phase once all callers migrate to the new hierarchy.
ProcessingConfig = CWTConfig

# DEPRECATED: WaveletConfig was the pre-08 name for CWTConfig.
# Preserved so wavelet.py can be imported without modification until Plan 08-05.
WaveletConfig = CWTConfig

# =============================================================================
# DEFAULT CONFIG FACTORY
# =============================================================================

def get_default_config() -> CWTConfig:
    """
    Return default pipeline configuration as a CWTConfig instance.

    CWTConfig is the most parameter-rich subclass and is the appropriate
    default for the full pipeline (audio → CWT → renderer). Callers that
    previously used ProcessingConfig still work because ProcessingConfig
    is aliased to CWTConfig.

    Returns:
        CWTConfig: Default configuration

    Raises:
        ValueError: If configuration validation fails (e.g., file not found)
    """
    config = CWTConfig()

    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        log.error(error_msg)
        raise ValueError(error_msg)
    else:
        log.info("Configuration validation passed")

    return config
