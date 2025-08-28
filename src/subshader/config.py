"""
Configuration module for SubShader components.

This module provides centralized configuration classes to make wavelet 
normalization, plot rendering, and shader visualization agnostic of each other 
while maintaining sensible defaults and allowing easy customization.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import tkinter as tk


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
        # Fallback to common resolution if system detection fails
        return 1920, 1080





@dataclass
class WaveletConfig:
    """Configuration for wavelet transform and normalization."""
    
    # Normalization parameters (adjusted for better visibility)
    db_floor: float = -60.0  # Reduced range for brighter visualization
    db_ceil: float = 0.0
    epsilon: float = 1e-12
    output_dtype: np.dtype = np.float32

    # Downsampling parameters (higher detail for fullscreen)
    target_width: int = 128

    # Musical scale parameters
    notes_per_octave: int = 12
    num_octaves: int = 10
    root_note_a0_hz: float = 27.5

    # Audio parameters
    typical_sampling_freq: int = 44100


@dataclass
class VisualizationConfig:
    """Configuration for visualization rendering."""
    
    # Window parameters (auto-derived from system display - fullscreen)
    window_width: int = None
    window_height: int = None
    num_frames: int = 256  # More history for fullscreen
    
    # Shader parameters (optimized for fullscreen)
    scaling_factor: float = 0.25  # Boost dim audio signals for visibility
    gamma_correction: float = 0.35  # Gamma correction for visual enhancement
    
    # Colormap configuration (5 RGB color tuples)
    colormap_colors: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.3),   # Dark blue
        (0.3, 0.0, 0.5),   # Purple
        (1.0, 0.0, 0.0),   # Red
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 1.0, 1.0),   # White
    )
    
    # Transition points for color interpolation
    transition_points: Tuple[float, float, float, float] = (0.3, 0.6, 0.8, 1.0)
    
    # Texture parameters
    texture_slot: int = 0
    
    def __post_init__(self):
        """Initialize window dimensions from system if not provided."""
        if self.window_width is None or self.window_height is None:
            # Use fullscreen dimensions by default
            self.window_width, self.window_height = _get_system_display_size()


@dataclass  
class ProcessingConfig:
    """Configuration for audio processing pipeline."""
    
    # File parameters
    file_path: str = "assets/audio/songs/beltran_sc_rip.wav"
    
    # Processing parameters
    window_size: int = 1024
    
    # Component configurations
    wavelet: WaveletConfig = None
    visualization: VisualizationConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.wavelet is None:
            self.wavelet = WaveletConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()


# Default configuration
def get_default_config() -> ProcessingConfig:
    """Get default fullscreen configuration with system-derived dimensions."""
    return ProcessingConfig()
