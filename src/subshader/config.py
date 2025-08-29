"""
Configuration module for SubShader components.

This module provides centralized configuration classes to make wavelet 
normalization, plot rendering, and shader visualization agnostic of each other 
while maintaining sensible defaults and allowing easy customization.

All configuration parameters are validated here. Classes can assume that any
config objects passed to them contain valid, pre-checked parameters.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import tkinter as tk
import os
from .utils.logging import get_logger

log = get_logger(__name__)


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


@dataclass
class WaveletConfig:
    """Configuration for wavelet transform and normalization."""
    
    # Normalization parameters
    # TODO : Experiment with floor and ceil values for better visualization
    db_floor: float = -80.0
    db_ceil: float = 0.0
    epsilon: float = 1e-12
    output_dtype: np.dtype = np.float32

    # Downsampling parameters 
    # TODO : Determine a cohesive relationship between target width and the window size and kernel legnth and plot resolution etc
    target_width: int = 512

    # Musical scale parameters
    notes_per_octave: int = 12
    num_octaves: int = 10
    root_note_a0_hz: float = 27.5

    # Audio parameters
    typical_sampling_freq: int = 44100
    
    def validate(self) -> List[str]:
        """
        Validate critical wavelet configuration parameters.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.db_floor >= self.db_ceil:
            errors.append(f"db_floor ({self.db_floor}) must be less than db_ceil ({self.db_ceil})")
        
        if self.target_width <= 0:
            errors.append(f"target_width ({self.target_width}) must be positive")
        
        if self.notes_per_octave <= 0:
            errors.append(f"notes_per_octave ({self.notes_per_octave}) must be positive")
            
        if self.num_octaves <= 0:
            errors.append(f"num_octaves ({self.num_octaves}) must be positive")
        
        return errors


@dataclass
class VisualizationConfig:
    """Configuration for visualization rendering."""
    
    # Window parameters (auto-derived from system display)
    window_width: int = None
    window_height: int = None
    # TODO : Determine a cohesive relationship between window dimensions and num_frames and plot resolution etc and the opengl texture size limit
    num_frames: int = 32 
    
    # Shader parameters 
    scaling_factor: float = 0.25  # Boost dim audio signals for visibility
    gamma_correction: float = 0.35  # Gamma correction for visual enhancement

    # TODO : Consolidate colormap and transition points into a cohesive uniform colorscheme    
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
            self.window_width, self.window_height = _get_system_display_size()
    
    def validate(self) -> List[str]:
        """
        Validate critical visualization configuration parameters.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.num_frames <= 0:
            errors.append(f"num_frames ({self.num_frames}) must be positive")
        
        if self.scaling_factor <= 0:
            errors.append(f"scaling_factor ({self.scaling_factor}) must be positive")
            
        if self.gamma_correction <= 0:
            errors.append(f"gamma_correction ({self.gamma_correction}) must be positive")
        
        # colormap must have valid RGB values
        for i, color in enumerate(self.colormap_colors):
            if len(color) != 3:
                errors.append(f"colormap_colors[{i}] must be RGB tuple (3 values), got {len(color)}")
            else:
                for j, component in enumerate(color):
                    if not (0.0 <= component <= 1.0):
                        errors.append(f"colormap_colors[{i}][{j}] ({component}) must be between 0.0 and 1.0")
        
        return errors


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
    
    def validate(self, opengl_max_texture_size: int = 16384) -> List[str]:
        """
        Validate the complete processing configuration.
        
        Args:
            opengl_max_texture_size (int): Maximum texture size supported by OpenGL
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if not os.path.exists(self.file_path):
            errors.append(f"Audio file not found: {self.file_path}")
        
        if self.window_size <= 0:
            errors.append(f"window_size ({self.window_size}) must be positive")
        
        # Validate sub-configurations
        if self.wavelet:
            errors.extend(self.wavelet.validate())
        if self.visualization:
            errors.extend(self.visualization.validate())
        
        # CRITICAL: Validate OpenGL texture size limits
        if self.wavelet and self.visualization:
            texture_width = self.visualization.num_frames * self.wavelet.target_width
            if texture_width > opengl_max_texture_size:
                errors.append(
                    f"CRITICAL: Texture size exceeds OpenGL limit! "
                    f"num_frames ({self.visualization.num_frames}) × target_width ({self.wavelet.target_width}) "
                    f"= {texture_width} pixels > {opengl_max_texture_size} limit. "
                    f"Reduce num_frames to {opengl_max_texture_size // self.wavelet.target_width} or less."
                )
        
        return errors


# Default configuration
def get_default_config() -> ProcessingConfig:
    """
    Get default configuration with system-derived dimensions.
    
    Args:
        validate (bool): Whether to validate the configuration before returning
        
    Returns:
        ProcessingConfig: Default configuration
        
    Raises:
        ValueError: If validation is enabled and configuration is invalid
    """
    config = ProcessingConfig()
    
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        log.error(error_msg)
        raise ValueError(error_msg)
    else:
        log.info("✅ Configuration validation passed")
    
    return config


def validate_config(config: ProcessingConfig, opengl_max_texture_size: int = 16384) -> bool:
    """
    Validate a configuration and log results.
    
    Args:
        config (ProcessingConfig): Configuration to validate
        opengl_max_texture_size (int): Maximum OpenGL texture size
        
    Returns:
        bool: True if valid, False if invalid
    """
    errors = config.validate(opengl_max_texture_size)
    
    if errors:
        log.error("❌ Configuration validation failed:")
        for error in errors:
            log.error(f"  - {error}")
        return False
    else:
        log.info("✅ Configuration validation passed")
        texture_width = config.visualization.num_frames * config.wavelet.target_width
        log.info(f"  Texture size: {texture_width} pixels (within {opengl_max_texture_size} limit)")
        return True
