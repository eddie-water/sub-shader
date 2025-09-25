"""
Configuration Module for SubShader.

Centralized configuration management for SubShader module components:
 - Defines configuration classes for audio, wavelet, and visualization settings
 - Parameter validation
 - Sensible defaults while allowing easy customization
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
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
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class GlobalNormalizationConfig:
    """Configuration for global normalization across frames."""
    
    # Enable/disable global normalization
    enabled: bool = True
    
    # Robust statistics parameters
    percentile: float = 99.0
    
    # Adaptation parameters
    decay_rate: float = 0.001  # Exponential decay rate per frame
    floor_value: float = 1e-8  # Minimum global factor to prevent division by zero
    
    # Warm-up parameters
    warmup_frames: int = 10  # Frames to process before normalization is "ready"
    
    # Enhancement parameters
    log_mapping: bool = False  # Apply log1p mapping for perceptual detail
    
    def validate(self) -> List[str]:
        """Validate global normalization configuration parameters."""
        errors = []
        
        if not (0.0 < self.percentile <= 100.0):
            errors.append(f"percentile ({self.percentile}) must be between 0 and 100")
        
        if not (0.0 <= self.decay_rate < 1.0):
            errors.append(f"decay_rate ({self.decay_rate}) must be between 0 and 1")
        
        if self.floor_value <= 0:
            errors.append(f"floor_value ({self.floor_value}) must be positive")
        
        if self.warmup_frames < 0:
            errors.append(f"warmup_frames ({self.warmup_frames}) must be non-negative")
        
        return errors


@dataclass
class WaveletConfig:
    """Configuration for wavelet transform and normalization."""

    # Sampling parameters
    typical_sampling_freq: int = 44100

    # Scale parameters for chromatic (musical) scale
    notes_per_octave: int = 12
    num_octaves: int = 10
    root_note_a0_hz: float = 27.5

    # Result of CWT is either its magnitude ("mag") or power ("pow")
    cwt_out_type: str = "pow"

    # Percentage of the middle section of the CWT result to keep
    reliable_mid_section_p: float = 0.7 

    # Downsampling parameters - default for real-time rendering, reduce for better performance 
    target_width: int = 256

    # TODO 36 - Instead of being able to disable global normalization, determine
    # how I can use it for certain when in the main loop, but access the 
    # stages of the wavelet cwt call
    global_norm: GlobalNormalizationConfig = None

    # TODO 36 - If global normalization is disabled, would I really even need to
    # be using these? Think of when I would want these old legacy params?
    # Maybe they'd be useful for ^ but atm can't really think of any
    # Legacy normalization parameters (used when global normalization is disabled)
    db_floor: float = -80.0
    db_ceil: float = 0.0
    epsilon: float = 1e-12
    output_dtype: np.dtype = np.float32
    
    def __post_init__(self):
        """Initialize global normalization config if not provided."""
        if self.global_norm is None:
            self.global_norm = GlobalNormalizationConfig()
    
    def validate(self) -> List[str]:
        """
        Validate critical wavelet configuration parameters.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate global normalization config
        if self.global_norm:
            errors.extend(self.global_norm.validate())
        
        # Legacy normalization validation (still needed for fallback)
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
    # Frame buffer parameters - optimized for texture limits and performance  
    num_frames: int = 64  # Increased for smoother scrolling (64 * 256 = 16384, exactly at OpenGL limit) 
    
    # Shader parameters - optimized for visual clarity
    scaling_factor: float = 0.25  # Boost dim audio signals for visibility (currently unused in shader)
    gamma: float = 0.4  # Slightly increased gamma for better contrast

    # Note: Colormap is now handled by the inferno colormap in the fragment shader
    
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
            
        if self.gamma <= 0:
            errors.append(f"gamma ({self.gamma}) must be positive")
        
        # Note: colormap validation removed as colors are now handled by inferno shader
        
        return errors


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    # Audio File parameters
    file_path: str = "assets/audio/songs/beltran_sc_rip.wav"
    chunk_size: int = 2048
    overlap_factor: float = 0.5     # As a percentage (0.5 = 50% overlap)

    def validate(self) -> List[str]:
        """
        Validate critical audio configuration parameters.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if not os.path.exists(self.file_path):
            errors.append(f"Audio file not found: {self.file_path}")
        
        if self.chunk_size <= 0:
            errors.append(f"chunk_size ({self.chunk_size}) must be positive")
        
        if not (0.0 <= self.overlap_factor <= 0.9):
            errors.append(f"overlap_factor ({self.overlap_factor}) must be between 0.0 and 0.9")

        return errors


@dataclass
class ProcessingConfig:
    """Configuration for audio processing pipeline."""
    
    # Component configurations
    audio: AudioConfig = None
    wavelet: WaveletConfig = None
    viz: VisualizationConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.audio is None:
            self.audio = AudioConfig()
        if self.wavelet is None:
            self.wavelet = WaveletConfig()
        if self.viz is None:
            self.viz = VisualizationConfig()
    
    def validate(self, opengl_max_texture_size: int = 16384) -> List[str]:
        """
        Validate the complete processing configuration with comprehensive hardware checks.
        
        Args:
            opengl_max_texture_size (int): Maximum texture size supported by OpenGL
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate sub-configurations
        if self.audio:
            errors.extend(self.audio.validate())
        if self.wavelet:
            errors.extend(self.wavelet.validate())
        if self.viz:
            errors.extend(self.viz.validate())

        # TODO: Look through these and see how important they are@q

        # CRITICAL: Validate OpenGL texture size limits
        if self.wavelet and self.viz:
            texture_width = self.viz.num_frames * self.wavelet.target_width
            if texture_width > opengl_max_texture_size:
                errors.append(
                    f"CRITICAL: Texture size exceeds OpenGL limit! "
                    f"num_frames ({self.viz.num_frames}) × target_width ({self.wavelet.target_width}) "
                    f"= {texture_width} pixels > {opengl_max_texture_size} limit. "
                    f"Reduce num_frames to {opengl_max_texture_size // self.wavelet.target_width} or less."
                )
        
        # GPU Memory validation 
        gpu_memory_errors = self._validate_gpu_memory()
        errors.extend(gpu_memory_errors)
        
        # CPU Memory validation
        cpu_memory_errors = self._validate_cpu_memory()
        errors.extend(cpu_memory_errors)
        
        # Performance validation
        performance_errors = self._validate_performance_targets()
        errors.extend(performance_errors)
        
        return errors

    def _validate_gpu_memory(self) -> List[str]:
        """Estimate and validate GPU memory requirements."""
        errors = []
        
        if not (self.wavelet and self.viz):
            return errors
        
        # Estimate GPU memory usage in MB
        freq_bins = self.wavelet.num_octaves * self.wavelet.notes_per_octave  # Approximate
        
        # CWT coefficients: freq_bins × input_n × 4 bytes (float32)
        cwt_memory_mb = (freq_bins * self.audio.chunk_size * 4) / (1024 * 1024)
        
        # Texture memory: freq_bins × (num_frames × target_width) × 4 bytes  
        texture_memory_mb = (freq_bins * self.viz.num_frames * self.wavelet.target_width * 4) / (1024 * 1024)
        
        # Rolling frame buffer: num_frames × freq_bins × target_width × 4 bytes
        buffer_memory_mb = (self.viz.num_frames * freq_bins * self.wavelet.target_width * 4) / (1024 * 1024)
        
        total_gpu_mb = cwt_memory_mb + texture_memory_mb + buffer_memory_mb
        
        # Warn if estimated usage exceeds reasonable limits
        if total_gpu_mb > 2048:  # 2GB warning threshold
            errors.append(
                f"WARNING: High GPU memory usage estimated: {total_gpu_mb:.1f} MB "
                f"(CWT: {cwt_memory_mb:.1f}, Texture: {texture_memory_mb:.1f}, Buffer: {buffer_memory_mb:.1f}). "
                f"Consider reducing num_frames or target_width."
            )
        
        return errors

    def _validate_cpu_memory(self) -> List[str]:
        """Estimate and validate CPU memory requirements.""" 
        errors = []
        
        if not (self.wavelet and self.viz):
            return errors
        
        # Audio buffer memory (for file reading)
        audio_buffer_mb = (self.audio.chunk_size * 4) / (1024 * 1024)  # float32
        
        # Wavelets kernels storage (estimated)
        freq_bins = self.wavelet.num_octaves * self.wavelet.notes_per_octave
        wavelet_kernels_mb = (freq_bins * self.audio.chunk_size * 8) / (1024 * 1024)  # complex64
        
        total_cpu_mb = audio_buffer_mb + wavelet_kernels_mb
        
        # Warn if estimated usage exceeds reasonable limits
        if total_cpu_mb > 1024:  # 1GB warning threshold  
            errors.append(
                f"WARNING: High CPU memory usage estimated: {total_cpu_mb:.1f} MB "
                f"(Audio: {audio_buffer_mb:.1f}, Kernels: {wavelet_kernels_mb:.1f}). "
                f"Consider reducing chunk_size."
            )
        
        return errors

    def _validate_performance_targets(self) -> List[str]:
        """Validate configuration against performance targets."""
        errors = []
        
        if not (self.wavelet and self.viz):
            return errors
        
        # Estimate computational complexity
        freq_bins = self.wavelet.num_octaves * self.wavelet.notes_per_octave
        operations_per_frame = freq_bins * self.audio.chunk_size
        
        # Warn about potentially expensive configurations
        if operations_per_frame > 10_000_000:  # 10M operations per frame
            errors.append(
                f"WARNING: High computational load estimated: {operations_per_frame:,} operations/frame. "
                f"This may impact real-time performance. Consider reducing chunk_size or num_octaves."
            )
        
        # Check for reasonable frame rates
        frames_per_second = self.wavelet.typical_sampling_freq / (self.audio.chunk_size * (1 - self.audio.overlap_factor))
        
        if frames_per_second > 200:  # Very high frame rate
            errors.append(
                f"WARNING: Very high frame rate estimated: {frames_per_second:.1f} FPS. "
                f"This may cause excessive GPU updates. Consider increasing chunk_size."
            )
        elif frames_per_second < 10:  # Very low frame rate  
            errors.append(
                f"WARNING: Low frame rate estimated: {frames_per_second:.1f} FPS. "
                f"This may cause choppy visualization. Consider decreasing chunk_size."
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
        texture_width = config.viz.num_frames * config.wavelet.target_width
        log.info(f"  Texture size: {texture_width} pixels (within {opengl_max_texture_size} limit)")
        return True
