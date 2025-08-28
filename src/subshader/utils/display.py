"""
Display utility functions for detecting system capabilities and calculating
optimal window dimensions for the SubShader application.
"""

import tkinter as tk
from typing import Tuple, Optional
import logging

# Use standard logging to avoid circular imports during testing
log = logging.getLogger(__name__)


def get_system_display_info() -> Tuple[int, int]:
    """
    Get the system's primary display resolution.
    
    Returns:
        Tuple[int, int]: Screen width and height in pixels
    """
    try:
        # Create a temporary tkinter root to query display info
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        
        log.info(f"Detected system display: {width}×{height}")
        return width, height
    
    except Exception as e:
        log.warning(f"Could not detect system display: {e}, using fallback")
        # Fallback to common resolution
        return 1920, 1080


def calculate_optimal_window_size(scale_factor: float = 0.8, 
                                 min_width: int = 800, 
                                 min_height: int = 600,
                                 max_width: int = 3840,
                                 max_height: int = 2160) -> Tuple[int, int]:
    """
    Calculate optimal window dimensions based on system display.
    
    Args:
        scale_factor (float): Fraction of screen to use (0.8 = 80% of screen)
        min_width (int): Minimum window width
        min_height (int): Minimum window height  
        max_width (int): Maximum window width
        max_height (int): Maximum window height
        
    Returns:
        Tuple[int, int]: Optimal window width and height
    """
    screen_width, screen_height = get_system_display_info()
    
    # Calculate scaled dimensions
    window_width = int(screen_width * scale_factor)
    window_height = int(screen_height * scale_factor)
    
    # Apply constraints
    window_width = max(min_width, min(window_width, max_width))
    window_height = max(min_height, min(window_height, max_height))
    
    # Ensure reasonable aspect ratio for audio visualization
    # Audio spectrograms work well with wider aspect ratios
    if window_width / window_height < 1.5:  # Too square
        window_width = int(window_height * 1.6)  # Make it 16:10 ratio
        window_width = min(window_width, max_width)
    
    log.info(f"Calculated optimal window: {window_width}×{window_height} "
             f"({scale_factor*100:.0f}% of {screen_width}×{screen_height})")
    
    return window_width, window_height


def get_display_scaling_factor() -> float:
    """
    Attempt to detect system display scaling factor (DPI scaling).
    
    Returns:
        float: Display scaling factor (1.0 = 100%, 1.5 = 150%, etc.)
    """
    try:
        root = tk.Tk()
        
        # Get physical and logical DPI
        dpi = root.winfo_fpixels('1i')  # Physical DPI
        logical_dpi = 96.0  # Standard logical DPI
        
        scaling_factor = dpi / logical_dpi
        root.destroy()
        
        log.debug(f"Detected display scaling: {scaling_factor:.2f}x (DPI: {dpi:.1f})")
        return scaling_factor
        
    except Exception as e:
        log.debug(f"Could not detect display scaling: {e}")
        return 1.0


def calculate_performance_window_size() -> Tuple[int, int]:
    """Calculate window size optimized for performance."""
    return calculate_optimal_window_size(scale_factor=0.6, max_width=1920, max_height=1080)


def calculate_quality_window_size() -> Tuple[int, int]:
    """Calculate window size optimized for visual quality."""
    return calculate_optimal_window_size(scale_factor=0.9, max_width=3840, max_height=2160)


def get_window_size_for_config(config_type: str = "default") -> Tuple[int, int]:
    """
    Get window dimensions for different configuration types.
    
    Args:
        config_type (str): "default", "performance", "quality", or "fullscreen"
        
    Returns:
        Tuple[int, int]: Window width and height
    """
    if config_type == "performance":
        return calculate_performance_window_size()
    elif config_type == "quality":
        return calculate_quality_window_size()
    elif config_type == "fullscreen":
        return get_system_display_info()
    else:  # default
        return calculate_optimal_window_size()


if __name__ == "__main__":
    """Test the display detection functions."""
    print("=== Display Detection Test ===")
    
    screen_w, screen_h = get_system_display_info()
    print(f"System Display: {screen_w}×{screen_h}")
    
    scaling = get_display_scaling_factor()
    print(f"Display Scaling: {scaling:.2f}x")
    
    default_w, default_h = get_window_size_for_config("default")
    print(f"Default Window: {default_w}×{default_h}")
    
    perf_w, perf_h = get_window_size_for_config("performance")
    print(f"Performance Window: {perf_w}×{perf_h}")
    
    quality_w, quality_h = get_window_size_for_config("quality")
    print(f"Quality Window: {quality_w}×{quality_h}")
