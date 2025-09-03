"""
Global Normalization Utility for SubShader

This module provides a GlobalNormalizer class that tracks normalization factors
across frames to maintain consistent scaling and prevent flickering in the
visualization. It uses robust statistics (99th percentile) and exponential
decay to adapt to changing audio dynamics while maintaining stability.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from .logging import get_logger

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

log = get_logger(__name__)


class GlobalNormalizer:
    """
    Global normalization utility that tracks a global normalization factor across frames.
    
    This class maintains a global factor G that adapts to the dynamic range of the input
    data while providing stable normalization. It uses robust statistics (configurable
    percentile) instead of raw max values to avoid outlier sensitivity.
    
    Key features:
    - Robust per-frame statistics using configurable percentiles
    - Exponential decay to adapt to changing dynamics
    - Warm-up period to ensure stability
    - Support for both NumPy and CuPy arrays
    - State export/import for persistence across runs
    - Optional log-mapping for perceptual enhancement
    """
    
    def __init__(
        self,
        percentile: float = 99.0,
        decay_rate: float = 0.001,
        floor_value: float = 1e-8,
        warmup_frames: int = 10,
        log_mapping: bool = False
    ):
        """
        Initialize the GlobalNormalizer.
        
        Args:
            percentile (float): Percentile to use for robust statistics (default: 99.0)
            decay_rate (float): Exponential decay rate per frame (default: 0.001)
            floor_value (float): Minimum value for the global factor (default: 1e-8)
            warmup_frames (int): Number of frames before considering G "ready" (default: 10)
            log_mapping (bool): Whether to apply log1p mapping after normalization (default: False)
        """
        # Configuration parameters
        self.percentile = percentile
        self.decay_rate = decay_rate
        self.floor_value = floor_value
        self.warmup_frames = warmup_frames
        self.log_mapping = log_mapping
        
        # State variables
        self.global_factor = 0.0  # Global normalization factor G
        self.frame_count = 0      # Number of frames processed
        self.is_ready = False     # Whether normalization is ready (past warmup)
        
        # Statistics tracking
        self.recent_statistics = []  # Recent per-frame statistics for debugging
        self.max_recent_stats = 100  # Maximum number of recent stats to keep
        
        log.info(f"GlobalNormalizer initialized: percentile={percentile}, decay={decay_rate}, "
                f"floor={floor_value}, warmup={warmup_frames}, log_mapping={log_mapping}")
    
    def update(self, frame: Union[np.ndarray, 'cp.ndarray'], mask: Optional[Union[np.ndarray, 'cp.ndarray']] = None) -> float:
        """
        Update the global normalization factor with a new frame of data.
        
        Args:
            frame (np.ndarray or cp.ndarray): Input frame data (magnitude values)
            mask (np.ndarray or cp.ndarray, optional): Valid data mask
            
        Returns:
            float: Current global normalization factor
        """
        # Determine if we're working with CuPy arrays
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
        # Apply mask if provided
        if mask is not None:
            if is_cupy and not isinstance(mask, cp.ndarray):
                mask = cp.asarray(mask)
            elif not is_cupy and isinstance(mask, cp.ndarray):
                mask = cp.asnumpy(mask)
            
            valid_data = frame[mask]
        else:
            valid_data = frame.flatten()
        
        # Skip empty frames
        if valid_data.size == 0:
            log.warning("Empty frame received, skipping update")
            return self.global_factor
        
        # Compute robust per-frame statistic (percentile)
        frame_stat = float(xp.percentile(valid_data, self.percentile))
        
        # Track recent statistics for debugging
        self.recent_statistics.append(frame_stat)
        if len(self.recent_statistics) > self.max_recent_stats:
            self.recent_statistics.pop(0)
        
        # Apply exponential decay
        self.global_factor = (1.0 - self.decay_rate) * self.global_factor
        
        # Ensure floor value
        self.global_factor = max(self.global_factor, self.floor_value)
        
        # Update with new maximum
        self.global_factor = max(self.global_factor, frame_stat)
        
        # Update frame count and readiness
        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            self.is_ready = True
        
        log.debug(f"Frame {self.frame_count}: stat={frame_stat:.6f}, G={self.global_factor:.6f}, "
                 f"ready={self.is_ready}")
        
        return self.global_factor
    
    def normalize(self, frame: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize a frame using the current global factor.
        
        Args:
            frame (np.ndarray or cp.ndarray): Input frame to normalize
            
        Returns:
            np.ndarray or cp.ndarray: Normalized frame clipped to [0, 1]
        """
        # Determine if we're working with CuPy arrays
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
        if self.global_factor <= 0:
            log.warning("Global factor is zero or negative, returning zeros")
            return xp.zeros_like(frame)
        
        # Normalize by dividing by global factor
        normalized = frame / self.global_factor
        
        # Clip to [0, 1] range
        normalized = xp.clip(normalized, 0.0, 1.0)
        
        # Apply optional log mapping for perceptual enhancement
        if self.log_mapping:
            normalized = xp.log1p(normalized) / xp.log1p(1.0)  # Normalize log1p to [0, 1]
        
        return normalized
    
    def is_normalization_ready(self) -> bool:
        """
        Check if the normalizer has processed enough frames to be considered ready.
        
        Returns:
            bool: True if past warmup period, False otherwise
        """
        return self.is_ready
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current normalization statistics for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Dictionary containing current statistics
        """
        recent_mean = np.mean(self.recent_statistics) if self.recent_statistics else 0.0
        recent_std = np.std(self.recent_statistics) if len(self.recent_statistics) > 1 else 0.0
        
        return {
            'global_factor': self.global_factor,
            'frame_count': self.frame_count,
            'is_ready': self.is_ready,
            'recent_stat_mean': recent_mean,
            'recent_stat_std': recent_std,
            'recent_stat_count': len(self.recent_statistics),
            'config': {
                'percentile': self.percentile,
                'decay_rate': self.decay_rate,
                'floor_value': self.floor_value,
                'warmup_frames': self.warmup_frames,
                'log_mapping': self.log_mapping
            }
        }
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export the current normalizer state for persistence.
        
        Returns:
            Dict[str, Any]: State dictionary that can be saved and later imported
        """
        return {
            'global_factor': self.global_factor,
            'frame_count': self.frame_count,
            'is_ready': self.is_ready,
            'recent_statistics': self.recent_statistics.copy(),
            'config': {
                'percentile': self.percentile,
                'decay_rate': self.decay_rate,
                'floor_value': self.floor_value,
                'warmup_frames': self.warmup_frames,
                'log_mapping': self.log_mapping
            }
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import a previously exported normalizer state.
        
        Args:
            state (Dict[str, Any]): State dictionary from export_state()
        """
        # Restore state variables
        self.global_factor = state.get('global_factor', 0.0)
        self.frame_count = state.get('frame_count', 0)
        self.is_ready = state.get('is_ready', False)
        self.recent_statistics = state.get('recent_statistics', []).copy()
        
        # Optionally update configuration (be careful about this)
        config = state.get('config', {})
        if config:
            log.info(f"Importing state with config: {config}")
            # Only update if explicitly requested or if current config is default
            
        log.info(f"Imported normalizer state: G={self.global_factor:.6f}, "
                f"frames={self.frame_count}, ready={self.is_ready}")
    
    def reset(self) -> None:
        """
        Reset the normalizer to its initial state.
        """
        self.global_factor = 0.0
        self.frame_count = 0
        self.is_ready = False
        self.recent_statistics.clear()
        
        log.info("GlobalNormalizer reset to initial state")
    
    def __repr__(self) -> str:
        """String representation of the normalizer."""
        return (f"GlobalNormalizer(G={self.global_factor:.6f}, frames={self.frame_count}, "
                f"ready={self.is_ready}, percentile={self.percentile})")
