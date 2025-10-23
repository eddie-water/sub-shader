"""Global Normalization Utility for SubShader"""

import numpy as np
from typing import Optional, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# TODO 36 Rename this. The result of the CWT is "normalized" this more like a "max tracker" as a reference point for the shader color map
class PlotNormalizer:
    """Tracks global normalization factor across frames for stable scaling."""
    
    def __init__(
        self,
        percentile: float = 99.0,
        decay_rate: float = 0.001,
        floor_value: float = 1e-8,
        warmup_frames: int = 10,
        log_mapping: bool = False
    ):
        self.percentile = percentile
        self.decay_rate = decay_rate
        self.floor_value = floor_value
        self.warmup_frames = warmup_frames
        self.log_mapping = log_mapping
        
        self.global_factor = 0.0
        self.frame_count = 0
        self.is_ready = False

    def process(self, frame: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Process the frame and return the normalized frame.

        Args:
            frame (Union[np.ndarray, 'cp.ndarray']): The frame to process.

        Returns:
            Union[np.ndarray, 'cp.ndarray']: The normalized frame.
        """
        self.global_factor = self.update(frame)
        return self.apply_normalization(frame, self.global_factor)
    
    def update(self, frame: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Update the global factor with a new frame.

        Args:
            frame (Union[np.ndarray, 'cp.ndarray']): The frame to update the global factor with.

        Returns:
            float: The updated global factor.
        """
        flat_data = frame.flatten()
        frame_stat = float(np.percentile(flat_data, self.percentile))

        self.global_factor = (1.0 - self.decay_rate) * self.global_factor
        self.global_factor = max(self.global_factor, self.floor_value)
        self.global_factor = max(self.global_factor, frame_stat)

        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            self.is_ready = True

        return self.global_factor

    def apply_normalization(self, data: Union[np.ndarray, 'cp.ndarray'], norm_factor: float) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize the data with the normalization factor.

        Args:
            data (Union[np.ndarray, 'cp.ndarray']): The data to normalize.
            norm_factor (float): The normalization factor to apply to the frame.
        Returns:
            Union[np.ndarray, 'cp.ndarray']: The normalized data.
        """
        is_cupy = CUPY_AVAILABLE and isinstance(data, cp.ndarray)
        xp = cp if is_cupy else np

        if norm_factor <= 0:
            return xp.zeros_like(data)

        normalized = data / norm_factor
        normalized = xp.clip(normalized, 0.0, 1.0)

        if self.log_mapping:
            normalized = xp.log1p(normalized) / xp.log1p(1.0)

        return normalized
