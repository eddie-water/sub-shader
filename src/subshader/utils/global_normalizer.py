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
class GlobalNormalizer:
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
    
    def update(self, frame: Union[np.ndarray, 'cp.ndarray'], mask: Optional[Union[np.ndarray, 'cp.ndarray']] = None) -> float:
        """
        Update the global factor with a new frame.

        Args:
            frame (Union[np.ndarray, 'cp.ndarray']): The frame to update the global factor with.
            mask (Optional[Union[np.ndarray, 'cp.ndarray']]): The mask to apply to the frame.

        Returns:
            float: The updated global factor.
        """
        # TODO 36: What is isinstance even checking here?
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
        # TODO 36: I don't understnad why this is asrray vs numpy
        if mask is not None:
            if is_cupy and not isinstance(mask, cp.ndarray):
                mask = cp.asarray(mask)
            elif not is_cupy and isinstance(mask, cp.ndarray):
                mask = cp.asnumpy(mask)
            valid_data = frame[mask]
        else:
            valid_data = frame.flatten()
        
        if valid_data.size == 0:
            return self.global_factor
        
        frame_stat = float(xp.percentile(valid_data, self.percentile))
        
        # TODO 36: Feel like this is a little much
        self.global_factor = (1.0 - self.decay_rate) * self.global_factor
        self.global_factor = max(self.global_factor, self.floor_value)
        self.global_factor = max(self.global_factor, frame_stat)
        
        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            self.is_ready = True
        
        return self.global_factor
    
    def normalize(self, frame: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize the frame with the global factor.

        Args:
            frame (Union[np.ndarray, 'cp.ndarray']): The frame to normalize.

        Returns:
            Union[np.ndarray, 'cp.ndarray']: The normalized frame.
        """
        
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
        if self.global_factor <= 0:
            return xp.zeros_like(frame)
        
        normalized = frame / self.global_factor
        normalized = xp.clip(normalized, 0.0, 1.0)
        
        if self.log_mapping:
            normalized = xp.log1p(normalized) / xp.log1p(1.0)
        
        return normalized
