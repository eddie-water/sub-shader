"""Global Normalization Utility for SubShader"""

import numpy as np
from typing import Optional, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


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
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
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
        
        self.global_factor = (1.0 - self.decay_rate) * self.global_factor
        self.global_factor = max(self.global_factor, self.floor_value)
        self.global_factor = max(self.global_factor, frame_stat)
        
        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            self.is_ready = True
        
        return self.global_factor
    
    def normalize(self, frame: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        is_cupy = CUPY_AVAILABLE and isinstance(frame, cp.ndarray)
        xp = cp if is_cupy else np
        
        if self.global_factor <= 0:
            return xp.zeros_like(frame)
        
        normalized = frame / self.global_factor
        normalized = xp.clip(normalized, 0.0, 1.0)
        
        if self.log_mapping:
            normalized = xp.log1p(normalized) / xp.log1p(1.0)
        
        return normalized
