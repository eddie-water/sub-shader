"""DSP module — all time-frequency analysis backends."""

from .dsp import DSP
from .cwt import CWT, CpuCWT, GpuCWT
from .pywavelet import PywaveletCWT
from .stft import STFT

# DEPRECATED aliases — remove after all callers migrated (Plan 08-05)
NpWavelet = CpuCWT
CuWavelet = GpuCWT
NumPyWavelet = CpuCWT
CuPyWavelet = GpuCWT

__all__ = [
    "DSP",
    "CWT", "CpuCWT", "GpuCWT",
    "PywaveletCWT",
    "STFT",
    # Deprecated aliases
    "NpWavelet", "CuWavelet", "NumPyWavelet", "CuPyWavelet",
]
