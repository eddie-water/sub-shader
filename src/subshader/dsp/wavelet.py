"""
Wavelet Transform Module for SubShader.

This module provides GPU-accelerated Continuous Wavelet Transform (CWT) 
implementation for real-time audio analysis:
 - Computes time-frequency representations using the Morlet wavelet
 - Utilizes CuPy for GPU acceleration and performance optimization
 - Supports chromatic scale frequency mapping for musical visualization
 - Includes global normalization for consistent amplitude scaling
"""

# =============================================================================
# IMPORTS
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.fft import fft, ifft
import cupy as cp
from cupyx.scipy import fft as cp_fft
import pywt

from subshader.utils.global_normalizer import GlobalNormalizer
from subshader.utils.logging import get_logger

from ..config import WaveletConfig

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PI = np.pi

# =============================================================================
# WAVELET CLASSES
# =============================================================================

class Wavelet(ABC):
    def __init__(self, sample_rate: int, input_size: int, config: Optional[WaveletConfig] = None):
        """
        Wavelet base class that all other wavelet classes are derived from.
        Uses a list of frequencies that follows the chromatic scale starting at
        A0 to specify which frequencies to look for in the input audio data.

        Args:
            sample_rate (int): The rate the input data was sampled in Hz
            input_size (int): The length of the input data
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        if config is None:
            config = WaveletConfig()
        self.config = config
        
        # TODO: Confirm COI is the reliable portion of the CWT result (accounting for cone of influence wich might remove edge effects in the plot)
        # Cone of influence parameters - extract reliable center region
        self.coi_edge_percent = 0.15  # Remove 15% from each edge (30% total)
        
        # Runtime validation - sample rate must match expected frequency
        if sample_rate != self.config.typical_sampling_freq:
            log.error(f"Invalid sample rate: {sample_rate} Hz (expected {self.config.typical_sampling_freq} Hz)")
            raise ValueError(f"Sampling Rate: {sample_rate} Hz is not {self.config.typical_sampling_freq} Hz. "
                             f"The CWT may not work as expected.")
        
        self.sample_rate = sample_rate
        # TODO input_size -> input_size
        self.input_size = input_size
        
        # Store downsampling target width from config
        self.target_width = self.config.target_width
        
        # Initialize global normalizer if enabled
        self.global_normalizer = None
        if self.config.global_norm.enabled:
            self.global_normalizer = GlobalNormalizer(
                percentile=self.config.global_norm.percentile,
                decay_rate=self.config.global_norm.decay_rate,
                floor_value=self.config.global_norm.floor_value,
                warmup_frames=self.config.global_norm.warmup_frames,
                log_mapping=self.config.global_norm.log_mapping
            )

        # Sampling Parameters
        self.sample_rate = sample_rate
        self.nyquist_freq = (sample_rate / 2.0)
        self.sampling_period = (1.0 / self.sample_rate)
        
        # Calculate cone of influence boundaries for reliable region extraction
        self.coi_start_idx = int(self.input_size * self.coi_edge_percent)
        self.coi_end_idx = int(self.input_size * (1.0 - self.coi_edge_percent))
        self.coi_reliable_length = self.coi_end_idx - self.coi_start_idx

        # Generate list of frequencies in the chromatic scale
        self.freqs = self._generate_chromatic_scale(
            self.config.num_octaves,
            self.config.notes_per_octave,
            self.config.root_note_a0_hz)
        self.num_freqs = len(self.freqs)

        # Input and output dimensions
        self.input_shape = (self.num_freqs, self.input_size)
        self.output_shape = (self.num_freqs, self.target_width)
        
        log.info(f"Cone of influence: removing edges {self.coi_edge_percent:.1%} each side")
        log.info(f"Reliable region: samples {self.coi_start_idx}-{self.coi_end_idx} ({self.coi_reliable_length}/{self.input_size})")

    def _generate_chromatic_scale(self, root_note: float, num_octaves: int, notes_per_octave: int = 12) -> list[float]:
        """
        Generates a list of frequencies that follow the exponential step size of 
        the chromatic scale.

        Args:
            root_note (float): The root note of the chromatic scale
            num_octaves (int): The number of octaves to generate
            notes_per_octave (int): The number of notes per octave

        Returns:
            list[float]: A list of frequencies in the chromatic scale
        """
        # Frequencies double every octave
        scale_factor = 2 ** (1 / notes_per_octave)
        i = np.arange(0, notes_per_octave * num_octaves, 1)
        freqs = root_note * (scale_factor ** i)

        # Discard frequencies that are unmeasurable
        return freqs[freqs < self.nyquist_freq]

    def get_input_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the input data for CWT processing.

        Returns:
            np.ndarray.shape: Shape of the input data
        """
        return self.input_shape
    
    def get_output_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the output data (downsampled CWT coefficients).
            
        Returns:
            np.ndarray.shape: Shape of the output data
        """
        return self.output_shape
    
    # Preserve external interface for backward compatibility
    def get_downsampled_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the output data (downsampled CWT coefficients).
        
        Note: This method is kept for backward compatibility.
        Use get_output_shape() for new code.
            
        Returns:
            np.ndarray.shape: Shape of the output data
        """
        return self.get_output_shape()

    def get_num_freqs(self) -> int:
        """
        Get the number of frequencies in the used in the CWT

        Returns:
            int: Number of frequencies in the CWT
        """
        return self.num_freqs
  
    def compute_cwt(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the Continuous Wavelet Transform (CWT) on input audio data, 
        normalizes the results, and downsamples to produce output coefficients.

        Args:
            input_data (np.ndarray): Raw input audio signal data

        Returns:
            np.ndarray: The normalized and downsampled output coefficients
        """
        if len(input_data) != self.input_size:
            log.error(f"Input data length mismatch: {len(input_data)} != {self.input_size}")
            raise ValueError(f"Input data length {len(input_data)} "
                             f"does not match expected input data size {self.input_size}")

        # Increase precision
        data = input_data.astype(np.float64)

        cwt_coefs = self.class_specific_cwt(data)
        
        # Extract reliable region (avoid cone of influence edge artifacts)
        reliable_coefs = self.extract_reliable_region(cwt_coefs)
        
        # Downsample the reliable CWT coefficients to produce output data
        downsampled_coefs = self.downsample(reliable_coefs, self.target_width)
        
        # Apply normalization (global or legacy)
        if self.global_normalizer is not None:
            # Use global normalization
            return self.apply_global_normalization(downsampled_coefs)
        else:
            # Use legacy per-frame normalization
            return self.normalize_coefs(downsampled_coefs)

    @abstractmethod
    def class_specific_cwt(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the subclass-specific implementation of the CWT

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The CWT coefficients 
        """
        pass

    def extract_reliable_region(self, cwt_coefs: np.ndarray) -> np.ndarray:
        """
        Extract the reliable center region from CWT coefficients to avoid edge artifacts.
        
        The cone of influence in wavelet transforms creates unreliable results near
        the edges where the wavelet doesn't have sufficient data support. This method
        extracts only the center region where results are reliable.
        
        Args:
            cwt_coefs (np.ndarray): Full CWT coefficients (freq_bins, time_samples)
            
        Returns:
            np.ndarray: Reliable center region (freq_bins, reliable_time_samples)
        """
        # Extract the reliable center region
        reliable_region = cwt_coefs[:, self.coi_start_idx:self.coi_end_idx]
        
        log.debug(f"Cone of influence: extracted reliable region {cwt_coefs.shape} -> {reliable_region.shape}")
        return reliable_region

    def normalize_coefs(self, raw_coefs: np.ndarray) -> np.ndarray:
        """
        Normalize CWT coefficients for plotting using a fixed dB range.
        
        This avoids per-frame min/max scaling (which causes flicker and grain)
        by mapping magnitudes into a consistent dynamic range.

        Args:
            raw_coefs (np.ndarray): Raw CWT coefficients (complex or real).

        Returns:
            np.ndarray: Normalized magnitudes in [0, 1].
        """
        # Magnitude of the CWT coefficients, add epsilon to avoid log(0)
        mag = np.abs(raw_coefs) + self.config.epsilon

        # Convert to decibels
        db_vals = 20.0 * np.log10(mag)

        # Fixed display range (dB)
        db_floor = self.config.db_floor
        db_ceil = self.config.db_ceil

        # Clamp to dB range
        db_vals = np.clip(db_vals, db_floor, db_ceil)

        # Map to [0, 1]
        norm_vals = (db_vals - db_floor) / (db_ceil - db_floor)

        return norm_vals.astype(np.float32)
    
    def apply_global_normalization(self, raw_coefs: np.ndarray) -> np.ndarray:
        """
        Apply global normalization using the GlobalNormalizer.
        
        This method computes magnitude, updates the global normalization factor,
        and returns normalized values in [0, 1] range.
        
        Args:
            raw_coefs (np.ndarray): Raw CWT coefficients (complex or real).
            
        Returns:
            np.ndarray: Globally normalized magnitudes in [0, 1].
        """
        # Compute magnitude of the CWT coefficients
        mag = np.abs(raw_coefs)
        
        # Create a valid data mask (exclude very small values that might be noise)
        valid_mask = mag > self.config.epsilon
        
        # Update global normalization factor
        self.global_normalizer.update(mag, mask=valid_mask)
        
        # Apply global normalization
        normalized = self.global_normalizer.normalize(mag)
        
        # Ensure output type consistency
        return normalized.astype(self.config.output_dtype)
    
    def downsample(self, coefs: np.ndarray, target_width: int = None) -> np.ndarray:
        """
        Downsample CWT coefficients to produce final output data.
        
        This method reduces the time dimension while preserving frequency resolution
        to make the data suitable for real-time GPU rendering.

        Args:
            coefs (np.ndarray): Input CWT coefficients (freq_bins, time_samples)
            target_width (int): Target output width (uses config if None)
            
        Returns:
            np.ndarray: Output data suitable for visualization
        """
        # Use config target width if not specified
        if target_width is None:
            target_width = self.config.target_width
            
        freq_bins, time_samples = coefs.shape
        
        # If already at target size or smaller, return as-is
        if time_samples <= target_width:
            return coefs
        
        # Calculate downsampling factor
        downsample_factor = max(1, time_samples // target_width)

        # Simple downsampling strategy - take every Nth sample
        # This preserves the most recent data (right side of the buffer)
        downsampled = coefs[:, ::downsample_factor]
        
        # If still too wide, crop to target size
        if downsampled.shape[1] > target_width:
            downsampled = downsampled[:, -target_width:]  # Keep most recent data
        
        log.debug(f"Downsampled to output data: {coefs.shape} -> {downsampled.shape} (factor: {downsample_factor})")
        return downsampled
    
    @abstractmethod
    def cleanup(self):
        """
        Clean up any resources used by the wavelet implementation.
        
        This method should be overridden by subclasses that allocate
        significant resources (especially GPU memory).
        """
        pass

class PyWavelet(Wavelet):
    def __init__(self, sample_rate, input_size, config: Optional[WaveletConfig] = None):
        """
        The PyWavelet implementation of the CWT

        Args:
            sample_rate (int): The rate the input data was sampled in Hz
            input_size (int): The length of the input data
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        super().__init__(sample_rate, input_size, config)

        # Wavelet info TODO ISSUE-36 why 1.5-1.0?
        self.wavelet_name = "cmor1.5-1.0"

        # Scale array used to specify wavelet dilation amounts during CWT
        f_norm = (self.freqs / self.sample_rate)
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    def class_specific_cwt(self, data: np.ndarray) -> np.ndarray:
        """
        Produces the normalized CWT coefficients using PyWavelets. 

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The scale-based normalized CWT coefficients 
        """
        coefs_raw, _ = pywt.cwt(data = data,
                                scales = self.scales,
                                wavelet = self.wavelet_name,
                                sampling_period = self.sampling_period)
    
        """
        Scale-Based Normalization 
        
        This account for the energy bias that occurs at higher scales which 
        PyWavelets does not do internally it seems. The wavelet equation is 
        this: 
        
            Psi_s(t) = 1/sqrt(s) * Psi(t-T/s)
        
        Where Psi is the wavelet at a scale s, and localized in time by T. We 
        need the '1/sqrt(s)' term to account for the energy bias that occurs at
        higher scales for 'Psi(t-T/s)'. The 's' term, since it's in the 
        denominator, stretches the wavelet horizontally when 's' is large, and 
        compresses it horizontally when 's' is small. Higher-scale wavelets 
        will have more area under their curves, and will seem to contribute 
        more energy to the inner product that is going on inside the CWT. To 
        account for this, we normalize the coefficients by dividing it by
        the square root of the scale. Not sure why PyWavelets doesn't just do
        this under the hood.
        """
        # TODO ISSUE-36 Why aren't I doing this to all the wavelet subclasses
        # in the parent class?
        coefs_scaled = coefs_raw / np.sqrt(self.scales[:, None])

        return coefs_scaled
    
    def cleanup(self):
        """
        Clean up any resources used by PyWavelet.
        
        PyWavelet doesn't allocate significant resources, so this is a no-op.
        """
        pass

# TODO 36 - Use namesapce array namespace pattern 'xp = cp elif np' to 
# consolidate the code. Maybe if the ANTS wavelet is the parent class that 
# checks a bool, and if it's true, use cp, else use np and then CuWavelet sets 
# it true, NumPyWavelet sets its false. Except notice that the fft and ifft 
# function don't necessarily follow that pattern... 
class AntsWavelet(Wavelet):
    def __init__(self, sample_rate: int, input_size: int, m_cycles: float = 6.0, 
        fwhm_cycles: float = 3.0, config: Optional[WaveletConfig] = None):
        """
        ANTS-style CWT with true scale-dependent time support.

        Args:
            sample_rate (int): Input audio sample rate in Hz
            input_size (int): Input data length in samples
            m_cycles (float): number of carrier cycles per wavelet
            fwhm_cycles (float): Gaussian FWHM width in cycles
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        super().__init__(sample_rate, input_size, config)

        self.m_cycles = m_cycles
        self.fwhm_cycles = fwhm_cycles
        self.data_n = self.input_size

        # Store per-frequency kernels (variable length)
        self.wavelet_kernels: list[np.ndarray] = []
        self.half_kern_ns: list[int] = []

        # TODO 36 - Redo this section - standardize my naming conventions
        # dur_s vs time_s vs support_t vs time_support_s vs 
        # cmw_k vs cmw_t vs cmw_kernel_t - I want the _% to indicate the domain where _t is time domain and _f is freq but I also like _x for freq
        # TODO 36 - create a member variable for the wavelet kernels in the t domain and f domain
        for f in self.freqs:
            # Duration in seconds for m_cycles cycles at frequency f
            dur_s = m_cycles / f
            m_samples = int(np.round(dur_s * self.sample_rate))

            # Time vector centered at 0
            t = (np.arange(m_samples) / self.sample_rate) - (dur_s / 2)

            # Gaussian FWHM in seconds
            fwhm_s = fwhm_cycles / f

            # Complex Morlet wavelet
            cmw_k = np.exp(1j * 2 * PI * f * t) * np.exp(-4 * np.log(2) * (t ** 2) / fwhm_s ** 2)

            # Scale normalization
            # TODO 36 - Why do I normalize this here? What about the PyWavelet implenetation that does it in the normalization step?
            cmw_k *= np.sqrt(f) # f = 1/s? is that true???

            kern_n = len(cmw_k)
            conv_n = self.data_n + kern_n - 1
            half_kern_n = kern_n // 2
            self.half_kern_ns.append(half_kern_n)

            # FFT of kernel, normalized
            cmw_x = fft(cmw_k, conv_n)
            cmw_x = cmw_x / np.max(np.abs(cmw_x))

            # Store as numpy array (variable length OK)
            self.wavelet_kernels.append(np.asarray(cmw_x, dtype=np.complex64))

        self.num_wavelets = len(self.wavelet_kernels)

    def get_wavelet_kernels(self) -> list[np.ndarray]:
        """
        Access the wavelet kernels.
        """
        if hasattr(self, 'wavelet_kernels'):
            return self.wavelet_kernels
        else:
            return None


class NumPyWavelet(AntsWavelet):
    def class_specific_cwt(self, data) -> np.ndarray:
        """
        Perform CWT using variable-length wavelets, CPU version.
        Returns: (num_freqs, input_size) matrix.
        """
        tf = np.zeros((self.num_freqs, self.input_size), dtype=np.float32)
        for i, cmw_x in enumerate(self.wavelet_kernels):
            conv_n = cmw_x.shape[0]
            data_x = fft(data, conv_n)
            conv = ifft(data_x * cmw_x)
            conv = np.abs(conv) ** 2
            half_kern_n = self.half_kern_ns[i]
            conv_valid = conv[half_kern_n:half_kern_n + self.data_n]
            tf[i, :] = conv_valid
        return tf

    def cleanup(self):
        pass


class CuPyWavelet(AntsWavelet):
    def __init__(self, sample_rate, input_size,
                 m_cycles=6.0, fwhm_cycles=3.0, config: Optional[WaveletConfig] = None):
        super().__init__(sample_rate, input_size, m_cycles, fwhm_cycles, config)
        log.info(f"CPU→GPU: Uploading {len(self.wavelet_kernels)} wavelets to GPU")

        # Convert each kernel individually to CuPy
        self.wavelet_kernels = [cp.asarray(w) for w in self.wavelet_kernels]
        self.num_wavelets = len(self.wavelet_kernels)

        # Allocate GPU time-frequency matrix
        self.tf_gpu = cp.zeros((self.num_freqs, self.input_size), dtype=cp.float32)

    def class_specific_cwt(self, data) -> np.ndarray:
        """
        Perform CWT using variable-length wavelets, GPU version.
        Returns: (num_freqs, input_size) matrix.
        """
        for i, cmw_x in enumerate(self.wavelet_kernels):
            conv_n = cmw_x.shape[0]
            data_x = cp_fft.fftn(cp.asarray(data, dtype=cp.complex64), conv_n)
            conv = cp_fft.ifft(data_x * cmw_x)
            conv = cp.abs(conv) ** 2
            half_kern_n = self.half_kern_ns[i]
            conv_valid = conv[half_kern_n:half_kern_n + self.data_n]
            self.tf_gpu[i, :] = conv_valid

        return cp.asnumpy(self.tf_gpu)
    
    def apply_global_normalization(self, raw_coefs: np.ndarray) -> np.ndarray:
        """
        Apply global normalization for CuPy-based wavelets.
        
        This method handles the GPU-to-CPU transfer needed for the GlobalNormalizer
        while maintaining efficiency.
        
        Args:
            raw_coefs (np.ndarray): Raw CWT coefficients (already converted from GPU).
            
        Returns:
            np.ndarray: Globally normalized magnitudes in [0, 1].
        """
        if self.global_normalizer is None:
            # Fallback to parent implementation
            return super().apply_global_normalization(raw_coefs)
        
        # Since raw_coefs is already numpy (converted in class_specific_cwt),
        # we can use the parent implementation directly
        return super().apply_global_normalization(raw_coefs)

    def cleanup(self):
        try:
            if hasattr(self, 'tf_gpu'):
                del self.tf_gpu
                self.tf_gpu = None
            if hasattr(self, 'wavelet_kernels'):
                del self.wavelet_kernels
                self.wavelet_kernels = None
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"Warning: Error during GPU cleanup: {e}")
    
class CuWavelet(CuPyWavelet):
    """
    This just kind of renames the class becuase it sounds ~cool~
    """
    pass