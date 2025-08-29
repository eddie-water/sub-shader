from abc import ABC, abstractmethod
import numpy as np
from numpy.fft import fft, ifft
import cupy as cp
from cupyx.scipy import fft as cp_fft
import pywt
from subshader.utils.logging import get_logger
from typing import Optional
from ..config import WaveletConfig

log = get_logger(__name__)

# Math Constants
pi = np.pi

class Wavelet(ABC):
    def __init__(self, sample_rate: int, window_size: int, config: Optional[WaveletConfig] = None):
        """
        Wavelet base class that all other wavelet classes are derived from.
        Uses a list of frequencies that follows the chromatic scale starting at
        A0 to specify which frequencies to look for in the audio data.

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        if config is None:
            config = WaveletConfig()
        self.config = config
        
        if sample_rate != self.config.typical_sampling_freq:
            log.error(f"Invalid sample rate: {sample_rate} Hz (expected {self.config.typical_sampling_freq} Hz)")
            raise ValueError(f"Sampling Rate: {sample_rate},", 
                             f"is not {self.config.typical_sampling_freq} Hz.",
                             f"The CWT may not work as expected.")
        self.sample_rate = sample_rate

        if window_size <= 0:
            log.error(f"Invalid window size: {window_size} (must be > 0)")
            raise ValueError(f"Window Size: {window_size},",
                             f"must be greater than 0.")
        self.window_size = window_size
        
        # Store downsampling target width from config
        self.target_width = self.config.target_width

        # Sampling Parameters
        self.sample_rate = sample_rate
        self.nyquist_freq = (sample_rate / 2.0)
        self.sampling_period = (1.0 / self.sample_rate)

        # Generate list of frequencies in the chromatic scale
        self.freqs = self._generate_chromatic_scale(
            self.config.num_octaves,
            self.config.notes_per_octave,
            self.config.root_note_a0_hz)
        self.num_freqs = len(self.freqs)

        # Resultant Shape of the CWT Data 
        self.result_shape = (self.num_freqs, self.window_size)

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

    def get_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the resultant CWT data.

        Returns:
            np.ndarray.shape: Shape of the computed CWT data
        """
        return self.result_shape
    
    def get_downsampled_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the downsampled CWT data.
            
        Returns:
            np.ndarray.shape: Shape of the downsampled CWT data
        """
        return (self.num_freqs, self.target_width)

    def get_num_freqs(self) -> int:
        """
        Get the number of frequencies in the used in the CWT

        Returns:
            int: Number of frequencies in the CWT
        """
        return self.num_freqs
  
    def compute_cwt(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Performs the Continuous Wavelet Transform (CWT) on raw audio data, 
        normalizes the results, and downsamples to reduce the data transfer 
        size.

        Args:
            audio_data (np.ndarray): raw audio signal data

        Returns:
            np.ndarray: The normalized and downsampled CWT coefficients
        """
        if len(audio_data) != self.window_size:
            log.error(f"Audio data length mismatch: {len(audio_data)} != {self.window_size}")
            raise ValueError(f"Audio data length {len(audio_data)}",
                             f"does not match window size {self.window_size}")

        # Increase precision
        data = audio_data.astype(np.float64)

        cwt_coefs = self.class_specific_cwt(data)
        
        # Downsample the raw CWT coefficients bc there's a lot of data
        downsampled_coefs = self.downsample(cwt_coefs, self.target_width)
        
        # Normalize the results
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
    
    def downsample(self, coefs: np.ndarray, target_width: int = None) -> np.ndarray:
        """
        Downsample CWT coefficients for efficient visualization.
        
        This method reduces the time dimension while preserving frequency resolution
        to make the data suitable for real-time GPU rendering.

        Args:
            coefs (np.ndarray): Normalized CWT coefficients (freq_bins, time_samples)
            target_width (int): Target width for visualization (uses config if None)
            
        Returns:
            np.ndarray: Downsampled coefficients suitable for visualization
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
        
        log.debug(f"Downsampled CWT: {coefs.shape} -> {downsampled.shape} (factor: {downsample_factor})")
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
    def __init__(self, sample_rate, window_size, config: Optional[WaveletConfig] = None):
        """
        The PyWavelet implementation of the CWT

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        super().__init__(sample_rate, window_size, config)

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

class AntsWavelet(Wavelet):
    def __init__(self, sample_rate: int, window_size: int,
                 m_cycles: float = 6.0, fwhm_cycles: float = 3.0, config: Optional[WaveletConfig] = None):
        """
        ANTS-style CWT with true scale-dependent time support.

        Args:
            sample_rate (int): audio sample rate in Hz
            window_size (int): analysis window length in samples
            m_cycles (float): number of carrier cycles per wavelet
            fwhm_cycles (float): Gaussian FWHM width in cycles
            config (WaveletConfig, optional): Configuration object with wavelet parameters
        """
        super().__init__(sample_rate, window_size, config)

        self.m_cycles = m_cycles
        self.fwhm_cycles = fwhm_cycles
        self.data_n = self.window_size

        # Store per-frequency kernels (variable length)
        self.wavelet_kernels: list[np.ndarray] = []
        self.half_kern_ns: list[int] = []

        for f in self.freqs:
            # Duration in seconds for m_cycles cycles at frequency f
            dur_s = m_cycles / f
            m_samples = int(np.round(dur_s * self.sample_rate))

            # Time vector centered at 0
            t = (np.arange(m_samples) / self.sample_rate) - (dur_s / 2)

            # Gaussian FWHM in seconds
            fwhm_s = fwhm_cycles / f

            # Complex Morlet wavelet
            cmw_k = np.exp(1j * 2 * np.pi * f * t) \
                    * np.exp(-4 * np.log(2) * (t ** 2) / fwhm_s ** 2)

            # Scale normalization
            cmw_k *= np.sqrt(f)

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


class NumpyWavelet(AntsWavelet):
    def class_specific_cwt(self, data) -> np.ndarray:
        """
        Perform CWT using variable-length wavelets, CPU version.
        Returns: (num_freqs, window_size) matrix.
        """
        tf = np.zeros((self.num_freqs, self.window_size), dtype=np.float32)
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


class CupyWavelet(AntsWavelet):
    def __init__(self, sample_rate, window_size,
                 m_cycles=6.0, fwhm_cycles=3.0, config: Optional[WaveletConfig] = None):
        super().__init__(sample_rate, window_size, m_cycles, fwhm_cycles, config)
        log.info(f"CPU→GPU: Uploading {len(self.wavelet_kernels)} wavelets to GPU")

        # Convert each kernel individually to CuPy
        self.wavelet_kernels = [cp.asarray(w) for w in self.wavelet_kernels]
        self.num_wavelets = len(self.wavelet_kernels)

        # Allocate GPU time-frequency matrix
        self.tf_gpu = cp.zeros((self.num_freqs, self.window_size), dtype=cp.float32)

    def class_specific_cwt(self, data) -> np.ndarray:
        """
        Perform CWT using variable-length wavelets, GPU version.
        Returns: (num_freqs, window_size) matrix.
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
    
class CuWavelet(CupyWavelet):
    """
    This just kind of renames the class becuase it sounds ~cool~
    """
    pass