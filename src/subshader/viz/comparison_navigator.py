# subshader/viz/comparison_navigator.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from abc import ABC, abstractmethod

class NavigatorBase(ABC):
    """
    Abstract base class for plot navigators with figure and button setup
    """

    def __init__(self, title=None, cmap="magma"):
        self.window_title = title
        self.cmap = cmap
        self.i = 0
        self.fig: plt.Figure = None

        self._create_fig()
        self._init_buttons()
        self._init_plots()
        self._update()
        plt.show()

    # Public - Base Setup

    def _create_fig(self):
        """Create figure with window management"""
        self.fig = plt.figure(figsize=(16, 9), constrained_layout=False)

        if self.window_title:
            self.fig.canvas.manager.set_window_title(self.window_title)
        
        fig_manager = self.fig.canvas.manager
        if hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'showMaximized'):
            fig_manager.window.showMaximized()

    def _init_buttons(self):
        """Setup navigation buttons and key bindings"""
        ax_prev = self.fig.add_axes([0.1, 0.05, 0.06, 0.03])
        ax_next = self.fig.add_axes([0.84, 0.05, 0.06, 0.03])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(lambda _: self._on_prev())
        self.btn_next.on_clicked(lambda _: self._on_next())

    # Private - Navigation Callbacks

    def _on_prev(self):
        """Handle previous button click"""
        self.i = (self.i - 1) % self._get_num_items()
        self._update()

    def _on_next(self):
        """Handle next button click"""
        self.i = (self.i + 1) % self._get_num_items()
        self._update()

    # Abstract - Derived Class Interface

    @abstractmethod
    def _init_plots(self):
        """Initialize plots and data structures"""
        pass

    @abstractmethod
    def _update(self):
        """Update plots with current data"""
        pass

    @abstractmethod
    def _get_num_items(self):
        """Return total number of items to navigate through"""
        pass


class KernelNavigator(NavigatorBase):
    """
    Plot Navigator for kernel analysis:
      - Cycles through wavelet indices plots each kernel in the time domain (L) 
        and the frequency domain (R)
      - Plots three different time ranges / zoom levels for each kernel
    """

    def __init__(self, wavelet, title=None):
        self.wavelet = wavelet
        super().__init__(title)

    # Public - Plot Setup

    def _init_plots(self):
        """Initialize figure with 4x2 grid for kernel visualization"""
        self.wavelets = self.wavelet.wavelets

        self.num_kernels = len(self.wavelets)
        self.kernels_t = [w.kernel_t for w in self.wavelets]
        self.kernels_f = [w.kernel_f for w in self.wavelets]

        self.time_supports_n = [w.time_support_n for w in self.wavelets]

        self.sins_t = [w.sinusoid.real for w in self.wavelets]
        self.gaussians_t = [w.gaussian.real for w in self.wavelets]

        self.center_freqs_hz = np.asarray(self.wavelet.freqs)
        self.sampling_freq_hz = self.wavelet.sample_rate

        REAL_PART_COLOR = 'darkorange'
        IMAG_PART_COLOR = 'mediumslateblue'
        MAG_COLOR = 'black'
        GRID_ALPHA = 0.25
        LINE_WIDTH = 2

        self.fig.subplots_adjust(bottom=0.12, top=0.93, left=0.06, right=0.98, wspace=0.15, hspace=0.4)

        # Row 0: All components overlaid (time and frequency)
        self.ax_components_t = self.fig.add_subplot(4, 2, 1)
        (self.sin_t_line,) = self.ax_components_t.plot([], [], REAL_PART_COLOR, label='Sinusoid', lw=LINE_WIDTH)
        (self.gaus_t_line,) = self.ax_components_t.plot([], [], IMAG_PART_COLOR, label='Gaussian', lw=LINE_WIDTH)
        (self.kernel_t_real_line,) = self.ax_components_t.plot([], [], MAG_COLOR, label='Kernel (Real)', lw=LINE_WIDTH)
        self.ax_components_t.set_title('Time Domain Components')
        self.ax_components_t.set_xlabel('Time (s)')
        self.ax_components_t.set_ylabel('Amplitude')
        self.ax_components_t.legend(loc='upper right', frameon=False)
        self.ax_components_t.grid(True, alpha=GRID_ALPHA)

        self.ax_components_f = self.fig.add_subplot(4, 2, 2)
        (self.sin_f_line,) = self.ax_components_f.plot([], [], REAL_PART_COLOR, label='Sinusoid', lw=LINE_WIDTH)
        (self.gaus_f_line,) = self.ax_components_f.plot([], [], IMAG_PART_COLOR, label='Gaussian', lw=LINE_WIDTH)
        (self.kernel_f_line,) = self.ax_components_f.plot([], [], MAG_COLOR, label='Kernel', lw=LINE_WIDTH)
        self.ax_components_f.set_title('Frequency Domain Components')
        self.ax_components_f.set_xlabel('Freq (Hz)')
        self.ax_components_f.set_ylabel('Magnitude')
        self.ax_components_f.legend(loc='upper right', frameon=False)
        self.ax_components_f.grid(True, alpha=GRID_ALPHA)

        # Rows 1-3: Three zoom levels for kernel (time and frequency)
        self.ax_time = [self.fig.add_subplot(4, 2, 3 + i*2) for i in range(3)]
        self.ax_freq = [self.fig.add_subplot(4, 2, 4 + i*2) for i in range(3)]

        self.time_ranges_sec = [0.5, 0.05, 0.005]
        self.freq_ranges_hz = [(20.0, 200.0), (20.0, 2000.0), (20.0, 20000.0)]

        self.kernel_t_real_lines = []
        self.kernel_t_imag_lines = []
        self.kernel_t_mag_lines = []
        self.kernel_f_lines = []

        # Setup time domain plots
        for i, (ax, time_range_sec) in enumerate(zip(self.ax_time, self.time_ranges_sec)):
            half_range_sec = time_range_sec / 2
            time_label = f'{half_range_sec} s' if half_range_sec >= 0.1 else f'{half_range_sec*1000:.1f} ms'
            ax.set_title(f'Time Domain ±{time_label}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=GRID_ALPHA)

            (k_t_real_line,) = ax.plot([], [], REAL_PART_COLOR, label='Real', lw=LINE_WIDTH)
            (k_t_imag_line,) = ax.plot([], [], IMAG_PART_COLOR, label='Imag', lw=LINE_WIDTH)
            (k_t_mag_line,)  = ax.plot([], [], MAG_COLOR, label='Mag', lw=LINE_WIDTH)
            self.kernel_t_real_lines.append(k_t_real_line)
            self.kernel_t_imag_lines.append(k_t_imag_line)
            self.kernel_t_mag_lines.append(k_t_mag_line)
            

            if i == 0:
                ax.legend(loc='upper right', frameon=False)

        # Setup frequency domain plots
        for ax, (freq_lo_hz, freq_hi_hz) in zip(self.ax_freq, self.freq_ranges_hz):
            freq_label = f'{freq_lo_hz:.0f}–{freq_hi_hz/1000:.1f}k Hz' if freq_hi_hz >= 1000 else f'{freq_lo_hz:.0f}–{freq_hi_hz:.0f} Hz'
            ax.set_title(f'Frequency Domain {freq_label}')
            ax.set_xlabel('Freq (Hz)')
            ax.set_xscale('log')
            ax.set_ylabel('Magnitude')
            ax.grid(True, which='both', alpha=GRID_ALPHA)

            (k_f_line,) = ax.plot([], [], MAG_COLOR, lw=LINE_WIDTH)
            self.kernel_f_lines.append(k_f_line)
    
    def _update(self):
        """Update kernel visualization"""
        i = self.i

        '''
        Component Analysis: Sinusoid, Gaussian, and Kernel
        '''
        # Time axis (centered at t=0)
        axis_t = np.arange(self.time_supports_n[i]) / self.sampling_freq_hz
        axis_t = axis_t - axis_t[len(axis_t)//2]

        # Frequency axis (for kernel FFT)
        kernel_f = self.kernels_f[i]
        num_samples_f = len(kernel_f)
        axis_f_kernel = np.fft.fftfreq(num_samples_f, d=1/self.sampling_freq_hz)
        pos_f_slice = slice(0, num_samples_f//2)
        axis_f_kernel_pos = axis_f_kernel[pos_f_slice]
        
        # Center frequency and zoom range (±50%)
        center_f = self.center_freqs_hz[i]
        range_f = (0, 1.5 * center_f)
        i_lo_f = np.searchsorted(axis_f_kernel_pos, range_f[0], side='left')
        i_hi_f = np.searchsorted(axis_f_kernel_pos, range_f[1], side='right')
        axis_f_zoomed = axis_f_kernel_pos[i_lo_f:i_hi_f]

        # Time domain: all three components overlaid
        self.sin_t_line.set_data(axis_t, self.sins_t[i])
        self.gaus_t_line.set_data(axis_t, self.gaussians_t[i])
        kernel_t_real = np.real(self.kernels_t[i])
        self.kernel_t_real_line.set_data(axis_t, kernel_t_real)
        
        self.ax_components_t.set_xlim(axis_t[0], axis_t[-1])
        self.ax_components_t.relim()
        self.ax_components_t.autoscale(axis="y", tight=True)

        # Frequency domain: all three components overlaid (zoomed to ±50% of center)
        sin_f = np.fft.fft(self.wavelets[i].sinusoid, num_samples_f)
        sin_f_mag = (1/num_samples_f)* np.abs(sin_f[pos_f_slice])
        sin_f_mag_zoomed = sin_f_mag[i_lo_f:i_hi_f]
        
        gaus_f = np.fft.fft(self.wavelets[i].gaussian, num_samples_f)
        gaus_f_mag = (1/num_samples_f) * np.abs(gaus_f[pos_f_slice])
        gaus_f_mag_zoomed = gaus_f_mag[i_lo_f:i_hi_f]
        
        kernel_f_mag = np.abs(kernel_f[pos_f_slice])
        kernel_f_mag = (1/num_samples_f) * kernel_f_mag
        kernel_f_mag_zoomed = kernel_f_mag[i_lo_f:i_hi_f]
        
        self.sin_f_line.set_data(axis_f_zoomed, sin_f_mag_zoomed)
        self.gaus_f_line.set_data(axis_f_zoomed, gaus_f_mag_zoomed)
        self.kernel_f_line.set_data(axis_f_zoomed, kernel_f_mag_zoomed)
        
        y_min = 0
        y_max = np.max([np.max(sin_f_mag_zoomed), np.max(gaus_f_mag_zoomed), np.max(kernel_f_mag_zoomed)])
        self.ax_components_f.set_xlim(range_f[0], range_f[1])
        self.ax_components_f.set_ylim(y_min, y_max * 1.05)
        # self.ax_components_f.relim()
        self.ax_components_f.autoscale(axis="y", tight=True)

        '''
        ZoomedWavelet Kernels
        '''
        # Time and Frequency Axes
        zoom_axis_t = np.arange(self.time_supports_n[i]) / self.sampling_freq_hz
        zoom_axis_t = zoom_axis_t - zoom_axis_t[len(zoom_axis_t)//2]

        num_samples_f = len(self.kernels_f[i])
        zoom_axis_f = np.fft.fftfreq(num_samples_f, d=1/self.sampling_freq_hz)

        # Time Domain Plots at different zoom levels
        for ax, time_range_sec, k_t_real_line, k_t_imag_line, k_t_mag_line in zip(
            self.ax_time, self.time_ranges_sec, 
            self.kernel_t_real_lines, self.kernel_t_imag_lines, self.kernel_t_mag_lines
        ):
            kernel_t = self.kernels_t[i]
            k_t_real_line.set_data(zoom_axis_t, np.real(kernel_t))
            k_t_imag_line.set_data(zoom_axis_t, np.imag(kernel_t))
            k_t_mag_line.set_data(zoom_axis_t, np.abs(kernel_t))

            half_range_sec = time_range_sec / 2
            y_min = np.min([
                np.min(np.real(kernel_t)), 
                np.min(np.imag(kernel_t)), 
                np.min(np.abs(kernel_t))])

            y_max = np.max([
                np.max(np.real(kernel_t)), 
                np.max(np.imag(kernel_t)), 
                np.max(np.abs(kernel_t))])

            ax.set_ylim(y_min * 1.1, y_max * 1.1)
            ax.set_xlim(-half_range_sec, half_range_sec)
            # ax.relim()
            ax.autoscale(axis="y", tight=True)


        # Frequency Domain Plots at different zoom levels
        pos_freq_slice = slice(0, num_samples_f//2)
        zoom_axis_f = zoom_axis_f[pos_freq_slice]
        zoom_kernel_f_mag = np.abs(self.kernels_f[i][pos_freq_slice])

        for ax, (freq_lo_hz, freq_hi_hz), k_f_line in zip(self.ax_freq, self.freq_ranges_hz, self.kernel_f_lines):
            k_f_line.set_data(zoom_axis_f, zoom_kernel_f_mag)
            y_max = np.max(zoom_kernel_f_mag)
            ax.set_ylim(0, y_max * 1.05)
            ax.set_xlim(freq_lo_hz, freq_hi_hz)
            # ax.relim()
            ax.autoscale(axis="y", tight=True)

        self.fig.suptitle(f'Wavelet Kernel Visualization - Center Frequency {self.center_freqs_hz[i]:.1f} Hz ({i+1}/{self.num_kernels})', fontsize=12)
        self.fig.canvas.draw_idle()
    
    def _get_num_items(self):
        """Return number of kernels"""
        return self.num_kernels


class DspStageNavigator(NavigatorBase):
    """
    Plot Navigator for DSP stage analysis:
      - Plots every DSP stage of the cwt pipeline
      - Steps through audio chunks and plots the current audio time series (L)
        and each DSP stage (R)
    """
    def __init__(self, audio_input, wavelet, title=None):
        self.audio_input = audio_input
        self.wavelet = wavelet
        self.chunk_i = 0
        super().__init__(title)

    # Public - Plot Setup

    def _init_plots(self):
        """Initialize figure with 1x5 grid for DSP stage visualization"""
        self.ax_audio_t = self.fig.add_subplot(1, 2, 1)
        self.ax_cwt = self.fig.add_subplot(4, 2, 2)
        self.ax_scale_norm = self.fig.add_subplot(4, 2, 4)
        self.ax_coi = self.fig.add_subplot(4, 2, 6)
        self.ax_downsample = self.fig.add_subplot(4, 2, 8)

        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.15, hspace=0.25)

        self.current_audio_chunk = self.audio_input.get_chunk()

        # Audio time series plot
        (self.line_audio_t,) = self.ax_audio_t.plot(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_audio_t.set_title("Audio Time Series")
        self.ax_audio_t.set_xlabel("Samples")
        self.ax_audio_t.set_ylabel("Amplitude")
        self.ax_audio_t.margins(x=0, y=0)
        self.ax_audio_t.grid(True, alpha=0.15)

        # DSP stage plots
        cwt_coefs = self.wavelet.class_specific_cwt(self.current_audio_chunk)
        mag_coefs = self.wavelet.compute_mag(cwt_coefs)
        self.im_cwt = self.ax_cwt.imshow(mag_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_cwt.set_title("Raw CWT")
        self.ax_cwt.set_ylabel("Freq Bin")
        self.ax_cwt.set_xticks([])
        self.ax_cwt.set_xticklabels([])

        scale_norm_coefs = self.wavelet.normalize_by_scale(cwt_coefs)
        mag_coefs = self.wavelet.compute_mag(scale_norm_coefs)
        self.im_scale_norm = self.ax_scale_norm.imshow(mag_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_scale_norm.set_title("Scale Normalization")
        self.ax_scale_norm.set_ylabel("Freq Bin")
        self.ax_scale_norm.set_xticks([])
        self.ax_scale_norm.set_xticklabels([])

        coi_coefs = self.wavelet.discard_unreliable_coefs(mag_coefs)
        self.im_coi = self.ax_coi.imshow(coi_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_coi.set_title("Cone of Influence")
        self.ax_coi.set_ylabel("Freq Bin")
        self.ax_coi.set_xticks([])
        self.ax_coi.set_xticklabels([])

        downsample_coefs = self.wavelet.downsample(coi_coefs)
        self.im_downsample = self.ax_downsample.imshow(downsample_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_downsample.set_title("Downsampled")
        self.ax_downsample.set_xlabel("Time")
        self.ax_downsample.set_ylabel("Freq Bin")

        self.fig.colorbar(self.im_downsample, ax=[self.ax_cwt, self.ax_scale_norm, self.ax_coi, self.ax_downsample], fraction=0.025, pad=0.02)

    def _update(self):
        """Update DSP stage visualization"""
        self.current_audio_chunk = self.audio_input.get_chunk()
        self.chunk_i += 1

        self.line_audio_t.set_data(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_audio_t.set_xlim(0, len(self.current_audio_chunk))
        self.ax_audio_t.relim()
        self.ax_audio_t.autoscale(axis="y", tight=True)

        cwt_coefs = self.wavelet.class_specific_cwt(self.current_audio_chunk)
        mag_coefs = self.wavelet.compute_mag(cwt_coefs)
        self.im_cwt.set_data(mag_coefs)
        self.ax_cwt.set_xlim(0, cwt_coefs.shape[1])
        self.ax_cwt.set_ylim(0, cwt_coefs.shape[0])

        scale_norm_coefs = self.wavelet.normalize_by_scale(cwt_coefs)
        mag_coefs = self.wavelet.compute_mag(scale_norm_coefs)
        self.im_scale_norm.set_data(mag_coefs)
        self.ax_scale_norm.set_xlim(0, scale_norm_coefs.shape[1])
        self.ax_scale_norm.set_ylim(0, scale_norm_coefs.shape[0])

        coi_coefs = self.wavelet.discard_unreliable_coefs(mag_coefs)
        self.im_coi.set_data(coi_coefs)
        self.ax_coi.set_xlim(0, coi_coefs.shape[1])
        self.ax_coi.set_ylim(0, coi_coefs.shape[0])

        downsample_coefs = self.wavelet.downsample(coi_coefs)
        self.im_downsample.set_data(downsample_coefs)
        self.ax_downsample.set_xlim(0, downsample_coefs.shape[1])
        self.ax_downsample.set_ylim(0, downsample_coefs.shape[0])

        self.fig.suptitle(f"DSP Stage Visualization - Chunk {self.chunk_i}")
        self.fig.canvas.draw_idle()

    def _get_num_items(self):
        """Return infinite items for continuous audio stream"""
        return float('inf')


class TransformNavigator(NavigatorBase):
    """
    Plot Navigator for transform analysis:
      - Steps through audio chunks and updates time (L) + two CWTs (R)
      - Forward-only stepping by default
    """
    
    def __init__(self, audio_input, py_wavelet, cp_wavelet, cwt_function, title=None, cmap="magma"):
        self.audio_input = audio_input
        self.py_wavelet = py_wavelet
        self.cp_wavelet = cp_wavelet
        self.cwt_function = cwt_function
        self.chunk_i = 0
        super().__init__(title, cmap)
    
    # Public - Plot Setup

    def _init_plots(self):
        """Initialize CWT comparison plots"""
        self.ax_audio_t = self.fig.add_subplot(1, 2, 1)
        self.ax_pywt = self.fig.add_subplot(2, 2, 2)
        self.ax_cpwt = self.fig.add_subplot(2, 2, 4)
        
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.15, hspace=0.25)

        self.current_audio_chunk = self.audio_input.get_chunk()

        # Audio time series plot
        (self.line_audio_t,) = self.ax_audio_t.plot(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_audio_t.set_title("Audio Time Series")
        self.ax_audio_t.set_xlabel("Samples")
        self.ax_audio_t.set_ylabel("Amplitude")
        self.ax_audio_t.margins(x=0, y=0)
        self.ax_audio_t.grid(True, alpha=0.15)

        # CWT comparison plots
        pywt_coefs = self.cwt_function(self.py_wavelet, self.current_audio_chunk)
        self.im_pywt = self.ax_pywt.imshow(pywt_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_pywt.set_title("PyWavelet CWT")
        self.ax_pywt.set_xlabel("Time")
        self.ax_pywt.set_ylabel("Freq Bin")

        cpwt_coefs = self.cwt_function(self.cp_wavelet, self.current_audio_chunk)
        self.im_cpwt = self.ax_cpwt.imshow(cpwt_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_cpwt.set_title("CuPy CWT")
        self.ax_cpwt.set_xlabel("Time")
        self.ax_cpwt.set_ylabel("Freq Bin")

        self.fig.colorbar(self.im_cpwt, ax=[self.ax_pywt, self.ax_cpwt], fraction=0.025, pad=0.02)
    
    def _update(self):
        """Update CWT comparison visualization"""
        self.current_audio_chunk = self.audio_input.get_chunk()
        self.chunk_i += 1

        sample_indices = np.arange(len(self.current_audio_chunk))
        self.line_audio_t.set_data(sample_indices, self.current_audio_chunk)
        self.ax_audio_t.set_xlim(sample_indices[0], sample_indices[-1])
        self.ax_audio_t.relim()
        self.ax_audio_t.autoscale(axis="y", tight=True)

        pywt_coefs = self.cwt_function(self.py_wavelet, self.current_audio_chunk)
        cpwt_coefs = self.cwt_function(self.cp_wavelet, self.current_audio_chunk)

        self.im_pywt.set_data(pywt_coefs)
        self.im_cpwt.set_data(cpwt_coefs)

        self.ax_pywt.set_xlim(0, pywt_coefs.shape[1])
        self.ax_pywt.set_ylim(0, pywt_coefs.shape[0])
        self.ax_cpwt.set_xlim(0, cpwt_coefs.shape[1])
        self.ax_cpwt.set_ylim(0, cpwt_coefs.shape[0])

        self.fig.suptitle(f"Chunk {self.chunk_i}")
        self.fig.canvas.draw_idle()
    
    def _get_num_items(self):
        """Return infinite items for continuous audio stream"""
        return float('inf')
