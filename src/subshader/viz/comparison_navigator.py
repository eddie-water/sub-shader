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

        self._create_fig()
        self._init_buttons()
        self._init_plot_comparison()
        self._draw()
        plt.show()

    def _create_fig(self):
        """Create the basic figure with window management"""
        self.fig = plt.figure(figsize=(16, 9), constrained_layout=False)

        # Set Window Title and Maximize
        if self.window_title:
            self.fig.canvas.manager.set_window_title(self.window_title)
        fig_manager = self.fig.canvas.manager
        if hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'showMaximized'):
            fig_manager.window.showMaximized()

    def _init_buttons(self):
        """Setup buttons and key bindings"""
        ax_prev = self.fig.add_axes([0.1, 0.05, 0.06, 0.03])
        ax_next = self.fig.add_axes([0.84, 0.05, 0.06, 0.03])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))

    @abstractmethod
    def _init_plot_comparison(self):
        """Initialize the plot comparison"""
        pass

    @abstractmethod
    def _draw(self, step=0):
        """Draw the plot"""
        pass

    @abstractmethod
    def _step(self, d):
        """Handle stepping"""
        pass

class KernelNavigator(NavigatorBase):
    """
    Plot Navigator for kernel analysis:
      - Cycles through wavelet indices plots each kernel in the time domain (L) 
        and the frequency domain (R)
      - Plots three different time ranges / zoom levels for each kernel
    """

    def __init__(self, np_wavelet, title=None):
        self.np_wavelet = np_wavelet
        super().__init__(title)

    def _init_plot_comparison(self):
        """Initialize figure with 3x2 grid for kernel visualization"""
        # 3x2 Plot Grid: Wavelet Time Series (L), Wavelet Frequency Domain (R)
        self.ax_ts = [self.fig.add_subplot(3, 2, 1 + i*2) for i in range(3)]
        self.ax_fs = [self.fig.add_subplot(3, 2, 2 + i*2) for i in range(3)]

        # Layout
        self.fig.subplots_adjust(bottom=0.15, top=0.92, left=0.06, right=0.98, wspace=0.12, hspace=0.4)

        # Get wavelet kernels
        self.wavelets = self.np_wavelet.wavelets
        self.k_ts = [w.kernel_t for w in self.wavelets]
        self.k_fs = [w.kernel_f for w in self.wavelets]
        self.num_k = len(self.k_ts)
        self.center_freqs = np.asarray(self.np_wavelet.freqs)
        self.sampling_freq = self.np_wavelet.sample_rate  

        # Plot visual characteristics
        REAL_PART_COLOR = 'darkorange'
        IMAG_PART_COLOR = 'mediumslateblue'
        MAG_COLOR = 'black'
        GRID_ALPHA = 0.25
        LINE_WIDTH = 2

        # Ranges for time (s) and frequency (Hz) axes
        self.t_ranges_s = [0.5, 0.05, 0.005]
        self.f_ranges_hz = [(20.0, 200.0), (20.0, 2000.0), (20.0, 20000.0)]

        # Initialize time and frequency domain kernel line lists
        self.k_t_real_lines = []
        self.k_t_imag_lines = []
        self.k_t_mag_lines = []
        self.k_f_lines = []

        # For each time range, plot the kernel time series real + imaginary parts + magnitude 
        for i, (ax, t_range) in enumerate(zip(self.ax_ts, self.t_ranges_s)):
            # Decorate plot with tidy title and labels
            x_lim = t_range/2
            if x_lim >= 0.1:
                time_range = f'{x_lim} s'
            else:
                time_range = f'{x_lim*1000:.1f} ms'
            ax.set_title(f'Time Domain ±{time_range}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=GRID_ALPHA)

            # Create line artists and append to line lists
            (k_t_real_line,) = ax.plot([], [], REAL_PART_COLOR, label='Real', lw=LINE_WIDTH)
            (k_t_imag_line,) = ax.plot([], [], IMAG_PART_COLOR, label='Imag', lw=LINE_WIDTH)
            (k_t_mag_line,)  = ax.plot([], [], MAG_COLOR, label='Mag', lw=LINE_WIDTH)
            self.k_t_real_lines.append(k_t_real_line)
            self.k_t_imag_lines.append(k_t_imag_line)
            self.k_t_mag_lines.append(k_t_mag_line)
            
            # Color legend only on first time series plot
            if i == 0:
                ax.legend(loc='upper right', frameon=False)

        # For each frequency range, plot the kernel frequency series magnitude 
        for ax, (f_lo, f_hi) in zip(self.ax_fs, self.f_ranges_hz):
            # Format frequency range string for tidy title
            if f_hi >= 1000:
                f_range = f'{f_lo:.0f}–{f_hi/1000:.1f}k Hz'
            else:
                f_range = f'{f_lo:.0f}–{f_hi:.0f} Hz'
            ax.set_title(f'Frequency Domain {f_range}')
            ax.set_xlabel('Freq (Hz)')
            ax.set_xscale('log')
            ax.set_ylabel('Magnitude')
            ax.grid(True, which='both', alpha=GRID_ALPHA)

            # Create line artist and append to line list
            (k_f_line,) = ax.plot([], [], MAG_COLOR, lw=LINE_WIDTH)
            self.k_f_lines.append(k_f_line)
    
    def _draw(self, step=0):
        """Draw kernel visualization"""
        i = self.i % self.num_k
        k_t = self.k_ts[i]
        t = np.arange(len(k_t)) / self.sampling_freq
        t = t - t[len(t)//2]

        # Update time domain plots for each range
        for ax, t_range, k_t_real_line, k_t_imag_line, k_t_mag_line in zip(
            self.ax_ts, self.t_ranges_s, 
            self.k_t_real_lines, self.k_t_imag_lines, self.k_t_mag_lines
        ):
            # Plot all signal parts
            k_t_real_line.set_data(t, np.real(k_t))
            k_t_imag_line.set_data(t, np.imag(k_t))
            k_t_mag_line.set_data(t, np.abs(k_t))

            # Set time range
            x_lim = t_range/2
            ax.set_xlim(-x_lim, x_lim)
            ax.relim(); ax.autoscale(axis="y", tight=True)

        # Update frequency domain plots for each range
        k_f = self.k_fs[i]
        k_n = len(k_f)
        f_axis = np.fft.fftfreq(k_n, d=1/self.sampling_freq)

        # Create x axis and compute magnitudes just for "positive" frequencies
        pos_freq_slice = slice(0, k_n//2)
        f_axis = f_axis[pos_freq_slice]
        k_f_mag = np.abs(k_f[pos_freq_slice])

        # Plot the frequency series for each decade
        for ax, (f_lo, f_hi), k_f_line in zip(self.ax_fs, self.f_ranges_hz, self.k_f_lines):
            k_f_line.set_data(f_axis, k_f_mag)
            ax.set_xlim(f_lo, f_hi)
            ax.relim(); ax.autoscale(axis="y", tight=True)

        self.fig.suptitle(f'Wavelet Kernel Visualization - Center Frequency {self.center_freqs[i]:.1f} Hz ({i+1}/{self.num_k})', fontsize=12)
        self.fig.canvas.draw_idle()
    
    def _step(self, d):
        """Handle stepping through kernels"""
        self.i = (self.i + d)
        self._draw()

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
        self.gs = gridspec.GridSpec(nrows=5, ncols=2)
        super().__init__(title)

    def _init_plot_comparison(self):
        """Initialize figure with 1x5 grid for DSP stage visualization"""
        self.ax_t = self.fig.add_subplot(1, 2, 1)
        self.ax_cwt = self.fig.add_subplot(4, 2, 2)
        self.ax_scl = self.fig.add_subplot(4, 2, 4)
        self.ax_coi = self.fig.add_subplot(4, 2, 6)
        self.ax_ds = self.fig.add_subplot(4, 2, 8)

        # Layout
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.15, hspace=0.25)

        # Get audio chunk
        self.current_audio_chunk = self.audio_input.get_chunk()
        self.chunk_i = 0

        # Plot audio time series and decorate plots
        (self.line_audio_t,) = self.ax_t.plot(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_t.set_title("Audio Time Series")
        self.ax_t.set_xlabel("Samples")
        self.ax_t.set_ylabel("Amplitude")
        self.ax_t.margins(x=0, y=0)
        self.ax_t.grid(True, alpha=0.15)

        # Compute and plot each DSP stage and decorate plots
        self.cwt_coefs = self.wavelet.class_specific_cwt(self.current_audio_chunk)
        self.mag_coefs = self.wavelet.compute_mag(self.cwt_coefs)
        self.im_cwt = self.ax_cwt.imshow(self.mag_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_cwt.set_title("Raw CWT")
        self.ax_cwt.set_ylabel("Freq Bin")
        self.ax_cwt.set_xticks([])
        self.ax_cwt.set_xticklabels([])

        self.scl_coefs = self.wavelet.normalize_by_scale(self.cwt_coefs)
        self.mag_coefs = self.wavelet.compute_mag(self.scl_coefs)
        self.im_scl = self.ax_scl.imshow(self.mag_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_scl.set_title("Scale Normalization")
        self.ax_scl.set_ylabel("Freq Bin")
        self.ax_scl.set_xticks([])
        self.ax_scl.set_xticklabels([])

        self.coi_coefs = self.wavelet.extract_reliable_region(self.mag_coefs)
        self.im_coi = self.ax_coi.imshow(self.coi_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_coi.set_title("Cone of Influence")
        self.ax_coi.set_ylabel("Freq Bin")
        self.ax_coi.set_xticks([])
        self.ax_coi.set_xticklabels([])

        self.ds_coefs = self.wavelet.downsample(self.coi_coefs)
        self.im_ds = self.ax_ds.imshow(self.ds_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_ds.set_title("Downsampled")
        self.ax_ds.set_xlabel("Time")
        self.ax_ds.set_ylabel("Freq Bin")

        # Colorbar
        self.fig.colorbar(self.im_ds, ax=[self.ax_cwt, self.ax_scl, self.ax_coi, self.ax_ds], fraction=0.025, pad=0.02)

    def _draw(self, step=0):
        """Draw DSP stage visualization"""
        self.current_audio_chunk = self.audio_input.get_chunk()
        self.chunk_i += 1

        self.line_audio_t.set_data(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_t.set_xlim(0, len(self.current_audio_chunk))
        self.ax_t.relim(); self.ax_t.autoscale(axis="y", tight=True)

        self.cwt_coefs = self.wavelet.class_specific_cwt(self.current_audio_chunk)
        self.mag_coefs = self.wavelet.compute_mag(self.cwt_coefs)
        self.im_cwt.set_data(self.mag_coefs)
        self.ax_cwt.set_xlim(0, self.cwt_coefs.shape[1])
        self.ax_cwt.set_ylim(0, self.cwt_coefs.shape[0])

        self.scl_coefs = self.wavelet.normalize_by_scale(self.cwt_coefs)
        self.mag_coefs = self.wavelet.compute_mag(self.scl_coefs)
        self.im_scl.set_data(self.mag_coefs)
        self.ax_scl.set_xlim(0, self.scl_coefs.shape[1])
        self.ax_scl.set_ylim(0, self.scl_coefs.shape[0])

        self.coi_coefs = self.wavelet.extract_reliable_region(self.mag_coefs)
        self.im_coi.set_data(self.coi_coefs)
        self.ax_coi.set_xlim(0, self.coi_coefs.shape[1])
        self.ax_coi.set_ylim(0, self.coi_coefs.shape[0])

        self.ds_coefs = self.wavelet.downsample(self.coi_coefs)
        self.im_ds.set_data(self.ds_coefs)
        self.ax_ds.set_xlim(0, self.ds_coefs.shape[1])
        self.ax_ds.set_ylim(0, self.ds_coefs.shape[0])

        self.fig.suptitle(f"DSP Stage Visualization - Chunk {self.chunk_i}")
        self.fig.canvas.draw_idle()

    def _step(self, d):
        """Handle stepping through audio chunks"""
        self.i = (self.i + d)
        self._draw()

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

        super().__init__(title, cmap)
    
    def _init_plot_comparison(self):
        """Initialize CWT plot comparison"""
        # 2x2 Plot Grid: Audio Time Series (L), Multiple CWT Implementation Plots (R)
        self.ax_t = self.fig.add_subplot(1, 2, 1)
        self.ax_pywt = self.fig.add_subplot(2, 2, 2)
        self.ax_cpwt = self.fig.add_subplot(2, 2, 4)
        
        # Layout
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.15, hspace=0.25)

        # Get chunk of audio
        self.current_audio_chunk = self.audio_input.get_chunk()
        self.chunk_i = 0

        # Plot audio time series and decorate plots
        (self.line_audio_t,) = self.ax_t.plot(np.arange(len(self.current_audio_chunk)), self.current_audio_chunk)
        self.ax_t.set_title("Audio Time Series")
        self.ax_t.set_xlabel("Samples")
        self.ax_t.set_ylabel("Amplitude")
        self.ax_t.margins(x=0, y=0)
        self.ax_t.grid(True, alpha=0.15)

        # Compute and plot PyWavelet and CuPy CWT time-freq coefs and decorate plots
        pywt_coefs = self.cwt_function(self.py_wavelet, self.current_audio_chunk)
        self.im_py = self.ax_pywt.imshow(pywt_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_pywt.set_title("PyWavelet CWT")
        self.ax_pywt.set_xlabel("Time")
        self.ax_pywt.set_ylabel("Freq Bin")

        cpwt_coefs = self.cwt_function(self.cp_wavelet, self.current_audio_chunk)
        self.im_cp = self.ax_cpwt.imshow(cpwt_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_cpwt.set_title("CuPy CWT")
        self.ax_cpwt.set_xlabel("Time")
        self.ax_cpwt.set_ylabel("Freq Bin")

        # Colorbar
        self.fig.colorbar(self.im_cp, ax=[self.ax_pywt, self.ax_cpwt], fraction=0.025, pad=0.02)
    
    def _draw(self, step=0):
        """Draw CWT visualization"""
        if step > 0:
            self.current_audio_chunk = self.audio_input.get_chunk()
            self.chunk_i += 1

            x = np.arange(len(self.current_audio_chunk))
            self.line_audio_t.set_data(x, self.current_audio_chunk)
            self.ax_t.set_xlim(x[0], x[-1])
            self.ax_t.relim(); self.ax_t.autoscale(axis="y", tight=True)

            pywt_coefs = self.cwt_function(self.py_wavelet, self.current_audio_chunk)
            cpwt_coefs = self.cwt_function(self.cp_wavelet, self.current_audio_chunk)

            self.im_py.set_data(pywt_coefs)
            self.im_cp.set_data(cpwt_coefs)

            self.ax_pywt.set_xlim(0, pywt_coefs.shape[1]); self.ax_pywt.set_ylim(0, pywt_coefs.shape[0])
            self.ax_cpwt.set_xlim(0, cpwt_coefs.shape[1]); self.ax_cpwt.set_ylim(0, cpwt_coefs.shape[0])

            self.fig.suptitle(f"Chunk {self.chunk_i}")
        
        self.fig.canvas.draw_idle()
    
    def _step(self, d):
        """Handle stepping through audio chunks"""
        # Only implement forward stepping by default
        self._draw(step=+1 if d > 0 else -1)
