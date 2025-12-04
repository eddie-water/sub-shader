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

    def _on_prev(self):
        """Handle previous button click"""
        self.i = (self.i - 1) % self._get_num_items()
        self._update()

    def _on_next(self):
        """Handle next button click"""
        self.i = (self.i + 1) % self._get_num_items()
        self._update()

    @abstractmethod
    def _init_plots(self):
        """Initialize plots and data structures"""
        pass

    @abstractmethod
    def _update(self):
        """Update plots with current data"""
        pass

    def _get_num_items(self):
        """Return total number of items to navigate through"""
        pass

class AudioNavigator(NavigatorBase):
    """
    Plot Navigator for audio analysis:
      - Left: Full audio with chunk highlight box
      - Right: 4 stacked plots showing individual chunks
    """
    NUM_CHUNK_PLOTS = 4
    CHUNK_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    def __init__(self, audio_input, title=None):
        self.audio_input = audio_input
        super().__init__(title)
        
    def _init_plots(self):
        """Initialize audio time series plot with chunk subplots"""
        # Store chunk parameters
        self.chunk_size = self.audio_input.get_chunk_size()
        self.hop_size = self.audio_input.hop_size
        
        # Load entire audio once
        self.entire_audio = self.audio_input.get_entire_audio()
        self.total_samples = len(self.entire_audio)
        
        # Create grid: left side full audio, right side 4 chunk plots
        self.gs = gridspec.GridSpec(self.NUM_CHUNK_PLOTS, 2, figure=self.fig, width_ratios=[2, 1])
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.93, wspace=0.15, hspace=0.3)
        
        # Left: Full audio plot (spans all rows)
        self.ax_full = self.fig.add_subplot(self.gs[:, 0])
        self.ax_full.set_title("Full Audio")
        self.ax_full.set_xlabel("Samples")
        self.ax_full.set_ylabel("Amplitude")
        self.ax_full.grid(True, alpha=0.15)
        
        # Plot entire audio
        (self.line_full,) = self.ax_full.plot(np.arange(self.total_samples), self.entire_audio, 
                                               color='steelblue', linewidth=0.5)
        self.ax_full.set_xlim(0, self.total_samples)
        self.ax_full.margins(y=0.1)
        
        # Chunk highlight boxes (one for current, faded ones for previous)
        self.chunk_spans = []
        
        # Right: 4 chunk plots
        self.ax_chunks = []
        self.line_chunks = []
        self.chunk_data = [None] * self.NUM_CHUNK_PLOTS  # Store which chunk index is in each plot
        
        for idx in range(self.NUM_CHUNK_PLOTS):
            ax = self.fig.add_subplot(self.gs[idx, 1])
            ax.set_ylabel("Amp")
            ax.grid(True, alpha=0.15)
            if idx == self.NUM_CHUNK_PLOTS - 1:
                ax.set_xlabel("Samples")
            (line,) = ax.plot([], [], color=self.CHUNK_COLORS[idx], linewidth=0.8)
            self.ax_chunks.append(ax)
            self.line_chunks.append(line)
        
    def _update(self):
        """Update plots - fill chunk plots one by one"""
        # Determine which plot slot to fill (0-3)
        plot_idx = self.i % self.NUM_CHUNK_PLOTS
        
        # Get chunk data
        chunk_start = self.i * self.hop_size
        chunk_end = chunk_start + self.chunk_size
        
        # Handle case where chunk extends beyond audio
        if chunk_end > self.total_samples:
            chunk_end = self.total_samples
        
        chunk_audio = self.entire_audio[chunk_start:chunk_end]
        
        # Update the chunk plot
        x_data = np.arange(len(chunk_audio))
        self.line_chunks[plot_idx].set_data(x_data, chunk_audio)
        self.ax_chunks[plot_idx].set_xlim(0, len(chunk_audio))
        self.ax_chunks[plot_idx].set_ylim(np.min(chunk_audio) * 1.1, np.max(chunk_audio) * 1.1)
        self.ax_chunks[plot_idx].set_title(f"Chunk {self.i + 1} (samples {chunk_start}-{chunk_end})")
        self.chunk_data[plot_idx] = self.i
        
        # Clear old highlight boxes
        for span in self.chunk_spans:
            span.remove()
        self.chunk_spans.clear()
        
        # Draw highlight boxes for all visible chunks (faded for old, bright for current)
        for idx in range(self.NUM_CHUNK_PLOTS):
            if self.chunk_data[idx] is not None:
                chunk_i = self.chunk_data[idx]
                c_start = chunk_i * self.hop_size
                c_end = c_start + self.chunk_size
                
                # Current chunk is bright, others are faded
                is_current = (idx == plot_idx)
                alpha = 0.4 if is_current else 0.15
                color = self.CHUNK_COLORS[idx]
                
                span = self.ax_full.axvspan(c_start, c_end, alpha=alpha, color=color)
                self.chunk_spans.append(span)
        
        self.fig.suptitle(f"Audio Analysis - Step {self.i + 1}/{self._get_num_items()}")
        self.fig.canvas.draw_idle()

    def _get_num_items(self):
        """Return number of chunks that fit in the audio"""
        return max(1, (self.total_samples - self.chunk_size) // self.hop_size + 1)

class KernelNavigator(NavigatorBase):
    """
    Plot Navigator for kernel analysis:
      - Cycles through wavelet indices plots each kernel in the time domain (L) 
        and the frequency domain (R)
      - Plots three different time ranges / zoom levels for each kernel
    """
    SINUSOID_COLOR = 'black'
    PERIOD_COLOR = 'black'
    GAUSSIAN_COLOR = 'mediumslateblue'
    WAVELET_COLOR = 'darkorange'
    FWHM_COLOR = 'red'
    LINE_WIDTH = 2
    MARKER_ALPHA = 0.5
    MARKER_WIDTH = 3
    GRID_ALPHA = 0.25

    def __init__(self, wavelet, title=None):
        self.wavelet = wavelet
        self.freq_axis_mode = 'zoomed'  # 'zoomed', 'positive', 'nyquist', or 'log_positive'
        super().__init__(title)

    def _init_buttons(self):
        """Setup navigation buttons and frequency axis toggle"""
        super()._init_buttons()
        
        # Add frequency axis mode toggle button
        ax_toggle = self.fig.add_axes([0.45, 0.05, 0.10, 0.03])
        self.btn_toggle = Button(ax_toggle, "Freq: Zoomed")
        self.btn_toggle.on_clicked(lambda _: self._on_toggle_freq_axis())

    def _on_toggle_freq_axis(self):
        """Cycle through frequency axis modes: zoomed -> positive -> nyquist -> log_positive -> zoomed"""
        if self.freq_axis_mode == 'zoomed':
            self.freq_axis_mode = 'positive'
            self.btn_toggle.label.set_text("Freq: 20-20k")
        elif self.freq_axis_mode == 'positive':
            self.freq_axis_mode = 'nyquist'
            self.btn_toggle.label.set_text("Freq: Nyquist")
        elif self.freq_axis_mode == 'nyquist':
            self.freq_axis_mode = 'log_positive'
            self.btn_toggle.label.set_text("Freq: Log")
        else:  # 'log_positive'
            self.freq_axis_mode = 'zoomed'
            self.btn_toggle.label.set_text("Freq: Zoomed")
        self._update()

    def _init_plots(self):
        """Initialize figure with 4x2 grid for kernel visualization"""
        self.wavelets = self.wavelet.wavelets
        self.sample_rate = self.wavelet.sample_rate

        self.num_kernels = len(self.wavelets)
        self.kernels_t = [w.kernel_t for w in self.wavelets]
        self.kernels_f = [w.kernel_f for w in self.wavelets]

        self.time_supports_n = [w.time_support_n for w in self.wavelets]

        self.sins_t = [w.sin_t.real for w in self.wavelets]
        self.gaussians_t = [w.gauss_t.real for w in self.wavelets]

        fwhm_supports_s = [w.gauss.fwhm_support_s for w in self.wavelets]
        self.fwhm_supports_n = [int(np.round(fwhm_support_s * self.sample_rate)) for fwhm_support_s in fwhm_supports_s]
        self.fwhm_supports_t = [fwhm_n / self.sample_rate for fwhm_n in self.fwhm_supports_n]
        self.center_freqs_hz = np.asarray(self.wavelet.freqs)

        self.fig.subplots_adjust(bottom=0.12, top=0.93, left=0.06, right=0.98, wspace=0.15, hspace=0.4)
        self.gs = gridspec.GridSpec(3, 7, figure=self.fig)

        # Row 0: Sinusoid Component
        self.ax_sin_t = self.fig.add_subplot(self.gs[0, 1:3])
        self.ax_sin_t.grid(True, alpha=self.GRID_ALPHA)
        self.ax_sin_t.set_title('Time Domain Sinusoid Component')
        self.ax_sin_t.set_ylabel('Amplitude')

        (self.line_sin_t,) = self.ax_sin_t.plot([], [], self.SINUSOID_COLOR, label='Sinusoid', lw=self.LINE_WIDTH)
        self.ax_sin_t.legend(loc='upper right', frameon=False)
        self.sin_period_vlines = []

        self.ax_sin_f = self.fig.add_subplot(self.gs[0, 4:6])
        self.ax_sin_f.grid(True, alpha=self.GRID_ALPHA)
        self.ax_sin_f.set_title('Frequency Domain Sinusoid Component')

        (self.line_sin_f,) = self.ax_sin_f.plot([], [], self.SINUSOID_COLOR, label='Sinusoid', lw=self.LINE_WIDTH)
        self.ax_sin_f.legend(loc='upper right', frameon=False)
        self.sin_peak_vlines = []

        # Row 1: Gaussian Component
        self.ax_gauss_t = self.fig.add_subplot(self.gs[1, 1:3])
        self.ax_gauss_t.grid(True, alpha=self.GRID_ALPHA)
        self.ax_gauss_t.set_title('Time Domain Gaussian Component')
        self.ax_gauss_t.set_ylabel('Amplitude')
 
        (self.line_gauss_t,) = self.ax_gauss_t.plot([], [], self.GAUSSIAN_COLOR, label='Gaussian', lw=self.LINE_WIDTH)
        (self.line_fwhm_t,) = self.ax_gauss_t.plot([], [], self.FWHM_COLOR, label='FWHM', lw=self.MARKER_WIDTH, linestyle=':', alpha=self.MARKER_ALPHA)
        self.ax_gauss_t.legend(loc='upper right', frameon=False)
        self.gaus_fwhm_vlines = []

        self.ax_gauss_f = self.fig.add_subplot(self.gs[1, 4:6])
        self.ax_gauss_f.grid(True, alpha=self.GRID_ALPHA)
        self.ax_gauss_f.set_title('Frequency Domain Gaussian Component')

        (self.line_gauss_f,) = self.ax_gauss_f.plot([], [], self.GAUSSIAN_COLOR, label='Gaussian', lw=self.LINE_WIDTH)
        self.ax_gauss_f.legend(loc='upper right', frameon=False)

        # Row 2: Resulting Wavelet Kernel 
        self.ax_kernel_t = self.fig.add_subplot(self.gs[2, 1:3])
        self.ax_kernel_t.grid(True, alpha=self.GRID_ALPHA)
        self.ax_kernel_t.set_title('Time Domain Wavelet Kernel')
        self.ax_kernel_t.set_xlabel('Time (s)')
        self.ax_kernel_t.set_ylabel('Amplitude')

        (self.kernel_sin_t_line,) = self.ax_kernel_t.plot([], [], self.SINUSOID_COLOR, label='Sinusoid', lw=self.LINE_WIDTH, alpha=self.MARKER_ALPHA)
        (self.kernel_gaus_t_line,) = self.ax_kernel_t.plot([], [], self.GAUSSIAN_COLOR, label='Gaussian', lw=self.LINE_WIDTH, alpha=self.MARKER_ALPHA)
        (self.kernel_t_real_line,) = self.ax_kernel_t.plot([], [], self.WAVELET_COLOR, label='Kernel', lw=self.LINE_WIDTH)
        self.ax_kernel_t.legend(loc='upper right', frameon=False)

        self.ax_kernel_f = self.fig.add_subplot(self.gs[2, 4:6])
        self.ax_kernel_f.grid(True, alpha=self.GRID_ALPHA)
        self.ax_kernel_f.set_title('Frequency Domain Wavelet Kernel')
        self.ax_kernel_f.set_xlabel('Frequency (Hz)')

        (self.kernel_sin_f_line,) = self.ax_kernel_f.plot([], [], self.SINUSOID_COLOR, label='Sinusoid', lw=self.LINE_WIDTH, alpha=self.MARKER_ALPHA)
        (self.kernel_gaus_f_line,) = self.ax_kernel_f.plot([], [], self.GAUSSIAN_COLOR, label='Gaussian', lw=self.LINE_WIDTH, alpha=self.MARKER_ALPHA)
        (self.kernel_f_line,) = self.ax_kernel_f.plot([], [], self.WAVELET_COLOR, label='Kernel', lw=self.LINE_WIDTH)
        self.ax_kernel_f.legend(loc='upper right', frameon=False)
        self.kernel_peak_vlines = []
    
    def _update(self):
        """Update kernel visualization"""
        i = self.i

        '''
        Component Analysis: Sinusoid, Gaussian, and Kernel
        '''
        # Time axis (centered at t=0)
        axis_t = np.arange(self.time_supports_n[i]) / self.sample_rate
        axis_t = axis_t - axis_t[len(axis_t)//2]

        # Frequency axis (for kernel FFT) - full spectrum including negative frequencies
        kernel_f = self.kernels_f[i]
        num_samples_f = len(kernel_f)
        axis_f_kernel = np.fft.fftfreq(num_samples_f, d=1/self.sample_rate)
        axis_f_kernel = np.fft.fftshift(axis_f_kernel)
        
        # Frequency range based on mode
        center_f = self.center_freqs_hz[i]
        nyquist_f = self.sample_rate / 2
        
        if self.freq_axis_mode == 'zoomed':
            # Centered/zoomed around center frequency, capped at Nyquist limits
            freq_width = 1.5 * center_f
            freq_width = min(freq_width, nyquist_f)
            range_f = (-freq_width, freq_width)
        elif self.freq_axis_mode == 'positive':
            # Positive frequencies only (20 Hz to Nyquist)
            range_f = (20, nyquist_f)
        elif self.freq_axis_mode == 'nyquist':
            # Full Nyquist range (negative to positive)
            range_f = (-nyquist_f, nyquist_f)
        else:  # 'log_positive'
            # Positive frequencies only for log scale (20 Hz to Nyquist)
            range_f = (20, nyquist_f)
        
        i_lo_f = np.searchsorted(axis_f_kernel, range_f[0], side='left')
        i_hi_f = np.searchsorted(axis_f_kernel, range_f[1], side='right')
        axis_f_zoomed = axis_f_kernel[i_lo_f:i_hi_f]

        # Row 0: Sinusoid Component Time Domain
        y_data_min = np.min(self.sins_t[i])
        y_data_max = np.max(self.sins_t[i])
        y_range = y_data_max - y_data_min
        pad = 0.1
        y_min = y_data_min - pad * y_range
        y_max = y_data_max + pad * y_range

        self.line_sin_t.set_data(axis_t, self.sins_t[i])
        self.ax_sin_t.set_ylim(y_min, y_max)
        self.ax_sin_t.set_xlim(axis_t[0], axis_t[-1])
        
        # Vertical Period Lines
        for line in self.sin_period_vlines:
            line.remove()
        self.sin_period_vlines.clear()
        
        period_sec = 1.0 / center_f
        t_start = axis_t[0]
        t_end = axis_t[-1]
        
        # Draw period lines across entire sinusoid width
        num_periods = int(np.ceil((t_end - t_start) / period_sec))
        first_line_t = np.ceil(t_start / period_sec) * period_sec
        
        for j in range(num_periods + 1):
            line_t = first_line_t + j * period_sec
            if t_start <= line_t <= t_end:
                vline = self.ax_sin_t.axvline(line_t, color=self.PERIOD_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle='--')
                self.sin_period_vlines.append(vline)

        # Draw two red lines at 1.5 periods to the left and right of the center of axis_t
        mid_t = 0  # axis_t is centered at t=0
        offset = 1.5 * period_sec
        left_line_t = mid_t - offset
        right_line_t = mid_t + offset
        vline_left = self.ax_sin_t.axvline(left_line_t, color=self.FWHM_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle=':')
        vline_right = self.ax_sin_t.axvline(right_line_t, color=self.FWHM_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle=':')
        self.sin_period_vlines.extend([vline_left, vline_right])
        
        # Set x-ticks to range limits, zero, and red FWHM line positions
        xtick_positions = sorted([axis_t[0], left_line_t, 0, right_line_t, axis_t[-1]])
        self.ax_sin_t.set_xticks(xtick_positions)
        self.ax_sin_t.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))
        
        # Set y-ticks: data min, max, and 0 if in range
        ytick_positions = [y_data_min, y_data_max]
        if y_data_min < 0 < y_data_max:
            ytick_positions.insert(1, 0)
        self.ax_sin_t.set_yticks(ytick_positions)
        self.ax_sin_t.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

        # Row 0: Sinusoid Frequency Domain
        sin_f = np.fft.fft(self.sins_t[i], num_samples_f)
        sin_f = np.fft.fftshift(sin_f)
        sin_f_mag = (1/num_samples_f) * np.abs(sin_f)
        sin_f_mag_zoomed = sin_f_mag[i_lo_f:i_hi_f]

        y_data_min = 0
        y_data_max = np.max(sin_f_mag_zoomed)
        y_range = y_data_max - y_data_min
        pad = 0.1
        y_min = y_data_min
        y_max = y_data_max + pad * y_range

        self.line_sin_f.set_data(axis_f_zoomed, sin_f_mag_zoomed)
        self.ax_sin_f.set_xscale('log' if self.freq_axis_mode == 'log_positive' else 'linear')
        self.ax_sin_f.set_xlim(range_f[0], range_f[1])
        self.ax_sin_f.set_ylim(y_min, y_max)
        
        # Clear previous peak lines and draw new ones at frequency peaks
        for line in self.sin_peak_vlines:
            line.remove()
        self.sin_peak_vlines.clear()
        
        # Draw lines at known sinusoid frequency peaks (±center_f)
        peak_freqs = []
        if -center_f >= range_f[0] and -center_f <= range_f[1]:
            vline = self.ax_sin_f.axvline(-center_f, color=self.WAVELET_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle='--')
            self.sin_peak_vlines.append(vline)
            peak_freqs.append(-center_f)
        if center_f >= range_f[0] and center_f <= range_f[1]:
            vline = self.ax_sin_f.axvline(center_f, color=self.WAVELET_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle='--')
            self.sin_peak_vlines.append(vline)
            peak_freqs.append(center_f)
        
        # Set x-ticks: include peak frequencies in both modes
        if self.freq_axis_mode == 'log_positive':
            # In log mode: decade boundaries and positive peak frequencies
            xtick_positions = []
            # Add decade boundaries within range
            for decade in [20, 100, 1000, 10000, 20000]:
                if range_f[0] <= decade <= range_f[1]:
                    xtick_positions.append(decade)
            # Add positive peak frequencies
            xtick_positions.extend([f for f in peak_freqs if f > 0 and range_f[0] <= f <= range_f[1]])
            self.ax_sin_f.set_xticks(sorted(set(xtick_positions)))
        else:
            # In linear mode: range limits, zero, and all peak frequencies
            xtick_positions = [range_f[0], range_f[1]]
            if range_f[0] < 0 < range_f[1]:
                xtick_positions.append(0)
            xtick_positions.extend(peak_freqs)
            self.ax_sin_f.set_xticks(sorted(set(xtick_positions)))
            self.ax_sin_f.ticklabel_format(axis='x', style='plain', useOffset=False)
        
        # Set y-ticks: data min and max
        ytick_positions = [y_data_min, y_data_max]
        self.ax_sin_f.set_yticks(ytick_positions)
        self.ax_sin_f.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

        # Row 1: Gaussian Component Time Domain
        y_data_min = np.min(self.gaussians_t[i])
        y_data_max = np.max(self.gaussians_t[i])
        y_range = y_data_max - y_data_min
        pad = 0.1
        y_min = y_data_min - pad * y_range
        y_max = y_data_max + pad * y_range

        self.line_gauss_t.set_data(axis_t, self.gaussians_t[i])
        
        # Plot FWHM as horizontal line at y=0.5
        self.line_fwhm_t.set_data(axis_t, 0.5 * np.ones_like(axis_t))

        # Clear previous FWHM vertical lines and recalculate new ones
        for line in self.gaus_fwhm_vlines:
            line.remove()
        self.gaus_fwhm_vlines.clear()

        fwhm_half_width_t = self.fwhm_supports_t[i] / 2
        fwhm_t_left = -fwhm_half_width_t
        fwhm_t_right = fwhm_half_width_t

        vline_left = self.ax_gauss_t.axvline(fwhm_t_left, color=self.FWHM_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle=':')
        vline_right = self.ax_gauss_t.axvline(fwhm_t_right, color=self.FWHM_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle=':')
        self.gaus_fwhm_vlines.extend([vline_left, vline_right])

        self.ax_gauss_t.set_ylim(y_min, y_max)
        self.ax_gauss_t.set_xlim(axis_t[0], axis_t[-1])
        
        # Set y-ticks at key gaussian values: 0, FWHM (0.5), and peak (1.0)
        ytick_positions = [0.0, 0.5, 1.0]
        if y_data_min < 0:
            ytick_positions.insert(0, y_data_min)
        self.ax_gauss_t.set_yticks(ytick_positions)
        self.ax_gauss_t.ticklabel_format(axis='y', style='plain', useOffset=False)
        
        # Set x-ticks to range limits, zero, and FWHM boundary positions
        xtick_positions = sorted([axis_t[0], fwhm_t_left, 0, fwhm_t_right, axis_t[-1]])
        self.ax_gauss_t.set_xticks(xtick_positions)
        self.ax_gauss_t.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))

        # Row 1: Gaussian Frequency Domain
        gaus_f = np.fft.fft(self.gaussians_t[i], num_samples_f)
        gaus_f = np.fft.fftshift(gaus_f)
        gaus_f_mag = (1 / num_samples_f) * np.abs(gaus_f)
        gaus_f_mag_zoomed = gaus_f_mag[i_lo_f:i_hi_f]

        y_data_min = np.min(gaus_f_mag_zoomed)
        y_data_max = np.max(gaus_f_mag_zoomed)
        y_range = y_data_max - y_data_min
        pad = 0.1
        y_min = y_data_min - pad * y_range
        y_max = y_data_max + pad * y_range

        self.line_gauss_f.set_data(axis_f_zoomed, gaus_f_mag_zoomed)
        self.ax_gauss_f.set_xscale('log' if self.freq_axis_mode == 'log_positive' else 'linear')
        self.ax_gauss_f.set_xlim(range_f[0], range_f[1])
        self.ax_gauss_f.set_ylim(y_min, y_max)
        
        # Set x-ticks: decade boundaries for log mode, range limits for linear
        if self.freq_axis_mode == 'log_positive':
            # In log mode: decade boundaries within range
            xtick_positions = []
            for decade in [20, 100, 1000, 10000, 20000]:
                if range_f[0] <= decade <= range_f[1]:
                    xtick_positions.append(decade)
            self.ax_gauss_f.set_xticks(sorted(xtick_positions))
        else:
            # In linear mode: range limits and zero
            xtick_positions = [range_f[0], range_f[1]]
            if range_f[0] < 0 < range_f[1]:
                xtick_positions.append(0)
            self.ax_gauss_f.set_xticks(sorted(xtick_positions))
            self.ax_gauss_f.ticklabel_format(axis='x', style='plain', useOffset=False)
        
        # Set y-ticks: data min and max
        ytick_positions = [y_data_min, y_data_max]
        if y_data_min < 0 < y_data_max:
            ytick_positions.insert(1, 0)
        self.ax_gauss_f.set_yticks(ytick_positions)
        self.ax_gauss_f.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

        # Row 2: Resulting Wavelet Kernel Time Domain
        y_data_min = np.min(np.real(self.kernels_t[i]))
        y_data_max = np.max(np.real(self.kernels_t[i]))
        y_range = y_data_max - y_data_min
        pad = 0.1
        y_min = y_data_min - pad * y_range
        y_max = y_data_max + pad * y_range

        self.kernel_sin_t_line.set_data(axis_t, self.sins_t[i])
        self.kernel_gaus_t_line.set_data(axis_t, self.gaussians_t[i])
        self.kernel_t_real_line.set_data(axis_t, np.real(self.kernels_t[i]))
        self.ax_kernel_t.set_ylim(y_min, y_max)
        self.ax_kernel_t.set_xlim(axis_t[0], axis_t[-1])
        
        # Set x-ticks to range limits and zero
        self.ax_kernel_t.set_xticks([axis_t[0], 0, axis_t[-1]])
        self.ax_kernel_t.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))
        
        # Set y-ticks: data min, max, and 0 if in range
        ytick_positions = [y_data_min, y_data_max]
        if y_data_min < 0 < y_data_max:
            ytick_positions.insert(1, 0)
        self.ax_kernel_t.set_yticks(ytick_positions)
        self.ax_kernel_t.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

        # Row 2: Resulting Wavelet Kernel Frequency Domain
        kernel_f_shifted = np.fft.fftshift(kernel_f)
        kernel_f_mag = (1/num_samples_f) * np.abs(kernel_f_shifted)
        kernel_f_mag_zoomed = kernel_f_mag[i_lo_f:i_hi_f]

        self.kernel_sin_f_line.set_data(axis_f_zoomed, sin_f_mag_zoomed)
        self.kernel_gaus_f_line.set_data(axis_f_zoomed, gaus_f_mag_zoomed)
        self.kernel_f_line.set_data(axis_f_zoomed, kernel_f_mag_zoomed)

        y_data_min = 0
        y_data_max = np.max([np.max(sin_f_mag_zoomed), np.max(gaus_f_mag_zoomed), np.max(kernel_f_mag_zoomed)])
        pad = 0.05
        y_min = y_data_min
        y_max = y_data_max * (1 + pad)
        
        self.ax_kernel_f.set_xscale('log' if self.freq_axis_mode == 'log_positive' else 'linear')
        self.ax_kernel_f.set_xlim(range_f[0], range_f[1])
        self.ax_kernel_f.set_ylim(y_min, y_max)
        
        # Clear previous peak lines and draw new ones at kernel frequency peaks
        for line in self.kernel_peak_vlines:
            line.remove()
        self.kernel_peak_vlines.clear()
        
        # Draw lines at known kernel frequency peaks (±center_f)
        peak_freqs = []
        if -center_f >= range_f[0] and -center_f <= range_f[1]:
            vline = self.ax_kernel_f.axvline(-center_f, color=self.WAVELET_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle='--')
            self.kernel_peak_vlines.append(vline)
            peak_freqs.append(-center_f)
        if center_f >= range_f[0] and center_f <= range_f[1]:
            vline = self.ax_kernel_f.axvline(center_f, color=self.WAVELET_COLOR, alpha=self.MARKER_ALPHA, linewidth=self.MARKER_WIDTH, linestyle='--')
            self.kernel_peak_vlines.append(vline)
            peak_freqs.append(center_f)
        
        # Set x-ticks: include peak frequencies in both modes
        if self.freq_axis_mode == 'log_positive':
            # In log mode: decade boundaries and positive peak frequencies
            xtick_positions = []
            # Add decade boundaries within range
            for decade in [20, 100, 1000, 10000, 20000]:
                if range_f[0] <= decade <= range_f[1]:
                    xtick_positions.append(decade)
            # Add positive peak frequencies
            xtick_positions.extend([f for f in peak_freqs if f > 0 and range_f[0] <= f <= range_f[1]])
            self.ax_kernel_f.set_xticks(sorted(set(xtick_positions)))
        else:
            # In linear mode: range limits, zero, and all peak frequencies
            xtick_positions = [range_f[0], range_f[1]]
            if range_f[0] < 0 < range_f[1]:
                xtick_positions.append(0)
            xtick_positions.extend(peak_freqs)
            self.ax_kernel_f.set_xticks(sorted(set(xtick_positions)))
            self.ax_kernel_f.ticklabel_format(axis='x', style='plain', useOffset=False)
        
        # Set y-ticks: data min and max
        ytick_positions = [y_data_min, y_data_max]
        self.ax_kernel_f.set_yticks(ytick_positions)
        self.ax_kernel_f.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

        self.fig.suptitle(f'Wavelet Components - Center Frequency {self.center_freqs_hz[i]:.1f} Hz ({i+1}/{self.num_kernels})', fontsize=12)
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
