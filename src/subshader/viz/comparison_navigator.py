# subshader/viz/comparison_navigator.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Rectangle, FancyArrowPatch
from abc import ABC, abstractmethod

from subshader.viz.plotter import CircularFrameBuffer, AudioFrameBuffer
from subshader.config import ColorNormalizationConfig

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
        plt.show(block=True)

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
    Plot Navigator for audio overlap analysis:
      - Plot 1: Global view of original audio (~8 chunks)
      - Plot 2: Even-indexed chunks (lane A) showing staggered positions
      - Plot 3: Odd-indexed chunks (lane B) showing staggered positions  
      - Plot 4: Composite where each new chunk overwrites the overlap region
    """
    VISIBLE_CHUNKS = 6  # Number of chunks visible in the window
    AUDIO_COLOR = '#1A1A1A'  # Near-black for audio waveforms
    # EVEN_COLOR = '#6B7FDB'   # Even chunks
    # ODD_COLOR = '#FC8961'    # Odd chunks
    EVEN_COLOR = 'orangwe'   # Even chunks
    ODD_COLOR = '#FC8961'    # Odd chunks
    
    def __init__(self, audio_input, title=None):
        self.audio_input = audio_input
        super().__init__(title)
        
    def _init_plots(self):
        """Initialize 4-plot layout for overlap visualization"""
        # Store chunk parameters
        self.chunk_size = self.audio_input.get_chunk_size()
        self.hop_size = self.audio_input.hop_size
        self.overlap = self.chunk_size - self.hop_size
        self.sample_rate = self.audio_input.sample_rate
        
        # Create time formatter for x-axis (converts samples to seconds)
        self.time_formatter = FuncFormatter(lambda x, pos: f'{x / self.sample_rate:.3f}')
        
        # Load entire audio once
        self.entire_audio = self.audio_input.get_entire_audio()
        self.total_samples = len(self.entire_audio)
        
        # Calculate window size (visible chunks worth of samples)
        self.window_size = self.VISIBLE_CHUNKS * self.hop_size + self.chunk_size
        
        # Track current window boundaries (only update when we cycle through visible chunks)
        self.window_start = 0
        self.window_end = min(self.window_size, self.total_samples)
        self.window_base_chunk = 0  # First chunk index in current window
        
        # Create 4-row grid
        self.gs = gridspec.GridSpec(4, 1, figure=self.fig, height_ratios=[1, 1, 1, 1])
        self.fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.93, hspace=0.35)
        
        # Get initial global audio for y-limits
        global_audio = self.entire_audio[self.window_start:self.window_end]
        global_x = np.arange(self.window_start, self.window_start + len(global_audio))
        y_min, y_max = np.min(global_audio), np.max(global_audio)
        y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        
        # Plot 1: Global view of original audio - populate immediately
        self.ax_global = self.fig.add_subplot(self.gs[0])
        self.ax_global.set_title("Original")
        self.ax_global.grid(True, alpha=0.15)
        self.ax_global.xaxis.set_major_formatter(self.time_formatter)
        self.ax_global.xaxis.set_major_locator(MultipleLocator(int(0.1 * self.sample_rate)))
        self.ax_global.set_yticks([])
        (self.line_global,) = self.ax_global.plot(global_x, global_audio, color=self.AUDIO_COLOR, linewidth=1)
        self.ax_global.set_xlim(self.window_start, self.window_end)
        self.ax_global.set_ylim(y_min - y_pad, y_max + y_pad)
        # Only keep most recent highlight for each lane on global view
        self.global_even_highlight = None  # Single span for most recent even chunk
        self.global_odd_highlight = None   # Single span for most recent odd chunk
        
        # Plot 2: Even chunks (Lane A)
        self.ax_even = self.fig.add_subplot(self.gs[1])
        self.ax_even.set_title("Even")
        self.ax_even.grid(True, alpha=0.15)
        self.ax_even.xaxis.set_major_formatter(self.time_formatter)
        self.ax_even.xaxis.set_major_locator(MultipleLocator(int(0.1 * self.sample_rate)))
        self.ax_even.set_yticks([])
        self.ax_even.set_xlim(self.window_start, self.window_end)
        self.ax_even.set_ylim(y_min - y_pad, y_max + y_pad)
        self.even_lines = []  # List of line objects for even chunks
        self.even_chunks = {}  # chunk_index -> line object
        self.even_highlight = None  # Highlight only when even chunk is most recent
        
        # Plot 3: Odd chunks (Lane B)
        self.ax_odd = self.fig.add_subplot(self.gs[2])
        self.ax_odd.set_title("Odd")
        self.ax_odd.grid(True, alpha=0.15)
        self.ax_odd.xaxis.set_major_formatter(self.time_formatter)
        self.ax_odd.xaxis.set_major_locator(MultipleLocator(int(0.1 * self.sample_rate)))
        self.ax_odd.set_yticks([])
        self.ax_odd.set_xlim(self.window_start, self.window_end)
        self.ax_odd.set_ylim(y_min - y_pad, y_max + y_pad)
        self.odd_lines = []  # List of line objects for odd chunks
        self.odd_chunks = {}  # chunk_index -> line object
        self.odd_highlight = None  # Highlight only when odd chunk is most recent
        
        # Plot 4: Composite - maintains running composite with overwriting
        self.ax_composite = self.fig.add_subplot(self.gs[3])
        self.ax_composite.set_title("Composite (Overwrite Overlapping Regions)")
        self.ax_composite.set_xlabel("Time (s)")
        self.ax_composite.grid(True, alpha=0.15)
        self.ax_composite.xaxis.set_major_formatter(self.time_formatter)
        self.ax_composite.xaxis.set_major_locator(MultipleLocator(int(0.1 * self.sample_rate)))
        self.ax_composite.set_yticks([])
        self.ax_composite.set_xlim(self.window_start, self.window_end)
        self.ax_composite.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Initialize composite buffer with NaN (will be filled as chunks come in)
        self.composite_buffer = np.full(self.window_size, np.nan)
        self.composite_colors = [''] * self.window_size  # Track which lane contributed each sample
        (self.line_composite,) = self.ax_composite.plot([], [], color='gray', linewidth=0.8)
        self.composite_even_line = None
        self.composite_odd_line = None
        # Only keep most recent highlight for each lane on composite view
        
        # Arrow tracking
        self.arrow_objects = []  # Track all arrow patches for cleanup
        self.composite_even_highlight = None  # Single span for most recent even chunk
        self.composite_odd_highlight = None   # Single span for most recent odd chunk
    
    def _clear_arrows(self):
        """Remove all existing arrow annotations"""
        for arrow in self.arrow_objects:
            arrow.remove()
        self.arrow_objects.clear()
    
    def _add_windowing_arrows(self, chunk_idx, chunk_start):
        """
        Add arrows to show windowing flow for current chunk.
        Shows arrows on every update to indicate the current operation.
        """
        # Convert sample positions to data coordinates
        hop_samples = self.hop_size
        chunk_samples = self.chunk_size
        chunk_end = chunk_start + chunk_samples
        
        # Determine vertical positions (middle of y-axis range)
        ylim = self.ax_global.get_ylim()
        y_range = ylim[1] - ylim[0]
        y_mid = (ylim[0] + ylim[1]) / 2
        
        # Arrow styling - made bigger and more visible
        arrow_style = dict(
            arrowstyle='-|>',
            lw=4,
            mutation_scale=25,
            alpha=1.0
        )
        
        is_even = (chunk_idx % 2 == 0)
        target_ax = self.ax_even if is_even else self.ax_odd
        lane_color = self.EVEN_COLOR if is_even else self.ODD_COLOR
        
        # 1. Overlap indicator on composite (shows overwrite region)
        if chunk_idx > 0 and self.overlap > 0:
            overlap_start = chunk_start
            overlap_end = chunk_start + self.overlap
            
            # Draw bracket showing overlap region - bigger and more visible
            bracket_y = ylim[1] - y_range * 0.15
            bracket_height = y_range * 0.08
            
            # Vertical lines of bracket
            left_line = self.ax_composite.plot(
                [overlap_start, overlap_start],
                [bracket_y - bracket_height, bracket_y + bracket_height],
                color='#FF0000',
                lw=3,
                alpha=1.0,
                zorder=100
            )[0]
            right_line = self.ax_composite.plot(
                [overlap_end, overlap_end],
                [bracket_y - bracket_height, bracket_y + bracket_height],
                color='#FF0000',
                lw=3,
                alpha=1.0,
                zorder=100
            )[0]
            horizontal_line = self.ax_composite.plot(
                [overlap_start, overlap_end],
                [bracket_y, bracket_y],
                color='#FF0000',
                lw=3,
                alpha=1.0,
                zorder=100
            )[0]
            
            self.arrow_objects.extend([left_line, right_line, horizontal_line])
            
            # Label overlap amount - bigger font
            overlap_text = self.ax_composite.text(
                (overlap_start + overlap_end) / 2,
                bracket_y + bracket_height * 1.5,
                f'OVERWRITE\n{self.overlap} samples',
                ha='center',
                va='bottom',
                fontsize=10,
                color='#FF0000',
                weight='bold',
                zorder=101
            )
            self.arrow_objects.append(overlap_text)
        
    def _update(self):
        """Update plots showing staggered overlap effect with overwriting composite"""
        # Get chunk data
        chunk_start = self.i * self.hop_size
        chunk_end = chunk_start + self.chunk_size
        
        # Handle case where chunk extends beyond audio
        if chunk_end > self.total_samples:
            chunk_end = self.total_samples
        
        chunk_audio = self.entire_audio[chunk_start:chunk_end]
        
        # Check if we need to advance the window
        chunks_since_base = self.i - self.window_base_chunk
        if chunks_since_base >= self.VISIBLE_CHUNKS:
            # Advance window by VISIBLE_CHUNKS
            self.window_base_chunk += self.VISIBLE_CHUNKS
            self.window_start = self.window_base_chunk * self.hop_size
            self.window_end = min(self.window_start + self.window_size, self.total_samples)
            
            # Clear old chunk lines when window advances
            for line in self.even_lines:
                line.remove()
            for line in self.odd_lines:
                line.remove()
            self.even_lines.clear()
            self.odd_lines.clear()
            self.even_chunks.clear()
            self.odd_chunks.clear()
            
            # Clear highlights on global view
            if self.global_even_highlight is not None:
                self.global_even_highlight.remove()
                self.global_even_highlight = None
            if self.global_odd_highlight is not None:
                self.global_odd_highlight.remove()
                self.global_odd_highlight = None
            
            # Clear highlights on lane plots
            if self.even_highlight is not None:
                self.even_highlight.remove()
                self.even_highlight = None
            if self.odd_highlight is not None:
                self.odd_highlight.remove()
                self.odd_highlight = None
            
            # Clear highlights on composite view
            if self.composite_even_highlight is not None:
                self.composite_even_highlight.remove()
                self.composite_even_highlight = None
            if self.composite_odd_highlight is not None:
                self.composite_odd_highlight.remove()
                self.composite_odd_highlight = None
            
            # Clear arrows when advancing window
            self._clear_arrows()
            
            # Reset composite buffer
            self.composite_buffer = np.full(self.window_end - self.window_start, np.nan)
            self.composite_colors = [''] * (self.window_end - self.window_start)
            
            # Update global view for new window
            global_audio = self.entire_audio[self.window_start:self.window_end]
            global_x = np.arange(self.window_start, self.window_start + len(global_audio))
            self.line_global.set_data(global_x, global_audio)
            
            y_min, y_max = np.min(global_audio), np.max(global_audio)
            y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
            
            # Update all axis limits
            self.ax_global.set_xlim(self.window_start, self.window_end)
            self.ax_global.set_ylim(y_min - y_pad, y_max + y_pad)
            self.ax_even.set_xlim(self.window_start, self.window_end)
            self.ax_even.set_ylim(y_min - y_pad, y_max + y_pad)
            self.ax_odd.set_xlim(self.window_start, self.window_end)
            self.ax_odd.set_ylim(y_min - y_pad, y_max + y_pad)
            self.ax_composite.set_xlim(self.window_start, self.window_end)
            self.ax_composite.set_ylim(y_min - y_pad, y_max + y_pad)
        
        is_even = (self.i % 2 == 0)
        
        # Add chunk to appropriate lane (Plot 2 or 3)
        x_data = np.arange(chunk_start, chunk_start + len(chunk_audio))
        
        if is_even:
            # Add to even lane (Plot 2)
            if self.i not in self.even_chunks:
                (line,) = self.ax_even.plot(x_data, chunk_audio, color=self.EVEN_COLOR, linewidth=0.8, alpha=0.7)
                self.even_lines.append(line)
                self.even_chunks[self.i] = line
            # Highlight current chunk lines
            for chunk_i, line in self.even_chunks.items():
                line.set_alpha(1.0 if chunk_i == self.i else 1.0)
                line.set_linewidth(1.0 if chunk_i == self.i else 1.0)
            
            # Plot 2: Show highlight for most recent even chunk
            if self.even_highlight is not None:
                self.even_highlight.remove()
            self.even_highlight = self._add_outline_box(self.ax_even, chunk_start, chunk_end, self.EVEN_COLOR)
            
            # Plot 1: Replace most recent even highlight (only keep one)
            if self.global_even_highlight is not None:
                self.global_even_highlight.remove()
            self.global_even_highlight = self._add_outline_box(self.ax_global, chunk_start, chunk_end, self.EVEN_COLOR)
            
            # Plot 4: Replace most recent even highlight (only keep one)
            if self.composite_even_highlight is not None:
                self.composite_even_highlight.remove()
            self.composite_even_highlight = self._add_outline_box(self.ax_composite, chunk_start, chunk_end, self.EVEN_COLOR)
        else:
            # Add to odd lane (Plot 3)
            if self.i not in self.odd_chunks:
                (line,) = self.ax_odd.plot(x_data, chunk_audio, color=self.ODD_COLOR, linewidth=0.8, alpha=0.7)
                self.odd_lines.append(line)
                self.odd_chunks[self.i] = line
            # Highlight current chunk lines
            for chunk_i, line in self.odd_chunks.items():
                line.set_alpha(1.0 if chunk_i == self.i else 1.0)
                line.set_linewidth(1.0 if chunk_i == self.i else 1.0)
            
            # Plot 3: Show highlight for most recent odd chunk
            if self.odd_highlight is not None:
                self.odd_highlight.remove()
            self.odd_highlight = self._add_outline_box(self.ax_odd, chunk_start, chunk_end, self.ODD_COLOR)
            
            # Plot 1: Replace most recent odd highlight (only keep one)
            if self.global_odd_highlight is not None:
                self.global_odd_highlight.remove()
            self.global_odd_highlight = self._add_outline_box(self.ax_global, chunk_start, chunk_end, self.ODD_COLOR)
            
            # Plot 4: Replace most recent odd highlight (only keep one)
            if self.composite_odd_highlight is not None:
                self.composite_odd_highlight.remove()
            self.composite_odd_highlight = self._add_outline_box(self.ax_composite, chunk_start, chunk_end, self.ODD_COLOR)
        
        # Update Plot 4: Composite with overwriting
        # New chunk overwrites whatever was there before in its region
        buffer_start = chunk_start - self.window_start
        buffer_end = buffer_start + len(chunk_audio)
        
        # Clamp to buffer bounds
        buffer_start = max(0, buffer_start)
        buffer_end = min(len(self.composite_buffer), buffer_end)
        audio_offset = max(0, self.window_start - chunk_start)
        
        # Overwrite the composite buffer with new chunk data
        chunk_len = buffer_end - buffer_start
        self.composite_buffer[buffer_start:buffer_end] = chunk_audio[audio_offset:audio_offset + chunk_len]
        
        # Track which lane contributed each sample
        for idx in range(buffer_start, buffer_end):
            self.composite_colors[idx] = 'even' if is_even else 'odd'
        
        # Redraw composite with colored segments
        self._update_composite_plot()
        
        # Update title
        lane_str = "Lane A (Even)" if is_even else "Lane B (Odd)"
        self.fig.suptitle(f"Signal Overlap | {100*self.overlap/self.chunk_size:.0f}% Overlap | {self.i + 1} out of {self._get_num_items()}")
        self.fig.canvas.draw_idle()
    
    def _update_composite_plot(self):
        """Redraw composite plot with segments colored by source lane"""
        # Remove old composite lines
        if self.composite_even_line is not None:
            self.composite_even_line.remove()
            self.composite_even_line = None
        if self.composite_odd_line is not None:
            self.composite_odd_line.remove()
            self.composite_odd_line = None
        
        # Build separate arrays for even and odd contributions
        x_all = np.arange(self.window_start, self.window_start + len(self.composite_buffer))
        even_y = np.where([c == 'even' for c in self.composite_colors], self.composite_buffer, np.nan)
        odd_y = np.where([c == 'odd' for c in self.composite_colors], self.composite_buffer, np.nan)
        
        # Plot each lane's contribution
        if not np.all(np.isnan(even_y)):
            (self.composite_even_line,) = self.ax_composite.plot(
                x_all, even_y, color=self.EVEN_COLOR, linewidth=1.0
            )
        if not np.all(np.isnan(odd_y)):
            (self.composite_odd_line,) = self.ax_composite.plot(
                x_all, odd_y, color=self.ODD_COLOR, linewidth=1.0
            )
        
        # Add windowing arrows for first few chunks to show the pattern
        self._clear_arrows()
        self._add_windowing_arrows(self.i, self.i * self.hop_size)

    def _add_outline_box(self, ax, x_start, x_end, color, linewidth=3):
        """Add a rectangle outline (no fill) spanning the full y-range of the axis"""
        y_min, y_max = ax.get_ylim()
        rect = Rectangle(
            (x_start, y_min), x_end - x_start, y_max - y_min,
            fill=False, edgecolor=color, linewidth=linewidth, zorder=10
        )
        ax.add_patch(rect)
        return rect

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
    SINUSOID_COLOR = '#1A1A1A'  # Near-black
    PERIOD_COLOR = '#1A1A1A'    # Near-black
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

class TopLevelComparisonNavigator(NavigatorBase):
    """
    Plot Navigator for top level comparison:
      - Plots the audio time series (L) and the CWT coefficients (R)
    """
    def __init__(self, title = "Top Level Comparison", audio_input = None, pywt = None, cpwt = None, midi_img_path = None, num_frames = 128):
        self.audio_input = audio_input
        self.num_frames = num_frames
        
        # Circular buffer for 1D audio chunks
        self.audio_buffer = AudioFrameBuffer(
            chunk_size=self.audio_input.chunk_size,
            num_chunks=num_frames
        )

        self.pywt = pywt
        self.cpwt = cpwt

        # Circular buffers for 2D CWT frames
        color_norm_config = ColorNormalizationConfig()
        self.pywt_frames = CircularFrameBuffer(
            frame_shape=self.pywt.get_output_shape(),
            num_frames=num_frames,
            color_norm_config=color_norm_config
        )
        self.cpwt_frames = CircularFrameBuffer(
            frame_shape=self.cpwt.get_output_shape(),
            num_frames=num_frames,
            color_norm_config=color_norm_config
        )

        self.midi_img = plt.imread(midi_img_path) if midi_img_path else None
        self.chunk_i = 0

        num_bars = 16
        beats_per_bar = 4
        beats_per_min = 128
        sample_rate = audio_input.get_sample_rate()

        num_samples = num_bars * beats_per_bar / beats_per_min * 60 * sample_rate
        
        super().__init__(title, cmap="magma")

    def _init_plots(self):
        """Initialize top level comparison plots"""
        # Create 4x3 grid, use middle column for plots (centered layout with padding)
        gs = gridspec.GridSpec(4, 3, figure=self.fig, 
                               width_ratios=[1, 1, 1],  # Narrow side columns
                               height_ratios=[1, 1, 1, 1])  # Taller spectrograms
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.95, hspace=0.3)
        
        # All plots in middle column (column index 1)
        self.ax_image = self.fig.add_subplot(gs[0, 1])
        self.ax_image.imshow(self.midi_img, aspect="auto", origin="upper")
        self.ax_image.set_title("MIDI Reference")
        self.ax_image.axis('off')

        self.ax_audio_t = self.fig.add_subplot(gs[1, 1])
        self.ax_audio_t.set_title("Audio Waveform")
        self.ax_audio_t.set_ylabel("Amplitude")

        self.ax_pywt_spec = self.fig.add_subplot(gs[2, 1])
        self.ax_pywt_spec.set_title("PyWavelet CWT")
        self.ax_pywt_spec.set_ylabel("Frequency Bin")

        self.ax_cpwt_spec = self.fig.add_subplot(gs[3, 1])
        self.ax_cpwt_spec.set_title("CuPy CWT")
        self.ax_cpwt_spec.set_ylabel("Frequency Bin")
        self.ax_cpwt_spec.set_xlabel("Time")

    def _update(self):
        """Update top level comparison visualization"""

        for i in range(self.num_frames):
            audio_chunk = self.audio_input.get_chunk()
            
            # Push audio to 1D buffer
            self.audio_buffer.push_chunk(audio_chunk)
            
            # Compute CWT and push to 2D buffers
            pywt_coefs = self.pywt.cwt(audio_chunk)
            cpwt_coefs = self.cpwt.cwt(audio_chunk)
            self.pywt_frames.push_frame(pywt_coefs)
            self.cpwt_frames.push_frame(cpwt_coefs)

        # Get all data in chronological order
        pywt_spectrogram = self.pywt_frames.get_flattened_buffer()
        cpwt_spectrogram = self.cpwt_frames.get_flattened_buffer()
        
        # Downsample audio to match spectrogram width for aligned visualization
        spectrogram_width = pywt_spectrogram.shape[1]
        x, y_min, y_max = self.audio_buffer.get_downsampled(spectrogram_width)

        # Plot everything
        self.ax_image.imshow(self.midi_img, aspect="auto", origin="upper")
        
        # Plot audio as min/max envelope (shows waveform shape clearly)
        self.ax_audio_t.fill_between(x, y_min, y_max, color='#1A1A1A', alpha=0.7)
        self.ax_audio_t.set_xlim(0, self.audio_buffer.total_samples)
        self.ax_audio_t.set_ylim(y_min.min() * 1.1, y_max.max() * 1.1)
        
        self.ax_pywt_spec.imshow(pywt_spectrogram, cmap=self.cmap, aspect="auto", origin="lower", vmin=0, vmax=self.pywt_frames.get_intensity_max())
        self.ax_cpwt_spec.imshow(cpwt_spectrogram, cmap=self.cmap, aspect="auto", origin="lower", vmin=0, vmax=self.cpwt_frames.get_intensity_max())

    