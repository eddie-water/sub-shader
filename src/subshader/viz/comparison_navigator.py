# subshader/viz/comparison_navigator.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ComparisonNavigator:
    """
    A reusable Matplotlib navigator that:
      - mode="kernels": cycles a wavelet index and plots time (L) and FFT (R)
      - mode="cwt":     steps through audio chunks; updates time (L) + two CWTs (R)

    Notes:
      - Artists are created once; updates happen with set_data.
      - "Prev" in cwt mode is forward-only by default (easy to add a ring buffer later).
      - Keybindings: Right/N = next, Left/P = prev.
    """
    def __init__(
        self,
        mode,
        *,
        # For kernels mode:
        np_wavelet=None,               # instance with get_wavelet_kernels('time'|'freq'), .freqs, .sample_rate
        # For cwt mode:
        audio_input=None,              # instance with get_chunk(), get_sample_rate()
        py_wavelet=None,               # CWT impl with compute_cwt(audio)
        cp_wavelet=None,               # second CWT impl with compute_cwt(audio)
        cmap="magma",
        title=None
    ):
        assert mode in ("kernels", "cwt")
        self.mode = mode
        self.np_wavelet = np_wavelet
        self.audio_input = audio_input
        self.py_wavelet = py_wavelet
        self.cp_wavelet = cp_wavelet
        self.cmap = cmap
        self.title = title

        self.idx = 0
        self._init_fig()
        if self.mode == "kernels":
            self._init_artists_kernels()
        else:
            self._init_artists_cwt()

        self._draw()

        plt.show()

    # ---------- Figure & Buttons ----------
    def _init_fig(self):
        self.fig = plt.figure(figsize=(14, 7), constrained_layout=False)
        if self.mode == "kernels":
            # 1x2 grid: left=time, right=fft
            self.ax_time = self.fig.add_subplot(1, 2, 1)
            self.ax_fft  = self.fig.add_subplot(1, 2, 2)
            window_title = self.title or "Wavelet Kernel Comparison"
        else:
            # Left = audio time series (full height); right = two stacked CWTs
            self.ax_time = self.fig.add_subplot(1, 2, 1)
            self.ax_py   = self.fig.add_subplot(2, 2, 2)
            self.ax_cp   = self.fig.add_subplot(2, 2, 4)
            window_title = self.title or "Static CWT Plot Comparison"

        try:
            self.fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

        # Buttons
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.93, wspace=0.15, hspace=0.25)
        ax_prev = self.fig.add_axes([0.06, 0.04, 0.08, 0.05])
        ax_next = self.fig.add_axes([0.86, 0.04, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(lambda _ : self._step(-1))
        self.btn_next.on_clicked(lambda _ : self._step(+1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, ev):
        if ev.key in ("right", "n"):
            self._step(+1)
        elif ev.key in ("left", "p"):
            self._step(-1)

    # ---------- Kernels mode ----------
    def _init_artists_kernels(self):
        assert self.np_wavelet is not None, "np_wavelet is required for mode='kernels'"
        self.kernels_t = self.np_wavelet.get_wavelet_kernels("time")
        self.kernels_f = self.np_wavelet.get_wavelet_kernels("freq")
        self.freqs     = np.asarray(self.np_wavelet.freqs)
        self.fs        = self.np_wavelet.sample_rate
        self.N         = len(self.kernels_t)

        # Time axis artists
        self.ax_time.set_title("Wavelet (time)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        (self.l_real,) = self.ax_time.plot([], [], lw=2, label="Real")
        (self.l_imag,) = self.ax_time.plot([], [], lw=2, label="Imag")
        (self.l_mag,)  = self.ax_time.plot([], [], lw=2, label="Mag")
        self.ax_time.grid(True, alpha=0.25)
        self.ax_time.legend(loc="upper right", frameon=False)

        # FFT axis artists
        self.ax_fft.set_title("Wavelet (FFT magnitude)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_xscale("log")
        self.ax_fft.set_ylabel("|H(f)|")
        (self.l_fft,) = self.ax_fft.plot([], [], lw=2)
        self.ax_fft.grid(True, which="both", alpha=0.25)

    def _draw_kernels(self):
        i = self.idx % self.N
        kt = self.kernels_t[i]
        t = np.arange(len(kt)) / self.fs
        t = t - t[len(t)//2]

        self.l_real.set_data(t, np.real(kt))
        self.l_imag.set_data(t, np.imag(kt))
        self.l_mag.set_data(t,  np.abs(kt))
        self.ax_time.set_xlim(t[0], t[-1])
        self.ax_time.relim(); self.ax_time.autoscale(axis="y", tight=True)

        kf = self.kernels_f[i]
        n  = len(kf)
        f  = np.fft.fftfreq(n, d=1/self.fs)[: n//2]
        mag= np.abs(kf[: n//2])
        self.l_fft.set_data(f, mag + 1e-12)  # avoid log(0)
        self.ax_fft.set_xlim(max(20, f[1]), min(self.fs/2, f[-1]))
        self.ax_fft.relim(); self.ax_fft.autoscale(axis="y", tight=True)

        self.fig.suptitle(f"Kernel {i+1}/{self.N}  —  f₀ ≈ {self.freqs[i]:.1f} Hz")

    # ---------- CWT mode ----------
    def _init_artists_cwt(self):
        assert all([self.audio_input, self.py_wavelet, self.cp_wavelet]), \
            "audio_input, py_wavelet, and cp_wavelet are required for mode='cwt'"

        # Initial chunk sizes artists
        self.curr_audio = self.audio_input.get_chunk()

        self.ax_time.set_title("Audio (time)")
        self.ax_time.set_xlabel("Samples")
        self.ax_time.set_ylabel("Amplitude")
        (self.l_ts,) = self.ax_time.plot(np.arange(len(self.curr_audio)), self.curr_audio)
        self.ax_time.margins(x=0, y=0)
        self.ax_time.grid(True, alpha=0.15)

        py_coefs = self.py_wavelet.compute_cwt(self.curr_audio)
        cp_coefs = self.cp_wavelet.compute_cwt(self.curr_audio)

        self.ax_py.set_title("PyWavelet CWT")
        self.im_py = self.ax_py.imshow(py_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_py.set_xlabel("Time")
        self.ax_py.set_ylabel("Freq Bin")

        self.ax_cp.set_title("CuPy CWT")
        self.im_cp = self.ax_cp.imshow(cp_coefs, cmap=self.cmap, aspect="auto", origin="lower")
        self.ax_cp.set_xlabel("Time")
        self.ax_cp.set_ylabel("Freq Bin")

        # Shared colorbar for right column
        try:
            self.fig.colorbar(self.im_cp, ax=[self.ax_py, self.ax_cp], fraction=0.025, pad=0.02)
        except Exception:
            pass

        self.chunk_idx = 0

    def _draw_cwt(self, step):
        # Forward-only by default. Add a ring buffer if you want true "Prev".
        if step > 0:
            self.curr_audio = self.audio_input.get_chunk()
            self.chunk_idx += 1

            x = np.arange(len(self.curr_audio))
            self.l_ts.set_data(x, self.curr_audio)
            self.ax_time.set_xlim(x[0], x[-1])
            self.ax_time.relim(); self.ax_time.autoscale(axis="y", tight=True)

            py_coefs = self.py_wavelet.compute_cwt(self.curr_audio)
            cp_coefs = self.cp_wavelet.compute_cwt(self.curr_audio)

            self.im_py.set_data(py_coefs)
            self.im_cp.set_data(cp_coefs)

            self.ax_py.set_xlim(0, py_coefs.shape[1]); self.ax_py.set_ylim(0, py_coefs.shape[0])
            self.ax_cp.set_xlim(0, cp_coefs.shape[1]); self.ax_cp.set_ylim(0, cp_coefs.shape[0])

            self.fig.suptitle(f"Chunk {self.chunk_idx}")

    # ---------- Shared ----------
    def _draw(self, step=0):
        if self.mode == "kernels":
            self._draw_kernels()
        else:
            self._draw_cwt(step)
        self.fig.canvas.draw_idle()

    def _step(self, d):
        if self.mode == "kernels":
            self.idx = (self.idx + d)
            self._draw()
        else:
            # Only implement forward stepping by default
            self._draw(step=+1 if d > 0 else -1)
