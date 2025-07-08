import numpy as np
import matplotlib.pyplot as plt
import math

class QuickPlot:
    def __init__(self, data, title="Quick Plot"):
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(title)

        if data.ndim == 1:
            self._plot_1d(data)
        elif data.ndim == 2:
            self._plot_2d(data)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        plt.tight_layout()
        plt.show()

    def _plot_1d(self, data):
        self.ax.plot(data)
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")

    def _plot_2d(self, data):
        # TODO do I want to flip the data?
        flipped = np.flipud(data)  # Optional: flip vertically so low freqs are at bottom
        mesh = self.ax.pcolormesh(flipped, cmap="inferno", shading="auto", vmin=0.0, vmax=1.0)
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel("Frequency Bins")
        self.fig.colorbar(mesh, ax=self.ax, label="Magnitude")

class QuickMultiPlot(QuickPlot):
    def __init__(self, plot_frames, titles=None):
        if titles is None:
            titles = [f"Frame {i + 1}" for i in range(len(plot_frames))]

        num_plots = len(plot_frames)
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        self.fig, self.axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        self.fig.suptitle("Quick Multi Plot")

        # Flatten in case axes is a 2D array (which it is when rows and cols > 1)
        self.axes = np.array(self.axes).flatten()

        for i, (ax, data, title) in enumerate(zip(self.axes, plot_frames, titles)):
            if data.ndim == 1:
                ax.plot(data)
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
            elif data.ndim == 2:
                flipped = np.flipud(data)
                mesh = ax.pcolormesh(flipped, cmap="inferno", shading="auto", vmin=0.0, vmax=1.0)
                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Frequency Bins")
                self.fig.colorbar(mesh, ax=ax, label="Magnitude")
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
            ax.set_title(title)

        # Hide unused axes
        for j in range(i + 1, len(self.axes)):
            self.axes[j].axis('off')

        plt.tight_layout()
        plt.show()
