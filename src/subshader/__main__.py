# src/subshader/__main__.py

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from subshader.utils.frame_counter_pyqt5 import FrameCounter

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import ShadeWavelet
from subshader.viz.plotter import Plotter

# Constants: 
#   Ideally have the biggest frame size and the smallest downsample factor 
WINDOW_SIZE = 4096

FILE_PATH = "assets/audio/c4_and_c7_4_arps.wav"

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)

sample_rate = audio_input.get_sample_rate() # 44.1 kHz

# Wavelet Object
wavelet = ShadeWavelet(sample_rate = sample_rate, 
                       window_size = WINDOW_SIZE)

# Plotter Object
plotter = Plotter(file_path = FILE_PATH)

def main_loop():
    # Grab a frame of audio
    audio_data = audio_input.get_frame()

    # Compute CWT on that frame
    coefs = wavelet.compute_cwt(audio_data)

    # Update plot
    plotter.update_plot(coefs)

    # Update FPS Count
    fps_counter.update()

timer = QtCore.QTimer()
timer.timeout.connect(main_loop)
timer.start()

# FPS Counter (stolen from PyQtGraph)
fps_counter = FrameCounter()
fps_counter.sigFpsUpdate.connect(lambda fps: plotter.update_fps(fps))

# Main 
if __name__ == '__main__':
    pg.exec()
