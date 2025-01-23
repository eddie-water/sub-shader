import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet

"""
Constants
    Ideally we have the biggest frame size and the smallest downsample possible
"""
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

FILE_PATH = "audio_files/zionsville.wav"

"""
Audio Input, Characteristics 
"""
audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)

sampling_freq = audio_input.get_sample_rate() # 44.1 kHz
sampling_period = (1.0 / sampling_freq)

"""
Wavelet Object
"""
wavelet = Wavelet(sampling_freq = sampling_freq, 
                  frame_size = FRAME_SIZE,
                  downsample_factor = DOWNSAMPLE_FACTOR)
data_shape = wavelet.get_shape()

"""
Global Backend Config Options
    useOpenGL           - enables OpenGL (seems to make things ~2x faster)
    enableExperimental  - use PyOpenGL for curve drawing TODO LATER is this 
                          really even needed?
App, Widget, Plot
"""
pg.setConfigOptions(useOpenGL = True, enableExperimental = True)

app = pg.mkQApp("Sub Shader")
win = pg.GraphicsLayoutWidget()
win.show()  
win.setWindowTitle('Continuous Wavelet Transform')

plot = win.addPlot(row = 0, 
                   col = 0, 
                   rowspan = 1,
                   colspan = 1,
                   title = FILE_PATH,
                   enableMenu = False)

"""
Color Array and Plot

Color Array - The color values that fill the space between the points in the 
              mesh grid. Each dimension of c must be decreased by 1 to fit 
              inside the bounds of the mesh

PColorMeshItem
    colorMap    - color map ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    edgeColors  - color of the edge of the polygons
    antialias   - antialisasing when drawing edgelines if edgeColors == True
    levels      - min and max values of the color map
    autoLevels  - autoscales the colormap levels when setData() is called
"""
colorMap        = pg.colormap.get('inferno')
edgeColors      = None
antialiasing    = False 
levels          = (-1, 1)
autoLevels      = False

# Autoscaling is performed just once, when color data is first supplied
pcmi = pg.PColorMeshItem(colorMap = colorMap,
                         levels = levels,
                         enableAutoLevels = autoLevels,
                         edgeColors = edgeColors,
                         antialiasing = antialiasing)
plot.addItem(pcmi)

"""
Color Bar, Text Box

ColorBarItem
    label       - title for color bar
    interactive - True would override enableAutoLevels in PColorMesh
    rounding    - TODO EVENTUALLY how fine of a rounding do we need

TextItem
    anchor      - (x, y) what corner of the text box anchors the text's position
    fill        - the background color fill
"""
bar = pg.ColorBarItem(label = "Magnitude",
                      interactive = False,
                      rounding = 0.1)
bar.setImageItem([pcmi])
win.addItem(bar, 0, 1, 1, 1)

textBox = pg.TextItem(anchor = (0, 1),
                      fill = 'black')
textBox.setPos(1, 1)
plot.addItem(textBox)

"""
Update Plot
"""
def update_plot():
    # Grab a frame of audio
    audio_data = audio_input.get_frame()

    # Compute CWT on that frame
    coefs = wavelet.compute_cwt(audio_data)

    # Update the color mesh grid
    pcmi.setData(coefs)

    # Update FPS Count
    framecnt.update()

"""
Qt Timer and Frame Counter
    QTimer with timeout of 0 to time out ASAP
    Frame Counter is a utility from PyQtGraph that I stole
"""
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start()

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: textBox.setText(f'{fps:.1f} fps'))

"""
Main 
"""
if __name__ == '__main__':
    pg.exec()
