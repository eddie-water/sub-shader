import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet

"""
Defines

Assumptions:
    Ideally we have the biggest frame size and the smallest downsample possible
"""
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

"""
Audio Input, Characteristics 
"""
frame_size = FRAME_SIZE
file_path = "audio_files/zionsville.wav"

audio_input = AudioInput(path = file_path, frame_size = frame_size)

sampling_rate = audio_input.get_sample_rate() # 44.1 kHz
sampling_period = (1.0 / sampling_rate)

"""
Wavelet Object
"""
wavelet = Wavelet(frame_size = frame_size)
data_shape = wavelet.get_shape()

"""
Global Backend Config Options
    useOpenGL           - enables OpenGL (seems to make things ~2x faster)
    enableExperimental  - use PyOpenGL for curve drawing TODO LATER is this 
                          really even needed?
App, Widget, Plot
"""
pg.setConfigOptions(useOpenGL = True, enableExperimental = True)

app = pg.mkQApp("CWT of Zionsville")
win = pg.GraphicsLayoutWidget()
win.show()  
win.setWindowTitle('Continuous Wavelet Transform')

plot = win.addPlot(row = 0, 
                   col = 0, 
                   rowspan = 1,
                   colspan = 1,
                   title = "PColorMesh Plot",
                   enableMenu = False)

"""
Mesh Grid Points, Range, Density
    x_length - horizontal boundary of the mesh
    y_length - vertical boundary of the mesh
    x_length     - all the horizontal points evaluated within the range
    y_length     - all the vertical points evaluated within the range

Create two sets of 2D arrays, represent the horizontal and vertical grid points
    X Array - [1, 1, ... , 1]
              [2, 2, ... , 2]
              ...
              [x_length, x_length, ... , x_length]

    Y Array - [1, 2, ..., y_length]
              [1, 2, ..., y_length]
              ...
              [1, 2, ..., y_length]
"""
x_length = data_shape[1]
y_length = data_shape[0]

# Create X and Y Array
x = np.linspace(1, x_length, x_length)
x = np.repeat(x, y_length)
x = x.reshape(y_length, x_length)

y = np.linspace(1, y_length, y_length)
y = np.repeat(y, x_length)
y = y.reshape(x_length, y_length)

# Downsample the arrays becuase Python can't graph large things fast
x = x[::, ::(DOWNSAMPLE_FACTOR)]
y = y[::(DOWNSAMPLE_FACTOR), ::]

# Plot Boundaries
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

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
    rounding    - TODO LATER how fine of a rounding do we need

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
textBox.setPos(x_min + 1, y_min + 1)
plot.addItem(textBox)

"""
Update Plot
"""
def update_plot():
    # Grab a frame of audio
    audio_data = audio_input.get_frame()

    # Compute CWT 
    coefs = wavelet.compute_cwt(audio_data)

    # Downsample
    coefs = coefs[::, ::(DOWNSAMPLE_FACTOR)]
    coefs = coefs[:,:-1]
    coefs = np.transpose(coefs)

    # Update the color mesh grid
    pcmi.setData(coefs)

    # Update FPS Count
    framecnt.update()

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start()

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: textBox.setText(f'{fps:.1f} fps'))

if __name__ == '__main__':
    pg.exec()
