# TODO NOW figure out where I left off 
# TODO NOW am I working in model/main or model/main_loop
# TODO NOW figure out where I left off 
# TODO NOW am I working in model/main or model/main_loop
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet

"""
Audio Input, Characteristics and Wavelet Object
"""
sampling_rate = 44100.0
sampling_period = (1.0 / sampling_rate)
file_path = "models/audio_files/zionsville.wav"

# TODO LATER - determine what is an appropriate window size
frame_size = 1024

audio_input = AudioInput(path = file_path, frame_size = frame_size)

wavelet = Wavelet(frame_size = frame_size)

data_shape = wavelet.get_shape()

# TODO NOW > NEXT > SOON > LATER

"""
Global Backend Config Options
    useOpenGL           - enables OpenGL (seems to make things ~2x faster)
    enableExperimental  - use PyOpenGL for curve drawing TODO LATER is this 
                          really even needed?
"""
pg.setConfigOptions(useOpenGL = True, enableExperimental = True)

"""
App, Widget, Plot
"""
app = pg.mkQApp("PColorMesh Example")
win = pg.GraphicsLayoutWidget()
win.show()  
win.setWindowTitle('ColorMesh')

# TODO SOON make all positional args relative to plot variables
plot = win.addPlot(0, 0, 1, 1,
                   title = "PColorMesh Plot",
                   enableMenu = False)

"""
Mesh Grid Points, Range, Density
    x_range - horizontal boundary of the mesh
    y_range - vertical boundary of the mesh
    density - scalar for the resolution between each integer in the range
    x_n     - all the horizontal points evaluated within the range
    y_n     - all the vertical points evaluated within the range

Create two sets of 2D arrays, represent the horizontal and vertical axes
    X Array - [1, 1, ... , 1]
              [2, 2, ... , 2]
              ...
              [x_range, x_range, ... , x_range]

    Y Array - [1, 2, ..., y_range]
              [1, 2, ..., y_range]
              ...
              [1, 2, ..., y_range]
"""
x_range = data_shape[1]
y_range = data_shape[0]

# Increasing density increases processing workload
# TODO NOW dont think I need density, delete this for now
density = 1 
x_n = x_range * density
y_n = y_range * density

x = np.linspace(1, x_range, x_n)
x = np.repeat(x, y_n)
x = x.reshape(y_n, x_n)
x = x[::, ::64]
x_range = x.shape[1]

# TODO NOW fix the y rangeuyh+
y = np.linspace(1, y_range, y_n)
y = np.repeat(y, x_n)

# TODO NEXT reshape using x_shape, not x_n
y = y.reshape(x_n, y_n)
y = y[::64, ::]

# Plot Boundaries
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

plot.setYRange(y_min, y_max)

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
# TODO NOW why this equation?
# c = np.exp(-(x*x_range))**2/1000
c = np.sin(2*np.pi*x)
c = c[:-1,:-1]

colorMap        = pg.colormap.get('magma')
edgeColors      = None
antialiasing    = False 
# TODO SOON describe the explicit relationship between autoscaling and levels and etc
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
# TODO LATER make the color bar the same height as the z axis
bar = pg.ColorBarItem(label = "Z Value",
                      interactive = False,
                      rounding = 0.1)
bar.setImageItem([pcmi])

# TODO LATER make all positional args relative to plot variables
win.addItem(bar, 0, 1, 1, 1)

textBox = pg.TextItem(anchor = (0, 1),
                      fill = 'black')
textBox.setPos(x_min + 1, y_min + 1)
plot.addItem(textBox)

"""
Wave Parameters

TODO NEXT clean up these wave parameters
"""
wave_amplitude  = 3
wave_speed      = 0.01
wave_length     = 10
color_speed     = 0.32
color_noise_freq = 0.05

"""
Update Plot
"""
def update_plot():
    audio_data = audio_input.get_frame()
    coefs = wavelet.compute_cwt(audio_data)
    coefs = coefs[::, ::64]
    coefs = coefs[:,:-1]

    # TODO NOW Downsample x y and coefs to have a total size of ~10k points

    # Update the color plot
    pcmi.setData(coefs)

    framecnt.update()

# TODO SOON what's the timer freq/period? ASAP? Or a default period?
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start()

# TODO SOON where does 'fps' come from? Is it inside FrameCounter?
framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: textBox.setText(f'{fps:.1f} fps'))

if __name__ == '__main__':
    pg.exec()
