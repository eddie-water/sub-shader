import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter



class Plotter():
    def __init__(self, file_path: str):
        """
        Global Backend Config Options
            useOpenGL           - enables OpenGL (seems to make things ~2x 
                                  faster)
            enableExperimental  - use PyOpenGL for curve drawing TODO LATER is 
                                  this really even needed?

        App, Widget, Plot
        """
        pg.setConfigOptions(useOpenGL = True, enableExperimental = True)

        self.app = pg.mkQApp("Sub Shader")
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  
        self.win.setWindowTitle('Continuous Wavelet Transform')

        self.plot = self.win.addPlot(row = 0, 
                        col = 0, 
                        rowspan = 1,
                        colspan = 1,
                        title = file_path,
                        enableMenu = False)

        """
        Color Array and Plot

        Color Array - The color values that fill the space between the points 
                    in the mesh grid. Each dimension of c must be decreased by 
                    1 to fit inside the bounds of the mesh

        PColorMeshItem
            colorMap    - color map ['viridis', 'plasma', 'inferno', 'magma', 
                          'cividis']
            edgeColors  - color of the edge of the polygons
            antialias   - antialisasing when drawing edgelines if edgeColors == 
                          True
            levels      - min and max values of the color map
            autoLevels  - autoscales the colormap levels when setData() is 
                          called
        """
        colorMap        = pg.colormap.get('inferno')
        edgeColors      = None
        antialiasing    = False 
        levels          = (-1, 1)
        autoLevels      = False

        # Autoscaling is performed just once, when color data is first supplied
        self.pcolormesh = pg.PColorMeshItem(colorMap = colorMap,
                                levels = levels,
                                enableAutoLevels = autoLevels,
                                edgeColors = edgeColors,
                                antialiasing = antialiasing)

        self.plot.addItem(self.pcolormesh)

        """
        Color Bar, Text Box

        ColorBarItem
            label       - title for color self.bar
            interactive - True would override enableAutoLevels in PColorMesh
            rounding    - TODO EVENTUALLY how fine of a rounding do we need

        TextItem
            anchor  - x, y what corner of the text box anchors the text's 
                      position
            fill    - the background color fill
        """
        self.bar = pg.ColorBarItem(label = "Magnitude",
                                   interactive = False,
                                   rounding = 0.1)
        self.bar.setImageItem([self.pcolormesh])
        self.win.addItem(self.bar, 0, 1, 1, 1)

        self.textBox = pg.TextItem(anchor = (0, 1),
                                   fill = 'black')
        self.textBox.setPos(1, 1)
        self.plot.addItem(self.textBox)

    """
    Update Plot
        Args:
            coefs: coefficients to update the pcolormesh with
    """
    def update_plot(self, coefs):
        self.pcolormesh.setData(coefs)

    """
    Update FPS
        Args:
            fps: the value to update the FPS Textbox to
    """
    def update_fps(self, fps: int):
        self.textBox.setText((f'{fps:.1f} fps'))
