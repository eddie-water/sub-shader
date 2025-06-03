from abc import ABC, abstractmethod
import pyqtgraph as pg

class Plotter(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def update_plot(self, coefs):
        pass

    @abstractmethod
    def update_fps(self, fps: int):
        pass

class PyQtGrapher(Plotter):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        """
        Global Backend Config Options
            useOpenGL           - enables OpenGL (seems to make things ~2x 
                                  faster)
            enableExperimental  - use PyOpenGL for curve drawing 

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
            rounding    - decimal precision

        TextItem
            anchor  - x, y what corner of the text box anchors the text's 
                      position
            fill    - the background color fill
        """
        self.bar = pg.ColorBarItem(label = "Magnitude",
                                   interactive = False,
                                   limits = (0, 1),
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
        # TODO ISSUE-33 Plotter Improvements fix axes so we don't have to transpose 
        coefs = coefs.T
        self.pcolormesh.setData(coefs)

    """
    Update FPS
        Args:
            fps: the value to update the FPS Textbox to
    """
    def update_fps(self, fps: int):
        self.textBox.setText((f'{fps:.1f} fps'))

class Shader(Plotter):
    def __init__(self, file_path: str):
        super().__init__(file_path)

        # Implement OpenGL shader-based plotting here
        raise NotImplementedError("Shader-based plotting not yet implemented.")
        # This would involve creating OpenGL shaders and rendering the CWT
    
    def update_plot(self, coefs):
        raise NotImplementedError("Shader-based plotting not yet implemented.")

    def update_fps(self, fps: int):
        raise NotImplementedError("Shader-based plotting not yet implemented.")