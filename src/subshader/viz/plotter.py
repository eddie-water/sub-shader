from abc import ABC, abstractmethod
import numpy as np

import pyqtgraph as pg

import moderngl
import glfw

class Plotter(ABC):
    def __init__(self, file_path: str, shape: tuple[int:int]):
        # TODO ISSUE-33 check if file_path is a valid path
        self.file_path = file_path

        if len(shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(shape)}D with shape {shape}")        
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"2D array cannot have shape: {shape}")
        self.shape = shape
        self.x_n, self.y_n = self.shape

    @abstractmethod
    def update_plot(self, values):
        pass

    @abstractmethod
    def update_fps(self, fps: int):
        pass

    @abstractmethod
    def should_window_close(self):
        pass

class PyQtGrapher(Plotter):
    def __init__(self, file_path: str, shape: tuple[int, int]):
        super().__init__(file_path, shape)
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
            values: coefficients to update the pcolormesh with
    """
    def update_plot(self, values):
        # TODO ISSUE-33 Plotter Improvements fix axes so we don't have to transpose 
        values = values.T
        self.pcolormesh.setData(values)

    """
    Update FPS
        Args:
            fps: the value to update the FPS Textbox to
    """
    def update_fps(self, fps: int):
        self.textBox.setText((f'{fps:.1f} fps'))

    def should_window_close(self):
        raise NotImplementedError("PyQtGraph-based window-check not implemented yet.")

class Shader(Plotter):
    def __init__(self, file_path: str, shape: tuple[int, int]):
        super().__init__(file_path, shape)

        # Initialize GLFW window and OpenGL context 
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Set properties to OpenGL 3.3 core profile (for ModernGL)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(800, 600, "Sub Shader", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")

        # TODO ISSUE-33 NEXT Understand how using this context affects the rest of the code
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

        # Compile shaders and set up program 
        prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 position;
            out vec2 texCoord;
            void main() {
                texCoord = (position + 1.0) / 2.0;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            in vec2 texCoord;
            out vec4 fragColor;
            uniform sampler2D scalogram;
            void main() {
                float value = texture(scalogram, texCoord).r;
                fragColor = vec4(value, value, value, 1.0);
            }
            """,
        )

        # Set up quad vertex buffer object and vertex array object 
        quad = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0,  1.0,
            1.0,  1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(prog, self.vbo, 'position')

        # Create 2D texture placeholder for scalogram 
        self.texture = self.ctx.texture((self.x_n, self.y_n), 1, dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        prog['scalogram'] = 0  # texture unit 0
    
    def update_plot(self, values):
        if values.shape != (self.x_n, self.y_n):
            raise ValueError(f"Expected shape {(self.x_n, self.y_n)}, got {values.shape}")

        # Update the texture with new values
        self.texture.write(values.astype('f4').tobytes())

        # Render the updated texture
        self.texture.use(location=0)

        # Clear the context and render the quad with the texture
        self.ctx.clear(0.2, 0.2, 0.2)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Swap buffers to display the rendered frame
        glfw.swap_buffers(self.window)

    def update_fps(self, fps: int):
        raise NotImplementedError("Shader-based plotting not yet implemented.")
    
    def should_window_close(self):
        return glfw.window_should_close(self.window)