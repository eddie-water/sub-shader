from abc import ABC, abstractmethod
import numpy as np

import pyqtgraph as pg

import moderngl
import glfw
from matplotlib import cm

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

# TODO ISSUE-33 LATER: Exit gracefully when the window is closed / ctrl + c
class Shader(Plotter):
    def __init__(self, file_path: str, shape: tuple[int, int]):
        super().__init__(file_path, shape)

        # Initi GLFW and OpenGL context
        self._init_graphics()

    def _create_magma_texture(self, resolution = 256):
        """Create a 1D texture with magma colormap"""
        # Generate magma colormap data
        magma_cmap = cm.get_cmap('magma')
        x = np.linspace(0, 1, resolution)
        colors = magma_cmap(x)[:, :3]  # RGB only, no alpha
        
        # Create 1D texture
        colormap_texture = self.ctx.texture((resolution, 1), 3, dtype='f4')
        colormap_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        colormap_texture.write(colors.astype('f4').tobytes())
        
        return colormap_texture

    def _init_graphics(self):
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

        # TODO ISSUE-33 SOON Understand how using this context affects the rest of the code
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

        # Init vertex shader (specifies where all the points are in the quad)
        self.vertex_shader = self._init_vertex_shader()

        # Init fragment shader (specifies how to color in between the points)
        self.fragment_shader = self._init_fragment_shaders()

        # Compile and links the vertex and fragment shaders into a program
        prog = self.ctx.program(
            vertex_shader = self.vertex_shader,
            fragment_shader = self.fragment_shader,
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

        # Create magma colormap texture (only needed for lookup approach)
        self.colormap_texture = self._create_magma_texture()
        
        # Create 2D texture placeholder for scalogram 
        self.texture = self.ctx.texture((self.x_n, self.y_n), 1, dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Set up uniforms
        prog['scalogram'] = 0  # texture unit 0
        prog['colormap'] = 1   # Only needed for lookup approach
        self.colormap_texture.use(location=1)  # Bind colormap texture to unit 1
        self.value_min_uniform = prog['valueMin']
        self.value_max_uniform = prog['valueMax']
        
        # Initialize value range
        self.value_min = 0.0
        self.value_max = 1.0

    def _init_vertex_shader(self):
        # Vertex shader for rendering a quad
        vertex_shader = """
            #version 330
            in vec2 position;
            out vec2 texCoord;
            void main() {
                texCoord = (position + 1.0) / 2.0;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """
        return vertex_shader

    def _init_fragment_shaders(self):
        # Option 1: Fragment shader with 1D texture lookup
        fragment_shader_lookup = """
            #version 330
            in vec2 texCoord;
            out vec4 fragColor;
            uniform sampler2D scalogram;
            uniform sampler2D colormap;
            uniform float valueMin;
            uniform float valueMax;
            
            void main() {
                float value = texture(scalogram, texCoord).r;
                
                // Normalize value to [0, 1] range
                float normalized = clamp((value - valueMin) / (valueMax - valueMin), 0.0, 1.0);
                
                // Apply gamma correction for better perception
                normalized = pow(normalized, 0.8);
                
                // Sample the colormap
                vec3 color = texture(colormap, vec2(normalized, 0.5)).rgb;
                
                fragColor = vec4(color, 1.0);
            }
        """
        
        # Option 2: Hardcoded magma approximation (no texture lookup needed)
        fragment_shader_hardcoded = """
            #version 330
            in vec2 texCoord;
            out vec4 fragColor;
            uniform sampler2D scalogram;
            uniform float valueMin;
            uniform float valueMax;
            
            vec3 magma(float t) {
                // Magma colormap approximation using polynomial fits
                t = clamp(t, 0.0, 1.0);
                
                vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
                vec3 c1 = vec3(0.166383, 0.009605, 0.620465);
                vec3 c2 = vec3(0.109303, 0.718710, 0.040311);
                vec3 c3 = vec3(2.108782, -1.531415, -0.273740);
                vec3 c4 = vec3(-2.490635, 2.051947, 1.073524);
                vec3 c5 = vec3(1.313448, -1.214297, -0.472305);
                
                return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
            }
            
            void main() {
                float value = texture(scalogram, texCoord).r;
                
                // Normalize and apply gamma correction
                float normalized = clamp((value - valueMin) / (valueMax - valueMin), 0.0, 1.0);
                normalized = pow(normalized, 0.7);
                
                // Get magma color
                vec3 color = magma(normalized);
                
                fragColor = vec4(color, 1.0);
            }
        """
        
        # Choose the shader option based on your preference
        # return fragment_shader_hardcoded
        return fragment_shader_lookup
    
    def update_plot(self, values):
        if values.shape != (self.x_n, self.y_n):
            raise ValueError(f"Expected shape {(self.x_n, self.y_n)}, got {values.shape}")

        # Update value range for better contrast
        self.value_min = float(np.percentile(values, 1))  # 1st percentile
        self.value_max = float(np.percentile(values, 99)) # 99th percentile
        
        # TODO ISSUE-33 NEXT: What's the point of these?
        # Update uniforms
        self.value_min_uniform.value = self.value_min
        self.value_max_uniform.value = self.value_max

        # Update the texture with new values
        self.texture.write(values.astype('f4').tobytes())

        # Bind textures
        self.texture.use(location=0)
        # TODO ISSUE-33 SOON: Use a conditional to switch use this or not
        # self.colormap_texture.use(location=1)  # Only for lookup approach

        # Clear the context and render the quad with the texture
        self.ctx.clear(0.05, 0.05, 0.05)  # Dark background
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Swap buffers to display the rendered frame
        glfw.swap_buffers(self.window)

    def update_fps(self, fps: int):
        raise NotImplementedError("Shader-based plotting not yet implemented.")
    
    def should_window_close(self):
        return glfw.window_should_close(self.window)