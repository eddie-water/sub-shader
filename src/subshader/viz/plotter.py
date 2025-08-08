from abc import ABC, abstractmethod
import os
import numpy as np
import moderngl
import glfw
import pyqtgraph as pg
from .shaders import get_vertex_shader_source, get_fragment_shader_source
from ..utils.logging import get_logger

log = get_logger(__name__)

class Plotter(ABC):
    def __init__(self, file_path: str, frame_shape: tuple[int, int]):
        """
        Abstract base class for all plotters.

        Args:
            file_path (str): The path to the file to plot.
            frame_shape (tuple[int, int]): The shape of each data frame to plot.
        """
        self.file_path = file_path

        if len(frame_shape) != 2:
            log.error(f"Invalid frame shape: expected 2D array, got {len(frame_shape)}D with shape {frame_shape}")
            raise ValueError(f"Expected 2D array, got {len(frame_shape)}D with shape {frame_shape}")        
        if frame_shape[0] <= 0 or frame_shape[1] <= 0:
            log.error(f"Invalid frame dimensions: {frame_shape} (must be > 0)")
            raise ValueError(f"2D array cannot have shape: {frame_shape}")

        self.frame_shape = frame_shape
        self.y_n, self.x_n = self.frame_shape

    @abstractmethod
    def update_plot(self, plot_values):
        """
        Abstract method to update the plot with new data.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        pass

    @abstractmethod
    def should_window_close(self):
        """Check if the window should close based on user input."""
        pass

class ShaderPlot(Plotter):
    def __init__(self, file_path: str, frame_shape: tuple[int, int], num_frames: int = 64, fullscreen: bool = False):
        """
        2D data visualization using shaders

        Args:
            file_path (str): The path to the file to plot.
            frame_shape (tuple[int, int]): The shape of each data frame to plot.
            num_frames (int): The number of frames to use for the visualization.
            fullscreen (bool): Whether to display in fullscreen mode.
        """
        super().__init__(file_path, frame_shape)

        # Handles circular buffer for scrolling visualization
        self.rolling_frame_buffer = RollingFrameBuffer(num_frames, self.y_n, self.x_n)

        # Create GL Context and Shader Renderer       
        file_name = os.path.basename(file_path)
        self.gl_context = GLContext(title=f"SubShader - {file_name}", fullscreen=fullscreen)

        # Main GPU rendering component - handles shader compilation, 
        # texture management, and rendering
        texture_height, texture_width = self.rolling_frame_buffer.get_flattened_buffer_shape()
        self.renderer = ShaderRenderer(self.gl_context.ctx, texture_width, texture_height)

    def update_plot(self, plot_values: np.ndarray):
        """
        Updates the rolling plot with new data.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        # Add a new plot fram to the circular buffer
        self.rolling_frame_buffer.add_frame(plot_values)

        # Update the texture with the new data
        self.renderer.update_texture(self.rolling_frame_buffer.get_flattened_buffer())
        
        # Clear the
        self.gl_context.clear()
        self.renderer.render()
        self.gl_context.swap_buffers()
    
    def update_fps(self, fps: int):
        """
        GPU visualizer doesn't have FPS display
        """
        raise NotImplementedError("FPS display not implemented for GPU visualizer")
    
    def should_window_close(self):
        """
        Check if user wants to close the window
        """  
        return self.gl_context.should_close()
    
    def cleanup(self):
        """
        Clean shutdown
        """
        glfw.terminate()

class GLContext:
    def __init__(self, width=1920, height=1080, title="Audio Visualizer", fullscreen=False):
        """
        Handles GLFW window and OpenGL context setup
        
        Args:
            width (int): Window width (ignored if fullscreen=True)
            height (int): Window height (ignored if fullscreen=True)
            title (str): Window title
            fullscreen (bool): Whether to create a fullscreen window
        """
        self.width = width
        self.height = height
        self.title = title
        self.fullscreen = fullscreen
        self.window = None
        self.ctx = None
        self._init_graphics()
    
    def _init_graphics(self):
        """
        GLFW is a cross-platform library used for creating windows with OpenGL 
        contexts and handling input. It's the way OpenGL displays the graphics
        onto the screen.
        """
        # Set up GLFW error callback to redirect messages to log
        self._setup_glfw_error_callback()
        
        if not glfw.init():
            log.error("GLFW initialization failed")
            raise RuntimeError("Failed to initialize GLFW")
        
        # WSL-specific GLFW hints to reduce escape sequence issues
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        
        # Set OpenGL context version hints before creating window
        # Request OpenGL 3.3 Core Profile for modern shader support
        # Core profile (no deprecated features)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  

        # Create window - monitor (None = windowed), share context (None = no sharing)
        if self.fullscreen:
            # Get primary monitor and its video mode for fullscreen
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            self.width, self.height = mode.size.width, mode.size.height
            log.info(f"Creating fullscreen window: {self.width}×{self.height}")
            self.window = glfw.create_window(self.width, self.height, self.title, monitor, None)
        else:
            log.info(f"Creating windowed mode: {self.width}×{self.height}")
            self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            # Clean up GLFW before failing
            glfw.terminate()
            log.error("GLFW window creation failed")
            raise RuntimeError("Failed to create window")

        # Make OpenGL context current for this thread before any OpenGL calls
        glfw.make_context_current(self.window)
        
        """
        ModernGL is a Python wrapper around OpenGL that provides a more
        pythonic interface for OpenGL calls. It allows us to create shaders,
        buffers, textures, and other OpenGL objects without dealing with the
        low-level OpenGL API directly.
        """
        self.ctx = moderngl.create_context()
        
        # Log OpenGL info for debugging
        log.info(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
        log.debug(f"Viewport: {self.ctx.viewport}")

        # Setup viewport (area of window where OpenGL renders) to match window 
        self.ctx.viewport = (0, 0, self.width, self.height)
        
        # Disable depth testing (z-values) since rendering 2D content only 
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Disable face culling - we want to see both sides of triangles
        # Face culling normally hides back-facing triangles for performance
        self.ctx.disable(moderngl.CULL_FACE)
        
        log.info("Graphics context initialized successfully")
    
    def _setup_glfw_error_callback(self):
        """
        Set up GLFW error callback to redirect messages to log instead of terminal.
        """
        def glfw_error_callback(error_code, description):
            """
            GLFW error callback that redirects messages to log.
            
            Args:
                error_code: GLFW error code
                description: Error description string
            """
            # Log WSL-specific escape sequence messages at debug level
            if any(msg in description for msg in [
                "Dropped Escape call",
                "ulEscapeCode", 
                "Invalid escape sequence",
                "Unknown escape sequence"
            ]):
                log.debug(f"GLFW WSL escape sequence: {description}")
                return
            
            # Log other GLFW errors at warning level
            log.warning(f"GLFW Error {error_code}: {description}")
        
        glfw.set_error_callback(glfw_error_callback)
        log.debug("GLFW error callback configured")
    
    def should_close(self):
        """
        Checks if the window should close based on user input.
        
        Returns:
            bool: True if the window should close, False otherwise.
        """
        return glfw.window_should_close(self.window)
    
    def swap_buffers(self):
        """
        Swap the front and back buffers to display the rendered content.
        """
        glfw.swap_buffers(self.window)
        glfw.poll_events()  # Process window events
    
    def clear(self, r=0.05, g=0.05, b=0.05):
        """
        Clear the OpenGL context with a specified color.
        """
        self.ctx.clear(r, g, b)

class ShaderRenderer:

    SCALOGRAM_TEXTURE_UNIT = 0

    def __init__(self, ctx, texture_width, texture_height):
        """
        Main GPU rendering component that handles shader compilation, texture 
        management, and rendering

        Args:
            ctx (moderngl.Context): The ModernGL context to use for shader 
                compilation.
            texture_width (int): The width of the texture.
            texture_height (int): The height of the texture.
        """
        self.ctx = ctx
        
        # Compile and link shaders
        self.shader = self._compile_shaders()
        
        # Bind the graphics geometry to the shader program
        self._setup_rendering_geometry()

        # Prepare the texture 
        self._setup_texture(texture_width, texture_height)
    
    def _compile_shaders(self):
        """
        Compile and link vertex (geometry) and fragment (color) shaders into 
        a GPU-executable program
        """
        vertex_shader = get_vertex_shader_source()
        fragment_shader = get_fragment_shader_source()
        
        log.info("Compiling shaders...")
        
        shader = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        log.info("Shader compilation successful!")

        return shader

    def _setup_rendering_geometry(self):
        """
        Create quad geometry (rectangle) that covers the GLFW window, store in 
        OpenGL context/memory via VBO, and bind to the shader program via VAO.
        """
        quad_vertices = np.array([
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right
            -1.0,  1.0,  # Top-left
             1.0,  1.0,  # Top-right
        ], dtype=np.float32)
        
        # Vertex Buffer Object stores the quad vertices in GPU memory (tobytes()
        # removes NumPy stuff that GPU doesn't need)
        log.info(f"CPU→GPU: Uploading vertex buffer ({quad_vertices.shape}, {quad_vertices.dtype}, {quad_vertices.nbytes} bytes)")
        self.vbo = self.ctx.buffer(quad_vertices.tobytes()) 

        # Vertex Array Object tells the shader program how to use the data
        # stored in the VBO (position, color, etc.)
        self.vao = self.ctx.simple_vertex_array(self.shader, self.vbo, 'position')

    def _setup_texture(self, width, height):
        """
        Create 2D texture for scalogram data
        
        Args:
            width (int): Width of the texture.
            height (int): Height of the texture.
        """
        # Use actual data size - downsampling is handled at the source
        log.info(f"Creating texture: {width}x{height} (1 channel grayscale, float32)")
        log.info(f"CPU→GPU: Allocating texture buffer ({width}×{height}, f4, {width * height * 4} bytes)")
        self.texture = self.ctx.texture((width, height), 1, dtype='f4')

        # Linear interpolation for smooth scaling if the texture is resized
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Store texture size
        self.size = (width, height)
        
        # Set up uniforms for colormap
        self.shader['scalogram'] = self.SCALOGRAM_TEXTURE_UNIT
        try:
            self.shader['valueMin'] = 0.0
            self.shader['valueMax'] = 1.0
        except KeyError:
            log.warning("Scaling uniforms not found")
        
        log.info(f"Audio texture created: {width}x{height}")



    def update_texture(self, data):
        """
        Update texture with new data
        
        Args:
            data (np.ndarray): 2D array of scalogram data to upload to texture.
        """
        # Data is already downsampled from the wavelet class
        data_bytes = data.astype('f4').tobytes()

        log.debug(f"CPU→GPU: Uploading texture data ({data.shape}, f4, {len(data_bytes)} bytes)")
        self.texture.write(data_bytes)
        self.texture.use(location=self.SCALOGRAM_TEXTURE_UNIT)

        log.debug(f"Texture updated: {data.shape}, range {data.min():.3f}-{data.max():.3f}")
    
    def render(self):
        """
        Render the quad
        """
        try:
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Check for OpenGL errors
            error = self.ctx.error
            if error != 'GL_NO_ERROR':
                log.error(f"Render error: {error}")
            
        except Exception as e:
            log.error(f"Render exception: {e}")

class RollingFrameBuffer:
    def __init__(self, num_frames, height, width):
        """
        Handles circular buffer for scrolling visualization
        
        Args:
            num_frames (int): Number of frames to store
            height (int): Height of each frame (frequency bins)
            width (int): Width of each frame (time samples)
        """
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        # Store full frames (no overlap)
        self.frames = np.zeros((num_frames, height, width), dtype=np.float32)
        self.frame_index = 0
        
        # Pre-allocate flattened buffer
        self.flattened_buffer = np.zeros((height, num_frames * width), dtype=np.float32)
    
    def add_frame(self, frame_data):
        """Add new frame to circular buffer and update flattened buffer"""
        if frame_data.shape != (self.height, self.width):
            log.error(f"Frame data shape mismatch: expected {(self.height, self.width)}, got {frame_data.shape}")
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")
        
        self.frames[self.frame_index] = frame_data
        self.frame_index = (self.frame_index + 1) % self.num_frames
        
        # Update flattened buffer immediately
        self._update_flattened_buffer()
    
    def _update_flattened_buffer(self):
        """Update flattened buffer with correct coordinate mapping"""
        # Calculate the correct order of frames (oldest first)
        frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]
        
        # Use vectorized operations for better performance
        for i, frame_i in enumerate(frame_order):
            start_col = i * self.width
            end_col = start_col + self.width
            self.flattened_buffer[:, start_col:end_col] = self.frames[frame_i]
    
    def get_flattened_buffer_shape(self):
        """
        Get the shape of the flattened buffer

        Returns:
            tuple: Shape of the flattened buffer.
        """
        return self.flattened_buffer.shape
    
    def get_flattened_buffer(self):
        """
        Get time-ordered flattened buffer for texture
        
        Returns:
            np.ndarray: Time-ordered flattened buffer.
        """
        return self.flattened_buffer


# =============================================================================
# ALTERNATIVE IMPLEMENTATION: PyQtGraph-based visualizer
# =============================================================================
# This is a separate implementation that uses PyQtGraph instead of GPU shaders.
# It's kept at the bottom to clearly separate it from the main shader-based
# implementation above.
# =============================================================================

class PyQtGrapher(Plotter):
    def __init__(self, file_path: str, frame_shape: tuple[int, int]):
        """
        Traditional PyQtGraph-based audio visualizer
        """
        super().__init__(file_path, frame_shape)
        
        # PyQtGraph configuration
        pg.setConfigOptions(useOpenGL=True, enableExperimental=True)

        self.app = pg.mkQApp("Sub Shader")
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  
        self.win.setWindowTitle('Continuous Wavelet Transform')

        self.plot = self.win.addPlot(row=0, col=0, rowspan=1, colspan=1,
                                   title=file_path, enableMenu=False)

        # Configure plot appearance
        self.plot.setLabel('left', 'Frequency (Hz)')
        self.plot.setLabel('bottom', 'Time')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Set up color map for better visualization
        self.plot.setColorMap(pg.colormap.get('inferno'))
        
        # Initialize empty image item
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        
        # Set up timer for updates
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self._update_display)
        self.timer.start(16)  # ~60 FPS
        
        self.latest_data = None
    
    def update_plot(self, plot_values):
        """
        Updates the plot with new data.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        self.latest_data = plot_values
    
    def _update_display(self):
        """Update the display with latest data"""
        if self.latest_data is not None:
            self.img_item.setImage(self.latest_data)
    
    def update_fps(self, fps: int):
        """
        Updates the FPS display.

        Args:
            fps (int): The current FPS.
        """
        self.win.setWindowTitle(f'Continuous Wavelet Transform - {fps} FPS')
    
    def should_window_close(self):
        """
        Check if user wants to close the window
        """
        return self.win.isHidden()
