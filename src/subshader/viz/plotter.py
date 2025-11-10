"""
Visualization Module for SubShader.

This module provides GPU-accelerated shader-based plotting for real-time
audio visualization:
 - Renders time-frequency data using OpenGL shaders
 - Implements efficient texture-based data streaming
 - Supports customizable gamma correction and color mapping
 - Manages GLFW window lifecycle and input handling
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pyqtgraph as pg
import moderngl
import glfw

from .shaders import get_vertex_shader_source, get_fragment_shader_source
from .plot_normalizer import PlotNormalizer

from subshader.utils.logging import get_logger
from subshader.config import VisualizationConfig

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class WindowCloseException(Exception):
    """Raised when the window is closed."""
    def __init__(self, message="Window closed"):
        super().__init__(message)
        self.log_level = "warning"
        self.log_message = f"Graceful exit: {message}"

# =============================================================================
# PLOTTER CLASSES
# =============================================================================

class Plotter(ABC):
    def __init__(self, file_path: str, frame_shape: tuple[int, int]):
        """
        Abstract base class for all plotters.

        Args:
            file_path (str): Path to the file to plot.
            frame_shape (tuple[int, int]): Shape of each data frame to plot.
        """
        self.file_path = file_path
        self.frame_shape = frame_shape

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
    def __init__(self, file_path: str, frame_shape: tuple[int, int], frame_overlap: float, config: VisualizationConfig):
        """
        2D data visualization using shaders

        Args:
            file_path (str): Path to the file to plot.
            frame_shape (tuple[int, int]): Initial shape estimate - will be 
                updated with actual data
            config: Global configuration for the visualizer
        """
        super().__init__(file_path, frame_shape)
        self.config = config

        # Circular buffer to aggregate frames in chronological order
        self.frame_buffer = CircularFrameBuffer(frame_shape=self.frame_shape,
                                                                     num_frames=self.config.num_frames,
                                                                     frame_overlap=frame_overlap,
                                                                     color_norm_config=self.config.color_norm)

        # ModernGL Context - window creation and OpenGL context setup
        self.gl_context = GLContext(title=f"SubShader - {os.path.basename(file_path)}")

        # GPU Renderer - shader compilation, texture management, and rendering 
        self.renderer = Renderer(ctx=self.gl_context.ctx,
                                           texture_shape=self.frame_buffer.get_shape(),
                                           gamma=self.config.gamma)

        # log.info(f"Renderer initialized with texture shape: {frame_buffer_shape}")

    def update_plot(self, plot_values: np.ndarray):
        """
        Updates the rolling plot frame buffer with a new frame of data. Then 
        sends the entire buffer of frames to the texture. Clears the back buffer
        and renders the new graphic. Displays the graphic on the screen by
        swapping the front and back buffers.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        # Append new frame to circular buffer
        self.frame_buffer.add_frame(plot_values)

        # Upload the entire chronologically ordered buffer to the texture
        self.renderer.update_texture(self.frame_buffer.get_flattened_buffer())

        # Render pass 
        self.gl_context.clear_graphic()
        self.renderer.render_graphic()
        self.gl_context.display_graphic()

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
    def __init__(self, width=1920, height=1080, title="SubShader"):
        """
        Handles GLFW window and OpenGL context setup

        Args:
            width (int): Default window width 
            height (int): Default window height
            title (str): Window title
        """
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.ctx = None
        self._init_graphics()

    # =============================================================================
    # PUBLIC METHODS - External interface
    # =============================================================================

    def should_close(self):
        """
        Checks if the window should close based on user input.
        
        Returns:
            bool: True if the window should close, False otherwise.
        """
        return glfw.window_should_close(self.window)
    
    def display_graphic(self):
        """
        Display the rendered content (swap front/back buffers).
        """
        glfw.swap_buffers(self.window)
        glfw.poll_events()  # Process window events
    
    def clear_graphic(self, r=0.0, g=0.0, b=0.0):
        """
        Clear the OpenGL context with a specified color.
        """
        self.ctx.clear(r, g, b)

    # =============================================================================
    # PRIVATE METHODS - Internal implementation
    # =============================================================================

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

        # Create maximized window with decorations 
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.width, self.height = mode.size.width, mode.size.height
        
        log.info(f"Creating maximized window: {self.width}×{self.height}")
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        glfw.maximize_window(self.window)
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

class Renderer:

    # Texture slot that tells the fragment shader which texture to read from
    TEXTURE_SLOT = 0

    def __init__(self, ctx, texture_shape: tuple[int, int], gamma):
        """
        Main GPU rendering component that 
            - Compiles the shaders 
            - Creates the graphic geometry and connects it to the shaders
            - Creates the texture and connects it to the shaders
            - Renders the graphic 

        Args:
            ctx (moderngl.Context): The ModernGL context to use for shader 
                compilation.
            texture_shape (tuple[int, int]): Texture dimensions (height, width).
            gamma (float): The gamma correction factor.
        """
        self.ctx = ctx
        
        # Initialize core rendering components
        self.shader = self._compile_shaders(gamma)
        self.vbo, self.vao = self._setup_rendering_geometry(self.shader)
        self.texture = self._setup_texture(self.shader, texture_shape)

    # ==========================================================================
    # PRIVATE METHODS - Internal implementation
    # ==========================================================================

    def _compile_shaders(self, gamma):
        """
        Compile and link vertex (geometry) and fragment (color) shaders into 
        a GPU-executable program
        """

        vertex_shader = get_vertex_shader_source()
        fragment_shader = get_fragment_shader_source()

        log.info("Compiling shaders...")
        shader = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        log.info("Shader compilation successful!")

        shader['gamma'] = gamma

        return shader

    def _setup_rendering_geometry(self, shader: moderngl.Program):
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
        vbo = self.ctx.buffer(quad_vertices.tobytes()) 

        # Vertex Array Object tells the shader program how to use the data
        # stored in the VBO (position, color, etc.)
        vao = self.ctx.simple_vertex_array(shader, vbo, 'position')

        return vbo, vao

    def _setup_texture(self, shader: moderngl.Program, texture_shape: tuple[int, int]):
        """
        Create texture and connect it to shader uniform via texture slot

        Args:
            shader (moderngl.Program): The shader program to use for texture 
                creation.
            texture_shape (tuple[int, int]): Texture dimensions (height, width).
        """
        height, width = texture_shape

        # Create texture object in memory 
        log.info(f"Creating texture: {width}x{height} (1 channel grayscale, float32)")
        log.info(f"CPU→GPU: Allocating texture buffer ({width}×{height}, f4, {width * height * 4} bytes)")
        self.texture = self.ctx.texture((width, height), 1, dtype='f4')

        # Set texture filtering for smoothness between pixels
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Assign texture slot to shader uniform
        shader['texture_sampler'] = self.TEXTURE_SLOT
        log.info(f"Assigned texture slot {self.TEXTURE_SLOT} to shader uniform 'texture_sampler'")

        return self.texture

    def _validate_texture_data(self, texture_data):
        """
        Validate data before uploading it to the texture.
        """
        if texture_data is None:
            log.error("Texture data is None")
            return

        if not hasattr(texture_data, 'shape'):
            log.error(f"Texture data has no shape attribute: {type(texture_data)}")
            return

        if len(texture_data.shape) != 2:
            log.error(f"Expected 2D texture data, got shape: {texture_data.shape}")
            return

        if np.any(np.isnan(texture_data)):
            log.error("Texture data contains NaN values")
            return

        if np.any(np.isinf(texture_data)):
            log.error("Texture data contains infinite values")
            return

        # Validate data size matches texture size
        texture_bytes = texture_data.astype('f4').tobytes()
        expected_bytes = self.texture.size[0] * self.texture.size[1] * 4  # 4 bytes per float32
        if len(texture_bytes) != expected_bytes:
            log.error(f"Data size mismatch: got {len(texture_bytes)} bytes, expected {expected_bytes} bytes")
            log.error(f"Texture size: {self.texture.size}, Data shape: {texture_data.shape}")
            return

        return True

    def _check_gl_error(self, ctx: moderngl.Context, operation: str):
        """
        Check for OpenGL errors and log 
        """
        error = ctx.error

        if error != 'GL_NO_ERROR':
            log.error(f"GL error during '{operation}': {error}")
            return False
        else:
            log.debug(f"GL OK: {operation}")
            return True

    # ==========================================================================
    # PUBLIC METHODS - External interface
    # ==========================================================================

    def update_texture(self, texture_data):
        """
        Upload new data to texture and activate it in the assigned slot

        Args:
            texture_data (np.ndarray): 2D array of data to upload to texture.
        """
        # Validate data before upload
        if not self._validate_texture_data(texture_data):
            return

        if not self._check_gl_error(self.ctx, "before texture write"):
            return

        # Convert to bytes and upload to texture
        texture_bytes = texture_data.astype('f4').tobytes()
        self.texture.write(texture_bytes)

        log.debug(f"Texture size: {self.texture.size}, Expected data size: {self.texture.size[0] * self.texture.size[1] * 4} bytes")
        log.debug(f"CPU→GPU: Uploaded texture data ({texture_data.shape}, f4, {len(texture_bytes)} bytes)")

        # Check for OpenGL errors after texture write
        if not self._check_gl_error(self.ctx, "after texture write"):
            return

        self.texture.use(location=self.TEXTURE_SLOT)

        log.debug(f"Texture updated: {texture_data.shape}, range {texture_data.min():.3f}-{texture_data.max():.3f}")

    def render_graphic(self):
        """
        Render the quad - this one-shots the graphics pipeline from the source
        data stored in the texture to the back buffer.
        """
        try:
            if not self._check_gl_error(self.ctx, "before rendering"):
                return

            # Ensure texture is bound
            self.texture.use(location=self.TEXTURE_SLOT)

            if not self._check_gl_error(self.ctx, "after texture binding"):
                return

            self.vao.render(moderngl.TRIANGLE_STRIP)

            if not self._check_gl_error(self.ctx, "after rendering"):
                return

        except Exception as e:
            log.error(f"Render exception: {e}")

class CircularFrameBuffer:
    def __init__(self, frame_shape, num_frames, frame_overlap, color_norm_config):
        """
        Handles circular buffer for scrolling visualization

        Args:
            num_frames (int): Number of frames to store
            height (int): Height of each frame (frequency bins)
            width (int): Width of each frame (time samples)
        """
        self.num_frames = num_frames 
        self.height, self.width = frame_shape

        # TODO 36 - NOW how do we handle the frame overlap, reliable region, and 
        # downsampling?

        # TODO 36 Wait I don't see how we're handling for the overlap here, I 
        # think I'm just getting lucky 
        self.frame_overlap = frame_overlap

        log.info(f"Plotting {self.num_frames} {frame_shape} sized frames")

        # Store full frames (no overlap)
        self.frames = np.zeros((num_frames, self.height, self.width), dtype=np.float32)
        self.frame_index = 0

        # Pre-allocate flattened buffer
        self.flattened_buffer = np.zeros((self.height, self.width * num_frames), dtype=np.float32)

        self.plot_normalizer = PlotNormalizer(percentile=color_norm_config.percentile,
                                              decay_rate=color_norm_config.decay_rate,
                                              floor_value=color_norm_config.floor_value,
                                              warmup_frames=color_norm_config.warmup_frames,
                                              log_mapping=color_norm_config.log_mapping)

    # ==========================================================================
    # PUBLIC METHODS - External interface
    # ==========================================================================

    def add_frame(self, frame_data):
        """Add new frame to circular buffer and update flattened buffer"""
        if frame_data.shape != (self.height, self.width):
            log.error(f"Frame data shape mismatch: expected {(self.height, self.width)}, got {frame_data.shape}")
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")

        self.frames[self.frame_index] = self.plot_normalizer.process(frame_data)
        self.frame_index = (self.frame_index + 1) % self.num_frames

        # Calculate the correct order of frames (oldest first)
        frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]

        # Use vectorized operations for better performance
        for i, frame_i in enumerate(frame_order):
            start_col = i * self.width
            end_col = start_col + self.width
            self.flattened_buffer[:, start_col:end_col] = self.frames[frame_i]

    def get_shape(self):
        """
        Get the shape of the entire, flattened frame buffer

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

class PyQtPlotter(Plotter):
    def __init__(self, file_path: str, frame_shape: tuple[int, int]):
        """
        Traditional PyQtGraph-based audio visualizer

        Args:
            file_path (str): Path to the file to plot.
            frame_shape (tuple[int, int]): Shape of each data frame to plot.
        """
        super().__init__(file_path, frame_shape)

        # PyQtGraph configuration
        pg.setConfigOptions(useOpenGL=True, enableExperimental=True)

        self.app = pg.mkQApp("Sub Shader")
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  
        self.win.setWindowTitle('Continuous Wavelet Transform')

        self.plot = self.win.addPlot(
            row=0, col=0, rowspan=1, colspan=1, title=file_path, enableMenu=False)

        # Configure plot appearance
        self.plot.setLabel('left', 'Frequency (Hz)')
        self.plot.setLabel('bottom', 'Time')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Set up color map for better visualization (apply to ImageItem)
        # PlotItem does not support setColorMap; ImageItem does.
        cmap = pg.colormap.get('inferno')
        # setColorMap is available on ImageItem in modern pyqtgraph versions
        if hasattr(pg.ImageItem, 'setColorMap'):
            # Will be set after ImageItem is created
            pass
        else:
            # Fallback for very old pyqtgraph: use lookup table
            lut = cmap.getLookupTable(alpha=False)
            # Will be applied to ImageItem after creation
            self._fallback_lut = lut
        
        # Initialize empty image item
        self.img_item = pg.ImageItem()
        # Apply colormap to the image item
        try:
            if hasattr(self.img_item, 'setColorMap'):
                self.img_item.setColorMap(cmap)
            elif hasattr(self, '_fallback_lut'):
                self.img_item.setLookupTable(self._fallback_lut)
        except Exception:
            # Non-fatal if colormap application fails
            pass
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
