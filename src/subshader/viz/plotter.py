from abc import ABC, abstractmethod

import numpy as np
import moderngl
import glfw
import pyqtgraph as pg
import logging
from .shaders import get_vertex_shader_source, get_fragment_shader_source

# TODO ISSUE-41 Move this to main
# Set up clean logging - file only, no console spam
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shader_debug.log', mode='w')  # Overwrite log each run
    ]
)

logger = logging.getLogger('ShaderPlotter')

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
            raise ValueError(f"Expected 2D array, got {len(frame_shape)}D with shape {frame_shape}")        
        if frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError(f"2D array cannot have shape: {frame_shape}")

        self.frame_shape = frame_shape
        self.y_n, self.x_n = self.frame_shape

    @abstractmethod
    def update_plot(self, values):
        """
        Abstract method to update the plot with new data.

        Args:
            values (np.ndarray): The new data to plot.
        """
        pass

    @abstractmethod
    def should_window_close(self):
        """Check if the window should close based on user input."""
        pass

class Shader(Plotter):
    def __init__(self, file_path: str, frame_shape: tuple[int, int], num_frames: int = 64):
        """
        Clean, high-level audio visualization using GPU shaders

        Args:
            file_path (str): The path to the file to plot.
            frame_shape (tuple[int, int]): The shape of each data frame to plot.
            num_frames (int): The number of frames to use for the visualization.
        """
        super().__init__(file_path, frame_shape)

        # Create GL Context and Shader Renderer       
        logger.info(f"Initializing Shader for {file_path}")
        self.gl_context = GLContext(title=f"Audio Visualizer - {file_path}")

        # TODO NOW look at how we pass in the texture width and height into the ShaderRenderer but then the same data goes into the scrolling buffer - tuck it in? maybye
        texture_width = self.x_n * num_frames
        texture_height = self.y_n
        
        # Main GPU rendering component - handles shader compilation, 
        # texture management, and rendering
        self.shader_renderer = ShaderRenderer(
            self.gl_context.ctx, texture_width, texture_height
        )
        self.scrolling_buffer = ScrollingBuffer(num_frames, self.y_n, self.x_n)
        self.scaling = AdaptiveScaling()
        
        logger.info("Shader system ready")
        print("Visualizer started - logs in 'shader_debug.log'")
    
    def update_plot(self, values: np.ndarray):
        """
        Updates the scrolling plot with new data.

        Args:
            values (np.ndarray): The new data to plot.
        """
        self.scrolling_buffer.add_frame(values)
        buffer_data = self.scrolling_buffer.get_flattened_buffer()
        
        # Update adaptive scaling
        value_min, value_max = self.scaling.update_range(buffer_data)
        
        # Update scaling uniforms for colormap
        try:
            self.shader_renderer.shader_program['valueMin'] = value_min
            self.shader_renderer.shader_program['valueMax'] = value_max
        except KeyError:
            pass  # Uniforms might be optimized out
        
        # Update texture
        self.shader_renderer.update_texture(buffer_data)
        
        # Clear and render
        self.gl_context.clear()
        self.shader_renderer.render()
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
    def __init__(self, width=800, height=600, title="Audio Visualizer"):
        """
        Handles GLFW window and OpenGL context setup
        """
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.ctx = None
        self._init_graphics()
    
    def _init_graphics(self):
        """
        GLFW is a cross-platform library used for creating windows with OpenGL 
        contexts and handling input. It's the way OpenGL displays the graphics
        onto the screen.
        """
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Set OpenGL context version hints before creating window
        # Request OpenGL 3.3 Core Profile for modern shader support
        # Core profile (no deprecated features)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  

        # Create window - monitor (None = windowed), share context (None = no sharing)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            # Clean up GLFW before failing
            glfw.terminate()
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
        logger.info(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
        logger.debug(f"Viewport: {self.ctx.viewport}")

        # Setup viewport (area of window where OpenGL renders) to match window 
        self.ctx.viewport = (0, 0, self.width, self.height)
        
        # Disable depth testing (z-values) since rendering 2D content only 
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Disable face culling - we want to see both sides of triangles
        # Face culling normally hides back-facing triangles for performance
        self.ctx.disable(moderngl.CULL_FACE)
        
        logger.info("Graphics context initialized successfully")
    
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
        self.shader_program = self._compile_shaders()
        
        # Bind the graphics geometry to the shader program
        self._setup_rendering_geometry()

        # Setup texture TODO ISSUE-33 NOW Clean comment and clean up setup texture and update texture
        self._setup_texture(texture_width, texture_height)
    
    def _compile_shaders(self):
        """
        Compile and link vertex (geometry) and fragment (color) shaders into 
        a GPU-executable program
        """
        vertex_shader = get_vertex_shader_source()
        fragment_shader = get_fragment_shader_source()
        
        logger.info("Compiling shaders...")
        logger.debug(f"Fragment shader preview: {fragment_shader[:100]}...")
        
        program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        logger.info("Shader compilation successful!")
        
        return program

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
        self.vbo = self.ctx.buffer(quad_vertices.tobytes()) 

        # Vertex Array Object tells the shader program how to use the data
        # stored in the VBO (position, color, etc.)
        self.vao = self.ctx.simple_vertex_array(self.shader_program, self.vbo, 'position')

    def _setup_texture(self, width, height):
        """
        Create 2D texture for scalogram data
        
        Args:
            width (int): Width of the texture.
            height (int): Height of the texture.
        """
        # TODO ISSUE-33 NOW Clean comment and clean up setup texture and update texture
        # Use a much smaller texture size to avoid WSL/OpenGL issues
        small_width = min(width, 1024)  # Max 1024 pixels wide
        small_height = min(height, 256)  # Max 256 pixels tall
        
        logger.info(f"Creating smaller texture: {small_width}x{small_height} (original: {width}x{height})")
        
        # Create texture - 1 = single channel (grayscale), f4 = float32
        # This allocates GPU memory for the texture
        self.texture = self.ctx.texture((small_width, small_height), 1, dtype='f4')

        # Linear interpolation for smooth scaling if the texture is resized
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Store original and actual sizes
        self.original_size = (width, height)
        self.actual_size = (small_width, small_height)
        
        # Set up uniforms for colormap
        self.shader_program['scalogram'] = self.SCALOGRAM_TEXTURE_UNIT
        try:
            self.shader_program['valueMin'] = 0.0
            self.shader_program['valueMax'] = 1.0
        except KeyError:
            logger.warning("Scaling uniforms not found")
        
        logger.info(f"Audio texture created: {small_width}x{small_height}")
    
    def update_texture(self, data):
        """
        Update texture with new data
        
        Args:
            data (np.ndarray): 2D array of scalogram data to upload to texture.
        """
        # TODO ISSUE-33 NOW Clean comment and clean up setup texture and update texture
        height, width = data.shape
        target_width, target_height = self.actual_size
        
        # Downsample the data to fit the smaller texture
        if height != target_height or width != target_width:
            # Simple downsampling by taking every Nth sample
            h_step = max(1, height // target_height)
            w_step = max(1, width // target_width)
            
            downsampled = data[::h_step, ::w_step]
            
            # Crop or pad to exact size
            if downsampled.shape[0] > target_height:
                downsampled = downsampled[:target_height, :]
            if downsampled.shape[1] > target_width:
                downsampled = downsampled[:, :target_width]
                
            # Pad if needed
            if downsampled.shape != (target_height, target_width):
                padded = np.zeros((target_height, target_width), dtype=np.float32)
                h, w = downsampled.shape
                padded[:h, :w] = downsampled
                downsampled = padded
            
            data = downsampled
        
        # Send data to GPU connect the ??? texture to the shader ?? 
        self.texture.write(data.astype('f4').tobytes())
        self.texture.use(location = self.SCALOGRAM_TEXTURE_UNIT)
        
        logger.debug(f"Downsampled texture updated: {data.shape}, range {data.min():.3f}-{data.max():.3f}")
    
    def render(self):
        """
        Render the quad
        """
        try:
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Check for OpenGL errors
            error = self.ctx.error
            if error != 'GL_NO_ERROR':
                logger.error(f"Render error: {error}")
            
        except Exception as e:
            logger.error(f"Render exception: {e}")

class ScrollingBuffer:
    def __init__(self, num_frames, height, width):
        """
        Handles circular buffer for scrolling visualization
        """
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frames = np.zeros((num_frames, height, width), dtype=np.float32)
        self.frame_index = 0
        
        # Pre-allocate flattened buffer to avoid expensive reshape/roll operations
        self.flattened_buffer = np.zeros((height, num_frames * width), dtype=np.float32)
        self._update_flattened_buffer()
    
    def add_frame(self, frame_data):
        """Add new frame to circular buffer"""
        if frame_data.shape != (self.height, self.width):
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")
        
        self.frames[self.frame_index] = frame_data
        self.frame_index = (self.frame_index + 1) % self.num_frames
        
        # Update flattened buffer incrementally
        self._update_flattened_buffer()
    
    def _update_flattened_buffer(self):
        """Update flattened buffer without expensive roll/reshape operations"""
        # Calculate the correct order of frames
        frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]
        
        # Copy frames in correct order directly to flattened buffer
        for i, frame_i in enumerate(frame_order):
            start_col = i * self.width
            end_col = start_col + self.width
            self.flattened_buffer[:, start_col:end_col] = self.frames[frame_i]
    
    def get_flattened_buffer(self):
        """
        Get time-ordered flattened buffer for texture
        
        Returns:
            np.ndarray: Time-ordered flattened buffer.
        """
        return self.flattened_buffer

class AdaptiveScaling:
    def __init__(self, adaptation_rate=0.05, decay_rate=0.999, headroom=1.2):
        """
        Handles dynamic range scaling for audio visualization
        
        Args:
            adaptation_rate (float): Rate at which the global maximum is updated.
            decay_rate (float): Rate at which the global maximum decays when 
                no new peaks are detected.
        """
        self.adaptation_rate = adaptation_rate
        self.decay_rate = decay_rate
        self.headroom = headroom
        self.global_max = 0.01
        self.global_min = 0.0
    
    def update_range(self, data):
        """Update scaling range based on current data"""
        current_max = np.max(data)
        
        if current_max > self.global_max:
            self.global_max = (self.adaptation_rate * current_max + 
                             (1 - self.adaptation_rate) * self.global_max)
        else:
            self.global_max = (self.decay_rate * self.global_max + 
                             (1 - self.decay_rate) * current_max)
        
        self.global_max = max(self.global_max, 0.001)
        
        return self.global_min, self.global_max * self.headroom


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
    
    def update_plot(self, values):
        """
        Updates the plot with new data.

        Args:
            values (np.ndarray): The new data to plot.
        """
        self.latest_data = values
    
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
