"""
SubShader plotter module.

Handles real-time GPU rendering for the 2D data visualization:
 - Creates the window and OpenGL context with GLFW and ModernGL
 - Stores each 2D plot frame chronologically in a circular buffer 
 - Renders using a shader program all the frames as a single texture and applies
   minor gamma color correction 
 - Includes a deprecated PyQtGraph implementation for legacy or debug use
"""

# =============================================================================
# IMPORTS
# =============================================================================

from abc import ABC, abstractmethod

import glfw
import moderngl
import numpy as np
import os
import pyqtgraph as pg

from subshader.utils.logging import get_logger
from subshader.config import VisualizationConfig
from subshader.exceptions import WindowCloseException

from .shaders import get_vertex_shader_source, get_fragment_shader_source
from .plot_normalizer import IntensityTracker

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# PLOTTER CLASSES
# =============================================================================

class Plotter(ABC):
    def __init__(self, file_path: str, frame_shape: tuple[int, int]):
        """
        Abstract base class for all plotters.

        Args:
            file_path (str): Path to the data file to plot.
            frame_shape (tuple[int, int]): Shape of each data frame to plot.
        """
        self.file_path = file_path
        self.frame_shape = frame_shape

    @abstractmethod
    def update_plot(self, plot_values) -> None:
        """
        Abstract method to update the plot with new data.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        pass

    @abstractmethod
    def should_window_close(self) -> None:
        """Check if the window should close based on user input."""
        pass

class ShaderPlot(Plotter):
    def __init__(self, file_path: str, frame_shape: tuple[int, int], config: VisualizationConfig):
        """
        2D data visualization using shaders

        Args:
            file_path (str): Path to the file to plot.
            frame_shape (tuple[int, int]): The shape of each individual frame to
                add to the frame buffer
            config: Global configuration for the visualizer
        """
        super().__init__(file_path, frame_shape)

        self.config = config

        # Circular buffer to store data frames in chronological order
        self.frame_buffer = CircularFrameBuffer(frame_shape=self.frame_shape,
                                                num_frames=self.config.num_frames,
                                                color_norm_config=self.config.color_norm)

        # ModernGL Context - window creation and OpenGL context setup
        self.gl_context = GLContext(title=f"SubShader - {os.path.basename(file_path)}")

        # GPU Renderer - shader compilation, texture management, and rendering 
        self.renderer = Renderer(ctx=self.gl_context.ctx,
                                 texture_shape=self.frame_buffer.get_shape(),
                                 gamma=self.config.gamma)

    def update_plot(self, plot_values: np.ndarray) -> None:
        """
        Updates the circular buffer with a new frame of plot data. Then 
        sends the entire buffer (oldest on the left, newest on the right) of 
        frames to the texture. Clears the back bufferand renders the new 
        graphic. Displays the graphic on the screen by swapping the front and 
        back buffers.

        Args:
            plot_values (np.ndarray): The new data to plot.
        """
        # Append new frame to circular buffer (also updates intensity tracker)
        self.frame_buffer.push_frame(plot_values)

        # Upload and render entire chronologically ordered buffer to the texture
        self.renderer.update_texture(self.frame_buffer.get_flattened_buffer())
        self.renderer.set_intensity_max(self.frame_buffer.get_intensity_max())
        self.gl_context.clear_graphic()
        self.renderer.render_graphic()
        self.gl_context.display_graphic()

    def should_window_close(self) -> bool:
        """
        Check if user wants to close the window
        """  
        return self.gl_context.should_close()

    def cleanup(self) -> None:
        """
        Clean shutdown
        """
        glfw.terminate()

class GLContext:
    def __init__(self, title="SubShader"):
        """
        Handles GLFW window and OpenGL context setup

        Args:
            width (int): Default window width 
            height (int): Default window height
            title (str): Window title
        """
        self.window: object | None = None
        self.ctx: moderngl.Context | None = None

        self.window, width, height = self._init_window(title)

        # TODO-36 : self.ctx #3?
        self.ctx = self._init_opengl_context(width, height)

    # =========================================================================
    # PUBLIC METHODS - External interface
    # =========================================================================

    def should_close(self) -> bool:
        """
        Checks if the window should close based on user input.
        
        Returns:
            bool: True if the window should close, False otherwise.
        """
        return glfw.window_should_close(self.window)
    
    def display_graphic(self) -> None:
        """
        Display the rendered content (swap front/back buffers).
        """
        glfw.swap_buffers(self.window)
        glfw.poll_events()  # Process window events
    
    def clear_graphic(self, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None:
        """
        Clear the OpenGL context with a specified color.
        """
        self.ctx.clear(r, g, b)

    # =========================================================================
    # PRIVATE METHODS - Internal implementation
    # =========================================================================

    def _init_window(self, title: str) -> tuple[object, int, int]:
        """
        GLFW is a cross-platform library used for creating windows with OpenGL 
        contexts and handling input. It's the way OpenGL displays the graphics
        onto the screen.

        Args:
            title (str): Title of the window

        Returns:
            tuple[glfw.Window, int, int]: The window, width, and height.
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

        width, height = mode.size.width, mode.size.height

        log.info(f"Creating maximized window: {width}×{height}")

        window = glfw.create_window(width, height, title, None, None)
        glfw.maximize_window(window)

        if not window:
            # Clean up GLFW before failing
            glfw.terminate()
            log.error("GLFW window creation failed")
            raise RuntimeError("Failed to create window")

        # Make OpenGL context current for this thread before any OpenGL calls
        glfw.make_context_current(window)

        return window, width, height

    def _init_opengl_context(self, view_width: int, view_height: int) -> moderngl.Context:
        """
        ModernGL is a Python wrapper around OpenGL that provides a more
        pythonic interface for OpenGL calls. It allows us to create shaders,
        buffers, textures, and other OpenGL objects without dealing with the
        low-level OpenGL API directly.  
        Args:
            view_width (int): Width of the viewport
            view_height (int): Height of the viewport

        Returns:
            moderngl.Context: The ModernGL context.
        """
        ctx = moderngl.create_context()
        
        # Log OpenGL info for debugging
        log.info(f"OpenGL Version: {ctx.info['GL_VERSION']}")
        log.debug(f"Viewport: {ctx.viewport}")

        # Setup viewport (area of window where OpenGL renders) to match window 
        ctx.viewport = (0, 0, view_width, view_height)
        
        # Disable depth testing (z-values) since rendering 2D content only 
        ctx.disable(moderngl.DEPTH_TEST)
        
        # Disable face culling - we want to see both sides of triangles
        # Face culling normally hides back-facing triangles for performance
        ctx.disable(moderngl.CULL_FACE)
        
        log.info("Graphics context initialized successfully")

        return ctx
    
    def _setup_glfw_error_callback(self) -> None:
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

    def __init__(self, ctx: moderngl.Context, texture_shape: tuple[int, int], gamma: float):
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

    # =========================================================================
    # PRIVATE METHODS - Internal implementation
    # =========================================================================

    def _compile_shaders(self, gamma: float) -> moderngl.Program:
        """
        Compile and link vertex (geometry) and fragment (color) shaders into 
        a GPU-executable program

        Args:
            gamma (float): The gamma correction factor.

        Returns:
            moderngl.Program: The compiled shader program.
        """

        vertex_shader = get_vertex_shader_source()
        fragment_shader = get_fragment_shader_source()

        log.info("Compiling shaders...")
        # Why isn't context passed in as an arg?
        shader = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        log.info("Shader compilation successful!")

        shader['gamma'] = gamma
        shader['intensity_max'] = 1.0  # Initial value, updated each frame

        return shader

    def _setup_rendering_geometry(self, shader: moderngl.Program) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
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

    def _setup_texture(self, shader: moderngl.Program, texture_shape: tuple[int, int]) -> moderngl.Texture:
        """
        Create texture and connect it to shader uniform via texture slot

        Args:
            shader (moderngl.Program): The shader program to use for texture 
                creation.
            texture_shape (tuple[int, int]): Texture dimensions (height, width).
        """
        height, width = texture_shape

        # Create texture object in memory 
        self.texture = self.ctx.texture((width, height), 1, dtype='f4')
        log.info(f"Creating texture: {width}x{height} (1 channel grayscale, float32)")
        log.info(f"CPU→GPU: Allocating texture buffer with {width * height * 4} bytes)")

        # Set texture filtering for smoothness between pixels
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Assign texture slot to shader uniform
        shader['texture_sampler'] = self.TEXTURE_SLOT
        log.info(f"Assigned texture slot {self.TEXTURE_SLOT} to shader uniform 'texture_sampler'")

        return self.texture

    def _validate_texture_data(self, texture_data: np.ndarray) -> None:
        """
        Validate data before uploading it to the texture.

        Raises:
            ValueError: If texture data is invalid.
        """
        if texture_data is None:
            raise ValueError("Texture data is None — cannot upload to GPU texture")

        if not hasattr(texture_data, 'shape'):
            raise ValueError(f"Texture data has no shape attribute: {type(texture_data)}")

        if len(texture_data.shape) != 2:
            raise ValueError(f"Expected 2D texture data, got shape: {texture_data.shape}")

        if np.any(np.isnan(texture_data)):
            raise ValueError("Texture data contains NaN values")

        if np.any(np.isinf(texture_data)):
            raise ValueError("Texture data contains infinite values")

        # Validate data size matches texture size
        texture_bytes = texture_data.astype('f4').tobytes()
        expected_bytes = self.texture.size[0] * self.texture.size[1] * 4
        if len(texture_bytes) != expected_bytes:
            raise ValueError(
                f"Data size mismatch: got {len(texture_bytes)} bytes, "
                f"expected {expected_bytes} bytes. "
                f"Texture size: {self.texture.size}, Data shape: {texture_data.shape}"
            )

    def _check_gl_error(self, ctx: moderngl.Context, operation: str) -> bool:
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

    # =========================================================================
    # PUBLIC METHODS - External interface
    # =========================================================================

    def update_texture(self, texture_data: np.ndarray) -> None:
        """
        Upload new data to texture and activate it in the assigned slot

        Args:
            texture_data (np.ndarray): 2D array of data to upload to texture.
        """
        # Validate data before upload (raises ValueError on invalid data)
        self._validate_texture_data(texture_data)

        if not self._check_gl_error(self.ctx, "before texture write"):
            raise RuntimeError("OpenGL error before texture write")

        # Convert to bytes and upload to texture
        texture_bytes = texture_data.astype('f4').tobytes()
        self.texture.write(texture_bytes)

        log.debug(f"Texture size: {self.texture.size}, Expected data size: {self.texture.size[0] * self.texture.size[1] * 4} bytes")
        log.debug(f"CPU→GPU: Uploaded texture data ({texture_data.shape}, f4, {len(texture_bytes)} bytes)")

        if not self._check_gl_error(self.ctx, "after texture write"):
            raise RuntimeError("OpenGL error after texture write")

        log.debug(f"Texture updated: {texture_data.shape}, range {texture_data.min():.3f}-{texture_data.max():.3f}")

    def render_graphic(self) -> None:
        """
        Render the quad - this one-shots the graphics pipeline from the source
        data stored in the texture to the back buffer.
        """
        try:
            if not self._check_gl_error(self.ctx, "before rendering"):
                raise RuntimeError("OpenGL error before rendering")

            # Ensure texture is bound
            self.texture.use(location=self.TEXTURE_SLOT)

            if not self._check_gl_error(self.ctx, "after texture binding"):
                raise RuntimeError("OpenGL error after texture binding")

            self.vao.render(moderngl.TRIANGLE_STRIP)

            if not self._check_gl_error(self.ctx, "after rendering"):
                raise RuntimeError("OpenGL error after rendering")

        except Exception as e:
            log.error(f"Render exception: {e}")
            raise

    def set_intensity_max(self, intensity_max: float) -> None:
        """
        Update the intensity_max uniform for colormap scaling.

        Args:
            intensity_max: The global max intensity value for normalization.
        """
        self.shader['intensity_max'] = max(intensity_max, 1e-8)  # Avoid division by zero

class CircularFrameBuffer:
    def __init__(self, frame_shape: tuple[int, int], num_frames: int, color_norm_config) -> None:
        """
        Handles circular buffer for scrolling visualization

        Args:
            num_frames (int): Number of frames to store
            height (int): Height of each frame (frequency bins)
            width (int): Width of each frame (time samples)
        """
        self.num_frames = num_frames
        self.height, self.width = frame_shape

        log.info(f"Plotting {self.num_frames} {frame_shape} sized frames")

        # Store full frames (no overlap)
        self.frames = np.zeros((num_frames, self.height, self.width), dtype=np.float32)
        self.frame_index = 0

        # Pre-allocate flattened buffer
        self.flattened_buffer = np.zeros((self.height, self.width * num_frames), dtype=np.float32)

        self.intensity_tracker = IntensityTracker(percentile=color_norm_config.percentile,
                                                  decay_rate=color_norm_config.decay_rate,
                                                  floor_value=color_norm_config.floor_value,
                                                  warmup_frames=color_norm_config.warmup_frames)

    # =========================================================================
    # PUBLIC METHODS - External interface
    # =========================================================================

    def push_frame(self, frame_data) -> None:
        """
        Add new frame to circular buffer and update flattened buffer

        Args:
            frame_data (np.ndarray): The new frame data to add to the circular
                buffer.
        """
        if frame_data.shape != (self.height, self.width):
            log.error(f"Frame data shape mismatch: expected {(self.height, self.width)}, got {frame_data.shape}")
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")

        self.intensity_tracker.update(frame_data)
        self.frames[self.frame_index] = frame_data

        # Calculate the correct order of frames (oldest first)
        self.frame_index = (self.frame_index + 1) % self.num_frames
        frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]

        # Populate the flattened buffer with the correct order of frames
        for i, frame_i in enumerate(frame_order):
            start_col = i * self.width
            end_col = start_col + self.width
            self.flattened_buffer[:, start_col:end_col] = self.frames[frame_i]

    def pop_frame(self, index: int) -> np.ndarray:
        """
        Get a specific frame from the circular buffer
        """
        self.frame_index = (self.frame_index - 1) % self.num_frames
        return self.frames[self.frame_index]

    def get_shape(self) -> tuple[int, int]:
        """
        Get the shape of the entire, flattened frame buffer

        Returns:
            tuple: Shape of the flattened buffer.
        """
        return self.flattened_buffer.shape

    def get_flattened_buffer(self) -> np.ndarray:
        """
        Get time-ordered flattened buffer for texture

        Returns:
            np.ndarray: Time-ordered flattened buffer.
        """
        return self.flattened_buffer

    def get_intensity_max(self) -> float:
        """
        Get the tracked global max intensity for colormap scaling.

        Returns:
            float: The global max value to use as vmax in colormap.
        """
        return self.intensity_tracker.global_max


class AudioFrameBuffer:
    """
    Circular buffer for 1D audio chunks.
    
    Stores audio chunks chronologically and outputs a flattened 1D array
    of all samples in time order (oldest first, newest last).
    """
    
    def __init__(self, chunk_size: int, num_chunks: int) -> None:
        """
        Initialize audio circular buffer.

        Args:
            chunk_size: Number of samples per audio chunk
            num_chunks: Number of chunks to store in the buffer
        """
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.total_samples = chunk_size * num_chunks
        
        # Store chunks as 2D array (num_chunks, chunk_size) for easy indexing
        self.chunks = np.zeros((num_chunks, chunk_size), dtype=np.float32)
        self.chunk_index = 0
        
        # Pre-allocate flattened output buffer
        self.flattened_buffer = np.zeros(self.total_samples, dtype=np.float32)
        
        log.info(f"AudioFrameBuffer: {num_chunks} chunks × {chunk_size} samples = {self.total_samples} total samples")

    def push_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add new audio chunk to circular buffer.

        Args:
            audio_chunk: 1D array of audio samples (length must match chunk_size)
        """
        if audio_chunk.shape[0] != self.chunk_size:
            raise ValueError(f"Expected chunk size {self.chunk_size}, got {audio_chunk.shape[0]}")
        
        # Store chunk at current index
        self.chunks[self.chunk_index] = audio_chunk
        
        # Advance index
        self.chunk_index = (self.chunk_index + 1) % self.num_chunks
        
        # Rebuild flattened buffer in chronological order (oldest first)
        chunk_order = [(self.chunk_index + i) % self.num_chunks for i in range(self.num_chunks)]
        for i, idx in enumerate(chunk_order):
            start = i * self.chunk_size
            end = start + self.chunk_size
            self.flattened_buffer[start:end] = self.chunks[idx]

    def get_flattened_buffer(self) -> np.ndarray:
        """
        Get all audio samples in chronological order.

        Returns:
            1D array of all samples (oldest first, newest last)
        """
        return self.flattened_buffer
    
    def get_chunk(self, index: int) -> np.ndarray:
        """
        Get a specific chunk by its buffer index.

        Args:
            index: Index into the circular buffer (0 = oldest, num_chunks-1 = newest)
        
        Returns:
            1D array of audio samples for that chunk
        """
        actual_index = (self.chunk_index + index) % self.num_chunks
        return self.chunks[actual_index]

    def get_downsampled(self, target_width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get min/max envelope downsampled to target width for visualization.
        
        This preserves the visual shape of the waveform by keeping the min and max
        values for each window, which is what audio editors like Audacity use.

        Args:
            target_width: Number of output points (typically matches spectrogram width)
        
        Returns:
            Tuple of (x_values, y_min, y_max) for plotting with fill_between
        """
        audio = self.flattened_buffer
        num_samples = len(audio)
        
        if target_width >= num_samples:
            # No downsampling needed
            x = np.arange(num_samples)
            return x, audio, audio
        
        # Calculate window size
        window_size = num_samples // target_width
        
        # Trim to exact multiple of window_size
        trimmed_len = window_size * target_width
        trimmed_audio = audio[:trimmed_len]
        
        # Reshape into windows and compute min/max per window
        windowed = trimmed_audio.reshape(target_width, window_size)
        y_min = windowed.min(axis=1)
        y_max = windowed.max(axis=1)
        
        # X values at center of each window
        x = np.arange(target_width) * window_size + window_size // 2
        
        return x, y_min, y_max


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
