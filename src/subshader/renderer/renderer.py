"""
Renderer module — OpenGL-based real-time visualization.

Renderer   — top-level orchestrator: creates window, frame buffer, GPU renderer,
             and drives the render loop each frame.
GLContext  — GLFW window and OpenGL context lifecycle.
GPURenderer — low-level shader compilation, texture management, and draw calls.
"""

import os
from pathlib import Path

import glfw
import moderngl
import numpy as np

from subshader.utils.logging import get_logger
from subshader.utils.timing import timed
from subshader.config import RendererConfig
from subshader.exceptions import WindowCloseException
from .frame_buffer import CircularFrameBuffer

log = get_logger(__name__)


# =============================================================================
# GL CONTEXT
# =============================================================================

class GLContext:
    def __init__(self, title="SubShader"):
        """
        Handles GLFW window and OpenGL context setup

        Args:
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


# =============================================================================
# GPU RENDERER (low-level)
# =============================================================================

class GPURenderer:
    """
    Low-level GPU rendering component.

    Compiles shaders, manages geometry buffers, manages the texture, and
    issues draw calls. Operates entirely within a provided ModernGL context.
    """

    # Texture slot that tells the fragment shader which texture to read from
    TEXTURE_SLOT = 0

    def __init__(self, ctx: moderngl.Context, texture_shape: tuple[int, int], gamma: float):
        """
        Args:
            ctx (moderngl.Context): The ModernGL context to use for shader compilation.
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
        shader_dir = Path(__file__).parent / "shaders"
        vertex_source = (shader_dir / "vertex.glsl").read_text()
        fragment_source = (shader_dir / "fragment.glsl").read_text()

        log.info("Compiling shaders...")
        shader = self.ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
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
            shader (moderngl.Program): The shader program to use for texture creation.
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


# =============================================================================
# RENDERER (top-level orchestrator)
# =============================================================================

class Renderer:
    def __init__(self, frame_shape: tuple[int, int], config: RendererConfig):
        """
        Real-time GPU visualization orchestrator.

        Creates the GLFW window, frame buffer, and GPU rendering pipeline.
        Drives the full render cycle each frame: push frame → upload texture →
        set intensity max → clear → render → display.

        Args:
            file_path (str): Path to the audio file being visualized (used for window title).
            frame_shape (tuple[int, int]): Shape (height, width) of each CWT frame.
            config (RendererConfig): Renderer configuration (num_frames, color norm, gamma).
        """
        self.frame_shape = frame_shape
        self.config = config
        self.file_path = config.file_path

        # Fixed intensity reference — set by set_fixed_intensity_max() before run()
        self._fixed_intensity_max = 1.0

        # Circular buffer to store data frames in chronological order
        self.frame_buffer = CircularFrameBuffer(frame_shape=self.frame_shape,
                                                num_frames=self.config.num_frames)

        # ModernGL Context - window creation and OpenGL context setup
        self.gl_context = GLContext(title=f"SubShader - {os.path.basename(self.file_path)}")

        # GPU Renderer - shader compilation, texture management, and rendering
        self.gpu_renderer = GPURenderer(ctx=self.gl_context.ctx,
                                        texture_shape=self.frame_buffer.get_shape(),
                                        gamma=self.config.color_norm.gamma)

    @timed
    def update(self, coefs: np.ndarray) -> None:
        """
        Push a new CWT frame and render it.

        Appends the frame to the circular buffer, uploads the full buffer to
        the GPU texture, and issues a complete render cycle.

        Args:
            coefs (np.ndarray): 2D CWT coefficient array (height × width).
        """
        # Append new frame to circular buffer
        self.frame_buffer.push_frame(coefs)

        # Upload and render entire chronologically ordered buffer to the texture.
        # intensity_max was set once via set_fixed_intensity_max() — shader uniform persists.
        self.gpu_renderer.update_texture(self.frame_buffer.get_flattened_buffer())
        self.gl_context.clear_graphic()
        self.gpu_renderer.render_graphic()
        self.gl_context.display_graphic()

    def set_fixed_intensity_max(self, value: float) -> None:
        """
        Set the fixed intensity normalization reference for the shader.

        Called once after pre-scan, before the render loop starts. The shader
        uniform is set here and never updated during playback.

        Args:
            value: Fixed intensity max from the pre-scan percentile computation.
        """
        self._fixed_intensity_max = value
        self.gpu_renderer.set_intensity_max(value)
        log.info(f"Renderer: fixed intensity_max = {value:.4f}")

    def should_close(self) -> bool:
        """
        Check if user wants to close the window.

        Returns:
            bool: True if the window should close, False otherwise.
        """
        return self.gl_context.should_close()

    def cleanup(self) -> None:
        """
        Clean shutdown — terminate GLFW.
        """
        glfw.terminate()
