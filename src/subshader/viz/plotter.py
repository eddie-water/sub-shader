from abc import ABC, abstractmethod
import numpy as np
import moderngl
import glfw
import logging
import pyqtgraph as pg

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
    def __init__(self, file_path: str, shape: tuple[int, int]):
        self.file_path = file_path
        if len(shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(shape)}D with shape {shape}")        
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"2D array cannot have shape: {shape}")
        self.shape = shape
        self.y_n, self.x_n = self.shape

    @abstractmethod
    def update_plot(self, values):
        pass

    @abstractmethod
    def should_window_close(self):
        pass

class ShaderSystem:
    """Handles shader compilation and program creation"""
    
    @staticmethod
    def create_vertex_shader():
        return """
            #version 330
            in vec2 position;
            out vec2 texCoord;
            void main() {
                texCoord = (position + 1.0) / 2.0;
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """
    
    @staticmethod
    def create_fragment_shader():
        return """
            #version 330
            in vec2 texCoord;
            out vec4 fragColor;
            uniform sampler2D scalogram;
            uniform float valueMin;
            uniform float valueMax;
            
            vec3 inferno(float t) {
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
                
                // Normalize using adaptive scaling
                float normalized = clamp((value - valueMin) / (valueMax - valueMin), 0.0, 1.0);
                normalized = pow(normalized, 0.7);  // Gamma correction
                
                vec3 color = inferno(normalized);
                fragColor = vec4(color, 1.0);
            }
        """
    
    @classmethod
    def create_program(cls, ctx):
        """Create and return compiled shader program"""
        vertex_shader = cls.create_vertex_shader()
        fragment_shader = cls.create_fragment_shader()
        
        logger.info("Compiling shaders...")
        logger.debug(f"Fragment shader preview: {fragment_shader[:100]}...")
        
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        logger.info("Shader compilation successful!")
        
        return program

class GLContext:
    """Handles GLFW window and OpenGL context setup"""
    
    def __init__(self, width=800, height=600, title="Audio Visualizer"):
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.ctx = None
        self._init_graphics()
    
    def _init_graphics(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        
        # Log OpenGL info
        logger.info(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
        logger.debug(f"Viewport: {self.ctx.viewport}")
        
        # Setup viewport and disable unnecessary features
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        
        logger.info("Graphics context initialized successfully")
    
    def should_close(self):
        return glfw.window_should_close(self.window)
    
    def swap_buffers(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()  # Process window events
    
    def clear(self, r=0.05, g=0.05, b=0.05):
        self.ctx.clear(r, g, b)

class RenderTarget:
    """Handles quad geometry and texture setup"""
    
    def __init__(self, ctx, program, texture_width, texture_height):
        self.ctx = ctx
        self.program = program
        self._setup_quad()
        self._setup_texture(texture_width, texture_height)  # Restore texture setup
    
    def _setup_texture(self, width, height):
        """Create 2D texture for scalogram data"""
        # Use a much smaller texture size to avoid WSL/OpenGL issues
        small_width = min(width, 1024)  # Max 1024 pixels wide
        small_height = min(height, 256)  # Max 256 pixels tall
        
        logger.info(f"Creating smaller texture: {small_width}x{small_height} (original: {width}x{height})")
        
        self.texture = self.ctx.texture((small_width, small_height), 1, dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Store original and actual sizes
        self.original_size = (width, height)
        self.actual_size = (small_width, small_height)
        
        # Set up uniforms for colormap
        self.program['scalogram'] = 0
        try:
            self.program['valueMin'] = 0.0
            self.program['valueMax'] = 1.0
        except KeyError:
            logger.warning("Scaling uniforms not found")
        
        logger.info(f"Audio texture created: {small_width}x{small_height}")
    
    def _setup_quad(self):
        """Create fullscreen quad"""
        quad_vertices = np.array([
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right
            -1.0,  1.0,  # Top-left
             1.0,  1.0,  # Top-right
        ], dtype=np.float32)
        
        self.vbo = self.ctx.buffer(quad_vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'position')
        print("Quad setup complete")
    
    def update_texture(self, data):
        """Update texture with new data"""
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
        
        # Upload data and bind
        self.texture.write(data.astype('f4').tobytes())
        self.texture.use(location=0)
        
        logger.debug(f"Downsampled texture updated: {data.shape}, range {data.min():.3f}-{data.max():.3f}")
    
    def render(self):
        """Render the quad"""
        try:
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Check for OpenGL errors
            error = self.ctx.error
            if error != 'GL_NO_ERROR':
                logger.error(f"Render error: {error}")
            
        except Exception as e:
            logger.error(f"Render exception: {e}")

class ScrollingBuffer:
    """Handles circular buffer for scrolling visualization"""
    
    def __init__(self, num_frames, height, width):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frames = np.zeros((num_frames, height, width), dtype=np.float32)
        self.frame_index = 0
    
    def add_frame(self, frame_data):
        """Add new frame to circular buffer"""
        if frame_data.shape != (self.height, self.width):
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")
        
        self.frames[self.frame_index] = frame_data
        self.frame_index = (self.frame_index + 1) % self.num_frames
    
    def get_flattened_buffer(self):
        """Get time-ordered flattened buffer for texture"""
        rolled = np.roll(self.frames, -self.frame_index, axis=0)
        return rolled.transpose(1, 0, 2).reshape(self.height, self.num_frames * self.width)

class AdaptiveScaling:
    """Handles dynamic range scaling for audio visualization"""
    
    def __init__(self, adaptation_rate=0.05, decay_rate=0.999, headroom=1.2):
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

class Shader(Plotter):
    """Clean, high-level audio visualization using GPU shaders"""
    
    def __init__(self, file_path: str, shape: tuple[int, int], num_frames=64):
        super().__init__(file_path, shape)
        
        logger.info(f"Initializing Shader for {file_path}")
        self.gl_context = GLContext(title=f"Audio Visualizer - {file_path}")
        self.program = ShaderSystem.create_program(self.gl_context.ctx)
        
        texture_width = self.x_n * num_frames
        texture_height = self.y_n
        
        self.render_target = RenderTarget(
            self.gl_context.ctx, self.program, texture_width, texture_height
        )
        self.scrolling_buffer = ScrollingBuffer(num_frames, self.y_n, self.x_n)
        self.scaling = AdaptiveScaling()
        
        logger.info("Shader system ready")
        print("🎵 Visualizer started - logs in 'shader_debug.log'")
    
    def update_plot(self, values):
        """Main API: Feed in coefficients, get beautiful visualization"""
        
        self.scrolling_buffer.add_frame(values)
        buffer_data = self.scrolling_buffer.get_flattened_buffer()
        
        # Update adaptive scaling
        value_min, value_max = self.scaling.update_range(buffer_data)
        
        # Update scaling uniforms for colormap
        try:
            self.program['valueMin'] = value_min
            self.program['valueMax'] = value_max
        except KeyError:
            pass  # Uniforms might be optimized out
        
        # Update texture
        self.render_target.update_texture(buffer_data)
        
        # Clear and render
        self.gl_context.clear()
        self.render_target.render()
        self.gl_context.swap_buffers()
    
    def update_fps(self, fps: int):
        """GPU visualizer doesn't need FPS display"""
        pass
    
    def should_window_close(self):
        """Check if user wants to close the window"""
        return self.gl_context.should_close()
    
    def cleanup(self):
        """Clean shutdown"""
        glfw.terminate()


class PyQtGrapher(Plotter):
    """Traditional PyQtGraph-based audio visualizer"""
    
    def __init__(self, file_path: str, shape: tuple[int, int]):
        super().__init__(file_path, shape)
        
        # PyQtGraph configuration
        pg.setConfigOptions(useOpenGL=True, enableExperimental=True)

        self.app = pg.mkQApp("Sub Shader")
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  
        self.win.setWindowTitle('Continuous Wavelet Transform')

        self.plot = self.win.addPlot(row=0, col=0, rowspan=1, colspan=1,
                                   title=file_path, enableMenu=False)

        # Setup colormap and plot item
        colorMap = pg.colormap.get('inferno')
        self.pcolormesh = pg.PColorMeshItem(colorMap=colorMap,
                                          levels=(-1, 1),
                                          enableAutoLevels=False,
                                          edgeColors=None,
                                          antialiasing=False)
        self.plot.addItem(self.pcolormesh)

        # Add colorbar
        self.bar = pg.ColorBarItem(label="Magnitude",
                                 interactive=False,
                                 limits=(0, 1),
                                 rounding=0.1)
        self.bar.setImageItem([self.pcolormesh])
        self.win.addItem(self.bar, 0, 1, 1, 1)

        # Add FPS text
        self.textBox = pg.TextItem(anchor=(0, 1), fill='black')
        self.textBox.setPos(1, 1)
        self.plot.addItem(self.textBox)

    def update_plot(self, values):
        """Update plot with new CWT coefficients"""
        values = values.T  # Transpose for correct orientation
        self.pcolormesh.setData(values)

    def update_fps(self, fps: int):
        """Update FPS display"""
        self.textBox.setText(f'{fps:.1f} fps')

    def should_window_close(self):
        """PyQtGraph handles its own event loop"""
        raise NotImplementedError("PyQtGraph-based window-check not implemented yet.")