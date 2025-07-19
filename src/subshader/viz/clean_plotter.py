from abc import ABC, abstractmethod
import numpy as np
import moderngl
import glfw
from matplotlib import cm

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
            
            void main() {
                // Test texture coordinates - should show red-green gradient
                fragColor = vec4(texCoord.x, texCoord.y, 0.0, 1.0);
            }
        """
    
    @classmethod
    def create_program(cls, ctx):
        """Create and return compiled shader program"""
        vertex_shader = cls.create_vertex_shader()
        fragment_shader = cls.create_fragment_shader()
        return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

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
    
    def should_close(self):
        return glfw.window_should_close(self.window)
    
    def swap_buffers(self):
        glfw.swap_buffers(self.window)
    
    def clear(self, r=0.05, g=0.05, b=0.05):
        self.ctx.clear(r, g, b)

class RenderTarget:
    """Handles quad geometry and texture setup"""
    
    def __init__(self, ctx, program, texture_width, texture_height):
        self.ctx = ctx
        self.program = program
        self._setup_quad()
        self._setup_texture(texture_width, texture_height)
    
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
    
    def _setup_texture(self, width, height):
        """Create 2D texture for scalogram data"""
        self.texture = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Only set up texture unit if the uniform exists in the shader
        try:
            self.program['scalogram'] = 0
        except KeyError:
            print("Warning: 'scalogram' uniform not found in shader")
            pass
    
    def update_texture(self, data):
        """Update texture with new data"""
        self.texture.write(data.astype('f4').tobytes())
        self.texture.use(location=0)
    
    def render(self):
        """Render the quad"""
        self.vao.render(moderngl.TRIANGLE_STRIP)

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
        # Reorder to fix time sequence
        rolled = np.roll(self.frames, -self.frame_index, axis=0)
        # Reshape: (height, num_frames * width) for horizontal scrolling
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
            # Fast adaptation upward
            self.global_max = (self.adaptation_rate * current_max + 
                             (1 - self.adaptation_rate) * self.global_max)
        else:
            # Slow decay downward
            self.global_max = (self.decay_rate * self.global_max + 
                             (1 - self.decay_rate) * current_max)
        
        # Ensure minimum range
        self.global_max = max(self.global_max, 0.001)
        
        return self.global_min, self.global_max * self.headroom

class Shader(Plotter):
    """Clean, high-level audio visualization using GPU shaders"""
    
    def __init__(self, file_path: str, shape: tuple[int, int], num_frames=8):
        super().__init__(file_path, shape)
        
        # Initialize all subsystems
        self.gl_context = GLContext(title=f"Audio Visualizer - {file_path}")
        self.program = ShaderSystem.create_program(self.gl_context.ctx)
        
        # Calculate texture dimensions for scrolling
        texture_width = self.x_n * num_frames
        texture_height = self.y_n
        
        self.render_target = RenderTarget(
            self.gl_context.ctx, self.program, texture_width, texture_height
        )
        self.scrolling_buffer = ScrollingBuffer(num_frames, self.y_n, self.x_n)
        self.scaling = AdaptiveScaling()
    
    def update_plot(self, values):
        """Main API: Feed in coefficients, get beautiful visualization"""
        
        # Debug original input
        print(f"ORIGINAL CWT - Shape: {values.shape}, Min: {values.min():.6f}, Max: {values.max():.6f}")
        
        # TEMPORARY TEST: Replace with obvious pattern
        test_pattern = np.ones(values.shape, dtype=np.float32) * 0.5  # Solid middle value
        # OR try this for gradient:
        # test_pattern = np.linspace(0, 2.0, values.size).reshape(values.shape).astype(np.float32)
        
        print(f"TEST PATTERN - Shape: {test_pattern.shape}, Min: {test_pattern.min():.6f}, Max: {test_pattern.max():.6f}")
        
        # Use test pattern instead of real data
        values = test_pattern
        
        print(f"AFTER REPLACEMENT - Shape: {values.shape}, Min: {values.min():.6f}, Max: {values.max():.6f}")
        
        # Add to scrolling buffer
        self.scrolling_buffer.add_frame(values)
        
        # Get flattened data for texture
        buffer_data = self.scrolling_buffer.get_flattened_buffer()
        print(f"Buffer data - Shape: {buffer_data.shape}, Min: {buffer_data.min():.6f}, Max: {buffer_data.max():.6f}")
        
        # Update scaling range
        value_min, value_max = self.scaling.update_range(buffer_data)
        print(f"Scaling range: {value_min:.6f} to {value_max:.6f}")
        print("---")
        
        # Update GPU uniforms (only if they exist)
        try:
            self.program['valueMin'].value = value_min
            self.program['valueMax'].value = value_max
        except KeyError as e:
            print(f"Warning: Uniform {e} not found in shader (probably optimized out)")
            pass
        
        # Update texture and render
        self.render_target.update_texture(buffer_data)
        self.gl_context.clear()
        self.render_target.render()
        self.gl_context.swap_buffers()
    
    def should_window_close(self):
        """Check if user wants to close the window"""
        return self.gl_context.should_close()
    
    def cleanup(self):
        """Clean shutdown"""
        glfw.terminate()

# Clean PyQtGrapher remains the same but simplified
class PyQtGrapher(Plotter):
    def __init__(self, file_path: str, shape: tuple[int, int]):
        super().__init__(file_path, shape)
        # ... existing PyQtGrapher code stays the same ...
    
    def update_plot(self, values):
        values = values.T
        self.pcolormesh.setData(values)
    
    def should_window_close(self):
        raise NotImplementedError("PyQtGraph-based window-check not implemented yet.")