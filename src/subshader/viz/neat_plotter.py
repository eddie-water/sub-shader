from abc import ABC, abstractmethod
import numpy as np
import moderngl
import glfw

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
            
            void main() {
                // SOLID RED TEST - SHOULD SHOW BRIGHT RED SCREEN
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }
        """
    
    @classmethod
    def create_program(cls, ctx):
        """Create and return compiled shader program"""
        vertex_shader = cls.create_vertex_shader()
        fragment_shader = cls.create_fragment_shader()
        
        print("=== COMPILING NEW SHADERS ===")
        print(f"Fragment shader first line: {fragment_shader.split('main()')[1][:50]}...")
        
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        print("Shader compilation successful!")
        
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
        
        # Debug OpenGL state
        print(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
        print(f"Viewport: {self.ctx.viewport}")
        print(f"Context size: {self.width}x{self.height}")
        
        # Ensure viewport is set correctly
        self.ctx.viewport = (0, 0, self.width, self.height)
        print(f"Set viewport to: {self.ctx.viewport}")
        
        # Disable depth testing (shouldn't be needed for 2D)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
    
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
        # Skip texture setup for now since we're testing basic rendering
        print("RenderTarget initialized - skipping texture for red test")
    
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
        """Skip texture update for red test"""
        print(f"Skipping texture update for red test")
    
    def render(self):
        """Render the quad"""
        print("=== RENDER CALL ===")
        print(f"VAO: {self.vao}")
        print(f"Program: {self.program}")
        print(f"GL Context valid: {self.ctx}")
        
        # Try to render
        try:
            self.vao.render(moderngl.TRIANGLE_STRIP)
            print("Render call completed successfully")
        except Exception as e:
            print(f"Render failed: {e}")
        
        # Check for OpenGL errors
        error = self.ctx.error
        if error != 'GL_NO_ERROR':
            print(f"OpenGL Error: {error}")
        else:
            print("No OpenGL errors")

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
    
    def __init__(self, file_path: str, shape: tuple[int, int], num_frames=8):
        super().__init__(file_path, shape)
        
        print("=== INITIALIZING SHADER SYSTEM ===")
        self.gl_context = GLContext(title=f"Audio Visualizer - {file_path}")
        self.program = ShaderSystem.create_program(self.gl_context.ctx)
        
        texture_width = self.x_n * num_frames
        texture_height = self.y_n
        
        self.render_target = RenderTarget(
            self.gl_context.ctx, self.program, texture_width, texture_height
        )
        self.scrolling_buffer = ScrollingBuffer(num_frames, self.y_n, self.x_n)
        self.scaling = AdaptiveScaling()
        print("=== SHADER SYSTEM READY ===")
    
    def update_plot(self, values):
        """Main API: Feed in coefficients, get beautiful visualization"""
        
        self.scrolling_buffer.add_frame(values)
        buffer_data = self.scrolling_buffer.get_flattened_buffer()
        
        value_min, value_max = self.scaling.update_range(buffer_data)
        
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