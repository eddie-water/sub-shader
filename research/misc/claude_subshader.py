import numpy as np
import cupy as cp
import sounddevice as sd
import queue
import threading
import pygame
from pygame.locals import *
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.gl import graphics_map_fplags
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
from OpenGL.GL import *
from OpenGL.GL import shaders

class WaveletTransformCUDA:
    def __init__(self, scales, sampling_rate, wavelet='morlet'):
        """
        Initialize the wavelet transform with CUDA acceleration
        
        Parameters:
        -----------
        scales : array-like
            Scales for the wavelet transform
        sampling_rate : float
            Sampling rate of the audio signal
        wavelet : str
            Wavelet type (default: 'morlet')
        """
        self.scales = cp.asarray(scales, dtype=cp.float32)
        self.sampling_rate = sampling_rate
        self.wavelet_type = wavelet
        
        # Create PyCUDA kernels (for better interoperability with OpenGL)
        self._compile_cuda_kernels()
        
    def _compile_cuda_kernels(self):
        """Compile the CUDA kernels for the wavelet transform"""
        # Define CUDA kernel code
        cuda_code = """
        // Morlet wavelet kernel
        __global__ void morlet_wavelet(float *time, float *wavelet_real, float *wavelet_imag, 
                                      float scale, int n) 
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                float t = time[i] / scale;
                float norm = 1.0f / (powf(3.14159265358979f, 0.25f) * sqrtf(scale));
                float exp_term = expf(-0.5f * t * t);
                
                wavelet_real[i] = norm * exp_term * cosf(6.0f * t);
                wavelet_imag[i] = norm * exp_term * sinf(6.0f * t);
            }
        }
        
        // Convolution kernel for wavelet transform
        __global__ void wavelet_conv(float *signal, float *wavelet_real, float *wavelet_imag,
                                    float *coeffs_real, float *coeffs_imag, 
                                    int signal_len, int wavelet_len) 
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < signal_len) {
                float sum_real = 0.0f;
                float sum_imag = 0.0f;
                
                for (int j = 0; j < wavelet_len; j++) {
                    int idx = i + j - wavelet_len / 2;
                    if (idx >= 0 && idx < signal_len) {
                        float sig_val = signal[idx];
                        sum_real += sig_val * wavelet_real[j];
                        sum_imag += sig_val * wavelet_imag[j];
                    }
                }
                
                coeffs_real[i] = sum_real;
                coeffs_imag[i] = sum_imag;
            }
        }
        
        // Magnitude calculation kernel
        __global__ void calculate_magnitude(float *real, float *imag, float *magnitude, int n) 
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
            }
        }
        
        // Direct update of texture memory kernel
        __global__ void update_texture(float *coeffs, float *texture_data, 
                                     int width, int height, float min_val, float max_val)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            
            if (x < width && y < height) {
                int idx = y * width + x;
                // Normalize coefficient value to [0,1] range
                float normalized = (coeffs[idx] - min_val) / (max_val - min_val);
                normalized = fmaxf(0.0f, fminf(normalized, 1.0f));  // Clamp to [0,1]
                
                // Store value in texture format (RGBA)
                texture_data[idx * 4 + 0] = normalized;  // R
                texture_data[idx * 4 + 1] = normalized;  // G
                texture_data[idx * 4 + 2] = normalized;  // B
                texture_data[idx * 4 + 3] = 1.0f;        // A (opacity)
            }
        }
        """
        
        # Compile the kernels
        self.module = SourceModule(cuda_code)
        
        # Get function pointers
        self.morlet_kernel = self.module.get_function("morlet_wavelet")
        self.conv_kernel = self.module.get_function("wavelet_conv")
        self.magnitude_kernel = self.module.get_function("calculate_magnitude")
        self.update_texture_kernel = self.module.get_function("update_texture")
        
    def transform(self, signal_gpu):
        """
        Compute the continuous wavelet transform directly on GPU
        
        Parameters:
        -----------
        signal_gpu : pycuda.gpuarray.GPUArray
            Input signal on GPU
        
        Returns:
        --------
        coefficients : pycuda.gpuarray.GPUArray
            Wavelet coefficients as a 2D array [scales, signal_length]
        """
        signal_len = signal_gpu.shape[0]
        
        # Pre-allocate output array on GPU
        coeffs = gpuarray.zeros((len(self.scales), signal_len), dtype=np.float32)
        
        # Create time array for wavelet computation
        max_wavelet_len = int(10 * np.max(self.scales.get()))  # Length depends on max scale
        t = gpuarray.to_gpu(np.linspace(-5, 5, max_wavelet_len, dtype=np.float32))
        
        # Define grid and block dimensions
        threads_per_block = (256, 1, 1)
        
        # Process each scale
        for i, scale in enumerate(self.scales.get()):
            # Adjust wavelet length based on scale (larger scales need more points)
            wavelet_len = min(int(10 * scale), max_wavelet_len)
            if wavelet_len % 2 == 0:
                wavelet_len += 1  # Ensure odd length
                
            t_scale = gpuarray.to_gpu(np.linspace(-5, 5, wavelet_len, dtype=np.float32))
            wavelet_real = gpuarray.zeros(wavelet_len, dtype=np.float32)
            wavelet_imag = gpuarray.zeros(wavelet_len, dtype=np.float32)
            
            # Generate wavelet at this scale
            blocks_per_grid = ((wavelet_len + threads_per_block[0] - 1) // threads_per_block[0], 1)
            self.morlet_kernel(
                t_scale.gpudata, wavelet_real.gpudata, wavelet_imag.gpudata,
                np.float32(scale), np.int32(wavelet_len),
                block=threads_per_block, grid=blocks_per_grid
            )
            
            # Allocate memory for real and imaginary parts of convolution
            coeffs_real = gpuarray.zeros(signal_len, dtype=np.float32)
            coeffs_imag = gpuarray.zeros(signal_len, dtype=np.float32)
            
            # Perform convolution
            blocks_per_grid = ((signal_len + threads_per_block[0] - 1) // threads_per_block[0], 1)
            self.conv_kernel(
                signal_gpu.gpudata, wavelet_real.gpudata, wavelet_imag.gpudata,
                coeffs_real.gpudata, coeffs_imag.gpudata, 
                np.int32(signal_len), np.int32(wavelet_len),
                block=threads_per_block, grid=blocks_per_grid
            )
            
            # Calculate magnitude
            magnitude = gpuarray.zeros(signal_len, dtype=np.float32)
            self.magnitude_kernel(
                coeffs_real.gpudata, coeffs_imag.gpudata, magnitude.gpudata, np.int32(signal_len),
                block=threads_per_block, grid=blocks_per_grid
            )
            
            # Store result for this scale
            coeffs[i] = magnitude
        
        return coeffs
        
    def update_texture_direct(self, coeffs, texture_buffer, width, height, min_val=None, max_val=None):
        """
        Update OpenGL texture directly from GPU data without CPU transfer
        
        Parameters:
        -----------
        coeffs : pycuda.gpuarray.GPUArray
            2D array of wavelet coefficients
        texture_buffer : pycuda.gl.BufferObject
            OpenGL texture buffer mapped to CUDA
        width : int
            Texture width
        height : int
            Texture height
        min_val, max_val : float
            Min and max values for normalization (None for auto)
        """
        # Determine min and max values for normalization if not provided
        if min_val is None:
            min_val = float(cp.min(coeffs))
        if max_val is None:
            max_val = float(cp.max(coeffs))
            # Avoid division by zero
            if min_val == max_val:
                max_val = min_val + 1.0
        
        # Map the texture buffer to get GPU pointer
        mapping = texture_buffer.map()
        texture_ptr = mapping.device_ptr_and_size()[0]
        
        # Define grid and block dimensions for 2D texture update
        block_dim = (16, 16, 1)
        grid_dim = (
            (width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1]
        )
        
        # Call the kernel to update texture directly
        self.update_texture_kernel(
            coeffs.gpudata, 
            cuda.to_device(np.zeros((height, width, 4), dtype=np.float32)),  # Placeholder for texture data
            np.int32(width), np.int32(height),
            np.float32(min_val), np.float32(max_val),
            block=block_dim, grid=grid_dim
        )
        
        # Unmap the buffer
        mapping.unmap()


class AudioProcessor:
    def __init__(self, sample_rate=44100, chunk_size=1024, buffer_size=10):
        """
        Initialize audio processor for real-time streaming
        
        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        chunk_size : int
            Number of samples per chunk
        buffer_size : int
            Number of chunks to buffer
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stream = None
        self.is_running = False
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream processing"""
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata[:, 0]
            
        # Add to buffer if there's space
        try:
            self.buffer.put_nowait(data)
        except queue.Full:
            pass  # Skip if buffer is full
    
    def start_stream(self):
        """Start the audio stream"""
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        self.stream.start()
        self.is_running = True
    
    def stop_stream(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
    
    def get_audio_chunk(self):
        """Get the next audio chunk from the buffer"""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None


class WaveletVisualizer:
    def __init__(self, width=1024, height=768):
        """
        Initialize the OpenGL-based wavelet transform visualizer with CUDA-OpenGL interop
        
        Parameters:
        -----------
        width : int
            Window width
        height : int
            Window height
        """
        self.width = width
        self.height = height
        self.should_quit = False
        
        # Initialize pygame and OpenGL
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Real-time Wavelet Transform")
        
        # Set up OpenGL
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Create texture for wavelet coefficients
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, 
                     GL_RGBA, GL_FLOAT, None)
        
        # Create PBO for CUDA-OpenGL interop
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        
        # Register buffer with CUDA
        self.cuda_pbo = cuda_gl.RegisteredBuffer(int(self.pbo))
        
        # Compile shaders for visualization
        self._compile_shaders()
        
        # Initialize vertex buffer for rendering quad
        self._init_vertex_buffer()
        
    def _compile_shaders(self):
        """Compile the OpenGL shaders for visualization"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main()
        {
            gl_Position = vec4(position, 1.0);
            TexCoord = texCoord;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform sampler2D coeffTexture;
        
        // Viridis colormap function
        vec3 viridis(float t) {
            const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
            const vec3 c1 = vec3(0.283072, 0.130895, 0.449241);
            const vec3 c2 = vec3(0.228527, 0.268198, 0.546059);
            const vec3 c3 = vec3(0.122176, 0.411517, 0.534245);
            const vec3 c4 = vec3(0.004037, 0.554206, 0.388479);
            const vec3 c5 = vec3(0.090735, 0.687164, 0.326157);
            const vec3 c6 = vec3(0.572147, 0.773031, 0.150684);
            const vec3 c7 = vec3(0.988362, 0.719863, 0.137482);
            
            float tc = t * 7.0;
            int i = int(tc);
            float f = fract(tc);
            
            if (i == 0) return mix(c0, c1, f);
            else if (i == 1) return mix(c1, c2, f);
            else if (i == 2) return mix(c2, c3, f);
            else if (i == 3) return mix(c3, c4, f);
            else if (i == 4) return mix(c4, c5, f);
            else if (i == 5) return mix(c5, c6, f);
            else return mix(c6, c7, f);
        }
        
        void main()
        {
            // Get raw value from texture
            float value = texture(coeffTexture, TexCoord).r;
            
            // Apply colormap
            vec3 color = viridis(value);
            
            // Output final color
            FragColor = vec4(color, 1.0);
        }
        """
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        
        # Create shader program
        self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        # Get uniform locations
        self.texture_uniform = glGetUniformLocation(self.shader_program, "coeffTexture")
    
    def _init_vertex_buffer(self):
        """Initialize vertex buffer for rendering a fullscreen quad"""
        # Define quad vertices and texture coordinates
        quad_vertices = np.array([
            # positions        # texture coords
            -1.0, -1.0, 0.0,   0.0, 1.0,  # bottom left
             1.0, -1.0, 0.0,   1.0, 1.0,  # bottom right
             1.0,  1.0, 0.0,   1.0, 0.0,  # top right
            -1.0,  1.0, 0.0,   0.0, 0.0   # top left
        ], dtype=np.float32)
        
        quad_indices = np.array([
            0, 1, 2,  # first triangle
            2, 3, 0   # second triangle
        ], dtype=np.uint32)
        
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        
        # Set vertex attribute pointers
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * np.dtype(np.float32).itemsize, None)
        glEnableVertexAttribArray(0)
        
        # Texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * np.dtype(np.float32).itemsize, 
                             ctypes.c_void_p(3 * np.dtype(np.float32).itemsize))
        glEnableVertexAttribArray(1)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def update_coefficients(self, wavelet_transform, coeffs):
        """
        Update the visualization with new wavelet coefficients
        
        Parameters:
        -----------
        wavelet_transform : WaveletTransformCUDA
            Wavelet transform object with GPU kernels
        coeffs : pycuda.gpuarray.GPUArray
            Wavelet coefficients as a 2D array
        """
        # Update texture data directly on GPU
        wavelet_transform.update_texture_direct(
            coeffs, 
            self.cuda_pbo, 
            self.width, 
            self.height
        )
        
        # Transfer texture data from PBO to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                       GL_RGBA, GL_FLOAT, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    
    def render(self):
        """Render the visualization"""
        # Clear screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(self.shader_program)
        
        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.texture_uniform, 0)
        
        # Draw quad
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        # Swap buffers
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.should_quit = True
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.should_quit = True
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        # Delete buffers and textures
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteBuffers(1, [self.pbo])
        glDeleteTextures(1, [self.texture_id])
        glDeleteVertexArrays(1, [self.vao])
        
        # Delete shader program
        glDeleteProgram(self.shader_program)
        
        # Unregister CUDA buffer
        self.cuda_pbo.unregister()
        
        # Quit pygame
        pygame.quit().009605, 0.335427);
            const vec3 c2 = vec3(0.269944, 0.014625, 0.341379);
            const vec3 c3 = vec3(0.271305, 0.019942, 0.347269);
            const vec3 c4 = vec3(0.272594, 0.025563, 0.353093);
            const vec3 c5 = vec3(0.273809, 0.031497, 0.358853);
            const vec3 c6 = vec3(0.274952, 0.037752, 0.364543);
            const vec3 c7 = vec3(0.276022, 0.044167, 0.370164);
            const vec3 c8 = vec3(0.277018, 0.050344, 0.375715);
            const vec3 c9 = vec3(0.277941, 0.056324, 0.381191);
            const vec3 c10 = vec3(0.278791, 0.062145, 0.386592);
            const vec3 c11 = vec3(0.279566, 0.067836, 0.391917);
            const vec3 c12 = vec3(0.280267, 0.073417, 0.397163);
            const vec3 c13 = vec3(0.280894, 0.078907, 0.402329);
            const vec3 c14 = vec3(0.281446, 0.08432, 0.407414);
            const vec3 c15 = vec3(0.281924, 0.089666, 0.412415);
            const vec3 c16 = vec3(0.282327, 0.094955, 0.417331);
            const vec3 c17 = vec3(0.282656, 0.100196, 0.42216);
            const vec3 c18 = vec3(0.28291, 0.105393, 0.426902);
            const vec3 c19 = vec3(0.283091, 0.110553, 0.431554);
            const vec3 c20 = vec3(0.283197, 0.11568, 0.436115);
            const vec3 c21 = vec3(0.283229, 0.120777, 0.440584);
            const vec3 c22 = vec3(0.283187, 0.125848, 0.44496);
            const vec3 c23 = vec3(0.283072, 0.130895, 0.449241);
            const vec3 c24 = vec3(0.282884, 0.13592, 0.453427);
            const vec3 c25 = vec3(0.282623, 0.140926, 0.457517);
            const vec3 c26 = vec3(0.28229, 0.145912, 0.46151);
            const vec3 c27 = vec3(0.281887, 0.150881, 0.465405);
            const vec3 c28 = vec3(0.281412, 0.155834, 0.469201);
            const vec3 c29 = vec3(0.280868, 0.160771, 0.472899);
            const vec3 c30 = vec3(0.280255, 0.165693, 0.476498);
            const vec3 c31 = vec3(0.279574, 0.170599, 0.479997);
            const vec3 c32 = vec3(0.278826, 0.17549, 0.483397);
            const vec3 c33 = vec3(0.278012, 0.180367, 0.486697);
            const vec3 c34 = vec3(0.277134, 0.185228, 0.489898);
            const vec3 c35 = vec3(0.276194, 0.190074, 0.493001);
            const vec3 c36 = vec3(0.275191, 0.194905, 0.496005);
            const vec3 c37 = vec3(0.274128, 0.199721, 0.498911);
            const vec3 c38 = vec3(0.273006, 0.20452, 0.501721);
            const vec3 c39 = vec3(0.271828, 0.209303, 0.504434);
            const vec3 c40 = vec3(0.270595, 0.214069, 0.507052);
            const vec3 c41 = vec3(0.269308, 0.218818, 0.509577);
            const vec3 c42 = vec3(0.267968, 0.223549, 0.512008);
            const vec3 c43 = vec3(0.26658, 0.228262, 0.514349);
            const vec3 c44 = vec3(0.265145, 0.232956, 0.516599);
            const vec3 c45 = vec3(0.263663, 0.237631, 0.518762);
            const vec3 c46 = vec3(0.262138, 0.242286, 0.520837);
            const vec3 c47 = vec3(0.260571, 0.246922, 0.522828);
            const vec3 c48 = vec3(0.258965, 0.251537, 0.524736);
            const vec3 c49 = vec3(0.257322, 0.25613, 0.526563);
            const vec3 c50 = vec3(0.255645, 0.260703, 0.528312);
            const vec3 c51 = vec3(0.253935, 0.265254, 0.529983);
            const vec3 c52 = vec3(0.252194, 0.269783, 0.531579);
            const vec3 c53 = vec3(0.250425, 0.27429, 0.533103);
            const vec3 c54 = vec3(0.248629, 0.278775, 0.534556);
            const vec3 c55 = vec3(0.246811, 0.283237, 0.535941);
            const vec3 c56 = vec3(0.244972, 0.287675, 0.53726);
            const vec3 c57 = vec3(0.243113, 0.292092, 0.538516);
            const vec3 c58 = vec3(0.241237, 0.296485, 0.539709);
            const vec3 c59 = vec3(0.239346, 0.300855, 0.540844);
            const vec3 c60 = vec3(0.237441, 0.305202, 0.541921);
            const vec3 c61 = vec3(0.235526, 0.309527, 0.542944);
            const vec3 c62 = vec3(0.233603, 0.313828, 0.543914);
            const vec3 c63 = vec3(0.231674, 0.318106, 0.544834);
            const vec3 c64 = vec3(0.229739, 0.322361, 0.545706);
            const vec3 c65 = vec3(0.227802, 0.326594, 0.546532);
            const vec3 c66 = vec3(0.225863, 0.330805, 0.547314);
            const vec3 c67 = vec3(0.223925, 0.334994, 0.548053);
            const vec3 c68 = vec3(0.221989, 0.339161, 0.548752);
            const vec3 c69 = vec3(0.220057, 0.343307, 0.549413);
            const vec3 c70 = vec3(0.21813, 0.347432, 0.550038);
            const vec3 c71 = vec3(0.21621, 0.351535, 0.550627);
            const vec3 c72 = vec3(0.214298, 0.355619, 0.551184);
            const vec3 c73 = vec3(0.212395, 0.359683, 0.55171);
            const vec3 c74 = vec3(0.210503, 0.363727, 0.552206);
            const vec3 c75 = vec3(0.208623, 0.367752, 0.552675);
            const vec3 c76 = vec3(0.206756, 0.371758, 0.553117);
            const vec3 c77 = vec3(0.204903, 0.375746, 0.553533);
            const vec3 c78 = vec3(0.203063, 0.379716, 0.553925);
            const vec3 c79 = vec3(0.201239, 0.38367, 0.554294);
            const vec3 c80 = vec3(0.19943, 0.387607, 0.554642);
            const vec3 c81 = vec3(0.197636, 0.391528, 0.554969);
            const vec3 c82 = vec3(0.19586, 0.395433, 0.555276);
            const vec3 c83 = vec3(0.1941, 0.399323, 0.555565);
            const vec3 c84 = vec3(0.192357, 0.403199, 0.555836);
            const vec3 c85 = vec3(0.190631, 0.407061, 0.556089);
            const vec3 c86 = vec3(0.188923, 0.41091, 0.556326);
            const vec3 c87 = vec3(0.187231, 0.414746, 0.556547);
            const vec3 c88 = vec3(0.185556, 0.41857, 0.556753);
            const vec3 c89 = vec3(0.183898, 0.422383, 0.556944);
            const vec3 c90 = vec3(0.182256, 0.426184, 0.55712);
            const vec3 c91 = vec3(0.180629, 0.429975, 0.557282);
            const vec3 c92 = vec3(0.179019, 0.433756, 0.55743);
            const vec3 c93 = vec3(0.177423, 0.437527, 0.557565);
            const vec3 c94 = vec3(0.175841, 0.44129, 0.557685);
            const vec3 c95 = vec3(0.174274, 0.445044, 0.557792);
            const vec3 c96 = vec3(0.172719, 0.448791, 0.557885);
            const vec3 c97 = vec3(0.171176, 0.45253, 0.557965);
            const vec3 c98 = vec3(0.169646, 0.456262, 0.55803);
            const vec3 c99 = vec3(0.168126, 0.459988, 0.558082);
            const vec3 c100 = vec3(0.166617, 0.463708, 0.558119);
            const vec3 c101 = vec3(0.165117, 0.467423, 0.558141);
            const vec3 c102 = vec3(0.163625, 0.471133, 0.558148);
            const vec3 c103 = vec3(0.162142, 0.474838, 0.55814);
            const vec3 c104 = vec3(0.160665, 0.47854, 0.558115);
            const vec3 c105 = vec3(0.159194, 0.482237, 0.558073);
            const vec3 c106 = vec3(0.157729, 0.485932, 0.558013);
            const vec3 c107 = vec3(0.15627, 0.489624, 0.557936);
            const vec3 c108 = vec3(0.154815, 0.493313, 0.55784);
            const vec3 c109 = vec3(0.153364, 0.497, 0.557724);
            const vec3 c110 = vec3(0.151918, 0.500685, 0.557587);
            const vec3 c111 = vec3(0.150476, 0.504369, 0.55743);
            const vec3 c112 = vec3(0.149039, 0.508051, 0.55725);
            const vec3 c113 = vec3(0.147607, 0.511733, 0.557049);
            const vec3 c114 = vec3(0.14618, 0.515413, 0.556823);
            const vec3 c115 = vec3(0.144759, 0.519093, 0.556572);
            const vec3 c116 = vec3(0.143343, 0.522773, 0.556295);
            const vec3 c117 = vec3(0.141935, 0.526453, 0.555991);
            const vec3 c118 = vec3(0.140536, 0.530132, 0.555659);
            const vec3 c119 = vec3(0.139147, 0.533812, 0.555298);
            const vec3 c120 = vec3(0.13777, 0.537492, 0.554906);
            const vec3 c121 = vec3(0.136408, 0.541173, 0.554483);
            const vec3 c122 = vec3(0.135066, 0.544853, 0.554029);
            const vec3 c123 = vec3(0.133743, 0.548535, 0.553541);
            const vec3 c124 = vec3(0.132444, 0.552216, 0.553018);
            const vec3 c125 = vec3(0.131172, 0.555899, 0.552459);
            const vec3 c126 = vec3(0.129933, 0.559582, 0.551864);
            const vec3 c127 = vec3(0.128729, 0.563265, 0.551229);
            const vec3 c128 = vec3(0.127568, 0.566949, 0.550556);
            const vec3 c129 = vec3(0.126453, 0.570633, 0.549841);
            const vec3 c130 = vec3(0.125394, 0.574318, 0.549086);
            const vec3 c131 = vec3(0.124395, 0.578002, 0.548287);
            const vec3 c132 = vec3(0.123463, 0.581687, 0.547445);
            const vec3 c133 = vec3(0.122606, 0.585371, 0.546557);
            const vec3 c134 = vec3(0.121831, 0.589055, 0.545623);
            const vec3 c135 = vec3(0.121148, 0.592739, 0.544641);
            const vec3 c136 = vec3(0.120565, 0.596422, 0.543611);
            const vec3 c137 = vec3(0.120092, 0.600104, 0.54253);
            const vec3 c138 = vec3(0.119738, 0.603785, 0.5414);
            const vec3 c139 = vec3(0.119512, 0.607464, 0.540218);
            const vec3 c140 = vec3(0.119423, 0.611141, 0.538982);
            const vec3 c141 = vec3(0.119483, 0.614817, 0.537692);
            const vec3 c142 = vec3(0.119699, 0.61849, 0.536347);
            const vec3 c143 = vec3(0.120081, 0.622161, 0.534946);
            const vec3 c144 = vec3(0.120638, 0.625828, 0.533488);
            const vec3 c145 = vec3(0.12138, 0.629492, 0.531973);
            const vec3 c146 = vec3(0.122312, 0.633153, 0.530398);
            const vec3 c147 = vec3(0.123444, 0.636809, 0.528763);
            const vec3 c148 = vec3(0.12478, 0.640461, 0.527068);
            const vec3 c149 = vec3(0.126326, 0.644107, 0.525311);
            const vec3 c150 = vec3(0.128087, 0.647749, 0.523491);
            const vec3 c151 = vec3(0.130067, 0.651384, 0.521608);
            const vec3 c152 = vec3(0.132268, 0.655014, 0.519661);
            const vec3 c153 = vec3(0.134692, 0.658636, 0.517649);
            const vec3 c154 = vec3(0.137339, 0.662252, 0.515571);
            const vec3 c155 = vec3(0.14021, 0.665859, 0.513427);
            const vec3 c156 = vec3(0.143303, 0.669459, 0.511215);
            const vec3 c157 = vec3(0.146616, 0.67305, 0.508936);
            const vec3 c158 = vec3(0.150148, 0.676631, 0.506589);
            const vec3 c159 = vec3(0.153894, 0.680203, 0.504172);
            const vec3 c160 = vec3(0.157851, 0.683765, 0.501686);
            const vec3 c161 = vec3(0.162016, 0.687316, 0.499129);
            const vec3 c162 = vec3(0.166383, 0.690856, 0.496502);
            const vec3 c163 = vec3(0.170948, 0.694384, 0.493803);
            const vec3 c164 = vec3(0.175707, 0.6979, 0.491033);
            const vec3 c165 = vec3(0.180653, 0.701402, 0.488189);
            const vec3 c166 = vec3(0.185783, 0.704891, 0.485273);
            const vec3 c167 = vec3(0.19109, 0.708366, 0.482284);
            const vec3 c168 = vec3(0.196571, 0.711827, 0.479221);
            const vec3 c169 = vec3(0.202219, 0.715272, 0.476084);
            const vec3 c170 = vec3(0.20803, 0.718701, 0.472873);
            const vec3 c171 = vec3(0.214, 0.722114, 0.469588);
            const vec3 c172 = vec3(0.220124, 0.725509, 0.466226);
            const vec3 c173 = vec3(0.226397, 0.728888, 0.462789);
            const vec3 c174 = vec3(0.232815, 0.732247, 0.459277);
            const vec3 c175 = vec3(0.239374, 0.735588, 0.455688);
            const vec3 c176 = vec3(0.24607, 0.73891,