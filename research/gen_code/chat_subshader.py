import numpy as np
import cupy as cp
from scipy.io import wavfile
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
import threading
import time
from ctypes import c_void_p

# === PARAMETERS ===
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096
CWT_FREQS = cp.linspace(20, 8000, 128)
AUDIO_PATH = "audio_files/c4_and_c7_4_arps.wav"

# === CWT FUNCTION ===
def morlet_wavelet(freq, scale, n, sample_rate):
    t = cp.linspace(-1, 1, n)
    s = scale / sample_rate
    wavelet = cp.exp(2j * cp.pi * freq * t) * cp.exp(-t ** 2 / (2 * s ** 2))
    return wavelet / cp.sqrt(s)

def cwt_gpu(signal_gpu, freqs, sample_rate):
    n = signal_gpu.shape[0]
    output = cp.zeros((len(freqs), n), dtype=cp.complex64)
    signal_fft = cp.fft.fft(signal_gpu)
    for i, f in enumerate(freqs):
        wavelet = morlet_wavelet(f, n, n, sample_rate)
        output[i] = cp.fft.ifft(signal_fft * cp.fft.fft(wavelet))
    return cp.abs(output)

# === SHADER SETUP ===
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    texCoord = (position + 1.0) / 2.0;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 texCoord;
out vec4 fragColor;
uniform sampler2D scalogram;
void main() {
    float value = texture(scalogram, texCoord).r;
    fragColor = vec4(value, value, value, 1.0);
}
"""

def init_shader():
    return compileProgram(
        compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER),
    )

def init_window_shader_gl():
    # Init GLFW, create window and OpenGL context
    if not glfw.init():
        raise Exception("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "CWT Visualizer", None, None)
    print("Inside init_window. GLFW Context:", glfw.get_current_context())
    glfw.make_context_current(window)
    
    # Init Shader
    shader = init_shader()
    glUseProgram(shader)
    
    print("Before init_quad. GLFW Context:", glfw.get_current_context())
    vao = init_quad()

    return window, shader, vao

def init_quad():
    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(0)

    return vao

# === AUDIO BUFFER HANDLING ===
audio_gpu = None
chunk_queue = []
queue_lock = threading.Lock()

# === ASYNC CHUNK PRODUCER ===
def audio_producer():
    global audio_gpu
    pos = 0
    while pos + CHUNK_SIZE < len(audio_gpu):
        chunk = audio_gpu[pos:pos + CHUNK_SIZE]
        with queue_lock:
            chunk_queue.append(chunk)
        pos += CHUNK_SIZE
        time.sleep(CHUNK_SIZE / SAMPLE_RATE)  # simulate real-time pacing

# === MAIN ===
def main():
    # Detect and warn if not using an NVIDIA GPU for OpenGL rendering
    renderer = glGetString(GL_RENDERER)
    if renderer:
        decoded_renderer = renderer.decode()
        print("OpenGL Renderer:", decoded_renderer)
        if "NVIDIA" not in decoded_renderer:
            print("This may cause OpenGL context errors in WSL.")
            print("Try installing the full NVIDIA driver stack and restarting WSL.")

    global audio_gpu

    # === Load and normalize audio ===
    rate, audio = wavfile.read(AUDIO_PATH)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio))
    audio_gpu = cp.asarray(audio)

    # === Init window and shaders ===
    window, shader, vao = init_window_shader_gl()

    glClearColor(0.2, 0.2, 0.2, 1.0)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Set uniform for shader
    glUseProgram(shader)
    scalogram_loc = glGetUniformLocation(shader, "scalogram")
    glUniform1i(scalogram_loc, 0)

    # === Start async thread ===
    threading.Thread(target=audio_producer, daemon=True).start()

    while not glfw.window_should_close(window):
        with queue_lock:
            if chunk_queue:
                chunk = chunk_queue.pop(0)
            else:
                chunk = None

        if chunk is not None:
            cwt_out = cwt_gpu(chunk, CWT_FREQS, SAMPLE_RATE)
            cwt_np = cp.asnumpy(cwt_out)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, cwt_np.shape[1], cwt_np.shape[0], 0, GL_RED, GL_FLOAT, cwt_np)

        # Draw frame
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)
        glBindVertexArray(vao)
        glBindTexture(GL_TEXTURE_2D, texture)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
