import numpy as np
import cupy as cp
from scipy.io import wavfile
import moderngl
import glfw
import threading
import time

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

# === AUDIO BUFFER HANDLING ===
audio_gpu = None
chunk_queue = []
queue_lock = threading.Lock()

def audio_producer():
    global audio_gpu
    pos = 0
    while pos + CHUNK_SIZE < len(audio_gpu):
        chunk = audio_gpu[pos:pos + CHUNK_SIZE]
        with queue_lock:
            chunk_queue.append(chunk)
        pos += CHUNK_SIZE
        time.sleep(CHUNK_SIZE / SAMPLE_RATE)

# === MAIN ===
def main():
    global audio_gpu

    # Load and normalize audio
    rate, audio = wavfile.read(AUDIO_PATH)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio))
    audio_gpu = cp.asarray(audio)

    # Init GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "CWT Visualizer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    ctx = moderngl.create_context()

    # Shaders
    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 position;
        out vec2 texCoord;
        void main() {
            texCoord = (position + 1.0) / 2.0;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """,
        fragment_shader="""
        #version 330
        in vec2 texCoord;
        out vec4 fragColor;
        uniform sampler2D scalogram;
        void main() {
            float value = texture(scalogram, texCoord).r;
            fragColor = vec4(value, value, value, 1.0);
        }
        """,
    )

    # Quad setup
    quad = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'position')

    # Texture placeholder
    texture = ctx.texture((CHUNK_SIZE, len(CWT_FREQS)), 1, dtype='f4')
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    prog['scalogram'] = 0  # texture unit 0

    # Start producer thread
    threading.Thread(target=audio_producer, daemon=True).start()

    while not glfw.window_should_close(window):
        glfw.poll_events()

        with queue_lock:
            chunk = chunk_queue.pop(0) if chunk_queue else None

        if chunk is not None:
            cwt_out = cwt_gpu(chunk, CWT_FREQS, SAMPLE_RATE)
            cwt_np = cp.asnumpy(cwt_out).astype('f4')
            texture.write(cwt_np.tobytes())

        texture.use(location=0)
        ctx.clear(0.2, 0.2, 0.2)
        vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
