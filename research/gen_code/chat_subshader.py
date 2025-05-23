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
CWT_FREQS = cp.linspace(20, 8000, 128) # TODO NOW Fix frequency range
AUDIO_PATH = "audio_files/c4_and_c7_4_arps.wav"

# === CWT KERNEL ===
def morlet_wavelet(freq, scale, n, sample_rate):
    # TODO NEXT Verify wavelet setup
    t = cp.linspace(-1, 1, n)
    s = scale / sample_rate
    wavelet = cp.exp(2j * cp.pi * freq * t) * cp.exp(-t ** 2 / (2 * s ** 2))
    return wavelet / cp.sqrt(s) 

# === CWT FUNCTION ===
def cwt_gpu(signal_gpu, freqs, sample_rate):
    n = signal_gpu.shape[0]
    output = cp.zeros((len(freqs), n), dtype=cp.complex64)
    signal_fft = cp.fft.fft(signal_gpu)
    for i, f in enumerate(freqs):
        wavelet = morlet_wavelet(f, n, n, sample_rate)

        # TODO NEXT Weird way to do this, fix readability
        output[i] = cp.fft.ifft(signal_fft * cp.fft.fft(wavelet))
    return cp.abs(output)

# === AUDIO BUFFER HANDLING ===
audio_gpu = None
chunk_queue = []
queue_lock = threading.Lock()

# === AUDIO STREAM THREAD ===
def audio_producer():
    global audio_gpu
    i = 0
    while i + CHUNK_SIZE < len(audio_gpu):
        chunk = audio_gpu[i : i + CHUNK_SIZE]
        with queue_lock:
            chunk_queue.append(chunk)
        i += CHUNK_SIZE
        time.sleep(CHUNK_SIZE / SAMPLE_RATE)

# === MAIN LOOP ===
def main():
    global audio_gpu

    # Load and normalize audio 
    rate, audio = wavfile.read(AUDIO_PATH)
    if audio.ndim > 1:
        # Convert stereo to mono
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio))
    audio_gpu = cp.asarray(audio)

    # Initialize GLFW window and OpenGL context 
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

    # Compile shaders and set up program 
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

    # Set up quad vertex buffer and vertex array object 
    quad = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'position')

    # Create 2D texture placeholder for scalogram 
    texture = ctx.texture((CHUNK_SIZE, len(CWT_FREQS)), 1, dtype='f4')
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    prog['scalogram'] = 0  # texture unit 0

    # Start real-time audio chunking thread ===
    threading.Thread(target=audio_producer, daemon=True).start()

    # Timing and FPS setup 
    frame_times = []
    fps_timer = time.time()
    fps_value = 0.0

    while not glfw.window_should_close(window):
        frame_start = time.perf_counter()

        glfw.poll_events()

        # Get audio chunk from queue 
        with queue_lock:
            chunk = chunk_queue.pop(0) if chunk_queue else None

        # Perform CWT and update texture 
        if chunk is not None:
            t0 = time.perf_counter()
            cwt_out = cwt_gpu(chunk, CWT_FREQS, SAMPLE_RATE)
            t1 = time.perf_counter()

            cwt_np = cp.asnumpy(cwt_out).astype('f4')
            texture.write(cwt_np.tobytes())
            t2 = time.perf_counter()

            print(f"CWT: {(t1 - t0)*1000:.2f} ms \t | Texture update: {(t2 - t1)*1000:.2f} ms")

        # Render frame 
        texture.use(location=0)
        ctx.clear(0.2, 0.2, 0.2)
        vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(window)

        # FPS Calculation 
        frame_end = time.perf_counter()
        frame_times.append(frame_end - frame_start)
        if time.time() - fps_timer > 1.0:
            avg_frame = sum(frame_times) / len(frame_times)
            print(f"FPS: {1.0 / avg_frame:.2f}")
            frame_times.clear()
            fps_timer = time.time()

    glfw.terminate()

if __name__ == "__main__":
    main()
