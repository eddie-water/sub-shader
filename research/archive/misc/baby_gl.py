import glfw
from OpenGL.GL import *

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(640, 480, "Test", None, None)
glfw.make_context_current(window)

print("Renderer:", glGetString(GL_RENDERER))
print("Version:", glGetString(GL_VERSION))

glfw.terminate()
