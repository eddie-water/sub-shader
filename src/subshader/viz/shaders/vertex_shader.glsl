/*
 * Vertex Shader for Audio Visualizer
 * 
 * This shader transforms vertex positions and generates texture coordinates
 * for rendering the plot. It takes a 2D vertex position as input and maps
 * it to normalized texture coordinates [0, 1] for OpenGL texture sampling.
 * The texture coordinates are then passed to the fragment shader for
 * sampling the correct location from a 2D texture.
 * 
 * Inputs:
 *   position (vec2): The vertex position as a 2D vector (x, y)
 * 
 * Outputs:
 *   texCoord (vec2): The texture coordinate as a 2D vector (x, y)
 */

#version 330
in vec2 position;
out vec2 texCoord;

void main() {
    // Map the position from [-1, 1] to [0, 1] for texture sampling
    texCoord = (position + 1.0) / 2.0;

    // Set the final position of the vertex in the 3D space
    gl_Position = vec4(position, 0.0, 1.0);
} 