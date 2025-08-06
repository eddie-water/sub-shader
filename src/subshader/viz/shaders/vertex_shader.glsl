/*
 * Vertex Shader for Audio Visualizer
 * 
 * A vertex shader is called once for each vertex in the geometry being 
 * rendered. It transforms vertex data from object space to clip space and 
 * prepares data for the fragment shader.
 * 
 * The vertex shader's job is to:
 * 1. Transform the input vertex position to clip space (gl_Position)
 * 2. Pass data to the fragment shader through output variables
 * 
 * Since the quad being rendered forms a 2D rectangle, it's already in the 
 * right coordinate system (-1,1 to 1,1). The main transformation here is
 * mapping the quad's corner positions to texture coordinates (0,0 to 1,1)
 * so the fragment shader can sample the source data texture correctly.
 * Vertex shaders typically transform vertex positions into the texture's 
 * coordinate system.
 * 
 * Inputs:
 *   position (vec2): Vertex position in normalized device coordinates (-1 to 1)
 * 
 * Outputs:
 *   texCoord (vec2): Texture coordinates for sampling (0 to 1)
 *   gl_Position (vec4): Vertex position in clip space (built-in GLSL env 
        variable that must be set)
 */

#version 330
in vec2 position;
out vec2 texCoord;

void main() {
    // 2D vertex is already in clip space, so pass through
    gl_Position = vec4(position, 0.0, 1.0);

    // position + 1 : range goes from [-1, 1] to [0, 2]
    // divide by 2 : range goes from [0, 2] to [0, 1]
    texCoord = (position + 1.0) / 2.0;
} 
