/*
 * Fragment Shader for Audio Visualizer
 * 
 * This shader determines the color of each pixel in the rendered plot.
 * It samples the scalogram texture at the provided texture coordinates,
 * normalizes the sampled value using adaptive scaling uniforms (valueMin
 * and valueMax), applies gamma correction, and maps the normalized
 * value to a color using the inferno colormap.
 * 
 * Inputs:
 *   texCoord (vec2): The texture coordinate for sampling the scalogram
 *   scalogram (sampler2D): The 2D texture containing the scalogram data
 *   valueMin (float): The minimum value for normalization
 *   valueMax (float): The maximum value for normalization
 * 
 * Outputs:
 *   fragColor (vec4): The final RGBA color for the pixel
 */

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
    // Read the value from the scalogram data at the given texture coordinate
    float value = texture(scalogram, texCoord).r;
    
    // Normalize using adaptive scaling
    float normalized = clamp((value - valueMin) / (valueMax - valueMin), 0.0, 1.0);
    normalized = pow(normalized, 0.7);  // Gamma correction
    
    // Grab the color from the inferno colormap using the normalized value
    vec3 color = inferno(normalized);
    fragColor = vec4(color, 1.0);
} 