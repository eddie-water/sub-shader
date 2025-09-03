/*
 * Fragment Shader for Audio Visualizer
 * 
 * Determines the color of each pixel by reading the values from the source
 * data texture. The texture contains coefficient frames stacked
 * side-by-side in a scrolling buffer. Each pixel samples one coefficient
 * value (already globally normalized to [0,1] range) and maps it to a color
 * using matplotlib's inferno colormap for accurate data representation.
 * 
 * Inputs:
 *   texCoord (vec2): Which coefficient to read from the texture
 *   texture_sampler (sampler2D): Texture containing globally normalized coefficient frames
 *   gamma (float): Gamma correction factor for perceptual enhancement
 * 
 * Outputs:
 *   fragColor (vec4): Final color for this pixel
 */

#version 330
in vec2 texCoord;
out vec4 fragColor;
uniform sampler2D texture_sampler;
uniform float gamma;

vec3 inferno_colormap(float t) {
    // Matplotlib inferno colormap - continuous perceptually uniform colormap
    // Data from matplotlib's inferno colormap lookup table
    t = clamp(t, 0.0, 1.0);
    
    // High-resolution inferno colormap control points (16 samples for smooth interpolation)
    const vec3 colors[16] = vec3[](
        vec3(0.001462, 0.000466, 0.013866),  // 0.0 - near black
        vec3(0.087411, 0.044556, 0.224813),  // 0.067 - dark purple
        vec3(0.258234, 0.038571, 0.406485),  // 0.133 - purple
        vec3(0.416331, 0.090203, 0.432943),  // 0.2 - purple-red
        vec3(0.562861, 0.156942, 0.359557),  // 0.267 - red-purple
        vec3(0.692840, 0.165141, 0.207872),  // 0.333 - red
        vec3(0.798216, 0.280197, 0.469538),  // 0.4 - red-orange
        vec3(0.881443, 0.392529, 0.101718),  // 0.467 - orange-red
        vec3(0.951546, 0.510943, 0.052167),  // 0.533 - orange
        vec3(0.988362, 0.645450, 0.039886),  // 0.6 - yellow-orange
        vec3(0.995380, 0.786264, 0.197138),  // 0.667 - yellow
        vec3(0.992541, 0.917399, 0.472873),  // 0.733 - bright yellow
        vec3(0.992357, 0.999825, 0.644924),  // 0.8 - yellow-white
        vec3(0.998364, 0.998364, 0.745097),  // 0.867 - light yellow
        vec3(0.999643, 0.999643, 0.899410),  // 0.933 - very light
        vec3(1.000000, 1.000000, 1.000000)   // 1.0 - white
    );
    
    // Map t to array index range
    float scaled = t * 15.0;  // 0-15 range
    int index = int(floor(scaled));
    float frac = scaled - float(index);
    
    // Clamp index to valid range
    index = clamp(index, 0, 14);
    int next_index = min(index + 1, 15);
    
    // Linear interpolation between adjacent colors
    return mix(colors[index], colors[next_index], frac);
}

void main() {
    // Read the value from the texture data at the given texture coordinate
    float value = texture(texture_sampler, texCoord).r;
    
    float normalized = clamp(value, 0.0, 1.0);
    normalized = pow(normalized, gamma);
    
    // Grab the color from matplotlib inferno colormap using the normalized value
    vec3 color = inferno_colormap(normalized);
    fragColor = vec4(color, 1.0);
} 