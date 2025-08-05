/*
 * Fragment Shader for Audio Visualizer
 * 
 * Determines the color of each pixel by reading the values from the source
 * data texture. The texture contains coefficient frames stacked
 * side-by-side in a scrolling buffer. Each pixel samples one coefficient
 * value, normalizes it using adaptive scaling, and maps it to a color
 * using a custom colormap that includes orange and white for maximum intensity.
 * 
 * Inputs:
 *   texCoord (vec2): Which coefficient to read from the texture
 *   scalogram (sampler2D): Texture containing stacked coefficient frames
 *   valueMin (float): Minimum value for normalization
 *   valueMax (float): Maximum value for normalization
 * 
 * Outputs:
 *   fragColor (vec4): Final color for this pixel
 */

#version 330
in vec2 texCoord;
out vec4 fragColor;
uniform sampler2D scalogram;
uniform float valueMin;
uniform float valueMax;

vec3 custom_colormap(float t) {
    // Custom colormap that includes orange and white for maximum intensity
    // Similar to professional DAW spectrum analyzers
    t = clamp(t, 0.0, 1.0);
    
    if (t < 0.3) {
        // Dark blue to purple for quiet parts
        return mix(vec3(0.0, 0.0, 0.3), vec3(0.3, 0.0, 0.5), t / 0.3);
    } else if (t < 0.6) {
        // Purple to red for medium intensity
        return mix(vec3(0.3, 0.0, 0.5), vec3(1.0, 0.0, 0.0), (t - 0.3) / 0.3);
    } else if (t < 0.8) {
        // Red to orange for high intensity
        return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.6) / 0.2);
    } else {
        // Orange to white for maximum intensity
        return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.8) / 0.2);
    }
}

void main() {
    // Read the value from the scalogram data at the given texture coordinate
    float value = texture(scalogram, texCoord).r;
    
    // Since data is already normalized to [0,1], use a more reasonable scaling
    // This should allow the brightest parts to reach orange/white colors
    float normalized = clamp(value / 0.3, 0.0, 1.0);  // More reasonable scaling
    normalized = pow(normalized, 0.4);  // Less gamma to preserve bright colors
    
    // Grab the color from our custom colormap using the normalized value
    vec3 color = custom_colormap(normalized);
    fragColor = vec4(color, 1.0);
} 