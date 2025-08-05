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
    
    // Define color stops with their positions
    vec3 color1 = vec3(0.0, 0.0, 0.3);   // Dark blue at 0.0
    vec3 color2 = vec3(0.3, 0.0, 0.5);   // Purple at 0.3
    vec3 color3 = vec3(1.0, 0.0, 0.0);   // Red at 0.6
    vec3 color4 = vec3(1.0, 0.5, 0.0);   // Orange at 0.8
    vec3 color5 = vec3(1.0, 1.0, 1.0);   // White at 1.0
    
    // Use smoothstep to create smooth transitions between segments
    float segment1 = smoothstep(0.0, 0.3, t) * (1.0 - smoothstep(0.3, 0.6, t));
    float segment2 = smoothstep(0.3, 0.6, t) * (1.0 - smoothstep(0.6, 0.8, t));
    float segment3 = smoothstep(0.6, 0.8, t) * (1.0 - smoothstep(0.8, 1.0, t));
    float segment4 = smoothstep(0.8, 1.0, t);
    
    // Calculate interpolated colors for each segment
    vec3 segment_color1 = mix(color1, color2, smoothstep(0.0, 0.3, t));
    vec3 segment_color2 = mix(color2, color3, smoothstep(0.3, 0.6, t));
    vec3 segment_color3 = mix(color3, color4, smoothstep(0.6, 0.8, t));
    vec3 segment_color4 = mix(color4, color5, smoothstep(0.8, 1.0, t));
    
    // Combine segments using smoothstep weights
    return segment_color1 * segment1 + segment_color2 * segment2 + 
           segment_color3 * segment3 + segment_color4 * segment4;
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