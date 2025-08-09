"""
Shader source code for the audio visualizer.
"""

import os
from pathlib import Path

def get_shader_source(filename: str) -> str:
    """
    Read shader source code from a file.
    
    Args:
        filename: The name of the shader file (e.g., 'vertex_shader.glsl')
        
    Returns:
        str: The shader source code as a string
    """
    shader_dir = Path(__file__).parent
    shader_path = shader_dir / filename
    
    with open(shader_path, 'r') as f:
        return f.read()

def get_vertex_shader_source() -> str:
    """Get the vertex shader source code."""
    return get_shader_source('vertex_shader.glsl')

def get_fragment_shader_source() -> str:
    """Get the fragment shader source code."""
    return get_shader_source('fragment_shader.glsl') 