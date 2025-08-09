"""
Environment setup utilities for SubShader.

This module handles platform-specific environment configurations,
particularly for WSL graphics and OpenGL setup.
"""

import os
import sys

def env_init():
    """
    Configure environment for SubShader.
    
    This function handles all environment setup including:
    - WSL-specific graphics configuration
    - Development debug settings
    - Platform-specific optimizations
    """
    # WSL-specific setup
    if _is_wsl():
        # Set display for WSL if not already set
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':0'
        
        # WSL-specific OpenGL settings
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    
    # Development debug settings
    if os.environ.get('SUBSHADER_DEBUG', '0') == '1':
        os.environ['GL_DEBUG_OUTPUT'] = '1'
        os.environ['GL_DEBUG_OUTPUT_SYNCHRONOUS'] = '1'


def _is_wsl():
    """
    Detect if running in WSL 
    
    Returns:
        bool: True if running in WSL, False otherwise.
    """
    if os.name != 'posix':
        return False
    
    try:
        # Check for Microsoft in kernel release (WSL1)
        if 'microsoft' in os.uname().release.lower():
            return True
        
        # Check for WSL2 specific indicators
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info or 'wsl' in version_info:
                    return True
        
        return False
    except (AttributeError, OSError):
        return False