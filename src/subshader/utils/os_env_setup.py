"""
Environment setup utilities for SubShader.

This module handles platform-specific environment configurations,
particularly for WSL graphics and OpenGL setup.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def setup_wsl_environment():
    """
    Configure environment variables and settings for WSL graphics.
    
    This function detects if running in WSL and sets up appropriate
    environment variables for graphics and OpenGL compatibility.
    """
    if not _is_wsl():
        return
    
    logger.info("Detected WSL environment, configuring graphics...")
    
    # Set display for WSL if not already set
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
        logger.debug("Set DISPLAY=:0")
    
    # WSL-specific OpenGL settings
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    
    logger.info("WSL graphics environment configured")

def _is_wsl():
    """
    Detect if running in WSL (Windows Subsystem for Linux).
    
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

def setup_glfw_environment():
    """
    Configure GLFW-specific environment settings.
    
    This function sets up any GLFW-related environment variables
    that might be needed for proper operation.
    """
    # GLFW error callback is now handled in the plotter module
    # where GLFW is actually initialized
    logger.debug("GLFW environment configured")

def setup_development_environment():
    """
    Configure environment for development and debugging.
    
    This function sets up environment variables that are useful
    during development but might not be needed in production.
    """
    # Enable OpenGL debug output in development
    if os.environ.get('SUBSHADER_DEBUG', '0') == '1':
        os.environ['GL_DEBUG_OUTPUT'] = '1'
        os.environ['GL_DEBUG_OUTPUT_SYNCHRONOUS'] = '1'
        logger.info("Development debug mode enabled")

def setup_all_environments():
    """
    Run all environment setup functions.
    
    This is the main entry point for environment configuration.
    Call this early in your application startup.
    """
    setup_wsl_environment()
    setup_glfw_environment()
    setup_development_environment()
    
    logger.info("Environment setup complete") 