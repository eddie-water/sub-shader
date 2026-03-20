"""
OpenGL diagnostic utilities for debugging GL_INVALID_OPERATION errors.
"""

import moderngl
from subshader.utils.logging import get_logger

log = get_logger(__name__)

class GLDiagnostics:
    """OpenGL diagnostic wrapper for debugging rendering issues."""
    
    @staticmethod
    def check_gl_error(ctx: moderngl.Context, operation: str) -> bool:
        """
        Check for OpenGL errors and log detailed information.
        
        Args:
            ctx: ModernGL context
            operation: Description of the operation being performed
            
        Returns:
            bool: True if no error, False if error occurred
        """
        error = ctx.error
        if error != 'GL_NO_ERROR':
            log.error(f"GL Error during '{operation}': {error}")
            return False
        else:
            log.debug(f"GL OK: {operation}")
            return True
    
    @staticmethod
    def diagnose_shader_program(program: moderngl.Program) -> dict:
        """
        Diagnose shader program state and uniforms.
        
        Args:
            program: ModernGL shader program
            
        Returns:
            dict: Diagnostic information
        """
        info = {
            'valid': True,
            'uniforms': {},
            'attributes': {},
            'issues': []
        }
        
        try:
            # Check if program is valid
            if not hasattr(program, '_glo') or program._glo == 0:
                info['valid'] = False
                info['issues'].append("Program object is invalid")
                return info
            
            log.info(f"Shader program diagnostics:")
            log.info(f"  Program ID: {program._glo}")
            
            # Try to get uniform information (this might fail in some ModernGL versions)
            try:
                # ModernGL doesn't expose uniform enumeration easily, but we can try
                # to access the internal uniform dictionary if it exists
                if hasattr(program, '_uniforms'):
                    for name, uniform in program._uniforms.items():
                        info['uniforms'][name] = {
                            'location': uniform.location if hasattr(uniform, 'location') else 'unknown',
                            'type': str(type(uniform))
                        }
                        log.info(f"  Uniform: {name} -> location {info['uniforms'][name]['location']}")
                else:
                    log.info("  Uniforms: Cannot enumerate (ModernGL internal)")
            except Exception as e:
                info['issues'].append(f"Cannot enumerate uniforms: {e}")
                log.warning(f"Cannot enumerate uniforms: {e}")
            
        except Exception as e:
            info['valid'] = False
            info['issues'].append(f"Program diagnostic failed: {e}")
            log.error(f"Program diagnostic failed: {e}")
        
        return info
    
    @staticmethod
    def diagnose_texture(texture: moderngl.Texture) -> dict:
        """
        Diagnose texture state.
        
        Args:
            texture: ModernGL texture
            
        Returns:
            dict: Diagnostic information
        """
        info = {
            'valid': True,
            'size': None,
            'format': None,
            'issues': []
        }
        
        try:
            info['size'] = texture.size
            info['format'] = texture.components
            log.info(f"Texture diagnostics:")
            log.info(f"  Size: {info['size']}")
            log.info(f"  Components: {info['format']}")
            log.info(f"  Filter: {texture.filter}")
            
        except Exception as e:
            info['valid'] = False
            info['issues'].append(f"Texture diagnostic failed: {e}")
            log.error(f"Texture diagnostic failed: {e}")
        
        return info
    
    @staticmethod
    def diagnose_vao(vao: moderngl.VertexArray) -> dict:
        """
        Diagnose vertex array object state.
        
        Args:
            vao: ModernGL vertex array
            
        Returns:
            dict: Diagnostic information
        """
        info = {
            'valid': True,
            'vertices': None,
            'issues': []
        }
        
        try:
            # Try to get vertex count if available
            if hasattr(vao, 'vertices'):
                info['vertices'] = vao.vertices
            
            log.info(f"VAO diagnostics:")
            log.info(f"  Vertices: {info['vertices']}")
            
        except Exception as e:
            info['valid'] = False
            info['issues'].append(f"VAO diagnostic failed: {e}")
            log.error(f"VAO diagnostic failed: {e}")
        
        return info
    
    @staticmethod
    def safe_uniform_set(program: moderngl.Program, name: str, value, ctx: moderngl.Context) -> bool:
        """
        Safely set a uniform with error checking.
        
        Args:
            program: Shader program
            name: Uniform name
            value: Uniform value
            ctx: OpenGL context
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Check for errors before setting uniform
            if not GLDiagnostics.check_gl_error(ctx, f"before setting uniform '{name}'"):
                return False
            
            # Try to set the uniform
            program[name] = value
            log.debug(f"Set uniform '{name}' = {value}")
            
            # Check for errors after setting uniform
            if not GLDiagnostics.check_gl_error(ctx, f"after setting uniform '{name}'"):
                return False
            
            return True
            
        except KeyError:
            log.error(f"Uniform '{name}' not found in shader program")
            return False
        except Exception as e:
            log.error(f"Failed to set uniform '{name}': {e}")
            return False
    
    @staticmethod
    def safe_render(vao: moderngl.VertexArray, ctx: moderngl.Context) -> bool:
        """
        Safely render with error checking.
        
        Args:
            vao: Vertex array to render
            ctx: OpenGL context
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Check for errors before rendering
            if not GLDiagnostics.check_gl_error(ctx, "before render"):
                return False
            
            # Attempt to render
            vao.render(moderngl.TRIANGLE_STRIP)
            log.debug("VAO render call completed")
            
            # Check for errors after rendering
            if not GLDiagnostics.check_gl_error(ctx, "after render"):
                return False
            
            return True
            
        except Exception as e:
            log.error(f"Render failed with exception: {e}")
            return False
