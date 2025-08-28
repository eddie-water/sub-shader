#!/usr/bin/env python3
"""
SubShader Diagnostic Mode
Run this script to test SubShader with comprehensive GL diagnostics enabled.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from subshader.viz.plotter import ShaderPlot
from subshader.dsp.wavelet import PyWavelet
from subshader.utils.gl_diagnostics import GLDiagnostics
from subshader.utils.logging import get_logger
import numpy as np

log = get_logger(__name__)

class DiagnosticRenderer:
    """Renderer with full GL diagnostics enabled."""
    
    def __init__(self, file_path: str, window_size: int = 1024, num_frames: int = 32, target_width: int = 512):
        self.file_path = file_path
        self.window_size = window_size
        self.num_frames = num_frames
        self.target_width = target_width
        
        log.info("🔍 Starting SubShader in DIAGNOSTIC MODE")
        log.info(f"Settings: window_size={window_size}, num_frames={num_frames}, target_width={target_width}")
        
        # Calculate expected texture size
        expected_width = num_frames * target_width
        log.info(f"Expected texture size: {expected_width} x ? pixels")
        
        self._check_gl_limits()
        self._init_components()
    
    def _check_gl_limits(self):
        """Check OpenGL limits before initialization."""
        import glfw
        import moderngl
        
        # Create temporary context to check limits
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        
        window = glfw.create_window(100, 100, "Diagnostic", None, None)
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        
        max_texture_size = int(ctx.info.get('GL_MAX_TEXTURE_SIZE', '0'))
        expected_width = self.num_frames * self.target_width
        
        log.info(f"OpenGL Max Texture Size: {max_texture_size}")
        log.info(f"Expected Texture Width: {expected_width}")
        
        if expected_width > max_texture_size:
            log.error(f"❌ TEXTURE SIZE EXCEEDS LIMIT!")
            log.error(f"   Expected: {expected_width} pixels")
            log.error(f"   Maximum:  {max_texture_size} pixels")
            log.error(f"   Ratio:    {expected_width / max_texture_size:.1f}x over limit")
            log.error(f"   Suggested fix: Reduce num_frames to {max_texture_size // self.target_width}")
        else:
            log.info(f"✅ Texture size within limits ({expected_width}/{max_texture_size})")
        
        glfw.destroy_window(window)
        glfw.terminate()
    
    def _init_components(self):
        """Initialize components with diagnostics."""
        log.info("🔧 Initializing wavelet processor...")
        self.wavelet = PyWavelet(44100, self.window_size)
        
        # Get expected frame shape
        frame_shape = self.wavelet.get_downsampled_result_shape()
        log.info(f"Frame shape: {frame_shape}")
        
        log.info("🔧 Initializing plotter...")
        self.plotter = ShaderPlot(self.file_path, frame_shape, self.num_frames)
        
        # Run diagnostics on the plotter's renderer
        self._diagnose_renderer()
    
    def _diagnose_renderer(self):
        """Run comprehensive diagnostics on the renderer."""
        log.info("🔍 Running comprehensive renderer diagnostics...")
        
        renderer = self.plotter.renderer
        
        # Diagnose shader program
        shader_info = GLDiagnostics.diagnose_shader_program(renderer.shader)
        if not shader_info['valid']:
            log.error("❌ Shader program issues:")
            for issue in shader_info['issues']:
                log.error(f"   - {issue}")
        else:
            log.info("✅ Shader program is valid")
        
        # Diagnose texture
        texture_info = GLDiagnostics.diagnose_texture(renderer.texture)
        if not texture_info['valid']:
            log.error("❌ Texture issues:")
            for issue in texture_info['issues']:
                log.error(f"   - {issue}")
        else:
            log.info("✅ Texture is valid")
        
        # Diagnose VAO
        vao_info = GLDiagnostics.diagnose_vao(renderer.vao)
        if not vao_info['valid']:
            log.error("❌ VAO issues:")
            for issue in vao_info['issues']:
                log.error(f"   - {issue}")
        else:
            log.info("✅ VAO is valid")
        
        # Check GL state
        GLDiagnostics.check_gl_error(renderer.ctx, "post-initialization state")
    
    def run_diagnostic_loop(self, duration_seconds: int = 5):
        """Run a short diagnostic loop."""
        log.info(f"🚀 Running diagnostic loop for {duration_seconds} seconds...")
        
        import time
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                # Process audio frame
                audio_frame = np.random.randn(self.window_size).astype(np.float32)  # Dummy data
                cwt_result = self.wavelet.compute_cwt(audio_frame)
                downsampled = self.wavelet.downsample(cwt_result, self.target_width)
                
                # Update visualization with diagnostics
                self._update_with_diagnostics(downsampled)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    log.info(f"Processed {frame_count} frames...")
                
                time.sleep(0.1)  # Slow down for diagnostics
        
        except KeyboardInterrupt:
            log.info("Diagnostic interrupted by user")
        
        log.info(f"✅ Diagnostic complete: {frame_count} frames processed")
    
    def _update_with_diagnostics(self, data):
        """Update visualization with full diagnostic checking."""
        # Check data validity
        if data is None or data.size == 0:
            log.error("❌ Invalid data for visualization")
            return
        
        data_min, data_max = data.min(), data.max()
        if data_min == data_max:
            log.warning(f"⚠️ Data has no variation: all values = {data_min}")
        
        # Update with GL error checking
        renderer = self.plotter.renderer
        
        GLDiagnostics.check_gl_error(renderer.ctx, "before frame update")
        
        # Update rolling buffer
        self.plotter.update_plot(data)
        
        GLDiagnostics.check_gl_error(renderer.ctx, "after frame update")
        
        # Safe render
        if not GLDiagnostics.safe_render(renderer.vao, renderer.ctx):
            log.error("❌ Render failed during diagnostic")

def main():
    """Run diagnostic mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SubShader Diagnostic Mode')
    parser.add_argument('--file', default='assets/audio/songs/beltran_sc_rip.wav', help='Audio file path')
    parser.add_argument('--window-size', type=int, default=1024, help='Window size')
    parser.add_argument('--num-frames', type=int, default=32, help='Number of frames')
    parser.add_argument('--target-width', type=int, default=512, help='Target width')
    parser.add_argument('--duration', type=int, default=5, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    try:
        diagnostic = DiagnosticRenderer(
            args.file, 
            args.window_size, 
            args.num_frames, 
            args.target_width
        )
        diagnostic.run_diagnostic_loop(args.duration)
        
    except Exception as e:
        log.error(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
