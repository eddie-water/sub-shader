# SubShader GL Diagnostics

## Overview
The GL diagnostics system helps debug OpenGL issues without cluttering the main application.

## Files
- `src/subshader/utils/gl_diagnostics.py` - Core diagnostic utilities
- `diagnostic_mode.py` - Standalone diagnostic script

## Usage

### Main Application (Clean)
```bash
python src/subshader/__main__.py
```
Runs normally with minimal GL error checking.

### Diagnostic Mode (Verbose)
```bash
python diagnostic_mode.py --duration 5
```
Runs with comprehensive GL diagnostics for 5 seconds.

### Options
```bash
python diagnostic_mode.py --help
```

## What Diagnostics Check

### OpenGL Limits
- Maximum texture size vs. required texture size
- Warns if texture dimensions exceed GPU capabilities

### Shader Program
- Program compilation status
- Uniform locations and types
- Program validity

### Textures
- Size and format validation
- Data range analysis (warns about suspicious values)
- Upload success verification

### Vertex Arrays
- VAO validity and vertex count
- Rendering success/failure

### Error Detection
- GL state before/after critical operations
- Detailed error context and suggestions

## Common Issues Detected

### GL_INVALID_VALUE
- Usually texture size exceeds `GL_MAX_TEXTURE_SIZE`
- Solution: Reduce `NUM_FRAMES` or `TARGET_WIDTH`

### GL_INVALID_OPERATION
- Shader uniform mismatch
- Invalid texture format/usage
- Rendering with incomplete state

### Data Issues
- All-zero texture data (no variation)
- Data outside expected [0,1] range
- Invalid array shapes

## Example Output
```
🔍 Starting SubShader in DIAGNOSTIC MODE
OpenGL Max Texture Size: 16384
Expected Texture Width: 16384
✅ Texture size within limits (16384/16384)
🔧 Initializing wavelet processor...
🔧 Initializing plotter...
✅ Shader program is valid
✅ Texture is valid
✅ VAO is valid
🚀 Running diagnostic loop for 5 seconds...
```

## Integration
The diagnostic utilities are available but not imported by default in the main application, keeping performance optimal while providing debugging capabilities when needed.
