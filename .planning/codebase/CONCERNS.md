# Codebase Concerns

**Analysis Date:** 2026-03-21

## Tech Debt

**Wavelet Time Support Calculation (TODO-36):**
- Issue: Frame rate validation in `src/subshader/config.py` (lines 348-357) is commented out because the calculation doesn't account for the smallest wavelet's time support requirement
- Files: `src/subshader/config.py:348-357`
- Impact: Cannot properly validate that chunk_size is large enough to capture the full time support of the lowest frequency wavelet. Users may not receive warnings when configuration is inadequate for proper CWT analysis
- Fix approach: Implement proper time support calculation for the lowest frequency wavelet and use it to validate minimum chunk_size requirements. This requires analyzing the wavelet kernel dimensions across all frequencies

**Wavelet Kernel Constant Unvalidated (TODO ISSUE-36):**
- Issue: The Complex Morlet wavelet constant "cmor1.5-1.0" is hardcoded without explanation in `src/subshader/dsp/wavelet.py:284`
- Files: `src/subshader/dsp/wavelet.py:283-284`
- Impact: Users cannot easily modify the wavelet characteristics. No documentation explains why these specific parameters (1.5 and 1.0) were chosen
- Fix approach: Make the constant configurable via WaveletConfig, add parametric support for different Morlet wavelet families, and document the parameter selection rationale

**Overlap and Plot Relationship (TODO-45):**
- Issue: Overlap factor calculation and visualization in `src/subshader/config.py:65` is incomplete
- Files: `src/subshader/config.py:65`
- Impact: The relationship between overlap_factor configuration (0-0.9 range) and actual plot rendering may be inconsistent
- Fix approach: Validate and document the precise relationship between audio overlap_factor and visual plot display. Ensure plot windows correctly represent overlapping regions

**Missing Documentation on Scale Normalization (TODO-37):**
- Issue: Scale-dependent normalization in `src/subshader/dsp/wavelet.py:427` uses sqrt scaling with incomplete explanation
- Files: `src/subshader/dsp/wavelet.py:427, 501, 517`
- Impact: Comments reference missing figures and incomplete mathematical explanations. Makes the code harder to understand and modify
- Fix approach: Add complete documentation with figures showing (1) why sqrt is needed (power = mag^2), (2) COI mask visualization, and (3) center keep slice visualization

## Known Bugs

**Typo in Color Constant:**
- Symptoms: One color value is misspelled as 'orangwe' instead of a valid matplotlib color code
- Files: `src/subshader/viz/comparison_navigator.py:89`
- Trigger: Running AudioNavigator will cause a matplotlib error when attempting to render the even-indexed chunks plot
- Impact: Visualization component fails with invalid color specification
- Workaround: Manually comment out the line and use a valid color hex code like '#FF9500'

**Orphaned OpenGL Context TODO:**
- Symptoms: Vague TODO comment about OpenGL context at line 154
- Files: `src/subshader/viz/plotter.py:154`
- Trigger: Context initialization in GLContext.__init__
- Impact: Unclear if there's an unresolved issue with the context setup or just a forgotten comment
- Workaround: None - clarify or remove the comment

## Security Considerations

**CuPy GPU Memory Management:**
- Risk: Incomplete cleanup of GPU memory in error conditions. The cleanup method in `src/subshader/dsp/wavelet.py:609-620` may not fully release GPU memory if exceptions occur during processing
- Files: `src/subshader/dsp/wavelet.py:609-620`
- Current mitigation: Try-except block catches general exceptions and logs warnings
- Recommendations: (1) Add context manager pattern for GPU resource allocation, (2) implement robust cleanup on exceptions during CWT computation, (3) add GPU memory monitoring to warn if memory leaks occur

**Bare Exception Handling:**
- Risk: Multiple bare or overly broad exception handlers that could mask real errors
- Files: `src/subshader/config.py:46`, `src/subshader/viz/plotter.py:501`, `src/subshader/viz/plotter.py:773`
- Current mitigation: Logging in most cases
- Recommendations: Replace broad Exception handlers with specific exception types (e.g., RuntimeError, FileNotFoundError, OpenGL-specific errors)

**Audio File Path Configuration:**
- Risk: Hardcoded audio file path in `src/subshader/__main__.py:45` overrides config - "assets/audio/daw/a2a3_a4_minor_scale.wav"
- Files: `src/subshader/__main__.py:45`
- Current mitigation: None - value is hardcoded
- Recommendations: Make audio file path configurable via CLI argument or environment variable instead of hardcoding

## Performance Bottlenecks

**GPU-CPU Data Transfer Logging Overhead:**
- Problem: Expensive logging statements on every frame in GPU wavelet implementation
- Files: `src/subshader/dsp/wavelet.py:593, 602`
- Cause: `log.info()` calls inside CuPyWavelet.class_specific_cwt() which runs per frame showing CPU→GPU and GPU→CPU transfer
- Improvement path: (1) Move logging to debug level, (2) log only on first execution or periodically, (3) implement batch logging that reports aggregate statistics

**Texture Size Exceeding OpenGL Limits:**
- Problem: Configuration can produce texture dimensions that exceed GPU limits (16384 max)
- Files: `src/subshader/config.py:240-249`
- Cause: Unbounded num_frames * target_width calculation
- Current impact: Critical error at runtime if texture exceeds limits
- Improvement path: (1) Add early validation before processing starts, (2) implement automatic downsampling if limits exceeded, (3) provide clear user guidance on parameter ranges

**IntensityTracker Percentile Calculation:**
- Problem: Computing percentile on every frame in `src/subshader/viz/plot_normalizer.py:44` may be expensive for large data
- Files: `src/subshader/viz/plot_normalizer.py:33-55`
- Cause: np.percentile() is called per frame without caching or approximation
- Improvement path: Consider using approximate percentile methods or precomputed histograms for faster updates

**Frame Buffer Flattening:**
- Problem: CircularFrameBuffer.get_flattened_buffer() called every frame to create contiguous texture data
- Files: `src/subshader/viz/plotter.py` (CircularFrameBuffer class)
- Cause: Efficient GPU upload requires contiguous memory, but frames are stored in circular manner
- Improvement path: Implement ring buffer with built-in flattening or use GPU side texture rotation

## Fragile Areas

**CuPy Availability at Runtime:**
- Files: `src/subshader/dsp/wavelet.py`, `src/subshader/viz/plot_normalizer.py`
- Why fragile: CuPy import is required but only conditionally imported in plot_normalizer. If CUDA/CuPy not available, system fails silently or with unclear errors
- Safe modification: (1) Add explicit GPU availability check at startup, (2) implement CPU fallback for plot_normalizer, (3) provide clear error messages with setup instructions
- Test coverage: No error handling tests for missing CUDA/CuPy

**Complex Configuration Validation Chain:**
- Files: `src/subshader/config.py:218-359`
- Why fragile: Multiple nested validate() methods (audio, wavelet, viz) with accumulated error messages. Easy to miss edge cases in the interaction between parameters
- Safe modification: Add unit tests for each validation method independently and for all combinations of invalid parameters
- Test coverage: No visible unit tests for configuration validation

**Plotter/Visualization Layer Cleanup:**
- Files: `src/subshader/viz/plotter.py:133-137`
- Why fragile: Multiple resources (GLFW window, ModernGL context, textures) need proper cleanup order. If cleanup fails, may leave GPU resources allocated or crash on exit
- Safe modification: Implement context manager pattern and explicit resource lifecycle management with tests
- Test coverage: Manual/integration testing only

**Hardcoded Overlap Factor Limits:**
- Files: `src/subshader/config.py:83-84`
- Why fragile: Overlap factor limited to 0.0-0.9 but rationale unclear. AudioInput calculates hop_size as `chunk_size * (1 - overlap_factor)` - if overlap=0.9, hop_size becomes 10% of chunk, causing very slow progression through audio
- Safe modification: Document why 0.9 is the maximum and test edge cases (overlap=0, 0.5, 0.9)
- Test coverage: No unit tests for edge case overlap values

## Scaling Limits

**GPU Memory Constraints:**
- Current capacity: Configuration estimates up to 2GB GPU memory usage
- Limit: OpenGL maximum texture size is 16384 pixels (configurable in validate() at 16384)
- Scaling path: (1) Implement tiled texture rendering for larger outputs, (2) add progressive detail levels, (3) implement GPU memory pooling and profiling

**Audio Chunk Processing:**
- Current capacity: Configuration tested with chunk_size up to 16384 samples
- Limit: Lowest frequency wavelets need significant time support (scales to 44100 Hz / 27.5 Hz = 1604 samples minimum). As num_octaves increases, memory grows linearly
- Scaling path: (1) Implement streaming wavelet computation, (2) add frequency band selection (don't compute all octaves), (3) implement progressive CWT with priority bands

**Real-Time Frame Rate:**
- Current capacity: Targets 30-60 FPS
- Limit: Configuration shows performance warnings at >200 FPS or >10M operations per frame
- Scaling path: (1) Implement adaptive downsampling based on frame rate targets, (2) use GPU compute shaders for downsampling instead of CPU, (3) decouple CWT computation from rendering

## Dependencies at Risk

**CuPy GPU Acceleration (Critical):**
- Risk: CuPy requires NVIDIA GPU with CUDA support. System has hard dependency on GPU acceleration with no CPU fallback
- Impact: Application cannot run on systems without CUDA-capable GPU or in CI/CD environments without GPU
- Migration plan: Implement CPU fallback path using NumPyWavelet. Make CuPy optional with auto-detection and graceful degradation

**PyQt5 and PyQtGraph (Medium Risk):**
- Risk: Legacy PyQt5 implementation kept for "debug use" according to docstring in `src/subshader/viz/plotter.py:9`
- Impact: Dead code path that's never tested, unused dependencies in requirements
- Migration plan: (1) Remove PyQtGraph-based plotter if not actively used, (2) add clear deprecation notice if needed, (3) migrate any required debug visualization to shader-based system

**Hard OpenGL 3.3 Requirement:**
- Risk: Application requires OpenGL 3.3 Core Profile (no compatibility mode)
- Impact: May fail on older systems or with certain GPU drivers
- Migration plan: (1) Add fallback to OpenGL 2.1, (2) improve error messages for incompatible systems

## Missing Critical Features

**Audio Input Format Support:**
- Problem: Only supports formats via soundfile library, but error handling doesn't clearly communicate unsupported formats
- Blocks: Users cannot easily understand why certain audio files fail to load
- Recommendation: Add explicit format validation with helpful error messages listing supported formats

**Real-Time Audio Input:**
- Problem: System only supports pre-loaded audio files, not real-time microphone input
- Blocks: Cannot create true real-time visualizer for live audio
- Recommendation: Add optional sounddevice integration for microphone/line-in input

**Configuration Persistence:**
- Problem: All configuration is hardcoded or passed via method parameters, no config file loading
- Blocks: Users cannot save/load preferred settings between runs
- Recommendation: Implement YAML/JSON config file loading with sensible defaults

## Test Coverage Gaps

**Configuration Validation:**
- What's not tested: Edge cases in config.validate() for all invalid parameter combinations
- Files: `src/subshader/config.py:218-359`
- Risk: Invalid configurations may be silently accepted or fail in unexpected ways during runtime
- Priority: High

**GPU Memory Estimation:**
- What's not tested: Accuracy of GPU memory calculations in _validate_gpu_memory()
- Files: `src/subshader/config.py:265-294`
- Risk: Estimates may be significantly off on different GPU architectures
- Priority: Medium

**Wavelet Reliability Region:**
- What's not tested: Correctness of reliable_slice calculation and its actual filtering effectiveness
- Files: `src/subshader/dsp/wavelet.py:395-417`
- Risk: Edge artifacts may still appear despite slicing
- Priority: Medium

**Colormap Application Robustness:**
- What's not tested: Exception handling in colormap application fallback
- Files: `src/subshader/viz/plotter.py:771-775`
- Risk: Silent failures could result in incorrect color visualization
- Priority: Low

**Frame Buffer Circular Logic:**
- What's not tested: Off-by-one errors or buffer overflow in circular frame buffer
- Files: `src/subshader/viz/plotter.py` (CircularFrameBuffer class)
- Risk: Memory corruption or visual artifacts in rendered output
- Priority: High

---

*Concerns audit: 2026-03-21*
