# Branch Goals
Accurate colors, smart sizing, clean edges - no crashes, real-time performance.

Implementaion strategy - get it working correctly and accurately first, THEN optimize performance.

## 1. Fix Color Visualization
Replace discrete 5-color system with continuous matplotlib inferno colormap. Current shader interpolation creates fake colors that don't represent actual data. Do first so plot is more accurate.

- [ ] Switch to continuous inferno colormap in fragment shader
- [ ] Figure out what scale_factor, gamma_correction, db_floor/db_ceil actually do
- [ ] What about min max normalization using  global max values trackers? When does that come into play
- [ ] Choose good defaults that don't lose visualization like before
- [ ] Config validation should prevent visualization data loss

## 2. Handle Cone of Influence
CWT edge effects mess up the plot. Only return the reliable center portion of CWT results to reduce artifacts. Do next since edge affects will be apparent after a more precise colormapping.

- [ ] Add overlapping audio windows in AudioInput
- [ ] Extract reliable CWT region (avoid edge effects)
- [ ] Keep wavelet and plotter changes minimal
- [ ] Test that edge artifacts are reduced

## 3. Smart Size Configurationc76+5556
Create optimal defaults for chunk_size, target_width, num_frames that balance performance vs accuracy. Validate against all hardware constraints upfront.

**Constraints to check:**
- OpenGL texture limit (16384 pixels) 
- GPU memory (CuPy arrays + texture)
- CPU memory (rolling buffer)
- Real-time performance targets

- [ ] Find optimal size defaults for best performance/accuracy tradeoff
- [ ] Add comprehensive constraint validation in config
- [ ] Estimate and validate GPU/CPU memory usage
- [ ] Target real-time FPS performance - highest FPS we can squeeze out of this
