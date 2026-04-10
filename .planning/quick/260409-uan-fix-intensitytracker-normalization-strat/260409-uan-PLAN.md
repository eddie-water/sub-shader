---
phase: quick
plan: 260409-uan
type: execute
wave: 1
depends_on: []
files_modified:
  - src/subshader/renderer/intensity.py
  - src/subshader/renderer/frame_buffer.py
  - src/subshader/pipeline.py
  - src/subshader/config.py
  - research/tests/viz/test_intensity_tracker.py
autonomous: true
must_haves:
  truths:
    - "The same audio level always produces the same brightness regardless of what came before"
    - "Quiet sections look dim relative to loud sections throughout the entire file"
    - "The normalization reference is determined before the visualization loop starts"
  artifacts:
    - path: "src/subshader/pipeline.py"
      provides: "Pre-scan step that computes fixed intensity_max before main loop"
    - path: "src/subshader/renderer/intensity.py"
      provides: "Simplified IntensityTracker holding a fixed reference value"
    - path: "research/tests/viz/test_intensity_tracker.py"
      provides: "Updated tests for fixed-reference normalization"
  key_links:
    - from: "src/subshader/pipeline.py"
      to: "src/subshader/renderer/renderer.py"
      via: "Fixed intensity_max passed to Renderer at construction or before run()"
      pattern: "renderer.*intensity_max|set_intensity_max"
    - from: "src/subshader/renderer/frame_buffer.py"
      to: "src/subshader/renderer/intensity.py"
      via: "IntensityTracker constructed with fixed value instead of tracking params"
      pattern: "IntensityTracker"
---

<objective>
Replace the dynamic global_max tracking in IntensityTracker with a fixed normalization
reference computed by pre-scanning the audio file before the visualization loop starts.

Purpose: The current adaptive normalization causes brightness to shift when audio dynamics
change (quiet intro looks bright, then dims when the beat drops). A fixed reference means
the same audio level always maps to the same brightness.

Output: Pipeline pre-scans a sample of CWT frames from the audio file, determines a fixed
intensity_max, and passes it through to the shader unchanged for the entire playback.
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/subshader/pipeline.py
@src/subshader/renderer/intensity.py
@src/subshader/renderer/frame_buffer.py
@src/subshader/renderer/renderer.py
@src/subshader/config.py
@src/subshader/audio/reader.py
@src/subshader/dsp/cwt.py
@src/subshader/renderer/shaders/fragment.glsl
@research/tests/viz/test_intensity_tracker.py

<interfaces>
<!-- Key types and contracts the executor needs. -->

From src/subshader/config.py:
```python
@dataclass
class ColorNormalizationConfig:
    gamma: float = 0.5
    global_intensity_smoothing_weight: float = 0.1
    global_intensity_percentile: float = 99.0
    frame_intensity_percentile: float = 95.0
    retention_rate: float = 0.95
    initial_intensity: float = 0.1
    frame_brightness_percentile: float = 99.0

@dataclass
class PipelineConfig:
    file_path: str = "assets/audio/reference/prospa_murda_baby_sc_rip.wav"
    chunk_size: int = 1 << 14
    overlap_factor: float = 0.5
    sample_rate: float = 44100.0
    total_samples: int = 0

@dataclass
class RendererConfig(PipelineConfig):
    num_frames: int = 32
    color_norm: ColorNormalizationConfig = field(default_factory=ColorNormalizationConfig)
```

From src/subshader/audio/reader.py:
```python
class AudioReader:
    def get_chunk(self) -> np.ndarray | None  # Returns chunk_size float64 samples, None at EOF
    def has_data(self) -> bool
    @property file_pos: int  # Seek position in samples
    @property total_samples: int
    @property sample_rate: float
```

From src/subshader/dsp/cwt.py:
```python
class CWT(DSP):
    def process(self, chunk) -> np.ndarray  # Returns (num_freqs, target_width) float32
    def get_output_shape(self) -> tuple
```

From src/subshader/renderer/renderer.py:
```python
class Renderer:
    def update(self, coefs: np.ndarray) -> None  # Push frame + render
    # Uses self.frame_buffer.push_frame(coefs) then self.gpu_renderer.set_intensity_max(...)

class GPURenderer:
    def set_intensity_max(self, intensity_max: float) -> None
```

From src/subshader/renderer/frame_buffer.py:
```python
class CircularFrameBuffer:
    def __init__(self, frame_shape, num_frames, color_norm_config)
    # Creates IntensityTracker with color_norm_config.global_intensity_percentile, retention_rate
    def push_frame(self, frame_data) -> None  # Calls intensity_tracker.update(frame_data)
    def get_intensity_max(self) -> float  # Returns intensity_tracker.global_max
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Simplify IntensityTracker to hold a fixed reference value and update tests</name>
  <files>
    src/subshader/renderer/intensity.py,
    src/subshader/config.py,
    research/tests/viz/test_intensity_tracker.py
  </files>
  <behavior>
    - Test: IntensityTracker(fixed_max=42.0).global_max == 42.0 after construction
    - Test: IntensityTracker(fixed_max=42.0).update(loud_frame) still returns 42.0 (no drift)
    - Test: IntensityTracker(fixed_max=42.0).update(quiet_frame) still returns 42.0 (no drift)
    - Test: IntensityTracker(fixed_max=0.0) clamps to floor_value (1e-8)
    - Test: ColorNormalizationConfig no longer has retention_rate field
    - Test: ColorNormalizationConfig.validate() still passes with valid params
  </behavior>
  <action>
    1. Rewrite IntensityTracker to accept a single `fixed_max: float` constructor param
       (plus `floor_value: float = 1e-8`). The class becomes a simple container:
       - `__init__(self, fixed_max, floor_value=1e-8)`: sets `self.global_max = max(fixed_max, floor_value)`
       - `update(self, frame)`: no-op, returns `self.global_max` unchanged. Keep the method
         signature so CircularFrameBuffer.push_frame() does not need changes yet.
       - `reset()`: no-op (or remove). The value is fixed.
       - Remove: percentile, retention_rate, warmup_frames, frame_count, is_ready.

    2. Clean up ColorNormalizationConfig in config.py:
       - Remove `retention_rate` field (no longer used).
       - Remove `global_intensity_smoothing_weight` field (no longer used).
       - Remove `initial_intensity` field (no longer used — the pre-scan provides the value).
       - Remove `frame_intensity_percentile` field (no longer used per-frame).
       - Remove `frame_brightness_percentile` field (no longer used per-frame).
       - Keep `gamma` (still used by shader).
       - Keep `global_intensity_percentile` (used by the pre-scan to compute the fixed max).
       - Update `validate()` to remove retention_rate check and any checks for removed fields.

    3. Rewrite research/tests/viz/test_intensity_tracker.py:
       - Replace all existing tests with the new behavior tests listed above.
       - Remove tests for retention_rate, decay, warmup, and config fields that no longer exist.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader && python -m pytest research/tests/viz/test_intensity_tracker.py -x -v</automated>
  </verify>
  <done>
    IntensityTracker holds a fixed value that never changes on update().
    ColorNormalizationConfig has no retention_rate or per-frame tracking fields.
    All tests pass.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add pre-scan to pipeline and wire fixed intensity_max through renderer</name>
  <files>
    src/subshader/pipeline.py,
    src/subshader/renderer/renderer.py,
    src/subshader/renderer/frame_buffer.py
  </files>
  <action>
    1. **Add pre-scan method to SubShader** (pipeline.py):
       Add a `_prescan_intensity(self) -> float` method that:
       - Uses `self.audio._reader` to read a sample of chunks across the file (not every chunk).
         Strategy: read ~10 evenly-spaced chunks from the audio file. For each chunk, run
         `self.dsp.process(chunk)` to get the CWT coefficients, then take
         `np.percentile(np.abs(coefs), config_percentile)` where config_percentile comes from
         `RendererConfig.color_norm.global_intensity_percentile` (default 99.0).
       - Track the max of all sampled percentile values across the 10 chunks.
       - Reset `self.audio._reader.file_pos = 0` after scanning so playback starts from the beginning.
       - Return the fixed intensity max value.
       - Log the pre-scan result: `log.info(f"Pre-scan complete: fixed intensity_max = {value:.4f}")`.
       - The number of sample chunks (10) can be a local constant `PRESCAN_NUM_CHUNKS = 10`.

    2. **Call pre-scan in SubShader.__init__()** after all stages are constructed:
       ```python
       self._fixed_intensity_max = self._prescan_intensity()
       ```

    3. **Pass fixed_intensity_max to Renderer**:
       Add a `set_fixed_intensity_max(self, value: float)` method to Renderer that:
       - Calls `self.gpu_renderer.set_intensity_max(value)` once.
       - Stores `self._fixed_intensity_max = value` for use in update().
       Call this from SubShader.__init__() after the pre-scan.

    4. **Update Renderer.update()** to use the fixed value:
       - Remove the per-frame `self.gpu_renderer.set_intensity_max(self.frame_buffer.get_intensity_max())` line.
       - Instead, set intensity_max from `self._fixed_intensity_max` (set once, never changes).
       - Actually, since set_intensity_max was already called with the fixed value and it does not
         change, the simplest approach is: call `set_intensity_max` once in `set_fixed_intensity_max`
         and remove the per-frame call entirely from update(). The shader uniform persists across
         draw calls.

    5. **Update CircularFrameBuffer**:
       - Change constructor to accept `fixed_intensity_max: float` instead of `color_norm_config`.
       - Pass `fixed_intensity_max` to IntensityTracker(fixed_max=fixed_intensity_max).
       - `get_intensity_max()` still returns `self.intensity_tracker.global_max` (unchanged API).
       - Actually, since Renderer.update() no longer reads intensity_max per-frame from the frame
         buffer, the IntensityTracker in CircularFrameBuffer becomes vestigial. Simplest approach:
         remove the IntensityTracker from CircularFrameBuffer entirely. Remove `get_intensity_max()`.
         Remove the `intensity_tracker.update()` call from `push_frame()`.

    6. **Update Renderer.__init__()** to match CircularFrameBuffer's new constructor:
       - CircularFrameBuffer no longer takes color_norm_config.
       - Just pass `frame_shape` and `num_frames`.
       - Initialize `self._fixed_intensity_max = 1.0` as default (overwritten by set_fixed_intensity_max).
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader && python -m pytest research/tests/viz/test_intensity_tracker.py -x -v && python -c "from subshader.pipeline import SubShader; from subshader.config import PipelineConfig; print('Import OK')"</automated>
  </verify>
  <done>
    SubShader pre-scans 10 evenly-spaced chunks to determine a fixed intensity_max.
    The fixed value is set once on the GPU shader uniform before the render loop starts.
    No per-frame intensity tracking occurs. CircularFrameBuffer no longer references IntensityTracker.
    The renderer uses the fixed value for the entire playback session.
    Imports succeed without error.
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest research/tests/viz/test_intensity_tracker.py -x -v` — all tests pass
2. `python -c "from subshader.pipeline import SubShader; print('OK')"` — no import errors
3. `grep -r "retention_rate" src/subshader/` — returns no results (fully removed)
4. `grep -r "IntensityTracker" src/subshader/renderer/frame_buffer.py` — returns no results (removed from frame buffer)
5. `grep "intensity_max" src/subshader/pipeline.py` — shows pre-scan sets the fixed value
</verification>

<success_criteria>
- The same CWT coefficient magnitude always maps to the same pixel brightness
- Pre-scan completes in under 5 seconds for a typical audio file
- No per-frame intensity tracking or adaptation occurs during playback
- All existing tests pass (updated for new behavior)
- No references to retention_rate, decay, or adaptive tracking remain in src/subshader/
</success_criteria>

<output>
After completion, create `.planning/quick/260409-uan-fix-intensitytracker-normalization-strat/260409-uan-SUMMARY.md`
</output>
