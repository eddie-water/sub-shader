---
phase: quick
plan: 260409-tpg
type: execute
wave: 1
depends_on: []
files_modified:
  - src/subshader/renderer/intensity.py
  - src/subshader/config.py
  - src/subshader/renderer/frame_buffer.py
  - src/subshader/renderer/RENDERER.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "decay_rate=0.95 means global_max retains 95% per frame (not drops to 5%)"
    - "Config field name communicates retention semantics clearly"
    - "All references to the old field name are updated consistently"
  artifacts:
    - path: "src/subshader/renderer/intensity.py"
      provides: "Fixed decay formula using retention_rate directly"
      contains: "self.retention_rate"
    - path: "src/subshader/config.py"
      provides: "Renamed field with correct docstring"
      contains: "retention_rate"
    - path: "src/subshader/renderer/frame_buffer.py"
      provides: "Updated field reference in IntensityTracker construction"
      contains: "retention_rate"
  key_links:
    - from: "src/subshader/config.py"
      to: "src/subshader/renderer/frame_buffer.py"
      via: "color_norm_config.retention_rate attribute access"
      pattern: "color_norm_config\\.retention_rate"
    - from: "src/subshader/renderer/frame_buffer.py"
      to: "src/subshader/renderer/intensity.py"
      via: "IntensityTracker(retention_rate=...) constructor kwarg"
      pattern: "retention_rate=color_norm_config\\.retention_rate"
---

<objective>
Fix the IntensityTracker decay formula semantic inversion. Currently `decay_rate=0.95`
computes `(1.0 - 0.95) * global_max = 0.05 * global_max`, dropping to 5% per frame.
The fix changes the formula to `retention_rate * global_max` so 0.95 means "retain 95%",
and renames the parameter from `decay_rate` to `retention_rate` for clarity.

Purpose: Eliminate brightness flickering caused by global_max resetting nearly to zero every frame.
Output: Corrected intensity tracking with stable colormap scaling.
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/subshader/renderer/intensity.py
@src/subshader/config.py
@src/subshader/renderer/frame_buffer.py
@src/subshader/renderer/RENDERER.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Fix formula and rename parameter across all source files</name>
  <files>src/subshader/renderer/intensity.py, src/subshader/config.py, src/subshader/renderer/frame_buffer.py, src/subshader/renderer/RENDERER.md</files>
  <behavior>
    - Test: IntensityTracker with retention_rate=0.95 and global_max=100.0 decays to 95.0 after one update with a zero-intensity frame (not 5.0)
    - Test: IntensityTracker with retention_rate=0.95 retains a high global_max when fed repeated high-intensity frames
    - Test: After 10 frames of silence, global_max decays exponentially (0.95^10 * initial ~ 59.9% retained)
  </behavior>
  <action>
    1. **src/subshader/renderer/intensity.py:**
       - Rename `decay_rate` parameter to `retention_rate` in __init__ (both param and self assignment)
       - Change default from `0.001` to `0.95` (matching config semantics)
       - Fix line 47 formula from `(1.0 - self.decay_rate) * self.global_max` to `self.retention_rate * self.global_max`
       - Update docstring on __init__ to explain retention semantics

    2. **src/subshader/config.py:**
       - Rename `decay_rate` field to `retention_rate` in ColorNormalizationConfig (line 75)
       - Update the comment from "Exponential decay rate" to "Fraction of global intensity retained per frame (0.95 = retain 95%)"
       - Update validate() references: `self.decay_rate` -> `self.retention_rate`, error message text accordingly

    3. **src/subshader/renderer/frame_buffer.py:**
       - Line 41: Change `decay_rate=color_norm_config.decay_rate` to `retention_rate=color_norm_config.retention_rate`
       - Line 25 docstring: Change `decay_rate` to `retention_rate`

    4. **src/subshader/renderer/RENDERER.md:**
       - Line 59: Update formula from `(1.0 - self.decay_rate)` to `self.retention_rate`
       - Line 138: Rename `decay_rate` to `retention_rate` and update description
       - Line 142: Update parameter guidance text to use `retention_rate`
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader && python -c "
from subshader.renderer.intensity import IntensityTracker
import numpy as np
# Test 1: retention semantics (0.95 retains 95%)
t = IntensityTracker(retention_rate=0.95)
t.global_max = 100.0
t.update(np.zeros((10, 10)))
assert abs(t.global_max - 95.0) < 0.01, f'Expected ~95.0, got {t.global_max}'
# Test 2: decay over 10 silent frames
t2 = IntensityTracker(retention_rate=0.95)
t2.global_max = 100.0
for _ in range(10):
    t2.update(np.zeros((10, 10)))
expected = 100.0 * (0.95 ** 10)
assert abs(t2.global_max - expected) < 0.01, f'Expected ~{expected:.2f}, got {t2.global_max}'
# Test 3: config field name exists
from subshader.config import ColorNormalizationConfig
c = ColorNormalizationConfig()
assert hasattr(c, 'retention_rate'), 'Missing retention_rate field'
assert not hasattr(c, 'decay_rate'), 'Old decay_rate field still exists'
print('All checks passed')
"</automated>
  </verify>
  <done>
    - IntensityTracker uses retention_rate semantics: 0.95 means retain 95% per frame
    - All three source files use retention_rate consistently (no decay_rate references remain in source)
    - RENDERER.md documentation updated to match
    - Verification script confirms correct decay behavior
  </done>
</task>

</tasks>

<verification>
Run the inline verification script above. Additionally confirm no stale `decay_rate` references remain in source files:

```bash
grep -rn "decay_rate" src/subshader/ --include="*.py"
```

Should return zero matches. (RENDERER.md may still mention decay conceptually — that is fine as long as the parameter name is `retention_rate`.)
</verification>

<success_criteria>
- `retention_rate=0.95` causes global_max to retain 95% per frame (not drop to 5%)
- No `decay_rate` attribute references in Python source files under src/subshader/
- Config validation still works for the renamed field
- RENDERER.md reflects the corrected formula and parameter name
</success_criteria>

<output>
After completion, create `.planning/quick/260409-tpg-fix-intensitytracker-decay-rate-semantic/260409-tpg-SUMMARY.md`
</output>
