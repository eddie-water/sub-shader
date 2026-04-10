---
phase: quick
plan: 260409-tpg
subsystem: renderer
tags: [bug-fix, intensity-tracker, color-normalization, tdd]
key-files:
  modified:
    - src/subshader/renderer/intensity.py
    - src/subshader/config.py
    - src/subshader/renderer/frame_buffer.py
    - src/subshader/renderer/RENDERER.md
  created:
    - research/tests/viz/test_intensity_tracker.py
decisions:
  - "retention_rate=0.95 means retain 95% per frame; formula is rate*max not (1-rate)*max"
  - "Default changed from 0.001 (decay) to 0.95 (retention) — consistent with config value"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-09"
  tasks: 1
  commits: 2
---

# Quick Task 260409-tpg: Fix IntensityTracker Decay Rate Semantic

**One-liner:** Renamed `decay_rate` to `retention_rate` and fixed formula inversion so 0.95 retains 95% per frame instead of collapsing global_max to 5%.

## What Was Done

`IntensityTracker.update()` had an inverted decay formula: `(1.0 - self.decay_rate) * self.global_max`. With `decay_rate=0.95` (the config default), this computed `0.05 * global_max` — dropping to 5% per frame. The result was aggressive brightness flickering as `global_max` collapsed nearly to zero every frame.

### Changes

**src/subshader/renderer/intensity.py**
- Renamed `decay_rate` parameter and attribute to `retention_rate`
- Changed default from `0.001` to `0.95`
- Fixed formula: `self.retention_rate * self.global_max` (was `(1.0 - self.decay_rate) * self.global_max`)
- Added docstring explaining retention semantics

**src/subshader/config.py**
- Renamed `decay_rate` field to `retention_rate` in `ColorNormalizationConfig`
- Updated comment: "Fraction of global intensity retained per frame (0.95 = retain 95%)"
- Updated `validate()` to reference `self.retention_rate`

**src/subshader/renderer/frame_buffer.py**
- Updated `IntensityTracker` construction: `retention_rate=color_norm_config.retention_rate`
- Updated docstring for `color_norm_config` parameter

**src/subshader/renderer/RENDERER.md**
- Updated formula snippet to use `self.retention_rate`
- Updated configuration table row (field name and description)

**research/tests/viz/test_intensity_tracker.py** (new)
- 7 tests covering retention semantics, exponential decay, config field existence, validation

## Commits

| Commit | Message |
|--------|---------|
| `489b5da` | `test(260409-tpg): add failing tests for retention_rate semantics` |
| `8acd68a` | `fix(260409-tpg): rename decay_rate to retention_rate and fix formula` |

## Verification

All 3 plan verification assertions pass:
- `retention_rate=0.95` → `global_max` decays to 95.0 after one silent frame (not 5.0)
- After 10 silent frames: `100.0 * 0.95^10 ≈ 59.87` retained
- `ColorNormalizationConfig` has `retention_rate`, no `decay_rate`

```
grep -rn "decay_rate" src/subshader/ --include="*.py"
# (zero matches)
```

## Deviations from Plan

None — plan executed exactly as written. TDD flow followed: RED commit (7 failing tests) then GREEN commit (implementation).

## Self-Check: PASSED

- `src/subshader/renderer/intensity.py` — exists, contains `self.retention_rate`
- `src/subshader/config.py` — exists, contains `retention_rate`
- `src/subshader/renderer/frame_buffer.py` — exists, contains `retention_rate`
- `research/tests/viz/test_intensity_tracker.py` — exists, 7 tests all pass
- Commits `489b5da` and `8acd68a` verified in git log
