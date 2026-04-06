---
phase: 08-codebase-refactoring-and-module-cleanup
plan: 03
subsystem: rendering
tags: [moderngl, glfw, opengl, circular-buffer, intensity-tracking, shader]

# Dependency graph
requires:
  - phase: 08-01
    provides: RendererConfig in config.py — used as the config type for the new Renderer class

provides:
  - renderer/ module with Renderer, GLContext, GPURenderer, CircularFrameBuffer, AudioFrameBuffer, IntensityTracker
  - renderer/shaders/vertex.glsl and fragment.glsl with short canonical names
  - ShaderPlot = Renderer deprecated alias in renderer/__init__.py

affects:
  - 08-05 (switchover plan — will migrate __main__.py to use subshader.renderer.Renderer)
  - 08-07 (final cleanup — will remove viz/ after switchover)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One class per file for renderer module — frame_buffer.py, intensity.py, renderer.py"
    - "Shaders loaded via Path(__file__).parent / 'shaders' — no __init__.py in shaders/"
    - "Deprecated alias pattern: ShaderPlot = Renderer in __init__.py for backward compat"

key-files:
  created:
    - src/subshader/renderer/__init__.py
    - src/subshader/renderer/renderer.py
    - src/subshader/renderer/frame_buffer.py
    - src/subshader/renderer/intensity.py
    - src/subshader/renderer/shaders/vertex.glsl
    - src/subshader/renderer/shaders/fragment.glsl
  modified:
    - src/subshader/viz/plotter.py

key-decisions:
  - "GPURenderer chosen as the name for the low-level GPU class (was Renderer in plotter.py) — Renderer is now the top-level orchestrator (was ShaderPlot), so the inner class needed a distinct name"
  - "ColorNormalizationConfig.global_intensity_percentile used (not .percentile) — field name matches actual config dataclass; old plotter.py used wrong attribute names"
  - "plotter.py fixed as Rule 1 auto-fix — VisualizationConfig was removed in 08-01 and the import was broken; must be fixed for the plan's own verification criterion"

patterns-established:
  - "Renderer is the canonical top-level name for the OpenGL visualization orchestrator"
  - "Shader files use short names (vertex.glsl, fragment.glsl) loaded via Path"

requirements-completed:
  - D-15
  - D-16
  - D-17

# Metrics
duration: 12min
completed: 2026-04-06
---

# Phase 08 Plan 03: Renderer Module Split Summary

**plotter.py (800 lines, 7 classes) split into renderer/ with three focused files: frame_buffer.py, intensity.py, renderer.py; ShaderPlot renamed to Renderer; shaders renamed to short canonical names**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-06T23:02:00Z
- **Completed:** 2026-04-06T23:14:53Z
- **Tasks:** 1
- **Files modified:** 7 (6 created, 1 modified)

## Accomplishments
- Created `renderer/` module with three focused files: one concern per file
- `Renderer` class (renamed from `ShaderPlot`) orchestrates GLContext + GPURenderer + CircularFrameBuffer
- `GLContext` and `GPURenderer` (low-level GPU operations, renamed from `Renderer`) in renderer.py
- `CircularFrameBuffer` and `AudioFrameBuffer` isolated in frame_buffer.py
- `IntensityTracker` isolated in intensity.py (from viz/plot_normalizer.py)
- Shader files copied as `vertex.glsl` / `fragment.glsl` — loaded via `Path(__file__).parent / "shaders"`
- `ShaderPlot = Renderer` alias in `__init__.py` for backward compatibility during migration
- viz/ directory preserved intact (Plan 08-05 handles switchover)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create renderer/ directory and split plotter.py into three files** - `ef0548a` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `src/subshader/renderer/__init__.py` - Module entry point with Renderer export and ShaderPlot alias
- `src/subshader/renderer/renderer.py` - GLContext + GPURenderer + Renderer (top-level orchestrator)
- `src/subshader/renderer/frame_buffer.py` - CircularFrameBuffer + AudioFrameBuffer
- `src/subshader/renderer/intensity.py` - IntensityTracker (copied from viz/plot_normalizer.py)
- `src/subshader/renderer/shaders/vertex.glsl` - Vertex shader (renamed from vertex_shader.glsl)
- `src/subshader/renderer/shaders/fragment.glsl` - Fragment shader (renamed from fragment_shader.glsl)
- `src/subshader/viz/plotter.py` - Fixed pre-existing broken import (VisualizationConfig → RendererConfig)

## Decisions Made
- `GPURenderer` chosen as the internal low-level GPU class name — the plan renames `ShaderPlot` to `Renderer`, so the old inner `Renderer` (which handled raw GL calls) needed a distinct name to avoid collision
- Used `ColorNormalizationConfig.global_intensity_percentile` in CircularFrameBuffer (not `.percentile`) — field name must match the actual dataclass
- `renderer.py` uses `from .frame_buffer import CircularFrameBuffer` and `from .intensity import IntensityTracker` — internal cross-file imports within the renderer package

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed broken VisualizationConfig import in viz/plotter.py**
- **Found during:** Task 1 (verification: `from subshader.viz.plotter import ShaderPlot`)
- **Issue:** `plotter.py` imported `VisualizationConfig` which was removed in Phase 08-01. The plan's own verification criterion requires `from subshader.viz.plotter import ShaderPlot` to exit 0.
- **Fix:** Replaced `VisualizationConfig` with `RendererConfig` in the import and type annotation; fixed `config.gamma` → `config.color_norm.gamma`; fixed `color_norm_config.percentile` → `color_norm_config.global_intensity_percentile` (matching actual dataclass field)
- **Files modified:** src/subshader/viz/plotter.py
- **Verification:** `from subshader.viz.plotter import ShaderPlot` exits 0
- **Committed in:** ef0548a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — pre-existing bug from 08-01 config refactor)
**Impact on plan:** Fix necessary to meet the plan's own verification criterion. No scope creep.

## Issues Encountered
- The low-level GPU class in `plotter.py` was also named `Renderer`, creating a naming conflict when promoting `ShaderPlot` to `Renderer`. Resolved by renaming the inner class to `GPURenderer`, which more accurately describes its role.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `from subshader.renderer import Renderer` works
- Both old (`from subshader.viz.plotter import ShaderPlot`) and new import paths work
- Plan 08-05 can now migrate `__main__.py` to use the new Renderer
- Plan 08-07 can remove viz/ after all callers are migrated

---
*Phase: 08-codebase-refactoring-and-module-cleanup*
*Completed: 2026-04-06*
