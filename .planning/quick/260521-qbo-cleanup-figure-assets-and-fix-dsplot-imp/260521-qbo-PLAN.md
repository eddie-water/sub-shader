---
phase: 260521-qbo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - assets/images/figures/**         # new tree (git mv targets)
  - assets/images/generated/         # source of some moves
  - assets/images/dsp/                # source of most moves
  - research/dsplot/figures/figure_1.py
  - research/dsplot/figures/components_recombine.py
  - research/dsplot/figures/lego_demo.py
  - research/dsplot/figures/style_skeleton.py
  - research/dsplot/figures/motivator.py
  - research/dsplot/figures/vector_basics.py
  - research/dsplot/figures/vector_projection_3d.py
  - research/dsplot/figures/dot_product_geometry.py
  - research/dsplot/figures/projection_reconstruction.py
  - research/dsplot/figures/alignment_diagnostic.py
  - research/dsplot/figures/style_override_demo.py
  - research/dsplot/figures/__main__.py
  - research/dsplot/figures/__init__.py
  - src/subshader/dsp/DSP.md
  - src/subshader/dsp/dsp.ipynb
autonomous: false
requirements: []
must_haves:
  truths:
    - "All figure PNGs are organized under assets/images/figures/<family>/<name>.png with their old prefixes stripped from the filename"
    - "No PNG is deleted; every move preserves git history via git mv"
    - "from research.dsplot.figures.figure_1 import render_hero succeeds from repo root with no PYTHONPATH hack"
    - "All render() / render_hero() / render_antihero() default output_dir values point at the new per-family subdir AND default output_filename has the family prefix stripped"
    - "DSP.md image links and dsp.ipynb image cells resolve to the new paths (no broken images)"
    - "Re-running each affected render produces a PNG at the new per-family location with the new stripped filename"
    - "No git commit is created by the executor; the user reviews and commits manually"
  artifacts:
    - path: "assets/images/figures/lego/"
      provides: "lego_demo v1..v52 + non-vN renames (current_state, post_xtick_fix, .png)"
    - path: "assets/images/figures/figure_1/"
      provides: "figure_1 chirp/hero/antihero/hero_candidate variants"
    - path: "assets/images/figures/sample_template/"
      provides: "sample_template_v1.png moved here as v1.png"
    - path: "assets/images/figures/style_skeleton/"
      provides: "style_skeleton + v3 + v8/v9/v12/v13 clean+guides variants"
    - path: "assets/images/figures/motivator/"
      provides: "all dsp_motivator_* PNGs (renamed to drop dsp_motivator_ prefix)"
    - path: "research/dsplot/figures/SUMMARY.md or PR-style mapping table"
      provides: "the full old→new PNG mapping for spot-checking"
      note: "Lives in the quick-task SUMMARY.md, not a new committed file"
  key_links:
    - from: "research/dsplot/figures/components_recombine.py"
      to: "research/dsplot/__init__.py"
      via: "relative import (from .. import …) or absolute (from research.dsplot import …)"
      pattern: "^from research\\.dsplot|^from \\.\\."
    - from: "research/dsplot/figures/figure_1.py::render_hero"
      to: "assets/images/figures/figure_1/hero_v1.png"
      via: "default output_dir kwarg points at the new per-family subdir"
    - from: "src/subshader/dsp/DSP.md"
      to: "assets/images/figures/<family>/<name>.png"
      via: "markdown image link"
      pattern: "assets/images/figures/"
    - from: "src/subshader/dsp/dsp.ipynb"
      to: "assets/images/figures/<family>/<name>.png"
      via: "markdown cell image refs"
      pattern: "assets/images/figures/"
---

<objective>
Reorganize the loose pile of generated figure PNGs into per-figure subdirectories, fix the broken `from dsplot import …` imports in the in-tree `research/dsplot/figures/` modules so they work from repo root without a PYTHONPATH hack, update render default `output_dir` / `output_filename` kwargs to the new locations + stripped filenames, and update DSP.md / dsp.ipynb image references to match. End with a visual verification rendering each affected family.

Purpose: The unorganized `assets/images/{generated,dsp}/` directories make it hard to find the latest of each figure family. The broken `from dsplot import …` pattern forces a PYTHONPATH workaround every render. This quick task closes both gaps without changing any figure rendering logic (especially not the locked v8 chirp pipeline).

Output:
- Reorganized `assets/images/figures/<family>/<name>.png` tree
- All `research/dsplot/figures/*.py` import statements rewritten to absolute `from research.dsplot import …` (or relative `from .. import …`) form
- Render-function `output_dir` / `output_filename` defaults updated in `figure_1.py`, `components_recombine.py`, `motivator.py`, `__main__.py`
- DSP.md + dsp.ipynb image refs updated
- Old→new PNG mapping table in SUMMARY.md
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/eddie-water/dev/python/sub-shader/CLAUDE.md
@/home/eddie-water/dev/python/sub-shader/.planning/STATE.md
@/home/eddie-water/dev/python/sub-shader/research/dsplot/figures/__init__.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/figures/__main__.py

<interfaces>
<!-- Key code paths the executor will modify. Extracted upfront so no codebase
     scavenger-hunt is needed. -->

Broken-import pattern (target of Task 2). 14 occurrences across these files,
discovered via:

    grep -rn "^from dsplot\|^import dsplot\|^from dsplot " research/dsplot/

  research/dsplot/figures/figure_1.py:28               from dsplot import (Figure, Heatmap, HeatmapPanel, Line, TextPanel, TimeSeries, TimeSeriesPanel, style,)
  research/dsplot/figures/components_recombine.py:24   from dsplot import (Annotation, Dropline, DynamicPanel, Figure, StaticPanel, Vector, VectorComponents, style,)
  research/dsplot/figures/vector_basics.py:13          from dsplot import Figure, StaticPanel, Vector, style
  research/dsplot/figures/vector_projection_3d.py:17   from dsplot import Annotation, Figure, Vector, style
  research/dsplot/figures/vector_projection_3d.py:18   from dsplot.panels import StaticPanel3D
  research/dsplot/figures/motivator.py:27              from dsplot import style
  research/dsplot/figures/alignment_diagnostic.py:22   from dsplot import style
  research/dsplot/figures/style_override_demo.py:51    import dsplot
  research/dsplot/figures/style_override_demo.py:52    from dsplot import Figure, StaticPanel, Vector, style
  research/dsplot/figures/dot_product_geometry.py:14   from dsplot import Annotation, Figure, StaticPanel, Vector, style
  research/dsplot/figures/projection_reconstruction.py:17  from dsplot import (...)
  research/dsplot/figures/lego_demo.py:13              from dsplot import (CompositePanel, Figure, Heatmap, HeatmapPanel, Line, StaticPanel, StaticPanel3D, Stem, TimeSeries, TimeSeriesPanel, Vector, VectorComponents, style,)
  research/dsplot/figures/style_skeleton.py:12         from dsplot import (...)

  EXCLUSIONS — DO NOT modify these (they live INSIDE the dsplot package and
  already work as relative-ish references in docstrings/internals):
    research/dsplot/__init__.py:16     (docstring example only)
    research/dsplot/style.py:11        (docstring example only)
    research/dsplot/notebooks/09_04_smoke_test.ipynb  (notebook outside figures/)

Render-function signatures to update (target of Task 3):

  # research/dsplot/figures/figure_1.py
  def render(output_dir: str = "assets/images/generated",
             output_filename: str = "figure_1_fourier_vs_wavelet.png") -> str
  def render_hero(output_dir: str = "assets/images/generated",
                  output_filename: str = "figure_1_hero_click_plus_tone_v1.png") -> str
  def render_antihero(output_dir: str = "assets/images/generated",
                      output_filename: str = "figure_1_antihero_low_vibrato_v1.png") -> str

  # research/dsplot/figures/components_recombine.py
  def render(output_dir: str, output_filename: str = "components_recombine_either_order_v19.png") -> str
  def render_vector_xy_reconstruction(output_dir: str, output_filename: str = "vector_xy_reconstruction.png") -> str

  # research/dsplot/figures/motivator.py — many MotivatorConfig entries with
  #   output_filename="dsp_motivator_v{N}_<freq-range>_<dur>.png"
  # Strategy: change render_one's default output_dir AND strip the
  # "dsp_motivator_" prefix from each MotivatorConfig's output_filename.

  # research/dsplot/figures/__main__.py
  _OUT_DIR  default                          = "assets/images/dsp"
  motivator_out (line ~95-97)                = "assets/images/generated"
  Hardcoded output_filename strings (lines 70-91, 102) — components_recombine,
  vector_xy_reconstruction, projection_reconstruction, vector_projection_3d,
  vector_basics, dot_product_geometry, alignment_diagnostic.

DSP.md image links to update (5 refs, target of Task 4):
  src/subshader/dsp/DSP.md:130  ../../../assets/images/dsp/components_recombine_either_order_v19.png
  src/subshader/dsp/DSP.md:140  ../../../assets/images/dsp/projection_reconstruction_either_order_v9.png
  src/subshader/dsp/DSP.md:169  ../../../assets/images/dsp/dot_product_geometry.png
  src/subshader/dsp/DSP.md:199  ../../../assets/images/dsp/vector_projection_3d_v2_combo5_palette.png
  src/subshader/dsp/DSP.md:245  ../../../assets/images/dsp/vector_xy_reconstruction.png

dsp.ipynb image cells (4 refs, target of Task 4):
  src/subshader/dsp/dsp.ipynb:323  assets/images/dsp/projection_reconstruction_either_order_v9.png
  src/subshader/dsp/dsp.ipynb:353  assets/images/dsp/dot_product_geometry.png
  src/subshader/dsp/dsp.ipynb:389  assets/images/dsp/vector_projection_3d_v2_combo5_palette.png
  src/subshader/dsp/dsp.ipynb:435  assets/images/dsp/vector_xy_reconstruction.png
</interfaces>

<inventory>
<!-- Current file counts in assets/images/{generated,dsp}/ for the relevant
     families. The executor MUST re-run `ls` to get the live list before
     moving; this block is only a sanity-check baseline. -->

assets/images/dsp/  (117 files total):
  lego_demo                       50 files (v8, v10..v52, .png, _current_state, _post_xtick_fix)
  style_skeleton                   9 files (.png, _v3, _v8/v9/v12/v13 clean+guides)
  components_recombine             20 files (.png + v2..v19, some _09_05 suffixes)
  projection_reconstruction        11 files
  projection_reference_directions   6 files
  vector_projection_3d              3 files (+ _09_05)
  vector_basics                    2 files
  dot_product_geometry             2 files (+ _09_05)
  dot_product_symmetry             1 file
  dot_product_twin_rectangles      2 files
  vector_projection                1 file
  vector_similarity                1 file
  vector_symmetric_projection      1 file
  vector_xy_projection             1 file
  vector_xy_reconstruction         2 files (+ _09_05)
  palette                          8 files (palette_*, palette_lego_*, palette_orange_*)
  style_override_demo/             SUBDIR — already organized, LEAVE IN PLACE

assets/images/generated/  (~60 files):
  figure_1                         17 files (fourier_vs_wavelet v1..v8, hero_candidate_2 v14..v18,
                                             hero_v41_style v1..v4, hero_click_plus_tone_v1,
                                             antihero_low_vibrato_v1)
  dsp_motivator                    24 files (v1..v6, vw1..vw4 variants, section1 variants, _09_05 dupes)
  dsp_alignment_diagnostic         4 files
  chirp                            3 files (chirp_random_walk, chirp_signal_comparison, chirp_sweep_200_20k)
  comparison_grid                  4 files (+ _STUB variants, chunksize16k)
  musical_signal_comparison        1 file
  polyphonic_signal_comparison     1 file
  timing_bar_chart                 1 file
  beltran_sc_rip_16_bar            1 file
  20-20k is bent                   1 file
  sample_template                  1 file (v1)
  stubs/                           SUBDIR — already organized, LEAVE IN PLACE
  dpi/                             SUBDIR — already organized, LEAVE IN PLACE

NOTE on lego_demo source dir:
  Scope brief says lego PNGs are in `assets/images/generated/`, but the live
  filesystem shows them in `assets/images/dsp/`. Trust the filesystem — use
  `git ls-files assets/images/ | xargs -I{} basename {}` to confirm before
  the move; mismatched-source scope-brief items belong wherever `git ls-files`
  places them.
</inventory>

<constraints>
- **NO `git commit`** — per user policy [[feedback-no-auto-commits]]. The
  executor MUST NOT run `git commit` at any point. The user reviews the
  staged changes and commits manually after spot-checking the SUMMARY.md
  old→new mapping table.
- **NO PNG deletions** — every move uses `git mv` (preserves history for
  tracked files) or `mv` followed by `git add -A` for any untracked files.
- **DO NOT touch the locked v8 chirp render LOGIC** in `figure_1.py` — only
  its `output_dir` / `output_filename` default kwargs.
- **DO NOT promote `dsplot` to a top-level package** — keep it under
  `research/dsplot/`. Imports inside `research/dsplot/figures/` should use
  the relative form `from .. import …` (preferred) or the absolute form
  `from research.dsplot import …`. Both should work; relative form is
  preferred because it doesn't require `research/` to be importable from
  repo root.
- **DO NOT modify** `research/dsplot/__init__.py:16` or
  `research/dsplot/style.py:11` — those `from dsplot import style` strings
  are docstring examples, not real imports.
- **DO NOT touch** `research/dsplot/figures/style_override_demo.py:51-52`
  unless its `import dsplot` is genuinely broken at module-load. Read its
  surrounding `sys.path` bootstrap block first — if the bootstrap is what
  makes the absolute `dsplot` import work, leave it OR convert it to the
  same relative form as the rest. Document the decision in SUMMARY.md.
- **DO NOT** move/rename PNGs inside already-organized subdirs:
  `assets/images/dsp/style_override_demo/`, `assets/images/generated/stubs/`,
  `assets/images/generated/dpi/`. Those stay put.
</constraints>

<tasks>

<task type="auto">
  <name>Task 1: Inventory + git mv PNGs into per-family subdirs under assets/images/figures/</name>
  <files>
    assets/images/figures/&lt;family&gt;/&lt;name&gt;.png  (new tree, populated via git mv)
  </files>
  <action>
**Step 1 — Live inventory (do NOT trust the &lt;inventory&gt; block blindly):**
Run:

    git ls-files assets/images/generated/ assets/images/dsp/ | sort &gt; /tmp/qbo_baseline.txt
    ls assets/images/generated/ assets/images/dsp/ | sort &gt; /tmp/qbo_disk.txt
    diff /tmp/qbo_baseline.txt /tmp/qbo_disk.txt

Untracked files (in disk but not in git ls-files) need `git add` after `mv`
rather than `git mv`.

**Step 2 — Build the mapping.**
Create a per-family move plan as a Python dict (write it inline in a
throwaway shell script under `/tmp/qbo_moves.sh` so the executor can review
before executing). The dest pattern is:

    assets/images/figures/&lt;family&gt;/&lt;stripped_name&gt;.png

Stripping rules (apply IN ORDER, first match wins):
  - `lego_demo_(v\d+)\.png`               -&gt; `lego/$1.png`
  - `lego_demo\.png`                      -&gt; `lego/baseline.png`
  - `lego_demo_(current_state|post_xtick_fix)\.png` -&gt; `lego/$1.png`
  - `style_skeleton\.png`                 -&gt; `style_skeleton/baseline.png`
  - `style_skeleton_(.+)\.png`            -&gt; `style_skeleton/$1.png`
  - `figure_1_fourier_vs_wavelet\.png`    -&gt; `figure_1/fourier_vs_wavelet.png`
  - `figure_1_fourier_vs_wavelet_(.+)\.png` -&gt; `figure_1/fourier_vs_wavelet_$1.png`
  - `figure_1_hero_(.+)\.png`             -&gt; `figure_1/hero_$1.png`
  - `figure_1_antihero_(.+)\.png`         -&gt; `figure_1/antihero_$1.png`
  - `figure_1_(.+)\.png`                  -&gt; `figure_1/$1.png`
  - `sample_template_(.+)\.png`           -&gt; `sample_template/$1.png`
  - `dsp_motivator_(.+)\.png`             -&gt; `motivator/$1.png`
  - `dsp_alignment_diagnostic\.png`       -&gt; `alignment_diagnostic/baseline.png`
  - `dsp_alignment_diagnostic_(.+)\.png`  -&gt; `alignment_diagnostic/$1.png`
  - `components_recombine_(.+)\.png`      -&gt; `components_recombine/$1.png`
  - `projection_reconstruction_(.+)\.png` -&gt; `projection_reconstruction/$1.png`
  - `projection_reference_directions(.*)\.png` -&gt; `projection_reference_directions/$1_or_baseline.png`
  - `vector_projection_3d(.*)\.png`       -&gt; `vector_projection_3d/$1_or_baseline.png`
  - `vector_(basics|projection|similarity|symmetric_projection|xy_projection|xy_reconstruction)(.*)\.png` -&gt; `vector_$1/$2_or_baseline.png`
  - `dot_product_(.+)\.png`               -&gt; `dot_product/$1.png` (group geometry/symmetry/twin_rectangles together)
  - `palette(.+)\.png`                    -&gt; `palette/$1.png`
  - `chirp_(.+)\.png`                     -&gt; `chirp/$1.png`
  - `comparison_grid(.*)\.png`            -&gt; `comparison_grid/$1_or_baseline.png`
  - everything else (`musical_signal_comparison`, `polyphonic_signal_comparison`,
    `timing_bar_chart`, `beltran_sc_rip_16_bar`, `20-20k is bent`)
                                          -&gt; `misc/&lt;original_name&gt;.png`

For names that strip to empty (e.g. `style_skeleton.png` -&gt; `style_skeleton/.png`),
use the literal stem `baseline.png`. Same for any other zero-suffix collision.

**Step 3 — Create destination dirs + execute moves.**

    mkdir -p assets/images/figures/{lego,figure_1,sample_template,style_skeleton,motivator,alignment_diagnostic,components_recombine,projection_reconstruction,projection_reference_directions,vector_projection_3d,vector_basics,vector_projection,vector_similarity,vector_symmetric_projection,vector_xy_projection,vector_xy_reconstruction,dot_product,palette,chirp,comparison_grid,misc}

For each tracked file in the mapping:
    git mv "&lt;src&gt;" "&lt;dst&gt;"
For each untracked file:
    mv "&lt;src&gt;" "&lt;dst&gt;" &amp;&amp; git add "&lt;dst&gt;"

**Step 4 — Verify NO files were lost.**
    git status --short | wc -l        # all moves must be R (rename) or new A
    find assets/images/figures/ -type f -name "*.png" | wc -l
    # Compare against pre-move count; difference must be 0.

**Step 5 — Emit the full old→new mapping to a stdout dump:**

    git status --short | grep "^R " | sed 's|R  ||' &gt; /tmp/qbo_mapping.txt

Print this list at end of task — it goes into SUMMARY.md for user spot-check.

**REMINDER: DO NOT `git commit`.** Stop after `git add` / `git mv` — the
user will commit manually after reviewing.
  </action>
  <verify>
    <automated>
# Coverage: every tracked PNG under assets/images/{generated,dsp}/ (excluding
# stubs/, dpi/, style_override_demo/) must have a counterpart under figures/.
LEFT=$(git ls-files assets/images/generated/ assets/images/dsp/ | grep -v "^assets/images/generated/stubs/\|^assets/images/generated/dpi/\|^assets/images/dsp/style_override_demo/" | wc -l)
RIGHT=$(git status --short | grep -E "^R  .*-&gt; assets/images/figures/" | wc -l)
test "$LEFT" -le "$RIGHT"  # all source files accounted for via renames
# Zero file deletions:
DELETIONS=$(git status --short | grep -c "^.D")
test "$DELETIONS" -eq 0
    </automated>
  </verify>
  <done>
- All in-scope PNGs staged as `R` (renames) under `assets/images/figures/&lt;family&gt;/`
- Zero `D` (deletions) in `git status --short`
- `assets/images/dsp/style_override_demo/`, `assets/images/generated/stubs/`,
  `assets/images/generated/dpi/` untouched
- Full old→new mapping captured for SUMMARY.md
  </done>
</task>

<task type="auto">
  <name>Task 2: Fix `from dsplot import …` imports in research/dsplot/figures/*.py</name>
  <files>
    research/dsplot/figures/figure_1.py
    research/dsplot/figures/components_recombine.py
    research/dsplot/figures/vector_basics.py
    research/dsplot/figures/vector_projection_3d.py
    research/dsplot/figures/motivator.py
    research/dsplot/figures/alignment_diagnostic.py
    research/dsplot/figures/dot_product_geometry.py
    research/dsplot/figures/projection_reconstruction.py
    research/dsplot/figures/lego_demo.py
    research/dsplot/figures/style_skeleton.py
    research/dsplot/figures/style_override_demo.py  (CONDITIONAL — see below)
  </files>
  <action>
For each file listed above (see the &lt;interfaces&gt; block for exact line
numbers), rewrite the broken import.

**Preferred form: relative imports (from .. import …)** — they work without
requiring `research/` to be importable as a package from repo root.

Rewrites (apply each on the noted line):

  figure_1.py:28
    FROM:  from dsplot import (
               Figure, Heatmap, HeatmapPanel, Line, TextPanel,
               TimeSeries, TimeSeriesPanel, style,
           )
    TO:    from .. import (
               Figure, Heatmap, HeatmapPanel, Line, TextPanel,
               TimeSeries, TimeSeriesPanel, style,
           )

  components_recombine.py:24
    FROM:  from dsplot import (Annotation, Dropline, DynamicPanel, Figure,
                               StaticPanel, Vector, VectorComponents, style,)
    TO:    from .. import (Annotation, Dropline, DynamicPanel, Figure,
                           StaticPanel, Vector, VectorComponents, style,)

  vector_basics.py:13
    FROM:  from dsplot import Figure, StaticPanel, Vector, style
    TO:    from .. import Figure, StaticPanel, Vector, style

  vector_projection_3d.py:17-18
    FROM:  from dsplot import Annotation, Figure, Vector, style
           from dsplot.panels import StaticPanel3D
    TO:    from .. import Annotation, Figure, Vector, style
           from ..panels import StaticPanel3D

  motivator.py:27
    FROM:  from dsplot import style
    TO:    from .. import style

  alignment_diagnostic.py:22
    FROM:  from dsplot import style
    TO:    from .. import style

  dot_product_geometry.py:14
    FROM:  from dsplot import Annotation, Figure, StaticPanel, Vector, style
    TO:    from .. import Annotation, Figure, StaticPanel, Vector, style

  projection_reconstruction.py:17 (whatever the multi-line tuple is)
    FROM:  from dsplot import (...)
    TO:    from .. import (...)

  lego_demo.py:13
    FROM:  from dsplot import (CompositePanel, Figure, Heatmap, HeatmapPanel,
                               Line, StaticPanel, StaticPanel3D, Stem,
                               TimeSeries, TimeSeriesPanel, Vector,
                               VectorComponents, style,)
    TO:    from .. import (CompositePanel, Figure, Heatmap, HeatmapPanel,
                           Line, StaticPanel, StaticPanel3D, Stem,
                           TimeSeries, TimeSeriesPanel, Vector,
                           VectorComponents, style,)

  style_skeleton.py:12 (whatever the multi-line tuple is)
    FROM:  from dsplot import (...)
    TO:    from .. import (...)

**style_override_demo.py — CONDITIONAL.** Read lines 1-60 first. The file
has a `sys.path` bootstrap that makes the absolute `import dsplot` work
when invoked as `python style_override_demo.py`. Decision:
  - If the bootstrap is required because the file is run as `__main__`
    rather than imported, KEEP the bootstrap + absolute imports as-is, but
    add a comment noting it's intentional.
  - If the file is only ever imported (never run as `__main__`), strip the
    bootstrap and switch to `from .. import …`.
  Look for `if __name__ == "__main__":` near the bottom — if present, it's
  run as `__main__`. Document the choice in SUMMARY.md.

**DO NOT modify:**
  - `research/dsplot/__init__.py:16` (docstring example only)
  - `research/dsplot/style.py:11` (docstring example only)
  - `research/dsplot/notebooks/09_04_smoke_test.ipynb` (notebook, not in scope)

**Verification grep:**
After all rewrites:

    grep -rn "^from dsplot\|^import dsplot\b" research/dsplot/figures/ \
        | grep -v '#' \
        | grep -v 'style_override_demo'   # if you kept the bootstrap
    # Expected output: empty (no matches), unless style_override_demo kept
    # its bootstrap for __main__ reasons.

**REMINDER: DO NOT `git commit`.**
  </action>
  <verify>
    <automated>
# No remaining `from dsplot` or `import dsplot` lines in figures/*.py
# (excluding comments and the conditionally-preserved style_override_demo).
HITS=$(grep -rn "^from dsplot\|^import dsplot\b" research/dsplot/figures/ | grep -v '^.*:#' | grep -v 'style_override_demo.py' | wc -l)
test "$HITS" -eq 0
# Import smoke test: each rewritten file's module-load succeeds from repo root.
python -c "from research.dsplot.figures.figure_1 import render_hero, render, render_antihero"
python -c "from research.dsplot.figures.components_recombine import render, render_vector_xy_reconstruction"
python -c "from research.dsplot.figures.lego_demo import show"
python -c "from research.dsplot.figures.style_skeleton import show"
python -c "from research.dsplot.figures.motivator import render_all"
python -c "from research.dsplot.figures.vector_basics import render"
python -c "from research.dsplot.figures.vector_projection_3d import render"
python -c "from research.dsplot.figures.dot_product_geometry import render"
python -c "from research.dsplot.figures.projection_reconstruction import render"
python -c "from research.dsplot.figures.alignment_diagnostic import render"
    </automated>
  </verify>
  <done>
- `from research.dsplot.figures.figure_1 import render_hero` succeeds from repo root with no PYTHONPATH hack
- Every figure module's top-level imports use `from ..` (relative) or stay absolute only where intentional (style_override_demo)
- The verification grep returns zero non-comment matches
  </done>
</task>

<task type="auto">
  <name>Task 3: Update render-function output_dir + output_filename defaults to new tree</name>
  <files>
    research/dsplot/figures/figure_1.py
    research/dsplot/figures/components_recombine.py
    research/dsplot/figures/motivator.py
    research/dsplot/figures/__main__.py
  </files>
  <action>
Update every render-function default `output_dir` AND default `output_filename`
to match the new `assets/images/figures/&lt;family&gt;/` tree. Use the exact same
filename stems that Task 1 produced.

**figure_1.py** — three render functions (lines 431, 615, 628):

  render():
    output_dir       "assets/images/generated"           -&gt; "assets/images/figures/figure_1"
    output_filename  "figure_1_fourier_vs_wavelet.png"   -&gt; "fourier_vs_wavelet.png"
                     # Note: this matches the v8 alias `fourier_vs_wavelet_v8_inst_freq_on_all_rows.png`
                     # form via the existing _09_05 / suffix mechanism in __main__.py.
                     # Default canonical name is the unsuffixed one.

  render_hero():
    output_dir       "assets/images/generated"                          -&gt; "assets/images/figures/figure_1"
    output_filename  "figure_1_hero_click_plus_tone_v1.png"             -&gt; "hero_click_plus_tone_v1.png"

  render_antihero():
    output_dir       "assets/images/generated"                          -&gt; "assets/images/figures/figure_1"
    output_filename  "figure_1_antihero_low_vibrato_v1.png"             -&gt; "antihero_low_vibrato_v1.png"

  # DO NOT touch the build_figure / _prepare / _prepare_hero / _prepare_antihero
  # logic. ONLY the default kwarg literals.

**components_recombine.py** — two render functions (lines 296, 318):

  render():
    output_filename  "components_recombine_either_order_v19.png" -&gt; "either_order_v19.png"
    # output_dir is positional with no default. Update __main__.py instead.

  render_vector_xy_reconstruction():
    output_filename  "vector_xy_reconstruction.png" -&gt; "vector_xy_reconstruction.png"
    # The filename has no `components_recombine_` prefix, but it's lived in
    # the components_recombine module historically. Strategy: route this to
    # a new family dir `vector_xy_reconstruction/`. So:
    output_filename stays "vector_xy_reconstruction.png" but moves to
    family `vector_xy_reconstruction/baseline.png` via __main__.py change.

**motivator.py** — multiple MotivatorConfig entries (lines 127, 136, 150, 165,
180, 219, 243). For EACH `output_filename` literal, strip the `dsp_motivator_`
prefix:

    output_filename="dsp_motivator_v4_100-2000hz_0.5s.png"
    -&gt; "v4_100-2000hz_0.5s.png"

    output_filename="dsp_motivator_v5_50-5000hz_1.0s.png"
    -&gt; "v5_50-5000hz_1.0s.png"

    output_filename="dsp_motivator_vw1_gentle_100-2000hz_0.5s.png"
    -&gt; "vw1_gentle_100-2000hz_0.5s.png"

    (same pattern for vw2, vw3, section1_20-20000hz_2.0s, vw4_aggressive_…)

  And update render_one's call site documentation / docstrings if they
  reference the old dir.

**__main__.py** — multiple hardcoded output paths (lines 28-29, 70-91, 95-103):

  _OUT_DIR (line 28):
    os.path.join(_RESEARCH_DIR, "..", "assets", "images", "dsp")
    -&gt; os.path.join(_RESEARCH_DIR, "..", "assets", "images", "figures")

  Per-call output_filename strings (lines 70-91):
    "vector_basics.png"                                  -&gt; family-dir override: "vector_basics/baseline.png"
    "dot_product_geometry.png"                           -&gt; "dot_product/geometry.png"
    "components_recombine_either_order_v19.png"          -&gt; "components_recombine/either_order_v19.png"
    "vector_xy_reconstruction.png"                       -&gt; "vector_xy_reconstruction/baseline.png"
    "projection_reconstruction_either_order_v9.png"      -&gt; "projection_reconstruction/either_order_v9.png"
    "vector_projection_3d_v2_combo5_palette.png"         -&gt; "vector_projection_3d/v2_combo5_palette.png"
    "dsp_alignment_diagnostic.png"                       -&gt; "alignment_diagnostic/baseline.png"

  motivator_out (line ~95):
    "assets/images/generated"  -&gt; "assets/images/figures/motivator"

  IMPORTANT: each `render(args.out, "&lt;family&gt;/&lt;file&gt;.png")` call concatenates
  args.out + filename. Since args.out is now `assets/images/figures`, the
  filename can be `&lt;family&gt;/&lt;file&gt;.png` and it Just Works. Alternative:
  pass the per-family dir as args.out for each call. Pick whichever is
  cleaner — the family-in-filename approach is the minimal diff.

  Update the docstring at the top of __main__.py too (lines 5-10) — replace
  `assets/images/dsp/` with `assets/images/figures/&lt;family&gt;/`.

**REMINDER: DO NOT `git commit`.**
  </action>
  <verify>
    <automated>
# Every render-function default points at the new tree.
grep -n 'output_dir.*=.*"assets/images/\(generated\|dsp\)"' research/dsplot/figures/ -r \
    | grep -v '#' \
    | wc -l \
    | xargs -I{} test {} -eq 0
# Module-level smoke: defaults are syntactically valid.
python -c "from research.dsplot.figures.figure_1 import render, render_hero, render_antihero; import inspect; \
  for f in (render, render_hero, render_antihero): \
    assert 'figures/figure_1' in inspect.signature(f).parameters['output_dir'].default, f.__name__"
    </automated>
  </verify>
  <done>
- All `render()` / `render_hero()` / `render_antihero()` defaults point at `assets/images/figures/&lt;family&gt;/`
- All motivator `MotivatorConfig.output_filename` literals have the `dsp_motivator_` prefix stripped
- `__main__.py` writes into the new tree by default; its docstring matches reality
- grep for old-tree string literals (`assets/images/generated`, `assets/images/dsp`) in `research/dsplot/figures/*.py` returns zero non-docstring hits
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>
Render-time verification + DSP.md / dsp.ipynb image-link updates + SUMMARY.md
mapping table.

Sub-steps the executor performs BEFORE the checkpoint:

1. **Update DSP.md image links (5 refs)** — apply the path rewrites listed in
   the &lt;interfaces&gt; block:

    DSP.md:130   components_recombine_either_order_v19.png  -&gt; components_recombine/either_order_v19.png
    DSP.md:140   projection_reconstruction_either_order_v9.png -&gt; projection_reconstruction/either_order_v9.png
    DSP.md:169   dot_product_geometry.png                    -&gt; dot_product/geometry.png
    DSP.md:199   vector_projection_3d_v2_combo5_palette.png  -&gt; vector_projection_3d/v2_combo5_palette.png
    DSP.md:245   vector_xy_reconstruction.png                -&gt; vector_xy_reconstruction/baseline.png

   Note the path prefix changes from `../../../assets/images/dsp/` to
   `../../../assets/images/figures/`.

2. **Update dsp.ipynb image refs (4 refs)** — same family-dir mapping. These
   are inside markdown cells; rewrite the JSON string literals carefully
   (preserve escaping). Use a python script or jq, NOT manual editing, to
   avoid mangling the notebook JSON. The notebook's nbformat structure must
   remain valid (verify with `jupyter nbconvert --to script --stdout > /dev/null`).

3. **Re-render verification (no PYTHONPATH hack):**

    python -c "from research.dsplot.figures.figure_1 import render_hero, render_antihero; \
               p1 = render_hero(); print('hero:', p1); \
               p2 = render_antihero(); print('antihero:', p2)"

    python -c "from research.dsplot.figures.figure_1 import render; print('chirp v8:', render())"

    python -c "from research.dsplot.figures.components_recombine import render; \
               import os; \
               print('components:', render('assets/images/figures/components_recombine', 'either_order_v19_reverify.png'))"

   For each render, confirm:
     - Imports succeed without PYTHONPATH=research
     - The PNG lands at the new per-family path
     - The old `assets/images/{generated,dsp}/` paths are NOT written to

4. **Write SUMMARY.md** at
   `.planning/quick/260521-qbo-cleanup-figure-assets-and-fix-dsplot-imp/260521-qbo-SUMMARY.md`
   with these sections:
     - Overview (3-line plain-English summary)
     - Decisions made (e.g. style_override_demo: kept absolute / converted to relative + rationale)
     - **Full old→new PNG mapping table** (from Task 1 /tmp/qbo_mapping.txt)
     - Files modified (with brief one-line per file)
     - How to run each render (one-liner per family)
     - Open questions / follow-ups for next quick
  </what-built>
  <how-to-verify>
The user reviews the visual + structural changes before deciding to commit.

**Step 1 — Spot-check the mapping table** in SUMMARY.md. Run this in a
terminal to compare against the live filesystem:

    git status --short | grep -E "^(R |A )" | head -40

The renames should be mechanical and predictable. Flag any surprises
(e.g. files moved into `misc/` that should have a dedicated family).

**Step 2 — Open DSP.md and confirm image previews render in the markdown
viewer.** All 5 image links should resolve. If any show a broken-image
icon, the path is wrong.

**Step 3 — Open dsp.ipynb in Jupyter (or VS Code's notebook UI) and confirm
all 4 image cells render.** Don't execute the notebook — just check the
inline markdown image previews.

**Step 4 — Re-run one of the figure_1 renders manually:**

    cd /home/eddie-water/dev/python/sub-shader
    source venv/bin/activate
    python -c "from research.dsplot.figures.figure_1 import render_hero; print(render_hero())"

Confirm:
  - No `ModuleNotFoundError`
  - Printed path is under `assets/images/figures/figure_1/`
  - The PNG was actually created at that path (`ls -lh &lt;printed_path&gt;`)

**Step 5 — Verify NO old-tree writes happened during re-render:**

    git status --short assets/images/generated/ assets/images/dsp/
    # Should show only the staged R (rename) entries from Task 1.
    # NO new files (Untracked or M) under generated/ or dsp/.

**Step 6 — Verify no unauthorized commit happened:**

    git log -1 --format="%s"   # Should NOT mention this quick's work
    git status                  # Should show staged renames + modifications,
                                # NOT a clean tree

If everything checks out, the user types "approved" and manually runs `git
commit` with their own message. If anything is broken, list the issues for
the executor to fix.
  </how-to-verify>
  <resume-signal>
Type "approved" if all renders, image previews, and the mapping table look
correct. Type "fix: &lt;issue&gt;" to send the executor back for a specific repair
(e.g. "fix: motivator dsp_motivator_section1 missed, still in __main__.py").
DO NOT type "approved" if you intend to commit the work yourself — the
checkpoint is purely a review gate; the executor never commits.
  </resume-signal>
</task>

</tasks>

<verification>
After all four tasks complete and the user approves the checkpoint:

  # 1. No PNGs lost (count check)
  TRACKED_BEFORE=$(git ls-files HEAD -- assets/images/generated/ assets/images/dsp/ | wc -l)
  TRACKED_AFTER=$(git ls-files -- assets/images/figures/ | wc -l)
  # TRACKED_AFTER should be >= TRACKED_BEFORE minus the un-touched
  # subdirs (stubs/, dpi/, style_override_demo/) which remain at their
  # original location.

  # 2. No broken image links in DSP.md
  while IFS= read -r path; do
      test -f "$path" || echo "BROKEN: $path"
  done &lt; &lt;(grep -oE 'assets/images/[^)]+\.png' src/subshader/dsp/DSP.md \
              | sed 's|^|/home/eddie-water/dev/python/sub-shader/|')

  # 3. No broken image refs in dsp.ipynb
  python -c "
  import json, os
  nb = json.load(open('src/subshader/dsp/dsp.ipynb'))
  import re
  refs = set()
  for cell in nb['cells']:
      for line in (cell.get('source') or []):
          refs |= set(re.findall(r'assets/images/[^\"\\\\)\\s]+\\.png', line))
  for r in refs:
      assert os.path.exists(r), f'BROKEN: {r}'
  print(f'{len(refs)} dsp.ipynb refs OK')
  "

  # 4. Import smoke from repo root, NO PYTHONPATH=research
  python -c "
  from research.dsplot.figures.figure_1 import render_hero, render, render_antihero
  from research.dsplot.figures.components_recombine import render as cr_render
  from research.dsplot.figures.lego_demo import show
  from research.dsplot.figures.style_skeleton import show as ss_show
  from research.dsplot.figures.motivator import render_all
  print('all imports OK')
  "

  # 5. Locked v8 chirp render still passes a byte-similarity check (visual)
  python -c "from research.dsplot.figures.figure_1 import render; print(render())"
  # User does a visual diff against pre-move v8 PNG.

  # 6. NO `git commit` was made by the executor
  git log -3 --format="%s%n%b" | grep -i "qbo\|figure-asset\|dsplot import" \
      && echo "UNAUTHORIZED COMMIT" \
      || echo "no executor commit — OK"
</verification>

<success_criteria>
- All PNGs from `assets/images/{generated,dsp}/` (excluding the 3 already-organized subdirs) moved into `assets/images/figures/&lt;family&gt;/` with prefixes stripped
- Zero PNG deletions in the diff
- `from research.dsplot.figures.figure_1 import render_hero` works from repo root without setting PYTHONPATH
- All render-function default `output_dir` / `output_filename` kwargs point at the new tree with stripped filenames
- DSP.md (5 refs) and dsp.ipynb (4 refs) image links resolve to the new locations; no broken-image previews
- Re-running each figure family's render writes to the new tree and does NOT write to the old tree
- SUMMARY.md contains the full old→new mapping table for spot-check
- NO `git commit` created by the executor — user reviews and commits manually
</success_criteria>

<output>
After the checkpoint approval, write SUMMARY.md to:
`.planning/quick/260521-qbo-cleanup-figure-assets-and-fix-dsplot-imp/260521-qbo-SUMMARY.md`

Include:
- Old→new mapping table (full list from /tmp/qbo_mapping.txt)
- Decisions log (style_override_demo path, any name collisions resolved as `baseline.png`, lego_demo source dir confirmation)
- Files modified (categorized by Task)
- How to run each family's render (one-liner per family — for future-self reference)
- Open questions for next quick (e.g. "should dsplot become a top-level package? — deferred")
</output>
