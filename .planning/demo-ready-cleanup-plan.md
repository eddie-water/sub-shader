# SubShader Demo-Ready Cleanup Plan

[Goal](#goal) · [Status](#status) · [How it works](#how-it-works) · [Workstreams](#workstreams) · [Decisions](#decisions) · [Worklog](#worklog)

---

## Goal

Get SubShader **mergeable to `develop` as v0.1** — a clean repo that renders as a
**landing page** (not a finished product). Five workstreams:

1. **Archive** the stale shit into one shallow `archive/` tree — duplicates, old frameworks,
   scratch. _Archive everything, delete nothing._
2. **Comment report** — audit `src/` comments, keep load-bearing DSP/math, strip litter.
3. **Docs stubbing** — every unfinished doc gets a 🚧 **UNDER CONSTRUCTION** banner so the repo
   reads as "v0.1, more coming" instead of broken.
4. **`dsp.ipynb` → `DSP.md`** — convert the notebook to standalone markdown with
   proportionally-spaced **gifs**, so it renders on GitHub/desktop without Jupyter. Convert the
   solid part (§1–2.5); stub §2.6+.
5. **Top-level `README`** — rewrite verbally + visually consistent with `DSP.md`.

v0.1 = landing page. **v1.0 = full hosted demo** (later). No merge to `develop` without explicit
user confirmation. No commits unless asked.

---

## Status

- [x] **A. Archive sweep** — stale files moved to shallow `archive/` (history-preserving). _Done._
- [x] **C1. Comment report** — `demo-ready-comment-report.md` written. _Done._
- [x] **C2a. Banner removal** — `# ===== SECTION =====` blocks stripped repo-wide (19 files); top-level spacing restored, all files compile, import OK. _Done (user: "remove")._
- [ ] **C2b. Comment fixes** — apply the rest of the punch-list (dead `_plot_kernel`, scratch, typos, breadcrumbs).
- [ ] **B. Docs stubbing** — 🚧 banners on README gaps; AUDIO/RENDERER as separate linked docs (see decision).
- [ ] **D. `DSP.md` from notebook** — convert §1–2.5 with gifs; resolve merge markers; stub §2.6+.
- [ ] **E. README rewrite** — match `DSP.md` voice + visual style; fill `[WRITE:]`/`[REWRITE:]` holes.

**Legend:** A + C1 + C2a done. B / C2b / D / E remain — **await user OK** (touch prose + tracked source).

---

## How it works

```
archive/          one shallow tree, history preserved. Delete nothing.
  legacy-figures/   pre-dsplot generators (comparison/figures/dsp_figures/palette_schema)
  scratch/          broken/temp source + drawio-cruft/ (WSL junk)
  scratch-images/   superseded PNGs from assets/images/claude/ top level
  old-docs/         superseded _v1 / architecture / refactor-notes docs

src/ stays the application. research/ keeps ONLY the live toolkit (timing_*, dsplot/, test_suite).
Docs link out: README → DSP.md / AUDIO.md / RENDERER.md, each stubbed where unfinished.
```

Notebook→markdown is **hand-authored**, not `nbconvert`: the notebook builds figures in code
cells, but the polished deliverables are the curated gifs under
`assets/images/dsp/figures/by_figure/`. Hand-authoring lets us embed gifs at proportional
`<img width>` widths (the README's existing `<p align="center">` idiom) and stub cleanly at the
§2.6 boundary.

---

## Workstreams

### A. Archive sweep ✅ (done)

Moved (import-free, verified by import-statement grep across `src/`, `tests/`, live `research/`):

| → archive/ | files |
|---|---|
| `legacy-figures/` | `comparison.py`, `figures.py`, `dsp_figures.py`, `palette_schema.py` |
| `scratch/` | root `dot.py`, `_temp_dot_product_symmetry_candidates.py` |
| `scratch/drawio-cruft/` | `figure 242 drawio.drawio`, `.dtmp`, `*:Zone.Identifier` |
| `scratch-images/` | 23 loose PNGs from `assets/images/claude/` (palette_*, bad-overlap, old_motivator_*, …) |
| `old-docs/` | `README_pipeline_timing.md`, `REFACTOR_NOTES.md`, `dsp_v1.md`, `architecture_flowchart.md`, `architecture_overview.md` |

**Resolved / held:**
- `claude-planning-guide.md` — **moved to `.claude/` (kept tracked on git)**, per user. _Done._
- `research/dot.py` — maintained sandbox; live figure `gen_figure_3_1` cites it as §3.1's source. **Keep.**
- `assets/images/claude/diagnostics/` — holds the **live** asset `optichrome_felipe_pantone.png`
  (`optichrome_overview.py:111`). Folder stays; deep-sorting its scratch is a later item.
- `__pycache__/` + `*.pyc` — regenerable; `.gitignore` them rather than archive (see Decision 3).

### B. Docs stubbing 🚧

Add a consistent banner to every unfinished area so v0.1 reads as intentional:

```markdown
> 🚧 **UNDER CONSTRUCTION** 🚧 — v0.1 landing page. Full write-up lands with v1.0.
```

Targets:
- `README.md` — replace `[WRITE:]` / `[REWRITE:]` / `[List future improvements]` placeholders
  (Benchmark, Installation, Future Improvements) with stub banners or real copy (see E).
- `AUDIO.md`, `RENDERER.md` — **kept as separate linked docs** (user decision). Each gets the 🚧
  banner **plus a short paragraph describing what the finished doc will cover** — not just "TBD".
- `DSP.md` §2.6+ — banner the boundary past the converted-solid content (see D).

### C. Comments

- **C1 ✅** — report written: `demo-ready-comment-report.md`.
- **C2a ✅** — `# ===== SECTION =====` banner blocks stripped repo-wide (user: "remove"). 19 files;
  verified only comments + blank lines removed (no code), PEP8 top-level spacing restored, all
  modules compile + import. The 5 files with pre-existing uncommitted edits were fixed in place.
- **C2b** — remaining punch-list: delete dead `_plot_kernel()` body (`wavelet_kernel.py:83–94`),
  strip `signal_generator.py` scratch, fix `renderer.py:43/246` breadcrumbs, `gaussian.py` typos.
  **Keep** all `D-xx`/`Pitfall`/L1 DSP comments — the "extremely important to call out" class.

### D. `DSP.md` from the notebook

The current `src/subshader/dsp/DSP.md` has **unresolved git merge-conflict markers**
(`<<<<<<< Updated upstream` …). The conversion **replaces** it wholesale (user decision).

Section → asset map (each has a curated gif + still under `by_figure/`):

| § | content | gif asset |
|---|---|---|
| 1 | Motivations (Fourier vs Wavelet) | `fig_1_fourier_vs_wavelet` hero/antihero |
| 2.1–2.3 | Decomposition / Inner Product / Dot Product | (text; still from `fig_2_4_*`) |
| 2.4.1 | Recombine x+y | `fig_2_4_1_xy_recombine_cycle_v1.gif` |
| 2.4.2 | Project a onto b | `fig_2_4_2_a_onto_b_cycle_v1.gif` |
| 2.4.3 | Dot product in 3D | `fig_2_4_3_dot_product_3d_cycle_v1.gif` |
| 2.5 | Sign accumulation | `fig_2_5_sign_accumulation_cycle_v49_border_flush.gif` |
| 2.6 | Sine basis | `fig_2_6_sine_basis_2hz_10hz_v21.gif` → 🚧 **stub boundary** |
| 3–4 | Fourier / Wavelet analysis | 🚧 UNDER CONSTRUCTION (notebook still draft) |

Embed idiom: `<p align="center"><img src="…gif" width="N%"></p>` at proportional widths.
Prose is the notebook's — Claude scaffolds embeds + stubs; **user authors final copy.**

### E. README rewrite

Match `DSP.md`'s voice + visual rhythm. Keep the strong existing material (Design,
Plot/Timing comparison). Resolve holes:
- Benchmark → short summary + link to `TIMING.md` / `DSP.md` deep-dive.
- Installation → real steps (Python 3.9+, CUDA, OpenGL 3.3+).
- Future Improvements → concrete list (hosted demo, live capture, GPU panel, color controls).
- Add the two missing flowchart placeholders or stub them 🚧.

---

## Decisions

1. **Replace `DSP.md`** with the notebook conversion (not a new top-level file). Fixes the
   merge-marker corruption in the same move.
2. **Convert solid (§1–2.5), stub the rest.** Notebook is only reliable through ~§2.5
   ("need figures for 2.5+"); §2.6+ ships as 🚧, not as messy draft prose.
3. **Archive everything, delete nothing** — except `__pycache__`/`.pyc`, which are regenerable
   bytecode → `.gitignore`, not archive.
4. **Safe parts now, heavy parts on approval.** A (archive) + C1 (report) are reversible /
   read-only and done. B/C2/D/E touch tracked source + author-voice prose → await user OK.
5. **No commit, no merge** without explicit user confirmation.

### Open questions — all resolved

- ✅ `claude-planning-guide.md` → moved to `.claude/` (kept on git).
- ✅ Strip `# ===== SECTION =====` banners → **removed** repo-wide.
- ✅ `AUDIO.md`/`RENDERER.md` → **separate linked docs**, each 🚧 + a paragraph describing intent.

---

## Worklog

- **2026-06-26** — Plan drafted. Two read-only assessments run in parallel:
  **(1) stale-file manifest** — import-statement grep confirmed the "big four" pre-dsplot
  generators are referenced only by each other; whole legacy set + scratch archivable. Caught
  one live asset (`optichrome_felipe_pantone.png`) inside `assets/images/claude/diagnostics/`
  → folder held back. **(2) comment audit** — src is clean; two scratch files
  (`signal_generator.py`, `wavelet_kernel._plot_kernel`) + minor breadcrumbs/typos; all DSP
  design comments keep. **Executed safe parts:** archive sweep (history-preserving `git mv`
  where tracked) → `archive/{legacy-figures,scratch,scratch-images,old-docs}`; comment report
  written to `demo-ready-comment-report.md`. Notebook→`DSP.md` section/gif map built.
- **2026-06-26** — Resolved the 3 open questions + executed two directed items.
  `claude-planning-guide.md` → `.claude/` (kept tracked). **Banner blocks removed** repo-wide:
  a strip pass deleted every `# ===== LABEL =====` triple (19 files), then a spacing pass restored
  2 blank lines before top-level `def`/`class`/`@` (the strip had over-collapsed them). Verified
  via diff (only comments + blanks changed — no code), `py_compile` on all files, and a package
  import smoke test. 5 of the files had pre-existing uncommitted edits → fixed in place, not via
  git restore, so that work is preserved. **Pending user OK:** C2b comment fixes, B docs stubbing
  (AUDIO/RENDERER as separate 🚧 docs w/ intent paragraphs), D `DSP.md` conversion, E README rewrite.
