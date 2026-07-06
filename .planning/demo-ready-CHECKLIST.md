# Demo-Ready Cleanup — Status Checklist

Quick view of where we are and what needs you. Tick as you review.
Full detail: [demo-ready-cleanup-plan.md](demo-ready-cleanup-plan.md) · [comment report](demo-ready-comment-report.md)

---

## ✅ Done — please review (nothing committed, all reversible)

- [ ] **Archive sweep** — skim `archive/` (legacy-figures · scratch · scratch-images · old-docs).
      Verify renames: `git status --short | grep '^R'`
- [ ] **Planning guide** moved → `.claude/claude-planning-guide.md` (still tracked)
- [ ] **Banner blocks removed** repo-wide — `git diff -- src/subshader` (only comments + blanks)
- [ ] **Pre-edited files intact** — spot-check your earlier work survived:
      `git diff src/subshader/renderer/renderer.py` · `dsp/cwt.py` · `pipeline.py`
- [ ] **Still imports** — `source venv/bin/activate && python3 -c "import subshader"`
- [ ] **Plan + comment report** read and look right

## 🟡 Needs your decision before I continue

- [ ] **Go-ahead to execute the rest?** Pick order:
      - [ ] C2b + B first (low-risk: comment fixes + doc stubs)
      - [ ] D first (DSP.md notebook conversion — biggest landing-page win)
- [ ] **`signal_generator.py`** — it's unimported research scratch sitting in `src/`.
      Keep + clean comments, or move to `archive/`?
- [ ] **README direction (E)** — confirm the three holes get *real copy you author* vs stub:
      Benchmark · Installation · Future Improvements

## ⏳ Queued — waiting on the go-ahead above

- [ ] **C2b** — dead `_plot_kernel()` body, `signal_generator.py` scratch, `renderer.py:43/246`,
      `gaussian.py` typos
- [ ] **B** — README gap stubs + `AUDIO.md`/`RENDERER.md` as separate 🚧 docs with intent paragraphs
- [ ] **D** — convert `dsp.ipynb` → `DSP.md` (§1–2.5 + gifs, fixes the merge-marker corruption, stubs §2.6+)
- [ ] **E** — README rewrite in `DSP.md`'s voice

## 🚩 Flags to keep in mind

- [ ] `DSP.md` currently has **unresolved git merge-conflict markers** — workstream D replaces the
      file and fixes this. Don't merge before D.
- [ ] **No commit / no merge** until you say so.
- [ ] Prose (D/E + AUDIO/RENDERER) — I scaffold; **you author the final copy.**
