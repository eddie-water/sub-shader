# Performance Ledger

Append-only record of performance fixes. Every entry pairs one code change
with a before/after timing run, so each claim about the pipeline getting
faster is backed by a recorded measurement and the exact diff that caused it.

## Index

| Entry | Whole frame | Headroom | Log |
| --- | --- | --- | --- |
| Fix 1 — FFT length padded to a fast size | 77.0 → 36.7 ms 🟢🟢 | 2.4× → 5.1× | [2026-08-25_21-34-36_44b67ce](ledger/2026-08-25_21-34-36_44b67ce_fix1_fft_padding.md) |
| Fix 2 — GPU clock floors locked (SM + memory) | 37.0 → 7.9 ms 🟢🟢🟢 | 5.0× → 23.5× | [2026-08-25_22-17-13_56c9f42](ledger/2026-08-25_22-17-13_56c9f42_fix2_gpu_clock_lock.md) |
| Fix 3 — down-sample aliasing: decimation → max-pool | 7.9 → 8.4 ms 🔴 (visual fix) | 23.6× → 22.1× | [2026-08-27_00-43-50_56c9f42](ledger/2026-08-27_00-43-50_56c9f42_fix3_downsample_max_pool.md) |
| Fix 4 — CWT post-processing moved to the GPU | 8.4 → 5.8 ms 🟢🟢 | 22.1× → 32.0× | [2026-08-28_21-39-01_16177fd](ledger/2026-08-28_21-39-01_16177fd_fix4_gpu_post.md) |

## How an entry is made

Each fix gets its own log file in `ledger/`, named by the **after**-run's
timestamp and the git sha the timing runs recorded:

```
ledger/yyyy-mm-dd_HH-MM-SS_<shortsha>_<slug>.md
```

1. Baseline: `python research/timing_campaign.py --full` (from repo root) —
   note the `run_id` it records into `timing_results.csv`.
2. Apply the fix. Save its raw patch: `git diff <files> > assets/timing/patches/<slug>.patch`
3. Re-run the same timing command — note the new `run_id`.
4. Generate the delta tables (module rollup + per-stage, diagram-colored):
   `python research/timing_diff.py <run_id_before> <run_id_after>` (from repo root)
5. Generate the timing diff figure (before / after / saved-or-cost on one axis):
   `python -m research.dsplot.figures.gen_timing_ledger_diff <run_id_before> <run_id_after> assets/timing/ledger_fix<N>_timing_v1.png`
6. If the fix changes what the pipeline *produces* (DSP, post-processing,
   rendering), render a before / after / difference figure from the real
   pipeline — one generator per fix under `research/dsplot/figures/gen_ledger_evidence_<slug>.py`
   (Fix 3's `gen_ledger_evidence_downsample.py` is the pattern). Pure-speed
   fixes state "none" instead.
7. Run the fix's tests verbosely and paste the output:
   `python -m pytest tests/<file>.py -v`
8. Write the log from the template below; add one row to the Index above.
   New figure iterations get new `_v<N>` files — never overwrite a PNG.

## Entry template

```markdown
## <Fix name — one line>

**Date:** yyyy-mm-dd · **Patch:** [patches/<slug>.patch](../patches/<slug>.patch)
**Runs:** `<run_id_before>` (before) → `<run_id_after>` (after)

**Root cause.** <2-5 sentences: what was slow, why, what the fix does, and
how correctness is proven (test name).>

### Effect on the output

<before / after / difference figure from the real pipeline, and one
paragraph on what the difference row shows — or "None" with the reason
the output is unchanged>

### Timing diff

<timing_ledger_diff figure; one line naming the saved/cost segments>

<paste timing_diff.py output: headline lines, legend, Modules table,
Process loop stages table, Start-up table>

**Notes.** <run conditions (pacing mode, GPU contention), anything that
qualifies the comparison, and what the result points at next.>

### Tests

<`pytest tests/<file>.py -v` output in a fence, with the date it was run>

## Code diff

<the patch, in a ```diff fence — makes the log self-contained>
```

## Reading the numbers

The loop deadline is 185.8 ms (8,192 new samples per frame at 44.1 kHz).
Real time = 44.1k samples/s; headroom = deadline / mean frame work.
`timing_diff.py` prints all three, plus glyphs: 🟢 faster / 🔴 slower /
⚪ same-ish (dots scale >5% / >25% / >60%), and module colors matching the
timing diagrams: 🟡 Audio · 🟣 DSP · 🟠 Render · ⚫ Setup.

## Long-term direction

(For future sessions.) This ledger is the seed of a timing-driven diagram
generator — pipeline/timing figures whose stage lengths are drawn from
recorded runs, with per-fix diffs highlighting which stages shrank. The
pieces that exist today: per-stage recordings keyed by run_id + git sha
(`timing_results.csv`, `timing_iterations.csv`), per-fix logs (`ledger/`)
and raw patches (`patches/`), the delta tool (`research/timing_diff.py`),
and the figure generators (`research/dsplot/figures/gen_timing_gantt.py`,
`research/gen_pipeline_flowchart.py`). The missing piece is wiring: a
stage-key → diagram-block mapping shared by all generators, driven from
this ledger's run pairs.
