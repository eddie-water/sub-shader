# Phase 09 — Deferred / Out-of-Scope Items

Items discovered during execution that are out of scope for the current plan.

## From Plan 09-01 Execution

### Orphan test file: `research/tests/dsplot/test_panel_composition.py`

- **Found:** Already untracked in the worktree at plan 09-01 start.
- **Owner:** Plan 09-02 (Panel composition contract).
- **Status:** Out of scope for 09-01. File currently fails collection because
  `dsplot.panels` does not exist yet (09-02 will create it). The file's
  `pytest.importorskip("dsplot", ...)` guards against the top-level package
  being absent but not against `dsplot.panels` specifically.
- **Action:** Leave in place — it belongs to 09-02 and the orchestrator will
  surface it when 09-02 runs. Plan 09-01's verify command targets
  `tests/dsplot/test_plottable_construction.py` specifically, so the orphan
  does not block 09-01.
