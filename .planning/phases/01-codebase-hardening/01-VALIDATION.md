---
phase: 1
slug: codebase-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | PIPE-02 | unit | `python -m pytest tests/test_gpu_fallback.py -v` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | PIPE-03 | unit | `python -m pytest tests/test_gpu_detection.py -v` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | QUAL-01 | manual | code review | N/A | ⬜ pending |
| TBD | TBD | TBD | QUAL-03 | manual | code review | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/` directory created
- [ ] `tests/conftest.py` — shared fixtures
- [ ] `pytest` installed in venv
- [ ] `tests/test_gpu_fallback.py` — stubs for PIPE-02
- [ ] `tests/test_gpu_detection.py` — stubs for PIPE-03

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Code readability in changed files | QUAL-01 | Subjective assessment | Review changed files for descriptive names, helpers, no comment litter |
| Existing readability maintained | QUAL-03 | Subjective assessment | Diff review to ensure no unnecessary refactoring |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
