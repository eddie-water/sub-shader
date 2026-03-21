---
phase: 2
slug: cwt-pipeline-polish
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=7.0 (installed as dev dependency) |
| **Config file** | none — pytest discovers `tests/` without config |
| **Quick run command** | `pytest tests/test_cwt_normalization.py -x` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_cwt_normalization.py -x`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | PIPE-01 | unit | `pytest tests/test_cwt_normalization.py::TestWaveletKernelNormalization -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | PIPE-01 | unit | `pytest tests/test_cwt_normalization.py::TestCwtBrightnessBias -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | PIPE-01 | unit | `pytest tests/test_cwt_normalization.py::TestNormalizeByScale -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | QUAL-02 | regression | `pytest tests/ -q` | ✅ (21 tests) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_cwt_normalization.py` — stubs for PIPE-01 (kernel norm, magnitude ratio, no-op)
- [ ] `tests/conftest.py` additions — `numpy_wavelet` fixture needed by PIPE-01 tests

*Existing `conftest.py` exists but lacks wavelet fixtures; additions required, not a new file.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Low-freq bands visually match high-freq brightness | PIPE-01 | Subjective visual check | Run `python -m subshader` with test audio, observe spectrogram for uniform brightness across bands |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
