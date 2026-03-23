---
phase: 5
slug: documentation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + manual file verification |
| **Config file** | none — documentation phase, validation is structural |
| **Quick run command** | `python -c "import subshader"` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Verify files exist and contain expected sections via grep
- **After every plan wave:** Run full pytest suite + verify all image placeholders reference existing files
- **Before `/gsd:verify-work`:** Full suite must be green, all scaffolds must have section headers
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | DOCS-01 | structural | `grep -c "##" README.md` | ✅ | ⬜ pending |
| 05-01-02 | 01 | 1 | DOCS-05 | structural | `grep -c "\[IMAGE" README.md` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | DOCS-02 | structural | `test -f DSP.md && grep -c "##" DSP.md` | ❌ W0 | ⬜ pending |
| 05-03-01 | 03 | 1 | DOCS-03, DOCS-04 | structural | `test -f AUDIO.md && test -f RENDERER.md` | ❌ W0 | ⬜ pending |
| 05-04-01 | 04 | 2 | DOCS-05 | execution | `python research/benchmark.py --figures` | ✅ | ⬜ pending |
| 05-05-01 | 05 | 2 | DOCS-06 | manual | visual inspection of scaffold quality | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements. No new test framework needed.
- Documentation validation is primarily structural (file existence, section presence, link validity).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Scaffold reads as guidance, not draft prose | DOCS-06 | Subjective quality check | Read each scaffold — verify bullet-point placeholders, not paragraphs |
| Comparison grid figure layout is readable | DOCS-05 | Visual quality check | Open generated figure, verify labels readable at README display size |
| User voice preserved in existing README sections | DOCS-06 | Subjective | Compare scaffold-modified sections against original draft voice |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
