---
phase: 3
slug: audio-visual-sync
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | none — pytest discovers tests/ directory automatically |
| **Quick run command** | `pytest tests/test_audio_player.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_audio_player.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | AUDIO-01 | setup | `pip install sounddevice` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | AUDIO-01 | unit | `pytest tests/test_audio_player.py::test_playback_position_advances -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | AUDIO-01 | unit | `pytest tests/test_audio_player.py::test_chunk_selection_from_position -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | AUDIO-01 | unit | `pytest tests/test_audio_player.py::test_cli_arg_overrides_path -x` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | AUDIO-02 | unit | `pytest tests/test_audio_player.py::test_invalid_file_raises -x` | ❌ W0 | ⬜ pending |
| 03-01-06 | 01 | 1 | AUDIO-02 | unit | `pytest tests/test_audio_player.py::test_loop_wrap_detection -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_audio_player.py` — stubs for AUDIO-01, AUDIO-02
- [ ] `sounddevice` added to `pyproject.toml` dependencies and installed in venv

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Audio plays through speakers in sync with visualization | AUDIO-01 | Requires audio hardware output | Run `python -m subshader demo.wav`, listen for sync between audio and visual transients |
| Sub-100ms perceived latency on transients | AUDIO-02 | Perceptual test requires human ear | Play file with sharp drum hits, observe if visual response feels immediate |
| 60-second drift-free playback | AUDIO-01 | Duration test with perceptual component | Run for 60+ seconds, check visualization doesn't drift ahead/behind |
| Seamless audio loop restart | AUDIO-01 | Perceptual continuity check | Let audio reach end, verify loop restart feels smooth |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
