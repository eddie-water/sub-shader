---
phase: 02-cwt-pipeline-polish
plan: "02"
subsystem: dsp
tags: [cwt, normalization, benchmark, figures, visualization]

requires:
  - phase: 02-cwt-pipeline-polish/02-01
    provides: [L1-normalized wavelet kernels, cwt-normalization-tests]
provides:
  - benchmark PNGs regenerated with normalized kernels
  - chirp_signal_comparison.png showing uniform frequency sweep brightness
  - polyphonic_signal_comparison.png showing balanced multi-tone intensity
affects: [README, documentation, assets/images/benchmarks]

tech-stack:
  added: []
  patterns: [benchmark figure regeneration after DSP changes]

key-files:
  created: []
  modified:
    - assets/images/benchmarks/chirp_signal_comparison.png
    - assets/images/benchmarks/polyphonic_signal_comparison.png
    - research/utilities/constants.py

key-decisions:
  - "AUDIO_POLYPHONIC constant updated to overlapping_A3_A4_A5.wav — polyphonic_audio_example.wav was an empty 0-frame placeholder"
  - "No intensity tracker tuning needed — color range looks reasonable after normalization fix"

patterns-established: []

requirements-completed: [PIPE-01, QUAL-02]

duration: 11min
completed: 2026-03-21
---

# Phase 02 Plan 02: Benchmark Figure Regeneration Summary

**Regenerated chirp and polyphonic CWT comparison figures using L1-normalized kernels; uniform brightness across full frequency range confirmed.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-21T22:55:41Z
- **Completed:** 2026-03-21T23:06:41Z
- **Tasks:** 2 (1 auto + 1 checkpoint auto-approved)
- **Files modified:** 3

## Accomplishments

- Regenerated `chirp_signal_comparison.png` (128 frames, 200 Hz to 20 kHz sweep) with L1-normalized kernels
- Regenerated `polyphonic_signal_comparison.png` (128 frames, A3/A4/A5 simultaneous tones) with L1-normalized kernels
- Fixed broken `AUDIO_POLYPHONIC` constant that pointed to an empty 0-frame placeholder file
- Confirmed SubShader CWT at 83 ms/frame is 14x faster than PyWavelet at 1152 ms/frame

## Task Commits

Each task was committed atomically:

1. **Task 1: Regenerate benchmark comparison figures** - `84175d5` (feat)
2. **Task 2: Visual verification** - auto-approved checkpoint (no commit)

**Plan metadata:** pending final commit

## Files Created/Modified

- `assets/images/benchmarks/chirp_signal_comparison.png` - Updated chirp sweep comparison (STFT | PyWavelet | SubShader) with normalized kernels
- `assets/images/benchmarks/polyphonic_signal_comparison.png` - Updated multi-tone comparison using overlapping A3/A4/A5 audio
- `research/utilities/constants.py` - Fixed AUDIO_POLYPHONIC to point to a real audio file

## Decisions Made

**AUDIO_POLYPHONIC constant updated:** `polyphonic_audio_example.wav` was an empty file (0 frames, 0.0 seconds duration). Updated to `overlapping_A3_A4_A5.wav` (59 seconds, simultaneous A3/A4/A5 tones) which is a semantically correct polyphonic test signal.

**No intensity tracker tuning needed:** After reviewing the generated figures, color range appears well-balanced. D-09 observation satisfied — no follow-up tuning required.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed AUDIO_POLYPHONIC pointing to empty placeholder file**
- **Found during:** Task 1 (Regenerate benchmark comparison figures)
- **Issue:** `research/utilities/constants.py` defined `AUDIO_POLYPHONIC = "assets/audio/daw/polyphonic_audio_example.wav"` but that file has 0 frames (empty placeholder). Benchmark reported "No frames processed" for polyphonic figure.
- **Fix:** Updated constant to `overlapping_A3_A4_A5.wav` — a 59-second file of simultaneous A3/A4/A5 arpeggios, semantically appropriate as a polyphonic test signal.
- **Files modified:** `research/utilities/constants.py`
- **Verification:** `polyphonic_signal_comparison.png` generated with 128 frames processed; 27/27 pytest tests still pass
- **Committed in:** `84175d5` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix)
**Impact on plan:** Necessary fix to make the polyphonic figure generation work at all. No scope creep.

## Issues Encountered

None beyond the AUDIO_POLYPHONIC bug above.

## Known Stubs

None — both PNGs are fully rendered with real DSP data from L1-normalized kernels.

## Next Phase Readiness

- CWT normalization fix is visually confirmed via benchmark figures
- Phase 02 (cwt-pipeline-polish) is complete — all 2 plans done
- Ready to proceed to Phase 03 (audio-visual sync / playback pipeline)

---
*Phase: 02-cwt-pipeline-polish*
*Completed: 2026-03-21*
