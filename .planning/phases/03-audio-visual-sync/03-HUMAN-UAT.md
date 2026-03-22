---
status: partial
phase: 03-audio-visual-sync
source: [03-VERIFICATION.md]
started: 2026-03-21
updated: 2026-03-21
---

## Current Test

[awaiting human testing]

## Tests

### 1. Simultaneous playback — audio and shader rendering start together
expected: Running `python -m subshader demo.wav` plays audio and renders CWT frames simultaneously, not sequentially
result: [pending]

### 2. Sub-100ms transient response
expected: Transient events in the audio (drum hit, sharp consonant) appear in the visualization within ~100ms of being heard
result: [pending]

### 3. No drift over 60 seconds
expected: The visualization does not drift ahead or behind the audio over a 60-second playback — sync holds for the duration
result: [pending]

### 4. Immediate audio stop on window close
expected: When user closes window, audio stops immediately — no orphaned audio playing after window closes
result: [pending]

### 5. Seamless audio loop
expected: When audio file ends, loop seamlessly — audio restarts from the beginning, visualization resets and re-renders as if starting fresh
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
