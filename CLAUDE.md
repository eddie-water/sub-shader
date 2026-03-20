# Claude Code Collaboration Guidelines

## Context Management

**Every Claude session displays a real-time context limit bar** with compact marker. This makes token usage visible and prevents mid-task context clears.

### Context Bar Format
Appears at the top of each response:
```
[████████████░░░░░░░░░░░░░ 52% | 104K/200K tokens]
```

### Thresholds & Actions
- **0-75%**: Normal operation
- **75%**: ⚠ Warning — begin planning compact/checkpoint for next context clear
- **90%+**: 🛑 Critical — COMPACT NOW before context limit forces session break

### Why This Matters
Context window fills quickly on complex projects. Visible tracking prevents surprise cutoffs and forces proactive planning instead of reactive scrambling.

---

## Git & Commits

**Never auto-commit code.** Always wait for explicit user request.
- User must explicitly say "commit", "save", or "create a PR"
- No destructive operations (reset --hard, force push) without confirmation
- Prefer creating new commits over amending published work

---

## Code Style

- No y-axis labels on subplot axes (commented out in `utilities/plotting.py`)
- PyWavelet computation can be stubbed with `--stub-pywt` flag for faster iteration
- Output redirects to `stubs/` folder when using `--stub-pywt`

---

## Useful Commands

```bash
# Fast iteration with stubbed PyWavelet (saves to stubs folder)
python research/benchmark.py --figures --stub-pywt

# Full figure generation with all methods
python research/benchmark.py --figures

# Run unit tests
python research/benchmark.py --unit-tests

# All modes at once
python research/benchmark.py --all
```
