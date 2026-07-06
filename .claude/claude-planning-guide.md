# Working with Claude — A Simple Planning Guide & Template

The whole idea: **agree on a good plan before Claude does any work.** Getting the plan right first avoids messy clean-ups later.

---

## The big picture

There are three kinds of prompts, and each is handled a little differently.

**Any prompt → `CLAUDE.md` (your always-on rules)**

One `CLAUDE.md` per repo. Claude reads it on *every* prompt. These are the global rules that apply no matter the task:

- "Don't commit without my permission."
- "Always show me the plan before changing anything."
- "Use plain English when describing complex or abstract concepts."

Add to it anytime: *"Add this to CLAUDE.md so you always do it: …"*

**Complex / difficult task → `plan.md` (agree on a plan, then run the loop)**

Make a plan first and agree on it *before* any work. A solid plan up front is far cheaper than cleaning up a loose one — and it keeps the session lean. (Long sessions bloat with stale, out-of-date context, so every reply costs more and the answers get worse.) Use the template below, then follow the loop.

**Small / easy task → usually no plan**

Just ask for it. If it needs a little thought, have Claude sketch a quick mini-plan right in the chat before proceeding. Truly trivial things — looking something up, changing a few lines — need nothing at all.

---

## The loop

```
1. PLAN      →  Claude writes a plan.md and STOPS. No work yet.
2. APPROVE   →  You read it, ask for changes, until it's right. Then say "approved."
3. EXECUTE   →  Claude does the work.
4. VALIDATE  →  You check the results.

   ✅ Correct?    →  Done.
   ❌ Not right?  →  Log what happened (what worked, what didn't, what's next),
                     then clear/compact the chat when context hits ~40%,
                     and start the next round from the plan + log.
```

The log is your memory between rounds. When the chat gets heavy (around **40% context full**), ask Claude to summarize into the log, start a **fresh session**, and paste the log back in. You keep the progress, you drop the stale clutter.

**Validating results:** tell Claude how to run things (the terminal command, the branch, the environment) and where results show up — or let Claude *ask you*. You can't confirm it worked if you don't know where to look.

---

## The Template (copy into `plan.md` for complex tasks)

```markdown
## Goal
<One or two plain sentences: what do I want by the end?>

## Background / Context
<What Claude needs to know. Why am I doing this? What happened before?>

## Key Files / References
<Files to read or change. Examples to copy the style of. Reference material.>

## The Plan (do this FIRST)
Write me a step-by-step plan and STOP.
Do not change, create, or delete anything until I reply "approved."

## How to Run It / See Results
<How do I check the work? The command to run, the branch, where output shows up.
 If you're not sure, ASK me before finishing.>

## Confirmation / Validation
When done, tell me: what changed, how to confirm it worked, what to double-check.
Then wait for me to confirm before calling it finished.

## Work Log / Tracking
Keep a running log and update it each round. Keep it short and plain:
 - What we did:
 - What worked / what we learned:
 - What did NOT work (so we don't repeat it):
 - Decisions made (and why):
 - What's left to do next:
```

---

## Three things to remember

1. **Plan first, approve, then work** — for anything complex.
2. **Log before you clear** — so a fresh session knows where you left off.
3. **When in doubt, ask** — type *"explain that simply"* or *"what are my options?"* anytime.
