# Working with Claude — A Simple Planning Guide & Template

The whole idea: **agree on a good plan before Claude does any work.** Getting the plan right first avoids messy clean-ups later.

---

## The big picture

Here is we manage Claude's context effectively, to give us the best results possible. It depends on the kind of prompt

### Any and Every Prompt 

Always reads `CLAUDE.md` → Always adheres to these guidelines

One `CLAUDE.md` per repo. Claude reads it on *every* prompt. These are the global rules that apply no matter the task:

- "Don't commit without my permission."
- "Always show me the plan before changing anything substantial."
- "Use plain English when describing complex or abstract concepts."

If you don't have one, make one using claude with /init

Add to it anytime by telling claude *"Add this rule: ... to CLAUDE.md so you always do it"*

Or just open the CLAUDE.md and write in it - try not to put anything too **specific** because it will probably be applied **every** time

### Complex Prompts / Difficult Tasks
Create and agree on a **specific** plan → `name-of-my-specific-prompt_plan.md` (agree on a plan, then run this workflow with)**

Make a plan first and agree on it *before* any work. A solid plan up front is far cheaper than cleaning up after a loose one — and it keeps the session lean. (Long sessions bloat with stale, out-of-date context, so every reply costs more and the answers get worse.) Use the template below, then follow the loop.

### Small Prompt  / Easy Task 

If its a really trivial task, you usally don't need a full on plan.

If it needs a little thought, have Claude sketch a quick mini-plan right in the chat before proceeding. Truly trivial things — looking something up, changing a few lines — dont really need a full scale written md plan

---

## The loop

```
1. ASK       →  Claude asks you a few clarifying questions.
2. PLAN      →  Claude writes a plan.md. No work yet. Modify or delete things that don't make sense.
3. APPROVE   →  Read the plan. Answer more questions and ask for changes until it's right. Then give Claude your approval.
4. EXECUTE   →  Tell Claude to execute the plan.
5. VALIDATE  →  You check the results.

   ✅ Correct?    →  Done.
   ❌ Not right?  →  Log what happened (what worked, what didn't, what's next),
                     then clear/compact the chat when context hits ~40%,
                     and start the next round from the plan + log.
```

**Clarifying questions first:** before finalizing the plan, have Claude ask you about anything it's unsure of. Keep these to the **key decisions that affect the goal** — not tiny details. A quick *"answer these before you plan"* up front prevents Claude from guessing wrong.

The log is your memory between rounds. When the chat gets heavy (around **40% context full**), ask Claude to summarize into the log, start a **fresh session**, and paste the log back in. You keep the progress, you drop the stale clutter.

**Validating results:** tell Claude how to run things (the terminal command, the branch, the environment) and where results show up — or let Claude *ask you*. You can't confirm it worked if you don't know where to look.

---

## Template 1 — `CLAUDE.md` (the always-on rules, one per repo)

These are global rules Claude follows on *every* prompt, so keep them **general** — nothing tied to one specific task.

```markdown
## About this project
<One or two lines: what is this repo / what am I working on?>

## Always
- For complex tasks, use a `plan.md` (see the plan.md template) and agree on it before any work.
- Show me the plan before changing anything substantial.
- Use plain English when explaining complex or abstract things.
- <add your own...>

## Never
- Don't commit without my permission.
- Don't delete files without asking first.
- <add your own...>

## How I like to work
<How should Claude talk to me? Short answers? Explain as you go? Ask before big moves?>

<!-- Feel free to add more sections — keep them general, not task-specific. -->
```

---

## Template 2 — `plan.md` (one per complex task)

```markdown
## Goal
<One or two plain sentences: what do I want by the end?>

## Background / Context
<What Claude needs to know. Why am I doing this? What happened before?>

## Key Files / References
<Files to read or change. Examples to copy the style of. Reference material.>

## The Plan (do this FIRST)
First, ask me any clarifying questions needed to get this right —
focus on the key decisions for reaching the goal, not tiny details.
Then write me a step-by-step plan and STOP.
Do not change, create, or delete anything until I reply "approved."

## How to Run It / See Results
<How do I check the work? The command to run, the branch, where output shows up.
 If you're not sure, ASK me before finishing.>

## Confirmation / Validation
When done, tell me: what changed, how to confirm it worked, what to double-check.
Then wait for me to confirm before calling it finished.

## Work Log / Tracking
Keep a running log and update it each round. Keep it short and plain:
 - Clarifying questions asked (and my answers):
 - What we did:
 - What worked / what we learned:
 - What did NOT work (so we don't repeat it):
 - Decisions made (and why):
 - What's left to do next:

<!-- Feel free to add more sections if this task needs them. -->
```

---

## Three things to remember

1. **Plan first, approve, then work** — for anything complex.
2. **Log before you clear** — so a fresh session knows where you left off.
3. **When in doubt, ask** — type *"explain that simply"* or *"what are my options?"* anytime.
