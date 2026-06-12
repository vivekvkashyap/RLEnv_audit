---
name: env-audit-problem-alignment
description: Problem-statement alignment check — given the user's problem statement (a required audit input), judge whether the environment actually tests what they claim they're trying to solve.
---

# Check 2 — problem-statement alignment

**Question:** does this environment actually test the thing the user says they
want to solve?

The problem statement is a **required audit input**. If you don't have it,
**stop and tell the user**: "A problem statement must be specified — what are
you trying to train or test with this environment?" Do not score this check, do
not substitute a guess from the env itself, and do not silently skip it — wait
for the user's answer. This check needs no model — only your judgement over the
env.

## Steps

1. Read the inspect JSON (`/tmp/envaudit_inspect.json`): system prompt, sample
   tasks (prompts + answers), and the reward function source.

2. Hold these against the user's problem statement and judge alignment:
   - **Dataset** — do the tasks actually exercise the claimed problem/domain/
     difficulty? (e.g. "I want to train competition math" but the dataset is
     grade-school arithmetic → misaligned.)
   - **Reward** — does it measure success *at the claimed problem*, or something
     adjacent/looser (e.g. rewards any formatted answer, not a correct one)? And
     at the right granularity — a single end-of-episode scalar for a claim about
     *process* (tool use, multi-step reasoning) rewards lucky final answers as
     much as real skill.
   - **System prompt / framing** — does the task posed to the model match the
     stated goal?
   - **Interaction shape** — does the action space match the claimed capability?
     "Train agentic coding / tool use / search" served by a single-turn
     text-completion env tests recall, not the claimed skill; conversely a heavy
     tool harness for a claim a plain answer would test adds noise.
   - **Scale & diversity for the goal** — if the user is *training*, is the
     dataset big and varied enough to support it (a few dozen rows, or hundreds
     of instances stamped from one template, will be memorized, not learned
     from)? An eval-only env offered for a training goal is a mismatch worth
     naming. Check `dataset_size` from the inspect JSON.

3. **Difficulty calibration.** Judge whether the difficulty band fits the stated
   goal: tasks a base model already aces give no training signal at the claimed
   level (reward will saturate at 1), tasks far beyond it give none either
   (saturates at 0). You're judging the *intent* here from the samples — the
   rollout-quality check measures the saturation empirically.

4. Call out concrete mismatches (over-broad, too easy/hard, measures the wrong
   thing, off-domain examples, wrong interaction shape, too small to train on)
   and genuine alignment.

## Output

Score 0–10 for how well the env tests the stated problem:

```json
{"name": "problem_alignment", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: aligned on X, misaligned on Y>"}
```
