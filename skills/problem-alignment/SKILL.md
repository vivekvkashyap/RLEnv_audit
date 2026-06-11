---
name: env-audit-problem-alignment
description: Problem-statement alignment check — given the user's problem statement (a required audit input), judge whether the environment actually tests what they claim they're trying to solve.
---

# Check 2 — problem-statement alignment

**Question:** does this environment actually test the thing the user says they
want to solve?

The problem statement is a **required audit input** — if you don't have it, ask
the user for it before scoring; never guess one from the env itself. This check
needs no model — only your judgement over the env.

## Steps

1. Read the inspect JSON (`/tmp/envaudit_inspect.json`): system prompt, sample
   tasks (prompts + answers), and the reward function source.

2. Hold three things against the user's problem statement and judge alignment:
   - **Dataset** — do the tasks actually exercise the claimed problem/domain/
     difficulty? (e.g. "I want to train competition math" but the dataset is
     grade-school arithmetic → misaligned.)
   - **Reward** — does it measure success *at the claimed problem*, or something
     adjacent/looser (e.g. rewards any formatted answer, not a correct one)?
   - **System prompt / framing** — does the task posed to the model match the
     stated goal?

3. Call out concrete mismatches (over-broad, too easy/hard, measures the wrong
   thing, off-domain examples) and genuine alignment.

## Output

Score 0–10 for how well the env tests the stated problem:

```json
{"name": "problem_alignment", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: aligned on X, misaligned on Y>"}
```
