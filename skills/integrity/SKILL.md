---
name: env-audit-integrity
description: Integrity check for an RL environment — verify it is written properly and actually runs. Confirms the dataset loads and is well-formed, the reward function is present and callable, the code follows verifiers/prime-intellect conventions, and there are no missing fields or broken imports. The "does it even run and is it shaped right" check.
---

# Check 1 — integrity

**Question:** is this environment written properly and does it run?

This check needs no model. Use the `inspect` tool plus reading the source.

## Steps

1. **Load.** Read `/tmp/envaudit_inspect.json` if the orchestrator already wrote
   it; otherwise run `rlenv-audit inspect <env> -n 20 --out /tmp/envaudit_inspect.json`.
   If `loaded` is false → **FAIL**, score ≤ 2, justification = the `error`.
   Stop here.

2. **Dataset well-formed.** From the JSON, check:
   - `dataset_size.train` or `.eval` is non-zero;
   - sampled rows have a `prompt` that is a chat-message list (`{role, content}`)
     or a non-empty string;
   - rows have an `answer` (or the reward clearly doesn't need one — e.g. a judge
     env; note which);
   - no obviously broken/empty rows.

3. **Reward present and callable.** `reward_funcs` is non-empty; read each
   `source`. Confirm it's a real function (not a stub returning a constant), and
   note its weight. Zero or all-zero weights is a serious problem.

4. **Conventions & imports.** Read the env source at `module_file`. Confirm it
   follows verifiers conventions: a `load_environment(...)` that returns an
   `Environment` with a `rubric`, `parser`, and dataset; imports resolve; no
   dead/missing fields; a parser is set if the reward relies on parsing.

5. **System prompt.** Note whether a system prompt is present (its absence is a
   real defect for format-sensitive envs, but judge in context).

## Output

Score 0–10: start at 10 and deduct for each defect by severity (won't load →
fail outright; missing reward / empty dataset → large deduction; no system
prompt / minor convention slips → small). Return:

```json
{"name": "integrity", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: what's right, and the most important defect if any>"}
```
