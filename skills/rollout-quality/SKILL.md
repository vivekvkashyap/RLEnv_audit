---
name: env-audit-rollout-quality
description: Rollout-quality check — inspect actual model rollouts and judge whether the environment is set up well in practice. Is the system prompt right, is anything missing from it, are the model's outputs sensible given the prompts, are there obvious failure modes the env setup is causing. Requires a model endpoint; reads the shared cached rollout set.
---

# Check 5 — rollout quality

**Question:** in practice, with a real model in the loop, is this environment set
up well — or is the setup itself causing failures?

**Requires a model endpoint.** If none was configured, output:

```json
{"name": "rollout_quality", "status": "N/A", "score": null, "justification": "no model endpoint provided"}
```

## Steps

1. Read the **shared** rollout cache (`/tmp/envaudit_rollouts.json`) — the same
   one the latency check uses. Each sample has the `prompt`, the gold `answer`,
   and `k` `rollouts` with their `text` and `reward`.
   If the cache was generated with `--dummy` (`"dummy": true`), the texts are
   placeholders — there is no real model behavior to judge. Output **N/A** with
   justification "dummy rollouts — no real model outputs".

2. Read a spread of actual rollouts and judge the **environment setup**, not the
   model's intelligence:
   - **System prompt** — does it clearly tell the model the task and the required
     output format? Is anything missing (no format instruction, ambiguous task,
     contradictory guidance)?
   - **Output sensibility** — given the prompts, are the model's outputs on-task
     and well-formed? If even reasonable outputs score 0, the env is likely
     mis-parsing or mis-rewarding them (cross-check with the rewards).
   - **Obvious failure modes caused by the env** — answers never match the parser,
     prompts truncated, tools/judge erroring, the task underspecified, the format
     impossible to satisfy, reward saturated at 0 or 1 for everything.

3. Tie observations to evidence: quote a rollout and its reward when flagging a
   problem (e.g. "rollout gives the correct value but in plain text and scores 0
   → parser only accepts \\boxed{}").

## Output

Score 0–10 for how well the env works in practice with a real model:

```json
{"name": "rollout_quality", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: system prompt verdict + the main practical failure mode>"}
```
