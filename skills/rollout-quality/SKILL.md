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
   one the latency check uses, generated through verifiers' own `vf-eval` engine
   so the rollouts follow the environment's *real* generation path (multi-turn /
   tool-use envs roll out correctly, the env's own sampling args apply). Each
   sample has the `prompt`, the gold `answer`, and `k` `rollouts`; each rollout
   carries `text`, `reward`, `error`, and — straight from vf-eval — `truncated`,
   `stop_reason`, and `output_tokens`. If the file is missing but an endpoint
   *was* configured, generate it once with
   `rlenv-audit rollouts <env> --endpoint <url> --model <name> -n 20 -k 8 --out /tmp/envaudit_rollouts.json`.
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
   - **Truncation / format reachability** — check `truncated` and `stop_reason`.
     Compute **one** number and use it everywhere: `truncated_frac` = (rollouts
     with `truncated == true`) / (total rollouts). If it is high and those
     rollouts stop on length / `max_turns_reached` before emitting the required
     answer format (high `output_tokens`, no `\\boxed{}`), the reward is sparse for
     a reason the env can fix (raise max tokens / context, or nudge for a shorter
     answer) rather than a true capability gap. Quote that single fraction in your
     analysis and in the justification — do not derive a second, differently-counted
     truncation number from logs or `stop_reason`.
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
