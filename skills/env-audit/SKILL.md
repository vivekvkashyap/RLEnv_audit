---
name: env-audit
description: Audit a Prime Intellect / verifiers RL environment before training on it. Runs six judgment-based checks (integrity, problem-statement alignment, reward design, latency, rollout quality, contamination) and produces a per-check scorecard with scores and written justifications. Use when the user wants to audit, review, vet, or quality-check an RL environment, or asks "is my env good / ready to train on?".
---

# env_audit — orchestrator

You are auditing one RL environment built on the `verifiers` framework (the Prime
Intellect Environments Hub standard). The audit is **six checks**, each its own
skill under `skills/`. Each check is judgment-heavy — you perform it yourself
using your own reasoning plus the deterministic `rlenv-audit` tools — and returns
a **score (0–100), a status, and a one-line justification**.

## 0. Gather inputs

Ask the user (or take from their request) and confirm:

1. **env id** (required) — e.g. `gsm8k`, `primeintellect/aime2024`. It must be
   installed (`vf-install <env>` or `vf-install <env> -r`).
2. **problem statement** (optional) — what the user says this env is meant to
   test/train. Enables check 2; if absent, check 2 is **N/A**.
3. **model endpoint** (optional) — an OpenAI-compatible endpoint + model name, or
   "dummy", or none. Enables checks 4 & 5; if absent, both are **N/A**.

## 1. Load the environment once

Run `rlenv-audit inspect <env> -n 20 --out /tmp/envaudit_inspect.json` and read
it. If `loaded` is false, the **integrity** check fails immediately — report that
as the scorecard and stop (the other checks can't run).

## 2. Run the no-endpoint checks (1, 2, 3, 6)

These need no model — your own judgement plus the tools. Run each by following its
skill, in order, and collect `{name, status, score, justification}`:

- `skills/integrity/SKILL.md`
- `skills/problem-alignment/SKILL.md` (N/A if no problem statement)
- `skills/reward-design/SKILL.md`
- `skills/contamination/SKILL.md`

## 3. Shared rollouts, then the endpoint checks (4, 5)

If the user gave an endpoint (or chose "dummy"):

1. Generate the rollouts **once**:
   `rlenv-audit rollouts <env> --endpoint <url> --model <name> -n 20 -k 8 --out /tmp/envaudit_rollouts.json`
   (or `--dummy` for a no-endpoint dry run). Eight rollouts over ~20 samples,
   scored and timed, cached to that file.
2. Run `skills/latency/SKILL.md` and `skills/rollout-quality/SKILL.md`, both
   reading that **single cache** — do not roll out again.

If there is no endpoint, mark **latency** and **rollout_quality** as `N/A`.

## 4. Assemble the scorecard

Write all six results to a JSON file:

```json
{"env_id": "<env>", "checks": [
  {"name": "integrity", "status": "PASS|WARN|FAIL", "score": 0-100, "justification": "..."},
  {"name": "problem_alignment", "status": "PASS|WARN|FAIL|N/A", "score": 0-100|null, "justification": "..."},
  {"name": "reward_design", "status": "...", "score": ..., "justification": "..."},
  {"name": "latency", "status": "PASS|WARN|N/A", "score": ...|null, "justification": "..."},
  {"name": "rollout_quality", "status": "...", "score": ..., "justification": "..."},
  {"name": "contamination", "status": "PASS|WARN|FAIL", "score": ..., "justification": "..."}
]}
```

Then `rlenv-audit scorecard /tmp/envaudit_results.json` to render it. The overall
rating averages the checks that actually ran (N/A excluded). Finally, give the
user a short prose summary: the grade, the biggest issue, and what to fix first.

## Rules

- A check is **N/A** only for the documented reasons (no problem statement; no
  endpoint). Never N/A a check just because it's hard.
- Every score needs a justification grounded in what you actually observed
  (tool output, completions you wrote, rollouts you read) — never a vibe.
- Statuses: **PASS** ≈ 75–100, **WARN** ≈ 40–74, **FAIL** ≈ 0–39 (use judgement
  at the edges). Be honest; the point is to catch faults before a training run.
