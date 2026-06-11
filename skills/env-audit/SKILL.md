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

1. **env id** (required) — e.g. `gsm8k`, `primeintellect/aime2024`.
2. **problem statement** (required) — what the user is trying to train/test with
   this env. Check 2 judges the env against it. If the user didn't give one, ask
   for it before starting — don't guess it from the env.
3. **model endpoint** (optional) — an OpenAI-compatible endpoint + model name, or
   "dummy", or none. Enables checks 4 & 5; if absent, both are **N/A**. If the
   user didn't mention one, ask once: "Do you have a model endpoint for the
   rollout checks, or should I skip them?"

## 1. Set up (self-bootstrapping)

Do whatever setup is missing — the user shouldn't have to prepare anything:

1. **Tools.** If `rlenv-audit` is not on PATH, install it:
   `pip install rlenv-audit` (use the active venv if there is one; fall back to
   `pip install git+https://github.com/vivekvkashyap/RLEnv_audit.git` if the
   PyPI package is unavailable). This also brings in `verifiers` and its
   `vf-install` command.
2. **The environment.** Try the inspect in step 2; if it fails with a
   load/import error, install the env and retry:
   `vf-install <env>` for Hub envs (e.g. `primeintellect/gsm8k`), or
   `vf-install <env> -r` for the verifiers example envs. Note: most Hub envs
   need Python ≥ 3.11. The env must land in the **same Python environment** as
   `rlenv-audit` (verifiers loads envs by importing them).

## 2. Load the environment once

Run `rlenv-audit inspect <env> -n 20 --out /tmp/envaudit_inspect.json` and read
it. If `loaded` is false (after the bootstrap above), the **integrity** check
fails immediately — report that as the scorecard and stop (the other checks
can't run).

## 3. Run the no-endpoint checks (1, 2, 3, 6)

These need no model — your own judgement plus the tools. Run each by following
its skill (installed alongside this one; in the repo they live under `skills/`),
in order, and collect `{name, status, score, justification}`:

- `env-audit-integrity`
- `env-audit-problem-alignment`
- `env-audit-reward-design`
- `env-audit-contamination`

## 4. Shared rollouts, then the endpoint checks (4, 5)

If the user gave an endpoint (or chose "dummy"):

1. Generate the rollouts **once**:
   `rlenv-audit rollouts <env> --endpoint <url> --model <name> -n 20 -k 8 --out /tmp/envaudit_rollouts.json`
   (or `--dummy` for a no-endpoint dry run). Eight rollouts over ~20 samples,
   scored and timed, cached to that file.
2. Run the `env-audit-latency` and `env-audit-rollout-quality` skills, both
   reading that **single cache** — do not roll out again.

If there is no endpoint, mark **latency** and **rollout_quality** as `N/A`.

## 5. Assemble the scorecard

Write all six results to `/tmp/envaudit_results.json`:

```json
{"env_id": "<env>", "checks": [
  {"name": "integrity", "status": "PASS|WARN|FAIL", "score": 0-100, "justification": "..."},
  {"name": "problem_alignment", "status": "PASS|WARN|FAIL", "score": 0-100, "justification": "..."},
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

- A check is **N/A** only for the documented reason (no endpoint). Never N/A a
  check just because it's hard.
- Every score needs a justification grounded in what you actually observed
  (tool output, completions you wrote, rollouts you read) — never a vibe.
- Statuses: **PASS** ≈ 75–100, **WARN** ≈ 40–74, **FAIL** ≈ 0–39 (use judgement
  at the edges). Be honest; the point is to catch faults before a training run.
