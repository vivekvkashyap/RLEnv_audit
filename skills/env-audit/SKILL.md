---
name: env-audit
description: Audit a Prime Intellect / verifiers RL environment before training on it. Runs six judgment-based checks (integrity, problem-statement alignment, reward design, latency, rollout quality, contamination) and produces a per-check scorecard with scores and written justifications. Use when the user wants to audit, review, vet, or quality-check an RL environment, or asks "is my env good / ready to train on?".
---

# env_audit — orchestrator

You are auditing one RL environment built on the `verifiers` framework (the Prime
Intellect Environments Hub standard). The audit is **six checks**, each its own
skill under `skills/`. Each check is judgment-heavy — you perform it yourself
using your own reasoning plus the deterministic `rlenv-audit` tools — and returns
a **score (0–10), a status, and a one-line justification**.

## 0. Gather inputs

Ask the user (or take from their request) and confirm:

1. **env id** (required) — the **fully qualified** Hub id `account/name`, e.g.
   `primeintellect/gsm8k`. Bare names like `gsm8k` are ambiguous (many accounts
   publish same-named envs on the Hub) — if the user gives one, ask for the full
   id before starting.
2. **problem statement** (required) — what the user is trying to train/test with
   this env. Check 2 judges the env against it. If the user didn't give one,
   **stop and ask** before running anything: "A problem statement is required —
   what are you trying to train or test with this environment?" Never guess one
   from the env, and never start the audit without it.
3. **model endpoint** (optional) — an OpenAI-compatible endpoint + model name, or
   "dummy", or none. Enables checks 4 & 5; if absent, both are **N/A**. If the
   user didn't mention one, ask once: "Do you have a model endpoint for the
   rollout checks, or should I skip them?"
4. **contamination datasets** (optional) — HuggingFace dataset ids or links
   (e.g. `openai/gsm8k`) to check the env's dataset against. Enables check 6;
   if none are given, contamination is **N/A** and carries no weight. Never
   substitute default benchmarks of your own.

## 1. Set up (self-bootstrapping)

Do whatever setup is missing — the user shouldn't have to prepare anything:

1. **Tools.** If `rlenv-audit` is not on PATH, install it:
   `pip install rlenv-audit` (use the active venv if there is one; fall back to
   `pip install git+https://github.com/vivekvkashyap/RLEnv_audit.git` if the
   PyPI package is unavailable). This also brings in `verifiers` and its
   `vf-install` command.
2. **Find rlenv-audit's Python environment — this is the one non-obvious step.**
   `rlenv-audit` is often installed as a `uvx` / `uv` tool in its **own isolated
   venv** (e.g. `~/.local/share/uv/tools/rlenv-audit/`), which is **not** the
   active shell Python. `verifiers` and `vf-install` live in that same venv, and
   the env package **must be installed into it** (verifiers loads an env by
   importing it, so they must share one interpreter). Resolve it once and reuse
   it for every install/inspect:
   - `head -1 "$(command -v rlenv-audit)"` shows the interpreter on the launcher's
     shebang; its directory is the venv's `bin/`. Use that `bin/vf-install` and
     `bin/python` (e.g. `VENV=~/.local/share/uv/tools/rlenv-audit;
     $VENV/bin/vf-install <account>/<env>`).
   - Sanity-check after install: `$VENV/bin/python -c "import <module>"` (module =
     the env name with `-`→`_`, last path segment), not the shell `python3`.
   Note: most Hub envs need Python ≥ 3.11 — the uv tool venv usually already is.
3. **The environment.** Try the inspect in section 2 below; if it fails with a
   load/import error, install the env into the venv found above and retry:
   `$VENV/bin/vf-install <account>/<env>` (e.g. `vf-install primeintellect/gsm8k`).

   **Classify an install failure — do not conflate two different things:**
   - **No such environment** — `vf-install` reports the id is *not found on the
     Hub* (404 / "no matching distribution" / "could not find environment"). This
     is a **wrong input**: **stop and tell the user** "There is no environment
     named `<account>/<name>` on the Prime Intellect Hub" (quote the error), ask
     them to check the id, and produce **no scorecard**.
   - **Exists but won't install** — `vf-install` *finds* the package but the
     install/build fails (dependency conflict, compile error, Python-version
     mismatch), or it installs yet the module still isn't importable. Do **not**
     claim the env doesn't exist. Report it as a setup/integrity failure: quote
     the build error, say the environment exists on the Hub but could not be
     installed in this environment, and stop. (If it imports but crashes on
     *load*, that is the integrity finding in step 2, not here.)

## 2. Load the environment once

Run `rlenv-audit inspect <env> -n 20 --out /tmp/envaudit_inspect.json` and read
it. If the env **installed but** `loaded` is false (it exists yet crashes on
load), that is a genuine audit finding: the **integrity** check fails
immediately — report that as the scorecard and stop (the other checks can't
run).

## 3. Run the no-endpoint checks (1, 2, 3, 6)

These need no model — your own judgement plus the tools. Run each by following
its skill (installed alongside this one; in the repo they live under `skills/`),
in order, and collect `{name, status, score, justification}`:

- `env-audit-integrity`
- `env-audit-problem-alignment`
- `env-audit-reward-design`
- `env-audit-contamination` (N/A if the user provided no datasets to check)

## 4. Shared rollouts, then the endpoint checks (4, 5)

If the user gave an endpoint (or chose "dummy"):

1. Generate the rollouts **once**:
   `rlenv-audit rollouts <env> --endpoint <url> --model <name> -n 20 -k 8 --out /tmp/envaudit_rollouts.json`
   (or `--dummy` for a no-endpoint dry run). This drives verifiers' own `vf-eval`
   engine, so rollouts follow the environment's real generation path; eight
   rollouts over ~20 samples, scored and timed (with per-rollout truncation and
   token usage), cached to that file. `vf-eval` is a client — the user's served
   model must be up; it does not start one.
2. Run the `env-audit-latency` and `env-audit-rollout-quality` skills, both
   reading that **single cache** — do not roll out again.

If there is no endpoint, mark **latency** and **rollout_quality** as `N/A`.

## 5. Assemble the scorecard + feedback

Each check scores **0–10** (one decimal allowed). Write all six results, plus a
written **feedback** section, to `/tmp/envaudit_results.json`:

```json
{"env_id": "<env>", "checks": [
  {"name": "integrity", "status": "PASS|WARN|FAIL", "score": 0-10, "justification": "..."},
  {"name": "problem_alignment", "status": "PASS|WARN|FAIL", "score": 0-10, "justification": "..."},
  {"name": "reward_design", "status": "...", "score": ..., "justification": "..."},
  {"name": "latency", "status": "PASS|WARN|FAIL|N/A", "score": ...|null, "justification": "..."},
  {"name": "rollout_quality", "status": "...", "score": ..., "justification": "..."},
  {"name": "contamination", "status": "PASS|WARN|FAIL|N/A", "score": ...|null, "justification": "..."}
], "feedback": "<1-3 paragraphs>"}
```

**feedback** is 1–3 short paragraphs for the env author: first what the
environment does *right* (be specific — cite what you observed), then what can
be improved and how, in priority order. This is the part a human acts on; make
every sentence earn its place.

Then `rlenv-audit scorecard /tmp/envaudit_results.json` to render it. The final
rating is a **weighted average out of 10** over the checks that ran (N/A
excluded): latency and contamination weigh 0.5, the other four checks 1.0 — the
tool computes this for you.

## 6. Save the report

Persist the audit so it outlives the session (skip only if the user says not to
save). Create `rlenv_audit_reports/<account>__<name>/` in the working directory
and write two files:

1. **`report.json`** — the machine-readable result:
   `rlenv-audit scorecard /tmp/envaudit_results.json --json > rlenv_audit_reports/<account>__<name>/report.json`
   (the computed scorecard: checks, grade, rating, feedback).
2. **`report.md`** — the human-readable report, which you author:
   - title (`# rlenv_audit — <account>/<name>`) and the date;
   - **Inputs**: env id, the user's problem statement, endpoint + model (or
     "none"), contamination datasets (or "none");
   - **Scorecard**: the six checks as a markdown table (check / status / score
     / justification) plus the final `rating: N.N/10`;
   - **Feedback**: the same feedback paragraphs from the results JSON.

End by telling the user where the report was saved.

## Rules

- A check is **N/A** only for the documented reasons (no endpoint → latency,
  rollout_quality; no contamination datasets → contamination). Never N/A a
  check just because it's hard.
- Every score needs a justification grounded in what you actually observed
  (tool output, completions you wrote, rollouts you read) — never a vibe.
- Statuses: **PASS** ≈ 7.5–10, **WARN** ≈ 4–7.4, **FAIL** ≈ 0–3.9 (use judgement
  at the edges). Be honest; the point is to catch faults before a training run.
