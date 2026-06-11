# env_audit

[![PyPI](https://img.shields.io/pypi/v/rlenv-audit?color=blue)](https://pypi.org/project/rlenv-audit/)
[![Python versions](https://img.shields.io/pypi/pyversions/rlenv-audit)](https://pypi.org/project/rlenv-audit/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**env_audit** audits [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
RL environments from the Prime Intellect Hub *before* you spend GPU hours
training on them. RL environments are treated like training data, but nobody
tests them first — a broken reward function doesn't crash, it silently teaches
the policy garbage. env_audit catches that: point an agent (Claude Code / Codex)
at an environment and it runs **six judgment-based checks** — each a skill file
the agent executes, backed by a small deterministic tool layer — and returns a
scorecard with a score (0–100), a status, and a written justification per check.

## Quickstart

```bash
# Install the skills (pick one)
uvx rlenv-audit install-skills
pip install rlenv-audit && rlenv-audit install-skills
```

Then ask your agent (Claude Code / Codex), giving the **environment name**, your
**problem statement**, and — if you have one — a **model endpoint**:

> "Audit `primeintellect/gsm8k`. I'm trying to train a grade-school math solver.
> Use my vLLM endpoint at `http://localhost:8000/v1`, model `Qwen2.5-7B`."

That's the whole interface. Everything else is self-bootstrapping: on the first
audit the skill installs the `rlenv-audit` tools (if missing) and `vf-install`s
the environment itself. The problem statement is required (the agent asks if
you don't give one); the endpoint is optional — without it the two rollout
checks are reported N/A.

## Output

The scorecard — one row per check with its status, score, and a one-line
justification — plus the overall grade, a 0–100 rating with a letter, and a
short prose summary of the biggest issue and what to fix first:

```
                               env_audit · gsm8k
┃ check             ┃ status ┃ score ┃ justification                           ┃
│ integrity         │ PASS   │    95 │ loads, reward callable, well-formed     │
│ problem_alignment │ PASS   │    90 │ dataset/reward match the stated goal    │
│ reward_design     │ PASS   │    88 │ discriminates; matches judgment 18/20   │
│ latency           │ N/A    │     — │ no endpoint                             │
│ rollout_quality   │ N/A    │     — │ no endpoint                             │
│ contamination     │ WARN   │    60 │ 3 near-matches with GSM8K test          │
overall: WARN   rating: B (83/100)
```

A `FAIL` on any check fails the audit. The rating averages only the checks that
ran (N/A excluded).

## The six checks

| # | Check | Needs | What it does |
|---|-------|-------|--------------|
| 1 | **integrity** | — | Does it even run and is it shaped right: dataset loads & is well-formed, reward present & callable, follows verifiers conventions, no missing fields / broken imports. |
| 2 | **problem-statement alignment** | — | Given your problem statement (a required input), judge whether the dataset + reward + prompt actually test that problem. |
| 3 | **reward design** | — | Stress-tests the reward without the policy: the agent writes ~20 synthetic completions (correct / wrong / edge / format perturbations), scores them through the real reward, and checks (a) the reward varies & discriminates sensibly and (b) each reward matches the agent's own judgment of quality. |
| 4 | **latency** | model endpoint | How long rollouts take end to end. Reads the shared cached rollouts. |
| 5 | **rollout quality** | model endpoint | Reads actual rollouts and judges whether the env is set up well in practice — system prompt right, outputs sensible, obvious env-caused failure modes. |
| 6 | **contamination** | — | Infers the domain, picks the public benchmarks for it, and checks whether dataset instances match/near-match benchmark instances. |

**Shared rollouts (checks 4 & 5).** Both need a model, so env_audit runs
rollouts **once** (8 rollouts over ~20 samples, scored + timed, cached) and both
checks read that single cache. No endpoint → 4 & 5 are **N/A**.

## Layout

```
skills/                 the six checks + the env-audit orchestrator (SKILL.md each)
.claude-plugin/         plugin + marketplace manifests (repo doubles as a Claude Code plugin)
rlenv_audit/
  adapters/verifiers.py EnvHandle — the only code that touches verifiers
  tools.py              inspect / score / rollouts / scorecard
  sandbox.py            Docker isolation (for executing risky completions)
  cli.py                the rlenv-audit / env-audit CLI (+ install-skills)
REWARD_DESIGN.md        the design guide the judgment checks cite
```

## Development

```bash
pip install -e ".[dev]" && pytest tests/
```

## License

MIT
