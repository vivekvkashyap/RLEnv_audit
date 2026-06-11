# env_audit

**A skill-based auditing system for RL environments.** Point an agent (Claude
Code / Codex) at a [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment from the Prime Intellect Hub and it runs **six checks** and produces
a scorecard — *before* you spend GPU hours training on a broken reward.

RL environments are treated like training data, but nobody tests them first. A
broken reward function doesn't crash — it silently teaches the policy garbage.
env_audit catches that.

## Why skills, not scripts

The six checks are **judgment-heavy, non-deterministic evaluations** — "does this
reward agree with a competent grader?", "is the system prompt missing something?",
"does this dataset overlap a benchmark?". Those are done well by an *agent*, not a
hard-coded script. So each check is a **skill file** (`skills/<check>/SKILL.md`)
that the agent reads and executes with its own reasoning, leaning on a small layer
of deterministic **tools** (`rlenv-audit ...`) for the exact parts: loading the
env, calling the reward function, running rollouts, rendering the scorecard.

Each check returns a **score (0–100), a status, and a written justification**.

## The six checks

| # | Check | Needs | What it does |
|---|-------|-------|--------------|
| 1 | **integrity** | — | Does it even run and is it shaped right: dataset loads & is well-formed, reward present & callable, follows verifiers conventions, no missing fields / broken imports. |
| 2 | **problem-statement alignment** | *(a problem statement)* | Given what the user says the env is *for*, judge whether the dataset + reward + prompt actually test that. **N/A** if no problem statement is provided. |
| 3 | **reward design** | — | Stress-tests the reward without the policy: the agent writes ~20 synthetic completions (correct / wrong / edge / format perturbations), scores them through the real reward, and checks (a) the reward varies & discriminates sensibly and (b) each reward matches the agent's own judgment of quality. |
| 4 | **latency** | model endpoint | How long rollouts take end to end. Reads the shared cached rollouts. |
| 5 | **rollout quality** | model endpoint | Reads actual rollouts and judges whether the env is set up well in practice — system prompt right, outputs sensible, obvious env-caused failure modes. |
| 6 | **contamination** | — | Infers the domain, picks the public benchmarks for it, and checks whether dataset instances match/near-match benchmark instances. |

**Shared rollouts (checks 4 & 5).** Both need a model, so env_audit asks once
which endpoint/model to use (or "dummy"), runs rollouts **once** (8 rollouts over
~20 samples, scored + timed, cached), and both checks read that single cache.
Checks 1, 2, 3, 6 need no endpoint. No endpoint → 4 & 5 are **N/A**.

## Usage

env_audit is run **by an agent**. With the skills available to Claude Code /
Codex, point it at an environment:

> "Audit the `gsm8k` environment." &nbsp; / &nbsp; "Audit `primeintellect/aime2024`
> — I'm trying to train a competition-math solver — using my vLLM at
> `http://localhost:8000/v1`."

The agent loads the `env-audit` skill, runs the six checks, and prints the
scorecard:

```
                               env_audit · gsm8k
┃ check             ┃ status ┃ score ┃ justification                           ┃
│ integrity         │ PASS   │    95 │ loads, reward callable, well-formed     │
│ problem_alignment │ N/A    │     — │ no problem statement provided           │
│ reward_design     │ PASS   │    88 │ discriminates; matches judgment 18/20   │
│ latency           │ N/A    │     — │ no endpoint                             │
│ rollout_quality   │ N/A    │     — │ no endpoint                             │
│ contamination     │ WARN   │    60 │ 3 near-matches with GSM8K test          │
overall: WARN   rating: B (81/100)
```

### Install the skills

```bash
pip install -e .              # installs the rlenv-audit / env-audit tools
vf-install primeintellect/gsm8k   # install an environment to audit
cp -r skills/* ~/.claude/skills/  # make the skills available to Claude Code
```

> Most Hub envs require **Python 3.11+**; `verifiers==0.1.14` (pinned) also runs
> on 3.10 for old-CUDA boxes, where you can install the older example envs.

### The tools (what the skills call)

```bash
rlenv-audit inspect <env> -n 20            # load + introspect -> JSON (reward source, samples, prompt)
rlenv-audit score <env> completions.json   # score agent-written completions through the reward fn
rlenv-audit rollouts <env> --endpoint <url> --model <m> -n 20 -k 8   # run+cache shared rollouts
rlenv-audit rollouts <env> --dummy         # fake rollouts, no endpoint (dry run)
rlenv-audit scorecard results.json         # render the final scorecard
```

These are deterministic and JSON-in/JSON-out — usable directly, but normally
driven by the skills.

## What good looks like

[`REWARD_DESIGN.md`](REWARD_DESIGN.md) is the reference the reward-design and
rollout-quality checks judge against — determinism, discrimination, baseline
floor, partial credit, bounds, anti-hacking, parser contract, contamination.

## Layout

```
skills/                 the six checks + the env-audit orchestrator (SKILL.md each)
rlenv_audit/
  adapters/verifiers.py EnvHandle — the only code that touches verifiers
  tools.py              inspect / score / rollouts / scorecard
  sandbox.py            Docker isolation (for executing risky completions)
  cli.py                the rlenv-audit / env-audit CLI
REWARD_DESIGN.md        the design guide the judgment checks cite
```

## Development

```bash
pip install -e ".[dev]" && pytest tests/
```

## License

MIT
