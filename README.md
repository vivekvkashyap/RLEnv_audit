# rlenv_audit

[![PyPI](https://img.shields.io/pypi/v/rlenv-audit?color=blue)](https://pypi.org/project/rlenv-audit/)
[![Python versions](https://img.shields.io/pypi/pyversions/rlenv-audit)](https://pypi.org/project/rlenv-audit/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

rlenv_audit audits [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
RL environments from the Prime Intellect Hub before you train on them. A broken
reward function doesn't crash, it silently teaches the policy garbage. Point an
agent (Claude Code / Codex) at an environment: it runs six checks and returns a
scorecard out of 10 with written feedback on what to improve.

## Quickstart

```bash
# Install the skills (pick one)
uvx rlenv-audit install-skills
pip install rlenv-audit && rlenv-audit install-skills
```

Then ask your agent, giving the **full environment id** (`account/name`; bare
names like `gsm8k` are ambiguous on the Hub), your **problem statement**, and
optionally a **model endpoint** and the **HuggingFace datasets** to check
contamination against:

**prompt**

```text
Audit primeintellect/gsm8k. I'm trying to train a grade-school math solver.
Use my vLLM endpoint at http://localhost:8000/v1, model Qwen2.5-7B.
Check contamination against openai/gsm8k.
```

*(in Claude Code or Codex)*

## Output

The scorecard, one row per check, each scored **out of 10**, plus one final
score and written feedback:

```
                       rlenv_audit · primeintellect/gsm8k
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check             ┃ status ┃ score ┃ justification                           ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ integrity         │ PASS   │   9.5 │ loads, reward callable, well-formed     │
│ problem_alignment │ PASS   │   9.0 │ dataset/reward match the stated goal    │
│ reward_design     │ PASS   │   8.8 │ discriminates; matches judgment 18/20   │
│ latency           │ N/A    │     — │ no endpoint                             │
│ rollout_quality   │ N/A    │     — │ no endpoint                             │
│ contamination     │ WARN   │   6.0 │ 3 near-matches with openai/gsm8k test   │
└───────────────────┴────────┴───────┴─────────────────────────────────────────┘
overall: WARN   rating: 8.7/10

feedback
The environment is solidly built: it loads cleanly, the reward is a real
verifier (boxed-answer extraction + math equivalence, not a stub), and it
discriminates well: correct completions scored 1.0 and every wrong or
malformed probe scored 0.0, matching my own judgment on 18 of 20 cases.

The main thing to improve is contamination: 3 of the sampled training
instances near-match the openai/gsm8k test split you asked me to check, so
benchmark gains may partly be memorization; either dedupe against that test
split or report on a different set. Second, the parser only accepts \boxed{}
answers; consider
accepting plain final-line answers too, or the policy gets zero reward for
correct-but-unformatted output early in training.
```

- **Final score**: a weighted average out of 10 over the checks that ran (N/A
  carries no weight). Latency and contamination weigh **0.5** each, the other
  four checks **1.0**.
- **Feedback**: 1 to 3 paragraphs, what the env does right first, then what to
  improve, in priority order.
- A `FAIL` on any check fails the audit.
- The full report is also saved to
  `rlenv_audit_reports/<account>__<name>/report.md` (human-readable) and
  `report.json` (machine-readable) in your working directory, so you can commit
  it, share it, or diff it against a re-audit after fixes.

## The six checks

| # | Check | Needs | What it does |
|---|-------|-------|--------------|
| 1 | **integrity** | - | Does it even run and is it shaped right: dataset loads & is well-formed, reward present & callable, follows verifiers conventions, no missing fields / broken imports. |
| 2 | **problem-statement alignment** | - | Given your problem statement (a required input), judge whether the dataset + reward + prompt actually test that problem. |
| 3 | **reward design** | - | Stress-tests the reward without the policy: the agent writes ~20 synthetic completions (correct / wrong / edge / format perturbations), scores them through the real reward, and checks (a) the reward varies & discriminates sensibly and (b) each reward matches the agent's own judgment of quality. |
| 4 | **latency** | model endpoint | How long rollouts take end to end. Reads the shared cached rollouts. |
| 5 | **rollout quality** | model endpoint | Reads actual rollouts and judges whether the env is set up well in practice: system prompt right, outputs sensible, obvious env-caused failure modes. |
| 6 | **contamination** | HF dataset ids | Compares the env's dataset against the HuggingFace datasets *you* name (e.g. `openai/gsm8k`) and flags matching / near-matching instances. **N/A** (carries no weight) if you don't provide any. |

**Shared rollouts (checks 4 & 5).** Both need a model, so rlenv_audit runs
rollouts **once** (8 rollouts over ~20 samples, scored + timed, cached) and both
checks read that single cache. No endpoint → 4 & 5 are **N/A**.

## Layout

```
skills/                 the six checks + the env-audit orchestrator (SKILL.md each)
.claude-plugin/         plugin + marketplace manifests (repo doubles as a Claude Code plugin)
rlenv_audit/
  adapters/verifiers.py EnvHandle, the only code that touches verifiers
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
