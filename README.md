# RLEnv_audit

**pytest for RL environments.** Point it at a [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment from the Prime Intellect Environments Hub and it runs a battery of
automated quality checks, then prints a scorecard and writes a machine-readable
`report.json`.

RL post-training environments are treated like training data — but unlike data,
nobody tests them before burning GPU hours. A broken reward function doesn't
crash; it silently teaches the policy garbage: rewards that vary run-to-run,
reward paid out to `sys.exit(0)`, all-zero reward distributions with no gradient,
a parser that drops the reward over a stray space, a dataset that's secretly the
eval set. RLEnv_audit catches these first.

```console
$ rlenv-audit run gsm8k

                              RLEnv_audit · gsm8k
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check         ┃ status ┃ summary                                             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ determinism   │ PASS   │ all 12 completions stable across 5 repeats          │
│ exploits      │ PASS   │ no hard cheats exploited; env rewards bare-answer …  │
│ parser        │ PASS   │ parser handled 21/24 perturbations (88%)            │
│ contamination │ PASS   │ no overlap found; checked 500 questions vs gsm8k, … │
│ latency       │ PASS   │ cold 203ms, warm mean 5ms (~209/s), 1.7x batched    │
│ distribution  │ SKIP   │ vLLM not installed (install rlenv-audit[gpu])       │
└───────────────┴────────┴─────────────────────────────────────────────────────┘
overall grade: PASS   (PASS 5  SKIP 1)

report written to: report.json
```

## Install

RLEnv_audit pins **`verifiers==0.1.14`** (deliberately — newer versions drag in
torch/vLLM; five of the six checks are CPU-only). Python 3.10+.

```bash
# with uv (recommended)
uv venv --python 3.10
uv pip install -e .            # or: uv pip install rlenv-audit

# install an environment to audit (any Hub env exposing load_environment)
vf-install gsm8k -r            # gsm8k example from the verifiers repo
# or a Hub env:  vf-install primeintellect/<env>
```

The **exploits** check needs Docker running (it executes hostile code in an
isolated container). The **distribution** check needs a GPU + vLLM
(`pip install rlenv-audit[gpu]`); without them it SKIPs.

## Usage

```bash
rlenv-audit run <env-id>                 # full battery
rlenv-audit run <env-id> --only determinism,parser
rlenv-audit run <env-id> --skip distribution
rlenv-audit run <env-id> --json out.json # also write JSON here
rlenv-audit run <env-id> --model Qwen/Qwen2.5-1.5B-Instruct   # distribution model
rlenv-audit list-checks                  # what each check needs
```

`rlenv-audit run` exits non-zero if the env earns an overall **FAIL** grade —
drop it into CI for environments.

### As a library

The CLI is a thin wrapper; everything is callable programmatically:

```python
import rlenv_audit

scorecard = rlenv_audit.audit("gsm8k")          # or pass a loaded Environment
print(scorecard.grade)                          # "PASS" / "WARN" / "FAIL" / "INCONCLUSIVE"
scorecard.write_json("report.json")

# run a subset
scorecard = rlenv_audit.audit("gsm8k", only=["determinism", "parser"])
```

## The six checks

| Check | Needs | What it catches |
| --- | --- | --- |
| **determinism** | — | Scores a fixed set of completions 5× each; **FAIL** if any reward varies. Non-deterministic rewards inject noise into the gradient. |
| **exploits** | Docker | Submits known cheats (`sys.exit(0)`, monkeypatch `assert`, read the expected-output file, empty solution) **instead of** real answers; **FAIL** if a no-solution cheat earns reward. Runs in a locked-down container because it executes hostile code. |
| **distribution** | GPU | Rollouts with a small reference model; **WARN** on all-zero / all-one / empty-rewarded distributions — shapes that produce no learning signal. |
| **parser** | — | Feeds the answer parser correct answers in perturbed formats (whitespace, `\boxed{}`, trailing punctuation, casing); score = fraction still extracted; **WARN** if brittle. |
| **contamination** | — | N-gram overlap of the dataset against popular eval sets (GSM8K, MATH-500, AIME, HumanEval, LiveCodeBench); **FAIL** listing matches. |
| **latency** | — | Times verification cold vs warm and probes batched scoring. Informational. |

`SKIP` ≠ `FAIL`: a check SKIPs when it can't run here (no GPU, Docker down, no
dataset). The overall grade is the worst meaningful result; all-SKIP is
`INCONCLUSIVE`.

## How it works

`rlenv-audit` loads the environment through `verifiers.load_environment`, then
the **adapter** (`adapters/verifiers.py`) normalizes it into an `EnvHandle` — the
only code that touches `verifiers`. The handle exposes the rubric's reward
functions, the parser, and the dataset, plus a synchronous `score()` over the
library's async scoring path (RubricGroup-aware; tears down rubric-owned process
pools). Each check is an independent function over that handle. See
[`DESIGN.md`](DESIGN.md) for the architecture and the verifiers-0.1.14 API notes.

## Scope (v0)

Six checks, one format (`verifiers`), one command — on purpose. No plugin system,
no config framework, no multi-format support. The `adapters/` and `checks/` seams
make extension *possible* later without building it now.

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/          # report unit tests + gsm8k integration tests
```

## License

MIT
