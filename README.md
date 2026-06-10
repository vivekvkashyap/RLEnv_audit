# RLEnv_audit

**pytest for RL environments.** Point it at a
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment from the
Prime Intellect Environments Hub and it runs a battery of automated quality
checks, prints a scorecard with a 0–100 rating, and tells you **what to fix** —
before you spend GPU hours training on a broken reward.

RL post-training environments are treated like training data, but unlike data
nobody tests them first. A broken reward function doesn't crash — it silently
teaches the policy garbage:

- rewards that vary run-to-run (noisy gradient),
- reward paid out to `sys.exit(0)` or to reading the answer off disk (reward hacking),
- a reward that scores correct answers no better than garbage (no signal),
- all-zero / all-one reward distributions (no gradient),
- a parser that drops the reward over a stray space,
- a dataset that's secretly the eval set (contamination).

RLEnv_audit catches these in seconds, on CPU, with one command.

```console
$ rlenv-audit run gsm8k

                              RLEnv_audit · gsm8k
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check         ┃ status ┃ summary                                             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ determinism   │ PASS   │ all 12 completions stable across 5 repeats          │
│ reward_design │ PASS   │ binary reward, separates gold from garbage on 5/5   │
│ exploits      │ PASS   │ no hard cheats exploited                            │
│ parser        │ PASS   │ parser handled 21/24 perturbations (88%)            │
│ contamination │ PASS   │ no overlap found; checked 500 questions             │
│ rollouts      │ SKIP   │ no model endpoint configured                        │
└───────────────┴────────┴─────────────────────────────────────────────────────┘
overall grade: PASS   rating: A (100/100)   (PASS 5  SKIP 1)

Recommendations (what to improve before training on this env):
  1. Reward is strictly 0-or-1. That's valid, but graded partial credit usually
     gives the policy a denser gradient on hard tasks (REWARD_DESIGN.md §partial-credit).

report written to: report.json
```

---

## Install

RLEnv_audit pins **`verifiers==0.1.14`** on purpose — newer versions drag in
torch/vLLM, and most of the nine checks run on CPU (the model-assisted ones just need an API endpoint).

```bash
# with uv (recommended)
uv venv --python 3.11
uv pip install -e .            # or: uv pip install rlenv-audit

# install an environment to audit (anything exposing load_environment)
vf-install primeintellect/gsm8k       # from the Environments Hub
vf-install gsm8k -r                    # or the example envs in the verifiers repo
```

> **Python version note.** Most current Hub environments (`primeintellect/*`)
> require **Python ≥3.11**, so use a 3.11 venv for those. The pinned
> `verifiers==0.1.14` also runs on 3.10 if you specifically need it (e.g. an old
> CUDA box) — but you'll only be able to install the older example envs there.

Optional extras:

- **exploits** needs **Docker** running (it executes hostile code in an isolated,
  network-disabled container).
- **distribution** needs a **GPU + vLLM**: `uv pip install -e ".[gpu]"`.
- **rollouts** needs an **OpenAI-compatible endpoint** (OpenAI, or a local vLLM /
  llama.cpp server).

Any of these missing → that check `SKIP`s; the audit still runs.

---

## Usage

```bash
rlenv-audit run <env-id>                          # full battery
rlenv-audit run <env-id> --only determinism,parser
rlenv-audit run <env-id> --skip distribution,rollouts
rlenv-audit run <env-id> --json out.json          # also write JSON here
rlenv-audit run <env-id> --model Qwen/Qwen2.5-1.5B-Instruct          # distribution model
rlenv-audit run <env-id> --endpoint http://localhost:8000/v1 --model qwen   # enable rollouts
rlenv-audit list-checks                           # what each check needs
```

`rlenv-audit run` writes `report.json` (machine-readable, full details +
recommendations) and **exits non-zero on an overall FAIL** — drop it straight
into CI for your environments.

### As a library

The CLI is a thin wrapper; everything is callable programmatically:

```python
import rlenv_audit

scorecard = rlenv_audit.audit("gsm8k")          # or pass an already-loaded Environment
print(scorecard.grade)                          # PASS / WARN / FAIL / INCONCLUSIVE
print(scorecard.rating)                         # {'score': 100, 'letter': 'A', ...}
print(scorecard.recommendations())              # ["...", ...]
scorecard.write_json("report.json")

scorecard = rlenv_audit.audit("gsm8k", only=["determinism", "reward_design"])
```

---

## The nine checks

| Check | Needs | What it catches |
| --- | --- | --- |
| **integrity** | — | Structural soundness, works on *any* env: reward functions present, dataset non-empty, answers populated, duplicate tasks, well-formed prompts, system prompt present. Pure introspection — no scoring. |
| **determinism** | model endpoint | A model writes ~20 diverse probes (gold / rewritten / wrong / edge) *in this env's own answer format*; each is scored 5×; **FAIL** if any reward varies. Non-determinism injects noise into the gradient. |
| **reward_design** | model endpoint | A model writes gold / partial / wrong / garbage completions; checks does correct out-score garbage, flat baseline floor, constant/binary/graded signal, bounds in [0,1], sane weights (the weight check runs without a model). **FAIL/WARN** with concrete fixes. |
| **exploits** | Docker | Submits known cheats (`sys.exit(0)`, monkeypatch `assert`, read the expected-output file, empty solution) *instead of* real answers; **FAIL** if a no-solution cheat earns reward above a junk baseline. Runs in a locked-down container — it executes hostile code. |
| **parser** | — | Feeds correct answers in perturbed formats (whitespace, `\boxed{}`, punctuation, casing); score = fraction still extracted; **WARN** if brittle. |
| **contamination** | — | N-gram overlap of the dataset against popular eval sets (GSM8K, MATH-500, AIME, HumanEval, LiveCodeBench), with boilerplate filtering; **FAIL** listing matches. |
| **rollouts** | model endpoint | Real mini-rollouts via any OpenAI-compatible endpoint: generate → parse → score, checking the pipeline works on real model text; **WARN** on zero-variance reward or a parser that extracts nothing. |
| **design_review** | model endpoint | Hands the env's *actual reward-function source code*, system prompt, and sample tasks to an LLM together with `REWARD_DESIGN.md`, and gets a structured expert review — the issues only reading the code reveals (swallowed exceptions, gameable judge prompts, fragile regexes). |
| **distribution** | GPU | Rollouts with a small reference model; **WARN** on all-zero / all-one / empty-rewarded distributions — shapes that produce no learning signal. |

`SKIP` ≠ `FAIL`. A check SKIPs when it can't run *here* (no GPU, Docker down, no
endpoint, no dataset) — it is never counted against the environment.

### Skill-file-driven probes

Several checks need *inputs* shaped like the environment under test — and a
static, hand-written battery only fits math/QA envs. So each such check ships a
**skill file** (`rlenv_audit/skills/<check>.md`): a prompt telling a model how to
read this specific env (system prompt, parser, sample tasks + gold answers,
reward source) and write the inputs that check needs:

| Check | Skill generates |
| --- | --- |
| determinism | ~20 diverse probes (gold / rewritten / wrong / edge) in the env's format |
| reward_design | gold / partial / wrong / garbage completions to measure reward shape |
| exploits | env-specific cheat completions (alongside the universal security battery) |
| parser | realistic format variants of the correct answer |

Point the audit at any OpenAI-compatible endpoint (OpenAI, a local vLLM, or a
Claude-compatible proxy) with `--endpoint` / `OPENAI_API_KEY`, and these checks
generate exactly the inputs they need for *your* env. **No endpoint → determinism
and reward_design SKIP** (there is no static fallback, by design); exploits and
parser still run their universal batteries.

## Rating & recommendations

Beyond per-check status, the report gives you:

- **Overall grade** — the worst meaningful result (any `FAIL` → FAIL).
- **Rating** — a weighted **0–100 score and A–F letter**, computed only over the
  checks that actually ran (`PASS` full credit, `WARN` half, `FAIL` none; `SKIP`
  excluded). So an env isn't penalized for a check that couldn't run.
- **Recommendations** — every failing check attaches a concrete fix, each citing a
  section of [`REWARD_DESIGN.md`](REWARD_DESIGN.md), the bundled guide to good
  reward design (determinism, discrimination, baseline floor, partial credit,
  bounds, weights, anti-hacking, parser contract, difficulty curriculum,
  contamination). That's what turns the audit from a gate into design feedback.

A clean scorecard means none of these failure modes were detected on the slice we
could measure — not a proof of correctness, but the cheap faults are ruled out.

## Auditing many environments

`scripts/survey.py` batch-audits a list of envs in isolated subprocesses (one
hang or dependency blow-up can't take down the run) and aggregates into
`survey.json` — the harness behind a Hub-wide survey.

```bash
python scripts/survey.py aime2024 math500 reverse-text wordle   # or edit the curated default list
```

## How it works

`rlenv-audit` loads the environment through `verifiers.load_environment`, then the
**adapter** (`adapters/verifiers.py`) normalizes it into an `EnvHandle` — the only
code that touches `verifiers`. The handle exposes the rubric's reward functions,
the parser, and the dataset, plus a synchronous `score()` over the library's async
scoring path (RubricGroup-aware, threads all dataset columns through, tears down
rubric-owned process pools). Each check is an independent function over that
handle, so checks run in any subset and degrade to `SKIP` rather than crash. See
[`DESIGN.md`](DESIGN.md) for the architecture and the verifiers-0.1.14 API notes.

## Limitations (honest)

- **Multi-turn / agentic / tool envs** — checks that score completions probe a
  single synthetic answer, so they give a weaker signal (or `SKIP`) where the
  reward depends on a real multi-turn trajectory.
- **Judge-based rewards** — need an API key to score; without one the scoring
  checks `SKIP`.
- **Sandbox / web / MCP envs** — may not load inside the offline exploits
  container → that check `SKIP`s (with the reason surfaced).
- **Install layer** — some Hub envs pin dependencies that conflict, or need
  Python/services you don't have; those are reported as a clean load failure, not
  a crash.

In all cases the tool runs and reports *why* it skipped — it does not break on
arbitrary environments.

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/          # report unit tests + gsm8k integration tests
```

## License

MIT
