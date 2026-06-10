# DESIGN — RLEnv_audit

> "pytest for RL environments." Point it at a `verifiers` environment from the
> Prime Intellect Environments Hub; it runs a battery of automated checks and
> prints a quality scorecard + a machine-readable `report.json`.

## 1. Why this exists

RL post-training environments are now treated like training data — but unlike
data, nobody tests them before burning GPU hours. A broken reward function does
not crash; it silently teaches the policy garbage:

- **Non-deterministic rewards** → noisy gradient, the policy chases randomness.
- **Exploitable rewards** → the policy learns to cheat (`sys.exit(0)`, read the
  answer off disk) instead of solving the task.
- **Degenerate reward distributions** (all-zero / all-one) → zero gradient, no
  learning signal at all.
- **Brittle parsers** → correct answers scored wrong because of a stray space.
- **Contaminated datasets** → "improvement" that is just memorized eval data.

`RLEnv_audit` catches these before they cost a training run. The end goal is a
survey: *"I audited N Hub environments; X failed determinism, Y are exploitable,
Z are contaminated."* So the tool must run on real Hub environments, emit clean
JSON, and be runnable by a stranger in 60 seconds.

## 2. The one architectural rule: CLI tool over a library, NOT a framework

The whole value is **zero adoption cost**. A person with an environment runs one
command and gets a verdict. No subclassing, no restructuring their env, no config
files to get started.

- **Primary interface — CLI:** `rlenv-audit run <env-id>` runs the battery,
  prints a scorecard, writes `report.json`.
- **Library underneath:** `import rlenv_audit; report = rlenv_audit.audit(env)`
  returns a structured `Scorecard`. **The CLI is a thin wrapper** — it parses
  args and renders output; every bit of real logic lives in the library.

The test for every design decision: *"can a stranger get value in 60 seconds
without reading docs?"* If no, it has drifted into framework territory — stop and
simplify.

## 3. Target format: `verifiers==0.1.14`

We build against one format — the `verifiers` library, Prime Intellect's standard
for the Hub. The version is **pinned to `0.1.14`** deliberately: the target box
has old/fragile CUDA, and newer `verifiers` drags in torch/vLLM. Five of the six
checks need only the verifiers *core*, which is CPU-only.

The adapter was written against the **actually installed source**, not remembered
API. The facts that shaped the design (all verified by reading
`site-packages/verifiers/`):

| Concern | Reality in 0.1.14 |
| --- | --- |
| Loading | `verifiers.load_environment(env_id, **env_args) -> Environment` is **synchronous**. It imports a module named `env_id.replace("-","_").split("/")[-1]` and calls its `load_environment()`. The env must be pip-installed as an importable module (via `vf-install`). |
| Rubric | Often a **`RubricGroup`**, whose own `.funcs` is empty — the real reward functions live in sub-rubrics and surface via `rubric._get_reward_func_names()`. |
| Scoring | **Async**, mutates a `state` dict in place. `score_rollout` asserts no group rewards; `score_group` handles both. We branch on `rubric.has_group_rewards`. |
| Reward funcs | May be bound methods using a `ProcessPoolExecutor` (e.g. `MathRubric`) → must `teardown()` when done or the process can hang. |
| Parser | `parser.parse_answer(messages) -> str | None` extracts the answer from a completion; `parser.parse(text)` is the lower-level hook. |
| Dataset | A HuggingFace `Dataset`; rows carry `prompt` (a `Messages` list of `{role,content}`), `answer` (str), plus env-specific columns. Read via `env.get_dataset(n)`. |

## 4. Abstraction mapping

```
verifiers concept      attacked by
-------------------    ----------------------------------------
rubric / reward fns →  determinism, exploits, distribution checks
parser             →  parser-robustness check
dataset            →  contamination check
(whole pipeline)   →  latency check
```

## 5. The adapter — `EnvHandle` (`adapters/verifiers.py`)

The adapter is the only code that touches `verifiers`. It normalizes a loaded
environment into an `EnvHandle` so checks never import `verifiers` directly:

```python
EnvHandle:
    env_id: str
    env: verifiers.Environment      # the raw loaded env
    rubric, parser                  # convenience handles
    dataset(n) -> list[dict]        # normalized rows: {prompt, answer, info, ...}
    reward_func_names() -> list[str]
    score(text, prompt, answer, info) -> (reward: float, metrics: dict[str,float])
    teardown()                      # best-effort rubric teardown (ProcessPool)
```

`score()` is the heart of the tool. It is a **synchronous** wrapper over the
async verifiers scoring path, so checks (and the public `audit()`) never deal
with asyncio:

1. Build a plain `state` dict:
   `{"prompt", "completion": [{"role":"assistant","content": text}], "answer",
   "info", "task": None, "input", "trajectory": []}`.
2. `if rubric.has_group_rewards: await rubric.score_group([state])`
   else `await rubric.score_rollout(state)`.
3. Return `state["reward"]` and `dict(state["metrics"])`.

This one path works uniformly across plain `Rubric`, `MathRubric`, and
`RubricGroup`. Loading is wrapped so a bad env-id yields a clean error, never a
traceback.

## 6. Data model (`checks/base.py`, `report.py`)

- `CheckStatus`: `PASS | FAIL | WARN | SKIP`. **SKIP** = the check could not run
  here (no GPU, Docker down, N/A for this env type) — distinct from FAIL.
- `CheckResult` (dataclass): `check_name`, `status`, `score: float | None`,
  `summary` (one human line for the table), `details: dict` (structured findings
  for JSON), `duration_s`.
- `Scorecard`: the env id + a list of `CheckResult` + a derived overall grade,
  with `to_terminal()` (rich table, color-coded) and `to_json()` (full details
  for the survey stage).

Each check is an independent function `check_x(handle, config) -> CheckResult`.
Independence is a requirement: some need a GPU, some need Docker, so
`--only exploits,contamination` and `--skip distribution` must work, and every
failure mode degrades to a clean SKIP/FAIL result rather than crashing the run.

## 7. The six checks (v0 scope — exactly these, no more)

1. **determinism** (no GPU) — score a fixed set of pre-generated completions 5×
   each; FAIL if any reward varies. Completions are derived from the dataset's
   own answers so the check works on any env.
2. **exploits** (no GPU, **Docker-mandatory**) — submit known cheat patterns
   instead of honest solutions (`sys.exit(0)`, monkeypatch `assert`, read the
   expected-output file, print the answer without computing, empty+success,
   parser-format tricks); FAIL listing which cheats scored reward. Runs the
   reward function inside a Docker sandbox because it executes hostile code.
3. **distribution** (needs GPU → SKIP) — vLLM rollouts with a small reference
   model, histogram the rewards; WARN on all-zero / all-one / empty-rewarded.
4. **parser** (no GPU) — feed correct answers in perturbed formats through the
   parser; score = fraction still extracted; WARN below threshold.
5. **contamination** (no GPU) — n-gram overlap of dataset questions against
   cached popular eval sets (AIME, MATH-500, GSM8K, HumanEval, LiveCodeBench);
   FAIL listing matches. SKIP eval sets that can't be fetched offline.
6. **latency** (no GPU) — time per verification call, cold vs warm, basic
   parallelism; informational (PASS/WARN only).

## 8. CLI surface

```
rlenv-audit run <env-id>                  # full battery
rlenv-audit run <env-id> --only a,b       # subset
rlenv-audit run <env-id> --skip c         # exclude
rlenv-audit run <env-id> --json out.json  # also write JSON report
rlenv-audit run <env-id> --model <name>   # reference model for distribution
rlenv-audit list-checks                   # checks + what each needs
```

## 9. Validated environment types & known limitations

The tool is built generically over the `verifiers` API and validated on four
structurally different env types:

| Env | Type | Notes |
| --- | --- | --- |
| `gsm8k` | `SingleTurnEnv` + `MathRubric`/`RubricGroup` | reference env |
| `reverse_text` | `SingleTurnEnv` + `XMLParser` | continuous LCS reward |
| `wordle` | `TextArenaEnv` (multi-turn) | game env |
| `math_group` | `EnvGroup` | aggregates sub-envs |

To stay env-agnostic the adapter threads *all* dataset columns into the scoring
state, and the parser/exploit checks discover each env's canonical answer format
from the parser itself (`parser.format(...)`), rather than assuming `\boxed{}`.

Honest limitations (each degrades to a clean SKIP, never a crash):

* **Multi-turn / tool / agentic rewards.** Scoring submits a single synthetic
  completion; rewards that depend on a real multi-turn trajectory or tool I/O
  can't be reproduced without a model, so determinism/exploits give a weaker
  signal there.
* **Offline exploit sandbox.** The sandbox runs with no network. Envs needing
  data not already cached (e.g. an NLTK corpus) or an external service fail to
  load *inside the container* → exploits SKIPs with the reason surfaced.
* **Pass-through parsers.** When `env.parser` does no extraction (the reward
  function extracts the answer itself), the parser check SKIPs — there's nothing
  parser-specific to perturb.
* **Incompatible env deps.** Some Hub envs pin dependencies that conflict with
  `verifiers==0.1.14` and won't install (e.g. `math_python`); those can't be
  audited under this pin.

## 10. Honest scope statement

v0 is **six checks, one format (`verifiers`), one command**. There is no plugin
system, no config-file framework, no multi-format support — on purpose. The
`adapters/` and `checks/` seams make those *possible* later without building them
now. Extension is a future concern; adoption cost is the present one.
