# DESIGN — env_audit

> A skill-based auditing system for RL environments. An agent (Claude Code /
> Codex) runs six judgment-based checks over a `verifiers` environment and emits
> a scorecard with scores + written justifications.

## 1. Why this exists

RL post-training environments are treated like training data, but nobody tests
them before burning GPU hours. A broken reward doesn't crash — it silently
teaches the policy garbage: non-deterministic rewards (noisy gradient),
exploitable rewards (the policy cheats), rewards that don't discriminate (no
signal), brittle parsers (correct answers scored wrong), contaminated datasets
(memorized eval). env_audit catches these first.

## 2. The core decision: skills, not scripts

The six checks are **judgment-heavy and non-deterministic** — "does this reward
agree with a competent grader?", "is the system prompt missing something?", "does
this dataset overlap a benchmark?". A deterministic script can only approximate
these with brittle heuristics. An agent does them well.

So each check is a **skill file** (`skills/<check>/SKILL.md`, SKILL.md style with
`name` + `description` frontmatter) that an agent reads and executes with its own
reasoning. The agent leans on a thin **tool layer** (`rlenv-audit ...`) only for
the parts that must be exact and reproducible:

- **load + introspect** the environment,
- **score** agent-written completions through the real reward function,
- run + cache a **shared set of rollouts**,
- **render** the scorecard.

Tools are pure JSON-in / JSON-out so a skill can shell out and read the result.
This split keeps judgment in the agent and determinism in the code.

## 3. Target format: `verifiers==0.1.14`

We build against the `verifiers` library (the Hub standard), **pinned to 0.1.14**
(newer versions drag in torch/vLLM; the target box has old CUDA). The adapter was
written against the actually-installed source. Facts that shaped it:

| Concern | Reality in 0.1.14 |
| --- | --- |
| Loading | `verifiers.load_environment(env_id)` is **synchronous**; it imports a module named `env_id.replace("-","_").split("/")[-1]` and calls its `load_environment()`. The env must be pip-installed (`vf-install`). |
| Rubric | Often a **`RubricGroup`** whose own `.funcs` is empty — real reward funcs surface via `rubric._get_reward_func_names()`. |
| Scoring | **Async**, mutates a `state` dict in place. Branch on `rubric.has_group_rewards` (`score_group` vs `score_rollout`). |
| Reward funcs | May own a `ProcessPoolExecutor` (e.g. `MathRubric`) → must `teardown()`. |
| Parser | `parser.parse_answer(messages) -> str | None`. |
| Dataset | HF `Dataset`; rows carry `prompt` (chat messages), `answer`, plus env columns. Many Hub envs are eval-only. |

## 4. The adapter — `EnvHandle` (`adapters/verifiers.py`)

The only code that touches `verifiers`. It normalizes a loaded env into a stable,
synchronous handle the tools use:

```python
EnvHandle:
    load_handle(env_id) -> EnvHandle          # clean EnvLoadError on failure
    reward_func_names() / reward_sources()    # names + getsource of reward fns (RubricGroup-aware)
    system_prompt() / module_file()           # env framing + source file
    dataset(n) / dataset_size()               # normalized rows, train↔eval fallback
    score(text, prompt, answer, columns) -> (reward, metrics)   # sync over async scoring
    canonical_answer(answer) / teardown()
```

`score()` builds a `state` dict, runs the rubric (group-aware), and returns the
reward — uniform across `Rubric`, `MathRubric`, and `RubricGroup`. All dataset
columns are threaded through so reward funcs that read custom fields work.

## 5. The tool layer (`tools.py`, `cli.py`)

Four commands, each JSON-in/JSON-out:

- `rlenv-audit inspect <env> -n 20` → `{loaded, env_type, parser_type, module_file,
  dataset_size, system_prompt, reward_funcs:[{name,weight,source}], sample:[...]}`.
  Load failures are captured as `{loaded: false, error}` so the integrity check
  sees them as data. Used by checks 1, 2, 3, 6.
- `rlenv-audit score <env> completions.json` → scores agent-written
  `[{prompt_index, label, text}]` through the reward function. Used by check 3.
- `rlenv-audit rollouts <env> --endpoint --model -n 20 -k 8` (or `--dummy`) →
  generates 8 rollouts over ~20 tasks **once**, scores + times them, caches to
  JSON. Checks 4 and 5 share this single cache.
- `rlenv-audit scorecard results.json` → computes the overall grade + rating
  (average of the checks that ran; N/A excluded) and renders the table.

## 6. The six checks (`skills/`)

1. **integrity** — does it run and is it shaped right (dataset, reward, conventions,
   imports). No endpoint.
2. **problem-alignment** — given the user's problem statement (a required audit
   input; the agent asks for it if missing), does the env actually test it. No
   endpoint.
3. **reward-design** — agent writes ~20 synthetic completions (correct / wrong /
   edge / format perturbations), scores them, and checks (a) variance &
   discrimination and (b) agreement between the reward and the agent's own quality
   judgment. No endpoint.
4. **latency** — end-to-end rollout timing from the shared cache. Needs an endpoint.
5. **rollout-quality** — reads actual rollouts and judges the env setup (system
   prompt, output sensibility, env-caused failure modes). Needs an endpoint.
6. **contamination** — infer domain → pick benchmarks → check dataset overlap. No
   endpoint.

The **env-audit** orchestrator skill gathers inputs (env id, problem statement,
optional endpoint), runs the no-endpoint checks, generates the shared
rollouts once if an endpoint is given, runs the endpoint checks from that cache,
and assembles the scorecard.

## 7. Scoring model

Each check returns `{name, status, score (0–100|null), justification}`.

- **status**: PASS (~75–100) / WARN (~40–74) / FAIL (~0–39) / **N/A** (documented
  skip: no endpoint).
- **rating**: the mean of the numeric scores over checks that actually ran (N/A
  excluded), mapped to an A–F letter.
- **grade**: the worst meaningful status (any FAIL → FAIL).

Every score must be grounded in observed evidence — tool output, completions the
agent wrote, rollouts it read — never a vibe. `REWARD_DESIGN.md` is the rubric the
reward-design and rollout-quality checks judge against.

## 8. Distribution

One repo, published two ways, so a user never clones it:

- **Claude Code plugin** — `.claude-plugin/marketplace.json` makes the repo a
  one-plugin marketplace; `/plugin install env-audit@rlenv-audit` ships the
  skill files straight from GitHub.
- **PyPI wheel** — the wheel force-includes `skills/` as package data;
  `rlenv-audit install-skills` (or `uvx rlenv-audit install-skills`) copies them
  into `~/.claude/skills/`.

The skills carry no other setup burden: the orchestrator's bootstrap step has
the agent pip-install the tools and `vf-install` the target environment itself
on first run. The user types one install command once, then just "audit <env>".

## 9. Honest scope

Six checks, one format (`verifiers`), agent-driven. Determinism lives in the
tools; judgment lives in the skills. No plugin system, no config framework — the
`adapters/` + `skills/` seams make extension possible without building it now.
