---
name: env-audit-integrity
description: Integrity check for an RL environment — verify it is written properly and actually runs. Confirms the dataset loads and is well-formed, the reward function is present and callable, the code follows verifiers/prime-intellect conventions, and there are no missing fields or broken imports. The "does it even run and is it shaped right" check.
---

# Check 1 — integrity

**Question:** is this environment written properly and does it run?

This check needs no model. Use the `inspect` tool plus reading the source.

**What a well-built env looks like** (grade against this anatomy): a clear task
specification (system prompt + dataset), an action interface matching the task
(plain completion, or tools for agentic work), each dataset row bundling
*everything its reward needs* (a task is more than a prompt — test cases,
oracles, expected outputs ride along in `answer`/`info`/columns), a real reward
mechanism, and a termination story (single-turn, or `max_turns` / a finish
condition that actually fires).

## Steps

1. **Load.** Read `/tmp/envaudit_inspect.json` if the orchestrator already wrote
   it; otherwise run `rlenv-audit inspect <env> -n 20 --out /tmp/envaudit_inspect.json`.
   If `loaded` is false → **FAIL**, score ≤ 2, justification = the `error`.
   Stop here. (A *nonexistent* env id is the orchestrator's problem — it stops
   and asks the user before this check runs. `loaded: false` here means the env
   exists and is installed but crashes on load: a genuine defect.)

2. **Dataset well-formed.** From the JSON, check:
   - `dataset_size.train` or `.eval` is non-zero;
   - sampled rows have a `prompt` that is a chat-message list (`{role, content}`)
     or a non-empty string;
   - rows have an `answer` (or the reward clearly doesn't need one — e.g. a judge
     env; note which);
   - no obviously broken/empty rows.

3. **Reward present and callable.** `reward_funcs` is non-empty; read each
   `source`. Confirm it's a real function (not a stub returning a constant), and
   note its weight. Zero or all-zero weights is a serious problem.

4. **Reward inputs exist in the data.** Cross-check what each reward function
   *reads* (its named args, `info[...]` / column accesses in the source) against
   what the sampled rows actually carry (`answer`, `info`, `extra_columns`). A
   reward that reads `info["test_cases"]` from rows that have no `test_cases`
   will crash or silently score a constant — a top-severity defect that load
   checks alone never catch.

5. **Hidden runtime dependencies.** If the rubric is judge-based (an LLM judge,
   a remote sandbox/executor), name what it needs at runtime — an API key, an
   endpoint, Docker, a service. An env that loads fine but cannot *score*
   without an undeclared dependency is an integrity defect; say which.

6. **Conventions & imports.** Read the env source at `module_file`. Confirm it
   follows verifiers conventions: a `load_environment(...)` that returns an
   `Environment` with a `rubric`, `parser`, and dataset; imports resolve; no
   dead/missing fields; a parser is set if the reward relies on parsing.

7. **Interaction shape & termination.** Check the env class fits the task:
   `SingleTurnEnv` for one-shot answers; for `MultiTurnEnv`/`ToolEnv`, confirm
   tools are defined and callable, `max_turns` is set to something sane, and a
   finish condition exists that the model can actually reach (a terminator tool,
   an answer format, a done state). An episode that can never end — or a "tool"
   env with no tools — is broken in a way single-turn checks miss.

8. **System prompt.** Note whether a system prompt is present (its absence is a
   real defect for format-sensitive envs, but judge in context).

## Output

Score 0–10: start at 10 and deduct for each defect by severity (won't load →
fail outright; missing reward / empty dataset → large deduction; no system
prompt / minor convention slips → small). Return:

```json
{"name": "integrity", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: what's right, and the most important defect if any>"}
```
