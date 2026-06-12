---
name: env-audit-reward-design
description: Reward-design check — stress-test the reward function without the policy model. Sample ~20 prompts, write synthetic completions for each (correct, wrong, edge cases, format perturbations), pass them through the reward function, and check (a) the reward varies and discriminates with sensible ordering, and (b) the assigned reward matches your own judgment of each completion's quality.
---

# Check 3 — reward design

**Question:** is the reward function a usable training signal — does it vary,
discriminate correctly, and agree with a competent judge?

No policy model needed. You write the completions yourself; the `score` tool runs
them through the real reward function.

## Steps

1. **Sample ~20 prompts.** From `/tmp/envaudit_inspect.json` take the sampled
   tasks (re-run `rlenv-audit inspect <env> -n 20` if you need more). Note the
   answer format the env expects (from the system prompt / parser type / reward
   source).

2. **Write synthetic completions.** For each sampled prompt, author several
   completions in *this env's format*, covering:
   - **correct** — a fully right answer that should earn max reward;
   - **wrong** — well-formed but the answer is incorrect;
   - **edge cases** — empty, whitespace, partial/near-miss, answer + extra text,
     extremely long, the answer stated twice;
   - **format perturbations** (when a format spec exists) — the *right answer in a
     different format*, and the *right format with a wrong answer*.
   Build a JSON list `[{"prompt_index": i, "label": "...", "text": "..."}, ...]`
   (~20–40 entries total) and save it to `/tmp/envaudit_completions.json`.

3. **Score them.** Run
   `rlenv-audit score <env> /tmp/envaudit_completions.json --out /tmp/envaudit_scores.json`
   and read the `(label, prompt_index, reward)` triples. For a code/agentic env
   the rubric *executes* your completion, so scoring defaults to a Docker sandbox
   — model-written code never runs on the host. The output carries a top-level
   `sandbox` block (`{requested, used, available, reason}`); glance at it.

   When entries come back with an `error` instead of a reward, **classify the
   error before grading**:
   - **Reward-function defect** — the rubric ran but mishandled an ordinary
     completion (empty text, long text, a near-miss): a real defect. Quote it. If
     **every** entry is this kind of error, the check is a **FAIL**.
   - **No execution backend** — the rubric couldn't run *anywhere*, so nothing
     about the reward signal was actually tested. This is **N/A, not FAIL**.
     Treat as backend-unavailable when: the top-level result has
     `"error": "sandbox required but unavailable: ..."`, or `sandbox.used` is
     `false` with `sandbox.available` `false`, or the per-entry errors are
     execution-infrastructure failures rather than logic — e.g. they mention
     `sandbox`, `docker`, a remote runner (`e2b`, `prime`, `modal`), `connection`
     / `network` / `timeout`, or a missing executor/binary. If **every** entry is
     this kind, the env's reward simply could not be exercised here.

   A working env whose code-runner is merely absent on this box must **not** be
   failed — say N/A and name the missing backend.

4. **Judge two things:**
   - **(a) Variance & discrimination.** Is there real spread in the rewards (not
     a constant)? Is the *ordering* sensible — correct ≥ partial > wrong ≈ edge
     junk? Does a right-answer-wrong-format or right-format-wrong-answer get
     scored the way it should?
   - **(b) Judgment agreement.** For each (prompt, completion, reward) triple,
     does the reward match *your own* assessment of that completion's quality?
     Count the agreements; flag every disagreement (e.g. a clearly-correct answer
     scored 0, or garbage scored high).

## Output

Combine the two into a 0–10 score (roughly: half variance/discrimination, half
judgment agreement). A constant reward, or correct answers not out-scoring
garbage, is a **FAIL**. If the reward could not be exercised at all because no
execution backend was available (see step 3), report **N/A** with a `null` score
and name the missing backend — do not FAIL.

```json
{"name": "reward_design", "status": "PASS|WARN|FAIL|N/A", "score": <int|null>,
 "justification": "<one line: signal type, discrimination, agreement N/total, worst disagreement — or, for N/A, the missing execution backend>"}
```

See `REWARD_DESIGN.md` for what good reward design looks like (discrimination,
baseline floor, partial credit, bounds, anti-hacking).
