# REWARD_DESIGN.md — how to design a good `verifiers` RL environment

This is the reference the `rlenv-audit` checks point at. Every recommendation in
a scorecard cites a section here. It's a checklist for building an environment
whose reward is actually a good training signal — written so you can fix a flagged
issue without guessing what "good" means.

The one-line principle: **the reward must go up if and only if the policy gets
better at the task.** Everything below is a way that principle breaks.

---

## §determinism — the same completion must always score the same

Score a fixed completion repeatedly; the reward must not move. Sources of
non-determinism to remove:

- unseeded RNG anywhere in scoring (`random`, `numpy`, sampling a judge);
- wall-clock or timeout-dependent scoring (a slow machine scores differently);
- network/API calls whose result varies (rate limits, model drift);
- LLM judges with `temperature > 0` — pin to `0` and ideally cache.

A reward that varies across identical re-scores injects pure noise into the
gradient: the policy chases randomness, not skill.

## §discrimination — correct must out-score garbage

A correct answer must score strictly higher than an empty or nonsense
completion. If it doesn't, the gradient points nowhere. The usual culprit is the
**parser/matcher rejecting the dataset's own gold answers** — if the env can't
reward its own answer key, no policy can learn from it. Test: run each row's gold
`answer` through the verifier and confirm it earns the max reward.

## §baseline-floor — don't pay everyone

If every response (even garbage) earns a constant positive reward — a
participation, length, or format bonus applied unconditionally — the *relative*
advantage of solving the task shrinks. Prefer a zero floor. If you keep a format
reward, weight it near zero and gate it on the answer being present, not on mere
formatting.

## §partial-credit — graded beats binary on hard tasks

Strict 0-or-1 reward is correct but sparse: on tasks the policy almost never
solves, the gradient is almost always zero. Where the task allows, give graded
partial credit (fraction of unit tests passing, edit-distance/LCS similarity,
sub-goal completion) so there's signal at the frontier.

## §bounds — keep the aggregate in [0, 1]

Keep the summed reward in a known, bounded range (normally `[0, 1]`). Unbounded or
wildly-scaled rewards complicate advantage normalization and make environments
incomparable. If you must use another range, document it.

## §weights — sanity-check the weighted sum

- At least one reward function must have a non-zero weight (all-zero weights →
  reward is 0 by construction → no learning).
- Negative-weight penalties are easy to get backwards; make sure a penalty can't
  dominate the positive signal and reward the policy for doing nothing.

## §anti-hacking — assume the policy will cheat

RL policies find every shortcut. A robust verifier:

- never trusts an exit code alone (a `sys.exit(0)` before tests must not pass);
- keeps expected outputs / test files **out of** the execution working directory
  (so the solution can't read the answer off disk);
- re-asserts results *after* the submission runs (so monkeypatching `assert`,
  overriding builtins, or printing the answer without computing it fails);
- rejects empty / no-op submissions.

If a no-solution completion earns reward, the policy will learn that shortcut
instead of the task.

## §parser-contract — parse what models actually emit

The parser must extract the answer from *real* model output, not just the
canonical format. Be liberal in what you accept: strip surrounding whitespace and
trailing punctuation, match case-insensitively, take the last occurrence when the
answer is restated, tolerate reasoning before the answer. And state the required
format explicitly in the system prompt — if the parser extracts nothing from real
rollouts, the format contract isn't being communicated.

## §difficulty-curriculum — avoid all-pass / all-fail batches

Generate rollouts with a reference policy and histogram the rewards. An all-zero
batch (too hard / broken) or all-one batch (trivial) gives zero gradient. Aim for
a spread of outcomes at the model's current ability — mix difficulties, or filter
the dataset to the band where the policy sometimes-but-not-always succeeds.

## §contamination — don't train on the eval set

N-gram/embedding-check your training tasks against the benchmarks you'll report
on (AIME, MATH, GSM8K, HumanEval, LiveCodeBench, …). Overlap means measured
"improvement" is partly memorization. For an explicitly *eval-only* environment,
overlap with its own benchmark is expected — just never use it for training.

---

### How the audit checks map to these sections

| Check | Sections it judges against |
| --- | --- |
| integrity | §parser-contract, §weights |
| reward_design | §determinism, §discrimination, §baseline-floor, §partial-credit, §bounds, §weights, §anti-hacking, §parser-contract |
| rollout_quality | §parser-contract, §difficulty-curriculum |
| contamination | §contamination |

(problem_alignment and latency judge things outside this guide: the user's stated
goal and throughput.)

A clean scorecard means none of these failure modes were detected on the slice we
could measure — not a proof of correctness, but the cheap faults are ruled out
before you spend GPU hours.
