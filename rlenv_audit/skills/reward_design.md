# Skill: generate reward-design probes

You are generating test completions to probe the **shape** of a reinforcement-
learning environment's reward function — not whether it's deterministic, but
whether it is a usable training signal: does a correct answer score higher than
garbage, is the reward graded or binary, is there a constant floor everyone gets.

Below this prompt is the environment under test (system prompt, parser, reward
functions, sample tasks with gold answers). Write completions in **this
environment's actual answer format** so the reward function's real branches are
exercised.

## What to produce

For **each** task, produce these kinds (so we can compare their rewards):

- **gold** — fully correct, in the expected format; should earn maximum reward.
- **gold_rewritten** — also fully correct, same answer, different phrasing/format.
- **partial** — partially correct (e.g. right approach, wrong final value; or
  some of several required parts correct). Used to detect graded vs binary reward.
- **wrong** — plausible, well-formed, but the answer is incorrect.
- **garbage** — irrelevant or nonsensical text that clearly does not address the
  task.

## Output format

Return ONLY this JSON object:

```
{"probes": [
  {"task_index": 0, "kind": "gold", "label": "short-label", "text": "the completion"},
  {"task_index": 0, "kind": "partial", "label": "...", "text": "..."},
  ...
]}
```

`task_index` must refer to one of the listed tasks. Cover every task with at
least gold / wrong / garbage. Keep each completion under 200 words.
