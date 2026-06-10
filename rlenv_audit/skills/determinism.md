# Skill: generate determinism probes

You are generating a battery of **test completions** to probe whether a
reinforcement-learning environment's reward function is **deterministic** — i.e.
whether scoring the *same* completion twice always yields the *same* reward.

Below this prompt you are given the environment under test: its system prompt,
its parser, its reward functions, and several real tasks with their gold answers.
Read them and write a diverse collection of assistant completions that, between
them, exercise as many branches of the reward function as possible.

## What to produce

Aim for **about 20 completions total**, spread across the given tasks, with a
healthy mix of these kinds:

- **gold** — a correct completion in the exact format this environment expects
  (tags, boxes, JSON, code, SQL, a move — whatever the system prompt/parser
  imply), which should earn full reward.
- **gold_rewritten** — also correct and full-reward, same final answer, but with
  noticeably different wording, reasoning order, or surface formatting.
- **wrong** — well-formed and plausible, but the final answer is incorrect.
- **edge** — completions that stress the reward/parser machinery: empty string,
  whitespace only, the answer followed by extra trailing text, the answer stated
  twice, an extremely long response, unusual unicode/escvape characters, a
  partial/near-miss answer, or the answer in a slightly-off format.

Correctness does **not** matter for this check — each completion is simply
re-scored several times and we check the reward never moves. So optimize for
**diversity** and **coverage of distinct code paths** (the reward-awarding path,
the zero path, error/timeout paths, and parser edge cases).

## Output format

Return ONLY this JSON object:

```
{"probes": [
  {"task_index": 0, "kind": "gold", "label": "short-unique-label", "text": "the completion"},
  {"task_index": 0, "kind": "edge", "label": "empty", "text": ""},
  ...
]}
```

`task_index` must refer to one of the tasks listed below. Keep each completion
under 200 words. Include at least a few of every kind.
