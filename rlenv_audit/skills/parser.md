# Skill: generate parser format variants

You are testing whether a reinforcement-learning environment's **answer parser**
is robust to the many ways a real model phrases the same correct answer. A
parser that only accepts one rigid format will score correct rollouts as wrong
and corrupt the training signal.

Below this prompt is the environment under test (system prompt, parser, sample
tasks with gold answers). For each task, write completions that **all contain
the same correct gold answer** but present it the way different real models
plausibly would.

## What to produce

For each task, several completions that should ALL parse to the gold answer:

- the answer in the env's canonical format
- with reasoning/chain-of-thought before it
- with extra surrounding text, trailing punctuation, or markdown
- the answer restated or emphasized ("**Answer:** ...")
- minor whitespace / casing / spacing variations of the format
- the answer wrapped in slightly different but reasonable markup

Every variant must be a genuinely correct answer to that task — only the
*presentation* changes. Do not produce wrong answers here.

## Output format

Return ONLY this JSON object:

```
{"variants": [
  {"task_index": 0, "label": "short-label", "text": "the completion"},
  ...
]}
```

`task_index` must refer to one of the listed tasks. Keep each completion under
150 words.
