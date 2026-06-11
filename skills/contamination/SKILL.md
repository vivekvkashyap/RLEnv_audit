---
name: env-audit-contamination
description: Contamination check — infer the environment's domain (math, coding, etc.), pick the common public benchmarks for that domain, and check whether dataset instances match or near-match benchmark instances. Matching instances lower the score; a clean dataset scores high.
---

# Check 6 — contamination

**Question:** does this env's dataset overlap public benchmarks, so that
"improvement" measured on those benchmarks would be partly memorization?

No model needed — your judgement plus the dataset.

## Steps

1. **Infer the domain.** From `/tmp/envaudit_inspect.json` (sample tasks + system
   prompt), decide the domain: grade-school math, competition math, general
   coding, competitive programming, QA/trivia, reasoning, etc.

2. **Pick the benchmarks that matter for that domain.** Examples:
   - grade-school math → GSM8K
   - competition/contest math → MATH, AIME, AMC, MATH-500
   - coding → HumanEval, MBPP, LiveCodeBench
   - QA/knowledge → TriviaQA, Natural Questions, SimpleQA
   - reasoning/MCQ → MMLU, MMLU-Pro, GPQA
   Name the specific ones you'll check against.

3. **Check for overlap.** Compare the env's dataset instances against those
   benchmarks' instances:
   - If you can load a benchmark (e.g. via `datasets`), do an n-gram / near-exact
     match of question text and report concrete matching pairs.
   - Otherwise reason from known benchmark contents and the dataset's stated
     source (many Hub envs literally *are* a benchmark — e.g. an `aime2024` env
     is AIME-2024; an env whose source loads `openai/gsm8k[test]` overlaps GSM8K
     by construction). Distinguish **same-template-different-instance** (fine)
     from **same instance** (contamination).
   - Note the train/eval distinction: an explicit *eval* env that *is* a
     benchmark is expected; a *training* env overlapping a benchmark you'll
     report on is the real problem.

## Output

Score 0–100 where 100 = clean, lower = more overlap:

```json
{"name": "contamination", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: domain, benchmarks checked, overlap found (counts) or clean>"}
```
