---
name: env-audit-contamination
description: Contamination check — compare the environment's dataset against the HuggingFace datasets the user explicitly provided. N/A (carries no weight in the rating) if the user provided none. Matching instances lower the score; a clean dataset scores high.
---

# Check 6 — contamination

**Question:** does this env's dataset overlap the datasets the user cares about,
so that "improvement" measured on them would be partly memorization?

**Runs only against user-provided datasets.** The user names the HuggingFace
datasets to check (ids like `openai/gsm8k`, or hf.co links — normalize a link to
its `org/name` id). Do **not** pick benchmarks yourself. If the user provided
none, output immediately:

```json
{"name": "contamination", "status": "N/A", "score": null,
 "justification": "no contamination datasets provided"}
```

(N/A is excluded from the rating, so the check carries no weight when skipped.)

## Steps

1. **Load each provided dataset** with the `datasets` library (it ships with
   verifiers), e.g.
   `python -c "from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main', split='test')"`.
   Use the split the user named; default to `test`, falling back to `train`.
   If a dataset can't be loaded (bad id, gated, offline), say so in the
   justification and judge from the ones that did load; if none load, output
   **N/A** with that reason.

2. **Get a good slice of the env's dataset.** Re-run
   `rlenv-audit inspect <env> -n 100` (or more) so the comparison isn't limited
   to the orchestrator's 20-sample inspect.

3. **Compare instances.** For each env instance vs. each provided dataset:
   - normalize the question text (lowercase, collapse whitespace, strip
     punctuation) and look for exact or near-exact matches (high n-gram
     containment), not just shared boilerplate;
   - **catch paraphrases the n-grams miss**: for near-misses, compare the
     *numbers, named entities, and the answer* — a reworded question with the
     same figures and the same answer is the same instance. Judge a sample of
     suspicious pairs yourself rather than trusting text overlap alone;
   - distinguish **same-template-different-instance** (fine) from **same
     instance** (contamination);
   - check **both splits** of each provided dataset where they exist — env
     tasks lifted from the benchmark's *train* split still poison eval-set
     conclusions when the splits share templates;
   - report concrete matching pairs (env row ↔ dataset row) with counts;
   - note the train/eval distinction: an explicit *eval* env that *is* one of
     the provided datasets is expected — say so; a *training* dataset
     overlapping the user's eval set is the real problem.

## Output

Score 0–10 where 10 = clean, lower = more overlap. Anchor the score to the
measured rate over the sampled rows: no same-instance matches → 9–10; isolated
matches (≲5% of the sample) → 6–8 (WARN); systematic overlap (tens of percent)
→ ≤5; the env *is* the user's eval set used for training → FAIL territory.
State the sample size — 0 matches in 100 rows is evidence, not proof.

```json
{"name": "contamination", "status": "PASS|WARN|FAIL", "score": <0-10>,
 "justification": "<one line: datasets checked, matches found (counts) or clean>"}
```
