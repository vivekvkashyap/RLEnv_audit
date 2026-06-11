---
name: env-audit-latency
description: Latency check — measure how long rollouts take end to end. Requires a model endpoint. Reads the shared cached rollout set (8 rollouts over ~20 samples) and reports timing; does not run its own rollouts.
---

# Check 4 — latency

**Question:** how expensive is a rollout end to end?

**Requires a model endpoint.** If none was configured, output:

```json
{"name": "latency", "status": "N/A", "score": null, "justification": "no model endpoint provided"}
```

## Steps

1. Read the **shared** rollout cache the orchestrator already generated
   (`/tmp/envaudit_rollouts.json`, produced by `rlenv-audit rollouts ... -n 20 -k 8`).
   Do **not** roll out again — checks 4 and 5 share this one cache.

2. From its `timing` block read `mean_s`, `p50_s`, `p90_s`, `max_s`, `total_s`
   and `calls`. (For Prime Intellect envs you may instead time `vf-eval <env>`
   end to end and report that — but prefer the shared cache.)

3. Judge the numbers in context: a per-rollout mean of a few seconds is normal
   for a hosted model; tens of seconds, heavy tail (p90 ≫ p50), or frequent
   errors mean the env's verification/tooling is slow and will bottleneck
   training throughput.

## Output

This check is informational — reserve **FAIL** for pathological cases (errors on
most rollouts, or absurd latency). Score reflects throughput health.

```json
{"name": "latency", "status": "PASS|WARN", "score": <int>,
 "justification": "<one line: mean/p90 per rollout, total, any error rate>"}
```
