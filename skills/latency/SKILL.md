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
   (`/tmp/envaudit_rollouts.json`, produced by `rlenv-audit rollouts ... -n 20 -k 8`,
   which drives verifiers' own `vf-eval` engine under the hood). Do **not** roll
   out again — checks 4 and 5 share this one cache. If the file is missing but an
   endpoint *was* configured, generate it once with
   `rlenv-audit rollouts <env> --endpoint <url> --model <name> -n 20 -k 8 --out /tmp/envaudit_rollouts.json`.

2. From its `timing` block read `mean_s`, `p50_s`, `p90_s`, `max_s`, `total_s`
   and `calls` — per-rollout end-to-end times vf-eval recorded (generation plus
   scoring). Note `max_concurrent` in the cache: timings reflect generation at
   that concurrency (the realistic batched-rollout setting), so read them as
   throughput-under-load rather than single-request latency.
   If the cache was generated with `--dummy` (`"dummy": true`, empty `timing`),
   there is no real timing to judge — output **N/A** with justification
   "dummy rollouts — no real timing".

3. Judge the numbers in context: a per-rollout mean of a few seconds is normal
   for a hosted model; tens of seconds, heavy tail (p90 ≫ p50), or frequent
   errors mean the env's verification/tooling is slow and will bottleneck
   training throughput.

## Output

This check is informational — reserve **FAIL** for pathological cases (errors on
most rollouts, or absurd latency). Score reflects throughput health.

```json
{"name": "latency", "status": "PASS|WARN|FAIL", "score": <int>,
 "justification": "<one line: mean/p90 per rollout, total, any error rate>"}
```
