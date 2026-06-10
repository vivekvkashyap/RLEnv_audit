# RLEnv_audit

**pytest for RL environments.** Point it at a [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment from the Prime Intellect Environments Hub and it runs a battery of
automated quality checks, then prints a scorecard and writes a machine-readable
`report.json`.

RL post-training environments are treated like training data — but unlike data,
nobody tests them before burning GPU hours. A broken reward function doesn't
crash; it silently teaches the policy garbage. RLEnv_audit catches that first.

```
rlenv-audit run gsm8k
```

> 🚧 **Status: under construction.** The project scaffold and design are in place;
> the checks are being built commit-by-commit (see [`COMMITS.md`](COMMITS.md)).
> Full usage docs land with the final docs commit. Architecture is in
> [`DESIGN.md`](DESIGN.md).

## The six checks (v0)

| Check | Needs | What it catches |
| --- | --- | --- |
| determinism | — | rewards that vary across identical re-scores |
| exploits | Docker | reward awarded to known cheats (`sys.exit(0)`, reading the answer off disk, …) |
| distribution | GPU | degenerate reward distributions (all-zero / all-one) |
| parser | — | brittle answer parsing (a stray space loses the reward) |
| contamination | — | dataset overlap with popular eval sets |
| latency | — | per-verification timing, cold vs warm |

v0 is six checks, one format (`verifiers`), one command — on purpose.
