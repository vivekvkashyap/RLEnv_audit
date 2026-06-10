"""latency check — how expensive is a verification call?

Reward latency bounds how fast an RL loop can run. We time a single reward call
cold (first invocation, which pays any one-time setup like a process-pool spin-up)
versus warm (steady state), and probe whether batched scoring overlaps work at
all. Purely informational: PASS, or WARN if a warm call is implausibly slow.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckStatus

_WARM_WARN_S = 1.0   # warn if the mean warm call exceeds this
_WARMUP = 1
_WARM_CALLS = 10
_BATCH = 8


def _build_state(prompt, answer, info, text, cols=None):
    cols = {k: v for k, v in (cols or {}).items()
            if k not in {"prompt", "answer", "info", "example_id"}}
    state = {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": text}],
        "answer": answer,
        "info": info or {},
        "task": None,
        "input": {"prompt": prompt, "answer": answer, "info": info or {}, **cols},
        "trajectory": [],
    }
    for k, v in cols.items():
        state.setdefault(k, v)
    return state


def _batched_score(handle: EnvHandle, states) -> None:
    rubric = handle.rubric
    if getattr(rubric, "has_group_rewards", False):
        handle._run(rubric.score_group(states))
    else:
        async def _all():
            await asyncio.gather(*[rubric.score_rollout(s) for s in states])

        handle._run(_all())


def check_latency(handle: EnvHandle, config: dict) -> CheckResult:
    warn_s = float(config.get("latency_warn_s", _WARM_WARN_S))
    warm_calls = int(config.get("latency_warm_calls", _WARM_CALLS))

    rows = handle.dataset(n=1)
    if not rows:
        return CheckResult("latency", CheckStatus.SKIP, "environment exposes no dataset")
    row = rows[0]
    prompt, answer, info = row["prompt"], row["answer"], row["info"]
    cols = row.get("raw", {})
    text = f"\\boxed{{{answer}}}"

    # Cold: first call pays any one-time setup cost.
    try:
        t0 = time.perf_counter()
        handle.score(text, prompt, answer, info, cols)
        cold_s = time.perf_counter() - t0
    except ScoringError as exc:
        return CheckResult("latency", CheckStatus.SKIP, f"could not score a completion: {exc}")

    # Warm: steady-state per-call timings.
    warm: list[float] = []
    for _ in range(warm_calls):
        t0 = time.perf_counter()
        handle.score(text, prompt, answer, info, cols)
        warm.append(time.perf_counter() - t0)

    warm_sorted = sorted(warm)
    mean = statistics.fmean(warm)
    median = statistics.median(warm)
    p90 = warm_sorted[min(len(warm_sorted) - 1, int(0.9 * len(warm_sorted)))]

    # Parallelism probe: sequential vs batched scoring of the same N completions.
    speedup = None
    parallel_detail: dict = {}
    try:
        states = [_build_state(prompt, answer, info, text, cols) for _ in range(_BATCH)]
        t0 = time.perf_counter()
        for _ in range(_BATCH):
            handle.score(text, prompt, answer, info, cols)
        seq_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        _batched_score(handle, states)
        batch_s = time.perf_counter() - t0
        speedup = seq_s / batch_s if batch_s > 0 else None
        parallel_detail = {
            "batch_size": _BATCH,
            "sequential_s": round(seq_s, 4),
            "batched_s": round(batch_s, 4),
            "speedup": round(speedup, 2) if speedup else None,
        }
    except Exception as exc:
        parallel_detail = {"error": str(exc)}

    details = {
        "cold_s": round(cold_s, 4),
        "warm_mean_s": round(mean, 4),
        "warm_median_s": round(median, 4),
        "warm_p90_s": round(p90, 4),
        "warm_min_s": round(min(warm), 4),
        "warm_max_s": round(max(warm), 4),
        "warm_calls": warm_calls,
        "calls_per_s": round(1.0 / mean, 1) if mean > 0 else None,
        "parallelism": parallel_detail,
    }

    speed_note = f", {speedup:.1f}x batched" if speedup and speedup >= 1.2 else ""
    summary = (
        f"cold {cold_s*1000:.0f}ms, warm mean {mean*1000:.0f}ms "
        f"(~{details['calls_per_s']}/s){speed_note}"
    )

    if mean > warn_s:
        return CheckResult(
            "latency", CheckStatus.WARN,
            f"slow verification: warm mean {mean*1000:.0f}ms/call (>{warn_s*1000:.0f}ms). " + summary,
            score=mean, details=details,
        )
    return CheckResult("latency", CheckStatus.PASS, summary, score=mean, details=details)


from rlenv_audit.checks import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="latency",
    func=check_latency,
    description="time per verification call (cold vs warm) and batched-scoring behavior",
    needs_gpu=False,
    needs_docker=False,
)
