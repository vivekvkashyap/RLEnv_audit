"""determinism check — does the reward function return the same reward for the
same completion every time?

A reward that varies across identical re-scores injects pure noise into the
gradient: the policy chases randomness instead of signal. We take a small fixed
set of completions, derived from each task's own gold answer so the check is
env-agnostic, score each one several times, and FAIL if any reward moves.

No GPU. No model. Pure replay of the rubric.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckStatus

# Treat reward differences below this as float noise, not non-determinism.
_EPSILON = 1e-9


def _completions_for(answer: str) -> list[tuple[str, str]]:
    """A fixed battery of (label, completion-text) for one task.

    Spans the reward function's branches: a plausibly-correct answer (exercises
    the reward-awarding path), a clearly-wrong one, and an empty completion. We
    only care that each is scored *consistently*, not whether it's correct.
    """
    ans = str(answer)
    return [
        ("gold_boxed", f"The answer is \\boxed{{{ans}}}"),
        ("gold_plain", f"The final answer is {ans}."),
        ("wrong", "The answer is \\boxed{-999999}"),
        ("empty", ""),
    ]


def check_determinism(handle: EnvHandle, config: dict) -> CheckResult:
    n_prompts = int(config.get("determinism_prompts", 3))
    repeats = int(config.get("determinism_repeats", 5))

    rows = handle.dataset(n=n_prompts)
    if not rows:
        return CheckResult(
            "determinism",
            CheckStatus.SKIP,
            "environment exposes no dataset to build completions from",
        )

    findings: list[dict] = []
    scored_any = False
    nondeterministic: list[str] = []

    for i, row in enumerate(rows):
        prompt, answer, info = row["prompt"], row["answer"], row["info"]
        cols = row.get("raw", {})
        for label, text in _completions_for(answer):
            rewards: list[float] = []
            try:
                for _ in range(repeats):
                    reward, _metrics = handle.score(text, prompt, answer, info, cols)
                    rewards.append(reward)
            except ScoringError:
                # This completion couldn't be scored at all — note and move on.
                findings.append(
                    {"prompt": i, "completion": label, "error": "could not score"}
                )
                continue

            scored_any = True
            spread = max(rewards) - min(rewards)
            is_det = spread <= _EPSILON
            tag = f"prompt[{i}]/{label}"
            if not is_det:
                nondeterministic.append(tag)
            findings.append(
                {
                    "prompt": i,
                    "completion": label,
                    "rewards": rewards,
                    "spread": spread,
                    "deterministic": is_det,
                }
            )

    if not scored_any:
        return CheckResult(
            "determinism",
            CheckStatus.SKIP,
            "could not score any completion through the rubric",
            details={"findings": findings},
        )

    total = sum(1 for f in findings if "deterministic" in f)
    stable = sum(1 for f in findings if f.get("deterministic"))
    score = stable / total if total else 0.0
    details = {
        "n_prompts": len(rows),
        "repeats": repeats,
        "completions_scored": total,
        "nondeterministic": nondeterministic,
        "findings": findings,
    }

    if nondeterministic:
        return CheckResult(
            "determinism",
            CheckStatus.FAIL,
            f"{len(nondeterministic)}/{total} completions gave varying rewards "
            f"across {repeats} repeats (e.g. {nondeterministic[0]})",
            score=score,
            details=details,
        )
    return CheckResult(
        "determinism",
        CheckStatus.PASS,
        f"all {total} completions stable across {repeats} repeats",
        score=score,
        details=details,
    )


from rlenv_audit.checks import CheckSpec  # noqa: E402  (avoid circular import at top)

SPEC = CheckSpec(
    name="determinism",
    func=check_determinism,
    description="rewards are identical across repeated scoring of the same completion",
    needs_gpu=False,
    needs_docker=False,
)
