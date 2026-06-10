"""reward_design check — is the reward function *shaped* well for RL?

A reward can be deterministic and unexploitable and still be a bad training
signal. This check probes the reward function's behavior across a structured
battery of completions (gold answer, wrong answer, empty, garbage) on several
tasks and audits the *design*:

* **discrimination** — does a correct answer score above garbage? If not, the
  policy gradient points nowhere.
* **baseline floor** — does every response (even garbage) earn a constant
  positive reward? A flat floor dilutes the learnable signal.
* **signal type** — constant / binary / graded. Binary 0-or-1 is fine but means
  sparse signal at the frontier; graded partial credit usually trains better.
* **bounds** — rewards far outside [0, 1] complicate advantage normalization.
* **weights** — all-zero weights mean the reward is 0 by construction; negative
  weights are surfaced for review.

Each finding maps to a concrete recommendation (see REWARD_DESIGN.md for the
full design guide). No GPU, no model — pure probing of the rubric.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckStatus

_EPS = 1e-9
_GARBAGE = "qwzx plover 5821 nthe"


def _wrong_answer(ans: str) -> str:
    """A clearly-wrong answer in the same general shape as the gold one."""
    s = str(ans).strip()
    if s and all(c.isdigit() or c in "+-./, " for c in s):
        return "-987654"
    return s[::-1] + " xq" if s else "xq"


def _gold_texts(handle: EnvHandle, ans: str) -> list[tuple[str, str]]:
    out = []
    canonical = handle.canonical_answer(str(ans))
    if canonical:
        out.append(("gold_canonical", canonical))
    out.append(("gold_boxed", f"The answer is \\boxed{{{ans}}}."))
    out.append(("gold_plain", f"The answer is {ans}."))
    return out


def _wrong_texts(handle: EnvHandle, ans: str) -> list[tuple[str, str]]:
    wrong = _wrong_answer(ans)
    out = []
    canonical = handle.canonical_answer(wrong)
    if canonical:
        out.append(("wrong_canonical", canonical))
    out.append(("wrong_boxed", f"The answer is \\boxed{{{wrong}}}."))
    return out


def check_reward_design(handle: EnvHandle, config: dict) -> CheckResult:
    n_tasks = int(config.get("reward_design_tasks", 5))
    rows = handle.dataset(n=n_tasks)
    if not rows:
        return CheckResult(
            "reward_design", CheckStatus.SKIP, "environment exposes no dataset"
        )

    # ---- structural facts about the rubric -------------------------------
    func_names = handle.reward_func_names()
    try:
        weights = list(handle.rubric._get_reward_weights())
    except Exception:
        weights = []
    nonzero_weights = [w for w in weights if abs(w) > _EPS]
    negative_weights = [w for w in weights if w < 0]

    recommendations: list[str] = []
    warnings: list[str] = []

    if weights and not nonzero_weights:
        return CheckResult(
            "reward_design", CheckStatus.FAIL,
            "all reward weights are zero — the aggregate reward is 0 by construction",
            score=0.0,
            details={
                "reward_funcs": func_names,
                "weights": weights,
                "recommendations": [
                    "Give at least one reward function a non-zero weight; with all "
                    "weights zero the policy receives no signal at all "
                    "(REWARD_DESIGN.md §weights)."
                ],
            },
        )

    # ---- behavioral probe -------------------------------------------------
    per_task: list[dict] = []
    all_rewards: list[float] = []
    discriminating = 0
    probed = 0
    floors: list[float] = []
    golds: list[float] = []

    for i, row in enumerate(rows):
        ans = str(row["answer"]).strip()
        prompt, info, cols = row["prompt"], row["info"], row.get("raw", {})

        def score(text: str) -> float | None:
            try:
                r, _ = handle.score(text, prompt, row["answer"], info, cols)
                return r
            except ScoringError:
                return None

        probes: dict[str, float | None] = {}
        if ans:
            for label, text in _gold_texts(handle, ans):
                probes[label] = score(text)
            for label, text in _wrong_texts(handle, ans):
                probes[label] = score(text)
        probes["empty"] = score("")
        probes["garbage"] = score(_GARBAGE)

        scored = {k: v for k, v in probes.items() if v is not None}
        if not scored:
            per_task.append({"task": i, "error": "no probe could be scored"})
            continue

        probed += 1
        all_rewards.extend(scored.values())
        gold = max((v for k, v in scored.items() if k.startswith("gold")), default=None)
        base = max((v for k, v in scored.items() if k in ("empty", "garbage")), default=0.0)
        floors.append(base)
        if gold is not None:
            golds.append(gold)
            if gold > base + _EPS:
                discriminating += 1
        per_task.append(
            {"task": i, "rewards": {k: round(v, 6) for k, v in scored.items()},
             "gold_max": gold, "junk_max": base}
        )

    if probed == 0:
        return CheckResult(
            "reward_design", CheckStatus.SKIP,
            "no probe completion could be scored through the rubric",
            details={"per_task": per_task},
        )

    # ---- analysis ----------------------------------------------------------
    distinct = sorted({round(r, 6) for r in all_rewards})
    rmin, rmax = min(all_rewards), max(all_rewards)
    if len(distinct) == 1:
        signal_type = "constant"
    elif len(distinct) == 2:
        signal_type = "binary"
    else:
        signal_type = "graded"

    discrimination_rate = discriminating / probed if golds else None
    flat_floor = bool(floors) and min(floors) > _EPS

    # constant reward: every probe (gold, wrong, garbage) scored the same
    if signal_type == "constant":
        msg = f"reward is constant ({distinct[0]}) for every probe — gold, wrong, and garbage all score the same"
        recommendations.append(
            "The reward returns the same value for correct, wrong, and garbage "
            "completions. If scoring needs a live service (judge/API), document it; "
            "otherwise the verifier is not reading the completion "
            "(REWARD_DESIGN.md §discrimination)."
        )
        status = CheckStatus.FAIL if abs(distinct[0]) > _EPS else CheckStatus.WARN
        summary = msg if status is CheckStatus.FAIL else msg + " (all zero — may need a live judge/service)"
        return CheckResult(
            "reward_design", status, summary, score=0.0,
            details={
                "reward_funcs": func_names, "weights": weights,
                "signal_type": signal_type, "reward_range": [rmin, rmax],
                "per_task": per_task, "recommendations": recommendations,
            },
        )

    if discrimination_rate is not None and discrimination_rate < 0.5:
        warnings.append(
            f"reward separated gold from garbage on only {discriminating}/{probed} tasks"
        )
        recommendations.append(
            "Correct answers often score no better than garbage. Check the parser/"
            "matcher accepts the dataset's own gold answers — if the env can't "
            "reward its own answer key, a policy can't learn from it "
            "(REWARD_DESIGN.md §discrimination)."
        )

    if flat_floor:
        warnings.append(
            f"every response earns a baseline reward (floor {min(floors):.3g})"
        )
        recommendations.append(
            "Remove constant participation/format rewards or weight them near zero: "
            "a flat floor shrinks the relative advantage of actually solving the task "
            "(REWARD_DESIGN.md §baseline-floor)."
        )

    if rmax > 1.0 + 1e-6 or rmin < -1e-6:
        warnings.append(f"reward range [{rmin:.3g}, {rmax:.3g}] outside [0, 1]")
        recommendations.append(
            "Normalize the aggregate reward into [0, 1] (or document the range): "
            "unbounded rewards complicate advantage normalization and comparisons "
            "across environments (REWARD_DESIGN.md §bounds)."
        )

    if negative_weights:
        warnings.append(f"{len(negative_weights)} reward weight(s) are negative")
        recommendations.append(
            "Negative-weight penalties are easy to get backwards — confirm each "
            "penalty can't dominate the positive signal (REWARD_DESIGN.md §weights)."
        )

    if signal_type == "binary":
        recommendations.append(
            "Reward is strictly 0-or-1. That's valid, but graded partial credit "
            "(e.g. per-test-case or similarity-based) usually gives the policy a "
            "denser gradient on hard tasks (REWARD_DESIGN.md §partial-credit)."
        )

    details = {
        "reward_funcs": func_names,
        "weights": weights,
        "signal_type": signal_type,
        "reward_range": [rmin, rmax],
        "discrimination_rate": discrimination_rate,
        "baseline_floor": min(floors) if floors else None,
        "tasks_probed": probed,
        "per_task": per_task,
        "recommendations": recommendations,
    }

    if discrimination_rate is not None:
        score = discrimination_rate
    else:
        score = None

    if warnings:
        return CheckResult(
            "reward_design", CheckStatus.WARN,
            "; ".join(warnings) + f" (signal: {signal_type})",
            score=score, details=details,
        )
    return CheckResult(
        "reward_design", CheckStatus.PASS,
        f"{signal_type} reward, separates gold from garbage on "
        f"{discriminating}/{probed} tasks, range [{rmin:.2g}, {rmax:.2g}]",
        score=score, details=details,
    )


from rlenv_audit.checks.base import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="reward_design",
    func=check_reward_design,
    description="reward shape: discrimination, baseline floor, signal type, bounds, weights",
    needs_gpu=False,
    needs_docker=False,
)
