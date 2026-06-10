"""determinism check — does the reward function return the same reward for the
same completion every time?

A reward that varies across identical re-scores injects pure noise into the
gradient: the policy chases randomness instead of signal.

The probe completions are **generated per-environment by a model** following the
``skills/determinism.md`` skill: it reads this env's system prompt, parser, and
sample tasks and writes ~20 diverse completions (gold, rewritten gold, wrong,
edge cases) in the environment's own answer format. That makes the check work on
*any* env — code, SQL, JSON, games — not just boxed-math. Each probe is then
scored several times and we check the reward never moves.

Needs a model endpoint (``--endpoint`` / ``OPENAI_API_KEY``). Without one there
is no battery to score, so the check SKIPs — there is no static fallback by
design.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckSpec, CheckStatus
from rlenv_audit.skills import endpoint_configured, run_skill

# Treat reward differences below this as float noise, not non-determinism.
_EPSILON = 1e-9


def check_determinism(handle: EnvHandle, config: dict) -> CheckResult:
    repeats = int(config.get("determinism_repeats", 5))
    n_tasks = int(config.get("determinism_tasks", 5))

    if not endpoint_configured(config):
        return CheckResult(
            "determinism", CheckStatus.SKIP,
            "needs a model endpoint to generate probe completions "
            "(set OPENAI_API_KEY / OPENAI_BASE_URL or pass --endpoint)",
        )

    rows = handle.dataset(n=n_tasks)
    if not rows:
        return CheckResult(
            "determinism", CheckStatus.SKIP,
            "environment exposes no dataset to build completions from",
        )

    skill = run_skill(handle, config, "determinism", rows)
    probes = (skill or {}).get("probes") or []
    probes = [p for p in probes if isinstance(p, dict) and isinstance(p.get("text"), str)]
    if not probes:
        return CheckResult(
            "determinism", CheckStatus.SKIP,
            "model did not generate any probe completions for this environment",
            details={"raw": skill},
        )

    findings: list[dict] = []
    scored_any = False
    nondeterministic: list[str] = []

    for p in probes:
        try:
            ti = int(p.get("task_index", 0))
        except (TypeError, ValueError):
            ti = 0
        ti = max(0, min(ti, len(rows) - 1))
        row = rows[ti]
        label = f"{p.get('kind', 'probe')}:{p.get('label', ti)}"
        text = p["text"]

        rewards: list[float] = []
        try:
            for _ in range(repeats):
                reward, _metrics = handle.score(
                    text, row["prompt"], row["answer"], row["info"], row.get("raw")
                )
                rewards.append(reward)
        except ScoringError:
            findings.append({"probe": label, "task": ti, "error": "could not score"})
            continue

        scored_any = True
        spread = max(rewards) - min(rewards)
        is_det = spread <= _EPSILON
        if not is_det:
            nondeterministic.append(label)
        findings.append(
            {
                "probe": label,
                "task": ti,
                "kind": p.get("kind"),
                "rewards": rewards,
                "spread": spread,
                "deterministic": is_det,
            }
        )

    if not scored_any:
        return CheckResult(
            "determinism", CheckStatus.SKIP,
            "none of the generated probes could be scored through the rubric",
            details={"findings": findings},
        )

    total = sum(1 for f in findings if "deterministic" in f)
    stable = sum(1 for f in findings if f.get("deterministic"))
    score = stable / total if total else 0.0
    details = {
        "model": (skill or {}).get("_model"),
        "repeats": repeats,
        "probes_generated": len(probes),
        "probes_scored": total,
        "kinds": sorted({p.get("kind") for p in probes if p.get("kind")}),
        "nondeterministic": nondeterministic,
        "findings": findings,
    }

    if nondeterministic:
        details["recommendations"] = [
            "Make the reward function deterministic: seed any RNG, avoid wall-clock "
            "or network calls in scoring, and pin judge temperature to 0. A reward "
            "that varies across identical re-scores injects pure noise into the "
            "gradient (REWARD_DESIGN.md §determinism)."
        ]
        return CheckResult(
            "determinism", CheckStatus.FAIL,
            f"{len(nondeterministic)}/{total} probes gave varying rewards across "
            f"{repeats} repeats (e.g. {nondeterministic[0]})",
            score=score, details=details,
        )
    return CheckResult(
        "determinism", CheckStatus.PASS,
        f"all {total} model-generated probes stable across {repeats} repeats",
        score=score, details=details,
    )


SPEC = CheckSpec(
    name="determinism",
    func=check_determinism,
    description="rewards are identical across repeated scoring of the same completion",
    needs_gpu=False,
    needs_docker=False,
    needs_model=True,
)
