"""integrity check — is the environment structurally sound?

The cheapest check, and the only one that needs nothing but the loaded env: pure
introspection, no scoring, no model, no Docker. It answers the questions a
reviewer would ask in the first minute:

* does the env expose reward functions at all?
* does the dataset exist, and how big is it?
* do tasks actually carry answers (and how many are empty)?
* are there duplicate questions inflating the dataset?
* are prompts well-formed chat messages, and is there a system prompt that
  tells the model what format to answer in?

Because it never calls the reward function, it works on *every* env type —
including judge-based and multi-turn envs where scoring checks degrade.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle
from rlenv_audit.checks.base import CheckResult, CheckSpec, CheckStatus

_SAMPLE = 200          # rows to inspect
_EMPTY_ANSWER_WARN = 0.2   # warn if >20% of rows have no answer
_DUP_WARN = 0.05           # warn if >5% of questions are duplicates


def _question_key(row: dict) -> str:
    """A normalized identity for a task, for duplicate detection.

    Includes the answer: game/stateful envs legitimately repeat the same setup
    prompt with a different hidden answer per row — those are distinct tasks,
    not duplicates.
    """
    raw = row.get("raw", {})
    text = ""
    for key in ("question", "problem", "prompt", "text"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if not text:
        prompt = row.get("prompt")
        if isinstance(prompt, list):
            users = [m.get("content", "") for m in prompt
                     if isinstance(m, dict) and m.get("role") == "user"]
            text = " ".join(users)
        else:
            text = str(prompt)[:500]
    return " ".join(text.lower().split()) + "||" + str(row.get("answer", "")).strip().lower()


def _prompt_well_formed(prompt) -> bool:
    if isinstance(prompt, str):
        return bool(prompt.strip())
    if isinstance(prompt, list):
        return all(
            isinstance(m, dict) and m.get("role") and isinstance(m.get("content"), str)
            for m in prompt
        )
    return False


def check_integrity(handle: EnvHandle, config: dict) -> CheckResult:
    env_type = type(handle.env).__name__
    parser_type = type(handle.parser).__name__ if handle.parser is not None else None
    func_names = handle.reward_func_names()

    warnings: list[str] = []
    recommendations: list[str] = []

    # --- rubric -------------------------------------------------------------
    if not func_names:
        return CheckResult(
            "integrity", CheckStatus.FAIL,
            f"{env_type} exposes no reward functions — nothing can be scored",
            score=0.0,
            details={
                "env_type": env_type,
                "recommendations": [
                    "Attach at least one reward function to the rubric; without one "
                    "the environment cannot produce a training signal."
                ],
            },
        )

    # --- dataset ------------------------------------------------------------
    rows = handle.dataset(n=_SAMPLE)
    if not rows:
        return CheckResult(
            "integrity", CheckStatus.FAIL,
            f"{env_type} exposes no usable dataset (train and eval are both empty)",
            score=0.0,
            details={
                "env_type": env_type,
                "reward_funcs": func_names,
                "recommendations": [
                    "The env loaded but yields zero tasks — check the dataset source "
                    "and any default filters in load_environment() that may filter "
                    "everything out."
                ],
            },
        )

    n = len(rows)
    empty_answers = sum(1 for r in rows if not str(r["answer"]).strip())
    keys = [_question_key(r) for r in rows]
    dup_rate = 1.0 - (len(set(keys)) / n) if n else 0.0
    malformed = sum(1 for r in rows if not _prompt_well_formed(r["prompt"]))
    sys_prompt = handle.system_prompt()

    empty_rate = empty_answers / n
    if empty_rate > _EMPTY_ANSWER_WARN:
        warnings.append(f"{empty_answers}/{n} tasks have an empty answer field")
        recommendations.append(
            f"{empty_rate:.0%} of sampled tasks carry no `answer`. If scoring is "
            "answer-based those tasks can never reward; if scoring is judge-based, "
            "ignore this."
        )
    if dup_rate > _DUP_WARN:
        warnings.append(f"~{dup_rate:.0%} duplicate questions in the sampled dataset")
        recommendations.append(
            "Deduplicate the dataset: repeated tasks over-weight those problems in "
            "training and inflate apparent dataset size."
        )
    if malformed:
        warnings.append(f"{malformed}/{n} prompts are not well-formed chat messages")
        recommendations.append(
            "Some prompts are not valid `[{role, content}]` message lists — most "
            "inference clients will reject them."
        )
    if not sys_prompt:
        warnings.append("no system prompt found")
        recommendations.append(
            "No system prompt communicates the expected answer format to the model. "
            "State it explicitly (e.g. 'put your final answer in \\boxed{}') or the "
            "parser will miss real outputs (REWARD_DESIGN.md §parser-contract)."
        )

    details = {
        "env_type": env_type,
        "parser_type": parser_type,
        "reward_funcs": func_names,
        "rows_sampled": n,
        "empty_answer_rate": round(empty_rate, 3),
        "duplicate_rate": round(dup_rate, 3),
        "malformed_prompts": malformed,
        "system_prompt_present": bool(sys_prompt),
        "system_prompt_excerpt": (sys_prompt[:160] if sys_prompt else None),
        "recommendations": recommendations,
    }

    facts = f"{env_type}, {len(func_names)} reward fn(s), {n} tasks sampled"
    if warnings:
        return CheckResult(
            "integrity", CheckStatus.WARN,
            f"{facts}; " + "; ".join(warnings),
            score=1.0 - min(1.0, 0.25 * len(warnings)),
            details=details,
        )
    return CheckResult(
        "integrity", CheckStatus.PASS,
        f"{facts}, answers present, prompts well-formed, system prompt set",
        score=1.0, details=details,
    )


SPEC = CheckSpec(
    name="integrity",
    func=check_integrity,
    description="structural soundness: rubric/dataset present, answers, duplicates, prompts",
    needs_gpu=False,
    needs_docker=False,
)
