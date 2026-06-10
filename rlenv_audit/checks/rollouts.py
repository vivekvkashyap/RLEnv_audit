"""rollouts check — does the full pipeline work on *real* model outputs?

Every other CPU check probes the env with synthetic completions. This one runs a
couple of genuine mini-rollouts: sample completions from a real model via any
OpenAI-compatible endpoint (OpenAI itself, or a local vLLM / llama.cpp server),
push them through the env's parser + reward, and verify the pipeline end-to-end:

* the parser extracts *something* from real model text (synthetic probes can't
  catch a parser tuned to a format models never actually produce);
* the rewards on real outputs aren't all identical (zero variance = zero
  learning signal for this model at this difficulty);
* nothing in the scoring path chokes on realistic, messy completions.

Configuration: ``--endpoint <url>`` / ``--model <name>`` on the CLI, or the
standard ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` env vars. SKIPs cleanly when no
endpoint is configured — it never fails an audit over missing credentials.
"""

from __future__ import annotations

import os

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckStatus

_EPS = 1e-9


def _norm_messages(prompt) -> list[dict] | None:
    """Coerce a dataset prompt into plain chat messages for the API."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        out = []
        for m in prompt:
            if not isinstance(m, dict):
                return None
            role, content = m.get("role"), m.get("content")
            if not isinstance(content, str):
                return None
            out.append({"role": role, "content": content})
        return out
    return None


def check_rollouts(handle: EnvHandle, config: dict) -> CheckResult:
    endpoint = config.get("endpoint") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not endpoint and not api_key:
        return CheckResult(
            "rollouts", CheckStatus.SKIP,
            "no model endpoint configured (set OPENAI_API_KEY / OPENAI_BASE_URL or "
            "pass --endpoint for a local vLLM server)",
        )

    try:
        from openai import OpenAI
    except Exception:
        return CheckResult(
            "rollouts", CheckStatus.SKIP, "openai client package not installed"
        )

    n_tasks = int(config.get("rollouts_tasks", 4))
    k = int(config.get("rollouts_samples", 2))
    max_tokens = int(config.get("rollouts_max_tokens", 1024))

    rows = handle.dataset(n=n_tasks)
    if not rows:
        return CheckResult("rollouts", CheckStatus.SKIP, "environment exposes no dataset")

    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=120)
    model = config.get("model")
    if not model:
        try:
            model = client.models.list().data[0].id
        except Exception:
            model = "gpt-4o-mini"

    rewards: list[float] = []
    extracted = 0
    samples: list[dict] = []
    errors: list[str] = []

    for i, row in enumerate(rows):
        messages = _norm_messages(row["prompt"])
        if messages is None:
            continue
        for _ in range(k):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=float(config.get("temperature", 0.8)),
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:160])
                continue

            parsed = None
            if handle.parser is not None:
                try:
                    parsed = handle.parser.parse_answer(
                        [{"role": "assistant", "content": text}]
                    )
                except Exception:
                    parsed = None
            if parsed is not None and str(parsed).strip():
                extracted += 1

            try:
                reward, _ = handle.score(
                    text, row["prompt"], row["answer"], row["info"], row.get("raw")
                )
            except ScoringError as exc:
                errors.append(f"scoring failed: {exc}"[:160])
                continue
            rewards.append(reward)
            samples.append(
                {
                    "task": i,
                    "reward": reward,
                    "parsed": (str(parsed)[:80] if parsed is not None else None),
                    "completion_excerpt": " ".join(text.split())[:140],
                }
            )

    if not rewards:
        reason = errors[0] if errors else "no completion could be generated/scored"
        return CheckResult(
            "rollouts", CheckStatus.SKIP,
            f"could not complete any rollout via {model}: {reason}",
            details={"model": model, "errors": errors},
        )

    mean = sum(rewards) / len(rewards)
    distinct = {round(r, 6) for r in rewards}
    extraction_rate = extracted / len(rewards)
    recommendations: list[str] = []
    warnings: list[str] = []

    if extraction_rate == 0:
        warnings.append("parser extracted nothing from any real model output")
        recommendations.append(
            "The parser found no answer in any real model completion — the expected "
            "output format likely isn't conveyed by the system prompt. State the "
            "format explicitly in the prompt or relax the parser "
            "(REWARD_DESIGN.md §parser-contract)."
        )
    if len(distinct) == 1:
        warnings.append(
            f"all {len(rewards)} rollouts scored identically ({rewards[0]:.3g})"
        )
        recommendations.append(
            "Every rollout earned the same reward — at this difficulty the policy "
            "gets zero gradient from this model. Mix difficulties or add partial "
            "credit (REWARD_DESIGN.md §difficulty-curriculum)."
        )

    details = {
        "model": model,
        "endpoint": endpoint or "api.openai.com",
        "n_rollouts": len(rewards),
        "reward_mean": round(mean, 4),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "parser_extraction_rate": round(extraction_rate, 3),
        "samples": samples,
        "errors": errors,
        "recommendations": recommendations,
    }

    if warnings:
        return CheckResult(
            "rollouts", CheckStatus.WARN, "; ".join(warnings) + f" (model: {model})",
            score=mean, details=details,
        )
    return CheckResult(
        "rollouts", CheckStatus.PASS,
        f"{len(rewards)} rollouts via {model}: mean reward {mean:.2f}, "
        f"parser extracted {extraction_rate:.0%}",
        score=mean, details=details,
    )


from rlenv_audit.checks.base import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="rollouts",
    func=check_rollouts,
    description="end-to-end mini-rollouts with a real model via an OpenAI-compatible endpoint",
    needs_gpu=False,
    needs_docker=False,
    needs_model=True,
)
