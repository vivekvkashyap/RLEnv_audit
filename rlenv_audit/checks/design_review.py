"""design_review check — an LLM reads the env's actual code against the guide.

The behavioral checks probe the reward from the *outside*. This one looks at it
from the *inside*: it hands a model the environment's reward-function source
code, system prompt, parser type, and a few sample tasks, together with the
REWARD_DESIGN.md design guide, and asks for a structured expert review — the
kind of issues only reading the code reveals (a try/except that swallows scoring
errors into 0.0, a judge prompt that can be gamed, a regex that anchors wrong, a
timeout that varies by machine).

Needs the same OpenAI-compatible endpoint as the rollouts check; SKIPs without
one. Findings come back as structured JSON (severity, issue, recommendation) and
land in the scorecard like any other check's.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rlenv_audit.adapters.verifiers import EnvHandle
from rlenv_audit.checks.base import CheckResult, CheckSpec, CheckStatus

# Compact fallback if the full REWARD_DESIGN.md isn't on disk (pip install).
_GUIDE_FALLBACK = """\
§determinism: same completion must always score the same (seed RNG, no clocks, judge temp 0).
§discrimination: a correct answer must out-score empty/garbage; the env must reward its own answer key.
§baseline-floor: don't give every response a constant participation/format reward.
§partial-credit: graded reward beats strict 0/1 on hard tasks (denser gradient).
§bounds: keep aggregate reward in [0,1] or document the range.
§weights: at least one non-zero weight; penalties must not dominate or invert the signal.
§anti-hacking: never trust exit codes alone; keep expected outputs out of the exec dir;
  re-assert after the submission runs; reject empty/no-op submissions; beware swallowed
  exceptions that silently return 0 or 1.
§parser-contract: parse what models actually emit (strip, case-insensitive, last occurrence);
  state the required format in the system prompt.
§difficulty-curriculum: avoid all-pass/all-fail batches; mix difficulties.
§contamination: don't train on tasks that overlap reported benchmarks.
"""

_PROMPT_TEMPLATE = """\
You are an expert reviewer of reinforcement-learning environments (the `verifiers` \
format). Below are design guidelines, then the actual implementation artifacts of \
one environment. Review the implementation against the guidelines.

Focus on issues only visible by READING THE CODE: swallowed exceptions that turn \
errors into a fixed reward, gameable judge prompts, fragile regexes/anchors, \
machine-dependent timeouts, reward branches that can't be reached, mismatch \
between the system prompt's promised format and what the parser/reward expects.

# Design guidelines
{guide}

# Environment: {env_id}
- env type: {env_type}
- parser: {parser_type}
- reward functions and weights: {funcs}

## System prompt
{system_prompt}

## Reward function source
{sources}

## Sample tasks (prompt -> expected answer)
{samples}

Respond with ONLY a JSON object, no prose, in this exact shape:
{{
  "issues": [
    {{"severity": "high|medium|low", "section": "<guideline section>",
      "finding": "<one-sentence issue>", "recommendation": "<one-sentence fix>"}}
  ],
  "strengths": ["<one sentence each>"],
  "verdict": "<one-sentence overall assessment>"
}}
List at most 6 issues, highest severity first. If the implementation is sound, \
return an empty issues list. Do not invent issues you cannot point to in the code.
"""


def _load_guide() -> str:
    for candidate in (
        Path(__file__).resolve().parents[2] / "REWARD_DESIGN.md",
        Path.cwd() / "REWARD_DESIGN.md",
    ):
        try:
            if candidate.exists():
                return candidate.read_text()[:12000]
        except Exception:
            continue
    return _GUIDE_FALLBACK


def _extract_json(text: str) -> dict | None:
    """Parse the model's JSON, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def check_design_review(handle: EnvHandle, config: dict) -> CheckResult:
    endpoint = config.get("endpoint") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not endpoint and not api_key:
        return CheckResult(
            "design_review", CheckStatus.SKIP,
            "no model endpoint configured (set OPENAI_API_KEY / OPENAI_BASE_URL or "
            "pass --endpoint) — code review needs a model",
        )
    try:
        from openai import OpenAI
    except Exception:
        return CheckResult(
            "design_review", CheckStatus.SKIP, "openai client package not installed"
        )

    # ---- gather the evidence ------------------------------------------------
    sources = handle.reward_sources()
    if not sources or all(s == "<source unavailable>" for s in sources.values()):
        return CheckResult(
            "design_review", CheckStatus.SKIP,
            "could not retrieve any reward-function source code to review",
        )

    try:
        weights = list(handle.rubric._get_reward_weights())
    except Exception:
        weights = []
    funcs_desc = ", ".join(
        f"{name} (w={weights[i]})" if i < len(weights) else name
        for i, name in enumerate(sources)
    )

    rows = handle.dataset(n=2)
    samples = "\n".join(
        f"- Q: {json.dumps(str(r['prompt']))[:400]}\n  A: {str(r['answer'])[:120]}"
        for r in rows
    ) or "(no dataset rows available)"

    src_block = "\n\n".join(f"### {name}\n```python\n{src}\n```" for name, src in sources.items())
    prompt = _PROMPT_TEMPLATE.format(
        guide=_load_guide(),
        env_id=handle.env_id,
        env_type=type(handle.env).__name__,
        parser_type=type(handle.parser).__name__ if handle.parser else "None",
        funcs=funcs_desc,
        system_prompt=(handle.system_prompt() or "(none)")[:1500],
        sources=src_block[:24000],
        samples=samples,
    )

    # ---- ask the model --------------------------------------------------------
    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=180)
    model = config.get("model")
    if not model:
        try:
            model = client.models.list().data[0].id
        except Exception:
            model = "gpt-4o-mini"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:
        return CheckResult(
            "design_review", CheckStatus.SKIP,
            f"model call failed ({model}): {exc}",
        )

    review = _extract_json(text)
    if review is None:
        return CheckResult(
            "design_review", CheckStatus.SKIP,
            f"model ({model}) did not return parseable JSON review",
            details={"raw_response": text[:1000]},
        )

    issues = [i for i in review.get("issues", []) if isinstance(i, dict)]
    high = [i for i in issues if i.get("severity") == "high"]
    medium = [i for i in issues if i.get("severity") == "medium"]
    recommendations = [
        f"[{i.get('severity', '?')}] {i.get('finding', '')} — {i.get('recommendation', '')} "
        f"(REWARD_DESIGN.md {i.get('section', '')})"
        for i in issues
    ]

    details = {
        "model": model,
        "reviewed_funcs": list(sources),
        "issues": issues,
        "strengths": review.get("strengths", []),
        "verdict": review.get("verdict", ""),
        "recommendations": recommendations,
    }
    score = max(0.0, 1.0 - 0.4 * len(high) - 0.15 * len(medium))

    if high:
        return CheckResult(
            "design_review", CheckStatus.WARN,
            f"model review ({model}) found {len(high)} high / {len(medium)} medium "
            f"issue(s): {high[0].get('finding', '')[:90]}",
            score=score, details=details,
        )
    if issues:
        return CheckResult(
            "design_review", CheckStatus.PASS,
            f"model review ({model}): no high-severity issues; "
            f"{len(issues)} minor note(s)",
            score=score, details=details,
        )
    return CheckResult(
        "design_review", CheckStatus.PASS,
        f"model review ({model}): no design issues found — "
        f"{review.get('verdict', 'implementation looks sound')[:100]}",
        score=score, details=details,
    )


SPEC = CheckSpec(
    name="design_review",
    func=check_design_review,
    description="LLM reads the reward source + prompts against REWARD_DESIGN.md",
    needs_gpu=False,
    needs_docker=False,
    needs_model=True,
)
