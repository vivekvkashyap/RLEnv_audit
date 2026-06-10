"""Model-generated, env-adaptive probe completions.

The static probe batteries in determinism/reward_design are math-shaped (boxed
answers, "The final answer is …"). That exercises the reward-awarding branch on
math/QA envs but not on code, SQL, JSON, or game envs, where a "gold completion"
looks completely different.

When a model endpoint is configured (same config as rollouts/design_review), we
ask the model ONCE per audit to read the env — system prompt, parser type, a few
sample tasks with their gold answers — and write realistic probe completions for
each task:

* ``gold``         — a full, realistic completion that should earn max reward
* ``gold_variant`` — a second correct completion, differently phrased
* ``wrong``        — plausible-looking, right format, incorrect answer

Generated gold probes are round-trip validated through the env's parser where
possible (does ``parse_answer(gold)`` recover the dataset's answer?), and the
validation result is carried on each probe. The battery is cached in the shared
``config`` dict so determinism and reward_design reuse one model call.

No endpoint, model failure, or unparseable output → returns ``{}`` and the
checks fall back to their static batteries. Strictly additive.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from rlenv_audit.adapters.verifiers import EnvHandle

_CACHE_KEY = "_generated_probes"

_PROMPT = """\
You are generating test completions to probe a reinforcement-learning \
environment's reward function. The environment gives a task to an AI assistant \
and scores the assistant's completion. For each task below, write THREE \
assistant completions:

1. "gold": a realistic, complete assistant response that fully solves the task \
and should earn MAXIMUM reward. It must present the expected answer in exactly \
the format the system prompt / task requires (tags, boxes, JSON, code block — \
whatever this environment expects).
2. "gold_variant": a second correct completion with noticeably different \
phrasing/reasoning, same expected answer, still in the required format.
3. "wrong": a plausible-looking completion in the correct format whose final \
answer is INCORRECT.

# Environment
- type: {env_type}
- parser: {parser_type}
- system prompt: {system_prompt}

# Tasks
{tasks}

Respond with ONLY a JSON object, no prose:
{{"tasks": [{{"index": 0, "gold": "...", "gold_variant": "...", "wrong": "..."}}, ...]}}
Keep each completion under 200 words.
"""


def _extract_json(text: str) -> dict | None:
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


def _validate_gold(handle: EnvHandle, text: str, answer: str) -> bool | None:
    """Round-trip a generated gold completion through the env's parser.

    Returns True/False, or None when validation isn't meaningful (no parser, or
    a pass-through parser that returns the whole message).
    """
    parser = handle.parser
    if parser is None:
        return None
    try:
        got = parser.parse_answer([{"role": "assistant", "content": text}])
    except Exception:
        return False
    if got is None:
        return False
    got_s, ans_s = str(got).strip(), str(answer).strip()
    if got_s == text.strip():  # pass-through parser — nothing was extracted
        return None
    return got_s.lower() == ans_s.lower()


def generate_probes(
    handle: EnvHandle,
    config: dict[str, Any],
    rows: list[dict],
) -> dict[int, list[dict]]:
    """Return ``{row_index: [{label, text, kind, validated}, ...]}``.

    Cached in ``config`` so multiple checks share one model call per audit.
    Empty dict means "not available — use the static battery".
    """
    cached = config.get(_CACHE_KEY)
    if cached is not None:
        return cached

    probes: dict[int, list[dict]] = {}
    config[_CACHE_KEY] = probes  # negative-cache by default; overwrite on success

    endpoint = config.get("endpoint") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if (not endpoint and not api_key) or not rows:
        return probes
    try:
        from openai import OpenAI
    except Exception:
        return probes

    n = min(len(rows), int(config.get("probe_tasks", 3)))
    task_lines = []
    for i in range(n):
        prompt_txt = json.dumps(str(rows[i]["prompt"]))[:600]
        task_lines.append(
            f'{i}. task: {prompt_txt}\n   expected answer: "{str(rows[i]["answer"])[:200]}"'
        )

    prompt = _PROMPT.format(
        env_type=type(handle.env).__name__,
        parser_type=type(handle.parser).__name__ if handle.parser else "None",
        system_prompt=(handle.system_prompt() or "(none)")[:1200],
        tasks="\n".join(task_lines),
    )

    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=120)
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
            max_tokens=2000,
        )
        data = _extract_json(resp.choices[0].message.content or "")
    except Exception:
        return probes
    if not data:
        return probes

    for entry in data.get("tasks", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n):
            continue
        answer = rows[idx]["answer"]
        batch = []
        for kind in ("gold", "gold_variant", "wrong"):
            text = entry.get(kind)
            if not isinstance(text, str) or not text.strip():
                continue
            validated = (
                _validate_gold(handle, text, answer) if kind.startswith("gold") else None
            )
            batch.append(
                {
                    "label": f"model_{kind}",
                    "text": text,
                    "kind": kind,
                    "validated": validated,
                }
            )
        if batch:
            probes[idx] = batch

    config[_CACHE_KEY] = probes
    if probes:
        config.setdefault("_probe_model", model)
    return probes
