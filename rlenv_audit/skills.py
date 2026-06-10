"""Skill-file-driven, env-adaptive input generation.

Several checks need *inputs* tailored to the environment under test — probe
completions for determinism/reward_design, cheat completions for exploits,
format variants for parser. A static, hand-written battery only fits math/QA
envs; it falls apart on code, SQL, JSON, or game environments whose answers look
nothing like ``\\boxed{42}``.

So each such check ships a **skill file** (``skills/<check>.md``): a prompt that
tells a model how to read this specific environment and write the inputs that
check needs. At run time we load the skill, append a structured description of
the actual environment (system prompt, parser, sample tasks + gold answers,
reward functions, optionally the reward source), send it to the configured
OpenAI-compatible endpoint, and parse the model's structured JSON back.

This is the bridge to "the user has Claude Code / Codex": point the audit at any
model endpoint and every skill-driven check generates exactly the inputs it
needs for *that* environment. No endpoint → the check SKIPs (these checks have
no static fallback, by design).

Results are cached per skill name in the shared ``config`` dict so a check never
generates twice in one audit.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rlenv_audit.adapters.verifiers import EnvHandle

_SKILL_DIR = Path(__file__).resolve().parent / "skills"


# --------------------------------------------------------------------------- io
def skill_path(name: str) -> Path:
    return _SKILL_DIR / f"{name}.md"


def load_skill(name: str) -> str | None:
    """Return the skill instructions for ``name``, or None if the file is absent."""
    path = skill_path(name)
    try:
        return path.read_text()
    except Exception:
        return None


def endpoint_configured(config: dict) -> bool:
    return bool(
        config.get("endpoint")
        or config.get("api_key")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
    )


# ----------------------------------------------------------------- env context
def _env_context(handle: EnvHandle, rows: list[dict], include_source: bool) -> str:
    try:
        weights = list(handle.rubric._get_reward_weights()) if handle.rubric else []
    except Exception:
        weights = []
    names = handle.reward_func_names()
    funcs = ", ".join(
        f"{n} (w={weights[i]})" if i < len(weights) else n for i, n in enumerate(names)
    ) or "(none)"

    lines = [
        "## Environment under test",
        f"- id: {handle.env_id}",
        f"- type: {type(handle.env).__name__}",
        f"- parser: {type(handle.parser).__name__ if handle.parser else 'None'}",
        f"- reward functions: {funcs}",
        "",
        "## System prompt",
        (handle.system_prompt() or "(none)")[:1500],
        "",
        "## Sample tasks (with their gold answers)",
    ]
    for i, row in enumerate(rows):
        lines.append(f"### task {i}")
        lines.append(f"prompt: {json.dumps(str(row['prompt']))[:700]}")
        lines.append(f"gold answer: {json.dumps(str(row['answer']))[:300]}")
    if include_source:
        srcs = handle.reward_sources()
        if srcs:
            lines.append("")
            lines.append("## Reward function source")
            for name, src in srcs.items():
                lines.append(f"### {name}\n```python\n{src}\n```")
    return "\n".join(lines)


def _client_and_model(config: dict):
    from openai import OpenAI

    endpoint = config.get("endpoint") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=180)
    model = config.get("model")
    if not model:
        try:
            model = client.models.list().data[0].id
        except Exception:
            model = "gpt-4o-mini"
    return client, model


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
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


# ------------------------------------------------------------------- the runner
def run_skill(
    handle: EnvHandle,
    config: dict[str, Any],
    name: str,
    rows: list[dict],
    *,
    include_source: bool = False,
    max_tokens: int = 3000,
) -> dict | None:
    """Run skill ``name`` against the environment; return the model's parsed JSON.

    Returns ``None`` when generation can't happen (no endpoint, no skill file,
    no rows, client/model error, or unparseable output) — the caller then SKIPs.
    Cached per skill name in ``config`` so each skill runs at most once per audit.
    """
    cache_key = f"_skill_{name}"
    if cache_key in config:
        return config[cache_key]
    config[cache_key] = None  # negative-cache; overwrite on success

    if not endpoint_configured(config) or not rows:
        return None
    instructions = load_skill(name)
    if not instructions:
        return None
    try:
        from openai import OpenAI  # noqa: F401
    except Exception:
        return None

    prompt = (
        instructions
        + "\n\n"
        + _env_context(handle, rows, include_source)
        + "\n\nReturn ONLY the JSON object specified above. No prose, no code fence."
    )
    try:
        client, model = _client_and_model(config)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        data = _extract_json(resp.choices[0].message.content or "")
    except Exception:
        return None
    if not data:
        return None

    data["_model"] = model
    config[cache_key] = data
    config.setdefault("_skill_model", model)
    return data


def validate_gold(handle: EnvHandle, text: str, answer: str) -> bool | None:
    """Round-trip a generated gold completion through the env's parser.

    True/False, or None when validation isn't meaningful (no parser, or a
    pass-through parser that returns the whole message unchanged).
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
    got_s = str(got).strip()
    if got_s == text.strip():  # pass-through parser — nothing extracted
        return None
    return got_s.lower() == str(answer).strip().lower()
