"""Mechanical tools the audit *skills* call.

env_audit's checks are judgment-heavy and run by an agent (Claude Code / Codex)
via skill files (see ``skills/``). This module is the deterministic backbone the
skills lean on for the parts that must be exact: loading the environment,
introspecting it, calling the reward function on agent-written completions,
running and caching a shared set of rollouts, and rendering the final scorecard.

Everything here is pure data-in / JSON-out so a skill can shell out to
``rlenv-audit <tool> ...`` and read the result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from rlenv_audit.adapters.verifiers import ScoringError, load_handle

CACHE_DIR = Path(os.environ.get("RLENV_AUDIT_CACHE", ".rlenv_audit_cache"))


# --------------------------------------------------------------------- inspect
def inspect_env(env_id: str, n: int = 20) -> dict[str, Any]:
    """Load an environment and return a structured description of it.

    Consumed by the integrity, problem-alignment, reward-design and contamination
    skills. Captures load status so the integrity check sees failures as data,
    not exceptions.
    """
    try:
        handle = load_handle(env_id)
    except Exception as exc:
        return {"env_id": env_id, "loaded": False, "error": str(exc)}

    try:
        try:
            weights = list(handle.rubric._get_reward_weights()) if handle.rubric else []
        except Exception:
            weights = []
        names = handle.reward_func_names()
        sources = handle.reward_sources()
        reward_funcs = [
            {
                "name": nm,
                "weight": weights[i] if i < len(weights) else None,
                "source": sources.get(nm, "<source unavailable>"),
            }
            for i, nm in enumerate(names)
        ]
        rows = handle.dataset(n=n)
        sample = [
            {
                "index": i,
                "prompt": r["prompt"],
                "answer": r["answer"],
                "info": r["info"],
                "extra_columns": {
                    k: v for k, v in (r.get("raw") or {}).items()
                    if k not in ("prompt", "answer", "info", "example_id")
                },
            }
            for i, r in enumerate(rows)
        ]
        return {
            "env_id": env_id,
            "loaded": True,
            "env_type": type(handle.env).__name__,
            "parser_type": type(handle.parser).__name__ if handle.parser else None,
            "module_file": handle.module_file(),
            "dataset_size": handle.dataset_size(),
            "system_prompt": handle.system_prompt(),
            "reward_funcs": reward_funcs,
            "sample_size": len(sample),
            "sample": sample,
        }
    finally:
        handle.teardown()


# ----------------------------------------------------------------------- score
def score_completions(env_id: str, completions: list[dict]) -> dict[str, Any]:
    """Score agent-written completions through the env's reward function.

    ``completions`` is a list of ``{"prompt_index": int, "label": str,
    "text": str}``. Each is scored against that dataset row. Returns the same
    list with ``reward`` and ``metrics`` filled in (or an ``error``). This is the
    reward-design check's measurement step.
    """
    handle = load_handle(env_id)
    try:
        max_idx = max((int(c.get("prompt_index", 0)) for c in completions), default=0)
        rows = handle.dataset(n=max_idx + 1)
        if not rows:
            return {"env_id": env_id, "error": "environment exposes no dataset"}
        results = []
        for c in completions:
            idx = max(0, min(int(c.get("prompt_index", 0)), len(rows) - 1))
            row = rows[idx]
            entry = {"prompt_index": idx, "label": c.get("label", ""), "text": c.get("text", "")}
            try:
                reward, metrics = handle.score(
                    c.get("text", ""), row["prompt"], row["answer"], row["info"], row.get("raw")
                )
                entry["reward"] = reward
                entry["metrics"] = metrics
            except ScoringError as exc:
                entry["error"] = str(exc)
            results.append(entry)
        return {"env_id": env_id, "n": len(results), "results": results}
    finally:
        handle.teardown()


# -------------------------------------------------------------------- rollouts
def _chat_messages(prompt) -> list[dict] | None:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        out = []
        for m in prompt:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                return None
            out.append({"role": m.get("role"), "content": m["content"]})
        return out
    return None


def run_rollouts(
    env_id: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    n_samples: int = 20,
    k: int = 8,
    max_tokens: int = 1024,
    dummy: bool = False,
    cache_path: str | None = None,
) -> dict[str, Any]:
    """Run (or fake, with ``dummy``) ``k`` rollouts over ``n_samples`` tasks once,
    score them, time them, and cache to JSON.

    Both the latency and rollout-quality skills read this single cache rather than
    each generating their own rollouts. Returns the cached object.
    """
    handle = load_handle(env_id)
    try:
        rows = handle.dataset(n=n_samples)
        if not rows:
            return {"env_id": env_id, "error": "environment exposes no dataset"}

        endpoint = endpoint or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        client = None
        if not dummy:
            if not endpoint and not api_key:
                return {"env_id": env_id, "error": "no model endpoint configured and --dummy not set"}
            from openai import OpenAI

            client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=180)
            if not model:
                try:
                    model = client.models.list().data[0].id
                except Exception:
                    model = "gpt-4o-mini"
        else:
            model = model or "dummy"

        samples = []
        latencies: list[float] = []
        for i, row in enumerate(rows):
            messages = _chat_messages(row["prompt"])
            rollouts = []
            for j in range(k):
                if dummy:
                    text, dt, err = f"(dummy rollout {j} for task {i})", 0.0, None
                elif messages is None:
                    text, dt, err = "", 0.0, "prompt is not chat messages"
                else:
                    t0 = time.perf_counter()
                    try:
                        resp = client.chat.completions.create(
                            model=model, messages=messages,
                            max_tokens=max_tokens, temperature=0.8,
                        )
                        text = resp.choices[0].message.content or ""
                        err = None
                    except Exception as exc:
                        text, err = "", f"{type(exc).__name__}: {exc}"[:200]
                    dt = time.perf_counter() - t0
                    latencies.append(dt)
                try:
                    reward, _ = handle.score(text, row["prompt"], row["answer"], row["info"], row.get("raw"))
                except ScoringError:
                    reward = None
                rollouts.append({"text": text, "latency_s": round(dt, 4), "reward": reward, "error": err})
            samples.append(
                {
                    "index": i,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "rollouts": rollouts,
                }
            )

        timing = {}
        if latencies:
            s = sorted(latencies)
            timing = {
                "calls": len(s),
                "mean_s": round(sum(s) / len(s), 3),
                "p50_s": round(s[len(s) // 2], 3),
                "p90_s": round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
                "max_s": round(s[-1], 3),
                "total_s": round(sum(s), 3),
            }
        out = {
            "env_id": env_id, "model": model, "endpoint": endpoint or "api.openai.com",
            "dummy": dummy, "n_samples": len(samples), "k": k,
            "timing": timing, "samples": samples,
        }

        path = Path(cache_path) if cache_path else CACHE_DIR / f"rollouts_{env_id.replace('/', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        out["cache_path"] = str(path)
        return out
    finally:
        handle.teardown()


# ------------------------------------------------------------------- scorecard
_STATUS_STYLE = {
    "PASS": "bold green", "WARN": "bold yellow", "FAIL": "bold red",
    "N/A": "dim", "SKIP": "dim", "ERROR": "bold red",
}
# Rating weights. Latency (informational) and contamination (user-opt-in,
# often expected for eval-style envs) count half as much as the core checks.
CHECK_WEIGHTS = {"latency": 0.5, "contamination": 0.5}


def build_scorecard(data: dict) -> dict:
    """Compute overall grade + rating from a ``{env_id, checks:[...], feedback?}``
    result.

    Each check is ``{name, status, score (0-10|null), justification}``. The
    rating (0-10) is the weighted average of the checks that actually ran (N/A
    and null-score checks are excluded); latency and contamination weigh 0.5,
    every other check 1.0.
    """
    checks = data.get("checks", [])
    scored = [
        (c["score"], CHECK_WEIGHTS.get(c.get("name"), 1.0))
        for c in checks
        if isinstance(c.get("score"), (int, float))
    ]
    total_w = sum(w for _, w in scored)
    rating = round(sum(s * w for s, w in scored) / total_w, 1) if total_w else None
    statuses = {c.get("status") for c in checks}
    if "FAIL" in statuses or "ERROR" in statuses:
        grade = "FAIL"
    elif "WARN" in statuses:
        grade = "WARN"
    elif any(s == "PASS" for s in statuses):
        grade = "PASS"
    else:
        grade = "INCONCLUSIVE"
    return {
        "env_id": data.get("env_id"),
        "grade": grade,
        "rating": rating,
        "checks": checks,
        "feedback": data.get("feedback"),
    }


def render_scorecard(data: dict) -> None:
    """Print the scorecard as a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    card = build_scorecard(data)
    console = Console()
    table = Table(title=f"rlenv_audit · {card['env_id']}", title_style="bold")
    table.add_column("check", style="bold", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("justification")
    for c in card["checks"]:
        status = str(c.get("status", "?"))
        score = c.get("score")
        table.add_row(
            c.get("name", "?"),
            Text(status, style=_STATUS_STYLE.get(status, "")),
            ("—" if not isinstance(score, (int, float)) else f"{score:.1f}"),
            c.get("justification", ""),
        )
    console.print(table)
    if card["rating"] is not None:
        console.print(Text.assemble("overall: ", Text(card["grade"], style=_STATUS_STYLE.get(card["grade"], "bold")),
                                     "   rating: ", Text(f"{card['rating']}/10", style="bold")))
    else:
        console.print(Text(f"overall: {card['grade']}   rating: N/A", style="bold"))
    if card.get("feedback"):
        console.print(Text("\nfeedback", style="bold"))
        console.print(str(card["feedback"]).strip())
