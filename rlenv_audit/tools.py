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
import shutil
from pathlib import Path
from typing import Any

from rlenv_audit.adapters.verifiers import DatasetLoadError, ScoringError, load_handle

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
        dataset_error = None
        try:
            rows = handle.dataset(n=n)
        except DatasetLoadError as exc:
            rows = []
            dataset_error = str(exc)
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
        result = {
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
        if dataset_error:
            result["dataset_error"] = dataset_error
        return result
    finally:
        handle.teardown()


# ----------------------------------------------------------------------- score
def score_completions(
    env_id: str, completions: list[dict], sandbox: str = "auto"
) -> dict[str, Any]:
    """Score agent-written completions through the env's reward function.

    ``completions`` is a list of ``{"prompt_index": int, "label": str,
    "text": str}``. Each is scored against that dataset row. Returns the same
    list with ``reward`` and ``metrics`` filled in (or an ``error``). This is the
    reward-design check's measurement step.

    ``sandbox`` controls *where* scoring runs, because for a code/agentic env the
    rubric **executes** the completion — model-written code we must not run on the
    host (see ``sandbox.py``):

    * ``"auto"`` (default) — score inside a locked-down Docker container when
      Docker is reachable; otherwise fall back to host scoring and flag it.
    * ``"on"`` — require the sandbox; if Docker is unavailable, score nothing and
      return a backend-unavailable result (so the caller can mark the check N/A
      rather than executing untrusted code on the host).
    * ``"off"`` — score on the host (the old behaviour; fine for non-code envs).

    Every return carries a top-level ``sandbox`` block
    (``{requested, used, available, reason}``), and every per-entry ``error``
    comes with an ``error_kind`` — ``"reward"`` (the rubric ran and raised) or
    ``"infra"`` (the completion could not be executed at all) — so the
    reward-design skill can tell "the reward function is broken" apart from
    "no execution backend here" without parsing error strings.
    """
    use_sandbox, sb = _decide_sandbox(sandbox)
    # Required isolation but no backend: refuse before even loading the env —
    # never execute untrusted completions on the host. This is the signal the
    # reward-design check turns into N/A, not FAIL.
    if sb["requested"] == "on" and not use_sandbox:
        return {
            "env_id": env_id, "n": 0, "results": [], "sandbox": sb,
            "error": f"sandbox required but unavailable: {sb['reason']}",
        }

    handle = load_handle(env_id)
    try:
        max_idx = max((int(c.get("prompt_index", 0)) for c in completions), default=0)
        try:
            rows = handle.dataset(n=max_idx + 1)
        except DatasetLoadError as exc:
            return {"env_id": env_id, "error": str(exc), "sandbox": sb}
        if not rows:
            return {"env_id": env_id, "error": "environment exposes no dataset", "sandbox": sb}

        if use_sandbox:
            results = _score_in_sandbox(env_id, completions, rows)
        else:
            results = _score_on_host(handle, completions, rows)
        return {"env_id": env_id, "n": len(results), "results": results, "sandbox": sb}
    finally:
        handle.teardown()


def _decide_sandbox(mode: str) -> tuple[bool, dict[str, Any]]:
    """Resolve the sandbox mode against Docker availability.

    Returns ``(use_sandbox, status)`` where ``status`` is the ``sandbox`` block
    surfaced to callers.
    """
    mode = (mode or "auto").lower()
    if mode == "off":
        return False, {"requested": "off", "used": False, "available": None,
                       "reason": "host scoring (sandbox disabled)"}

    from rlenv_audit.sandbox import docker_available

    ok, msg = docker_available()
    if ok:
        return True, {"requested": mode, "used": True, "available": True, "reason": msg}
    if mode == "on":
        return False, {"requested": "on", "used": False, "available": False, "reason": msg}
    # auto + no Docker: fall back to the host, but say so loudly.
    return False, {"requested": "auto", "used": False, "available": False,
                   "reason": f"docker unavailable, scored on host: {msg}"}


def _row_index(c: dict, rows: list[dict]) -> int:
    """Clamp a completion's ``prompt_index`` into the loaded row range."""
    return max(0, min(int(c.get("prompt_index", 0)), len(rows) - 1))


def _result_entry(idx: int, c: dict) -> dict[str, Any]:
    """The shared shape of one scored-completion result (host and sandbox paths)."""
    return {"prompt_index": idx, "label": c.get("label", ""), "text": c.get("text", "")}


def _score_on_host(handle: Any, completions: list[dict], rows: list[dict]) -> list[dict]:
    """Score each completion through the env's rubric on the host (no isolation)."""
    results = []
    for c in completions:
        idx = _row_index(c, rows)
        row = rows[idx]
        entry = _result_entry(idx, c)
        try:
            reward, metrics = handle.score(
                c.get("text", ""), row["prompt"], row["answer"], row["info"], row.get("raw")
            )
            entry["reward"] = reward
            entry["metrics"] = metrics
        except ScoringError as exc:
            # The rubric ran and raised — a reward-side failure, not infrastructure.
            entry["error"] = str(exc)
            entry["error_kind"] = "reward"
        results.append(entry)
    return results


def _score_in_sandbox(env_id: str, completions: list[dict], rows: list[dict]) -> list[dict]:
    """Score all completions in ONE Docker container: every row's task and its
    completions ship in a single payload, so the env (imports, rubric, dataset
    machinery) loads once in the container rather than once per row. The dataset
    is read on the host and passed in via the payload, so the container never
    needs the network — only the env's rubric, which may execute the code."""
    from rlenv_audit.sandbox import SandboxError, run_scoring

    if not completions:
        return []

    idx_of = [_row_index(c, rows) for c in completions]
    by_row: dict[int, list[int]] = {}
    for pos, idx in enumerate(idx_of):
        by_row.setdefault(idx, []).append(pos)

    # Use the list position as the sandbox label so duplicate human labels
    # (two "correct"s for one row) don't collide on the way back.
    items = []
    for idx, positions in by_row.items():
        row = rows[idx]
        items.append({
            "task": {"prompt": row["prompt"], "answer": row["answer"],
                     "info": row["info"], "columns": row.get("raw") or {}},
            "cheats": [{"label": str(pos), "text": completions[pos].get("text", "")}
                       for pos in positions],
        })

    infra_error = None
    try:
        # One container scores everything; give it headroom that scales with
        # how many completions it must execute.
        scored = run_scoring(env_id, items, timeout_s=max(300, 30 * len(completions)))
    except SandboxError as exc:
        scored, infra_error = {}, f"sandbox: {exc}"

    results = []
    for pos, c in enumerate(completions):
        entry = _result_entry(idx_of[pos], c)
        r = scored.get(str(pos))
        if r is None:
            entry["error"] = infra_error or "sandbox: no result returned"
            entry["error_kind"] = "infra"
        elif "error" in r:
            # The rubric ran inside the container and raised on this completion.
            entry["error"] = r["error"]
            entry["error_kind"] = "reward"
        else:
            entry["reward"] = r.get("reward")
            entry["metrics"] = r.get("metrics", {})
        results.append(entry)
    return results


# -------------------------------------------------------------------- rollouts
def _completion_text(completion: Any) -> str:
    """Pull the assistant's text out of a vf-eval ``completion`` (a list of chat
    messages, or already a string)."""
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for msg in completion:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):  # structured content parts
            parts.append("".join(p.get("text", "") for p in content if isinstance(p, dict)))
    return "\n".join(p for p in parts if p)


def _timing_value(timing: Any, key: str) -> float | None:
    """Read one duration (seconds) from a vf-eval rollout ``timing`` dict. Values
    are either floats or ``{"duration": float}``."""
    if not isinstance(timing, dict) or key not in timing:
        return None
    val = timing[key]
    if isinstance(val, dict):
        val = val.get("duration", 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "calls": len(s),
        "mean_s": round(sum(s) / len(s), 3),
        "p50_s": round(s[len(s) // 2], 3),
        "p90_s": round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
        "max_s": round(s[-1], 3),
        "total_s": round(sum(s), 3),
    }


def _dummy_rollouts(env_id: str, n_samples: int, k: int, model: str | None) -> dict[str, Any]:
    """Offline fake rollouts (no endpoint): real dataset rows, placeholder text,
    scored through the real reward. Keeps ``--dummy`` working for smoke tests."""
    handle = load_handle(env_id)
    try:
        try:
            rows = handle.dataset(n=n_samples)
        except DatasetLoadError as exc:
            return {"env_id": env_id, "error": str(exc)}
        if not rows:
            return {"env_id": env_id, "error": "environment exposes no dataset"}
        samples = []
        for i, row in enumerate(rows):
            rollouts = []
            for j in range(k):
                text = f"(dummy rollout {j} for task {i})"
                try:
                    reward, _ = handle.score(text, row["prompt"], row["answer"], row["info"], row.get("raw"))
                except ScoringError:
                    reward = None
                rollouts.append({"text": text, "latency_s": 0.0, "reward": reward, "error": None,
                                 "truncated": False, "stop_reason": None, "output_tokens": 0})
            samples.append({"index": i, "prompt": row["prompt"], "answer": row["answer"], "rollouts": rollouts})
        return {"env_id": env_id, "model": model or "dummy", "endpoint": "dummy",
                "engine": "dummy", "dummy": True, "n_samples": len(samples), "k": k,
                "timing": {}, "samples": samples}
    finally:
        handle.teardown()


def run_rollouts(
    env_id: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    n_samples: int = 20,
    k: int = 8,
    max_tokens: int | None = 1024,
    temperature: float | None = None,
    max_concurrent: int = 8,
    dummy: bool = False,
    cache_path: str | None = None,
) -> dict[str, Any]:
    """Generate a shared set of rollouts via verifiers' own ``vf-eval`` engine,
    then score, time, and cache them to JSON.

    Using vf-eval (rather than a hand-rolled chat loop) means rollouts run through
    the *environment's real generation path*: multi-turn / tool-use envs roll out
    correctly, the env's own sampling args apply, and vf-eval records per-rollout
    timing, truncation and token usage for us. Both the latency and rollout-quality
    skills read this single cache. ``vf-eval`` is a client over an OpenAI-compatible
    endpoint — it does not start a model, so the user's served model must be up.
    """
    cache = Path(cache_path) if cache_path else CACHE_DIR / f"rollouts_{env_id.replace('/', '_')}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if dummy:
        out = _dummy_rollouts(env_id, n_samples, k, model)
        if "error" not in out:
            cache.write_text(json.dumps(out, indent=2))
            out["cache_path"] = str(cache)
        return out

    endpoint = endpoint or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    if not endpoint:
        return {"env_id": env_id, "error": "no model endpoint configured and --dummy not set"}

    if not model:
        try:
            from openai import OpenAI

            model = OpenAI(base_url=endpoint, api_key=api_key, timeout=30).models.list().data[0].id
        except Exception as exc:
            return {"env_id": env_id, "error": f"could not determine a model from {endpoint}: {exc}"}

    import glob
    import subprocess
    import sys
    import tempfile

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="vfeval_", dir=CACHE_DIR))
    key_var = "RLENV_AUDIT_VLLM_KEY"
    sub_env = dict(os.environ, **{key_var: api_key})
    # Passing both -b and -k makes vf-eval use the endpoint directly (no endpoints.toml).
    cmd = [
        sys.executable, "-m", "verifiers.scripts.eval", env_id,
        "-m", model, "-b", endpoint, "-k", key_var,
        "-n", str(n_samples), "-r", str(k), "-c", str(max_concurrent),
        "-s", "-o", str(scratch), "--disable-tui",
    ]
    if max_tokens is not None:
        cmd += ["-t", str(max_tokens)]
    if temperature is not None:
        cmd += ["-T", str(temperature)]

    try:
        proc = subprocess.run(cmd, env=sub_env, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        shutil.rmtree(scratch, ignore_errors=True)
        return {"env_id": env_id, "error": "vf-eval timed out after 3600s"}

    matches = sorted(glob.glob(str(scratch / "**" / "results.jsonl"), recursive=True))
    if not matches:
        tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        shutil.rmtree(scratch, ignore_errors=True)
        return {"env_id": env_id,
                "error": f"vf-eval produced no results (exit {proc.returncode}): {tail}"}

    run_dir = Path(matches[-1]).parent
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    metadata: dict[str, Any] = {}
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            metadata = {}

    from collections import OrderedDict

    by_example: "OrderedDict[Any, dict]" = OrderedDict()
    latencies: list[float] = []
    for r in rows:
        ex = r.get("example_id", 0)
        dt = _timing_value(r.get("timing"), "total")
        if dt is None:
            dt = _timing_value(r.get("timing"), "generation")
        if dt is not None:
            latencies.append(dt)
        err = r.get("error")
        if isinstance(err, dict):
            err = err.get("error")
        rollout = {
            "text": _completion_text(r.get("completion")),
            "latency_s": round(dt, 4) if dt is not None else None,
            "reward": r.get("reward"),
            "error": err,
            "truncated": r.get("is_truncated"),
            "stop_reason": r.get("stop_condition"),
            "output_tokens": (r.get("token_usage") or {}).get("output_tokens"),
        }
        sample = by_example.setdefault(
            ex, {"index": ex, "prompt": r.get("prompt"), "answer": r.get("answer", ""), "rollouts": []}
        )
        sample["rollouts"].append(rollout)

    out = {
        "env_id": env_id, "model": model, "endpoint": endpoint,
        "engine": "vf-eval", "dummy": False,
        "n_samples": len(by_example), "k": k, "max_concurrent": max_concurrent,
        "timing": _percentiles(latencies),
        "metrics": {key: metadata[key]
                    for key in ("avg_reward", "avg_error", "pass_at_k", "pass_all_k")
                    if key in metadata},
        "samples": list(by_example.values()),
    }
    cache.write_text(json.dumps(out, indent=2))
    out["cache_path"] = str(cache)
    shutil.rmtree(scratch, ignore_errors=True)
    return out


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
