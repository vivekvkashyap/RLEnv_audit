"""Adapter: load a Prime Intellect Hub (`verifiers`) environment and normalize it
into an internal ``EnvHandle`` that the audit tools (``tools.py``) consume.

This is the *only* module that imports ``verifiers``. Everything verifiers-0.1.14
got surprising about is contained here (all verified against the installed
source — see DESIGN.md §3):

* ``verifiers.load_environment(env_id)`` is **synchronous** and imports an
  installed env package by module name.
* ``env.rubric`` is frequently a ``RubricGroup`` whose own ``.funcs`` is empty —
  the real reward functions surface via ``_get_reward_func_names()``.
* Reward scoring is **async** and mutates a ``state`` dict in place. We expose a
  synchronous ``score()`` so no caller ever touches asyncio.
* Some rubrics (e.g. ``MathRubric``) own a ``ProcessPoolExecutor`` that must be
  torn down or the interpreter can hang on exit.
"""

from __future__ import annotations

import asyncio
from typing import Any


class EnvLoadError(Exception):
    """Raised when an environment cannot be loaded (bad id, import failure)."""


class ScoringError(Exception):
    """Raised when a completion cannot be scored through the env's rubric."""


class DatasetLoadError(Exception):
    """Raised when an environment dataset exists but cannot be loaded."""


def load_handle(env_id: str, env_args: dict[str, Any] | None = None) -> "EnvHandle":
    """Load a verifiers environment by id and wrap it in an ``EnvHandle``.

    Converts verifiers' ``ValueError`` / ``RuntimeError`` (and the underlying
    ``ModuleNotFoundError``) into a single ``EnvLoadError`` with an actionable
    message — callers get a clean failure, never a traceback dump.
    """
    try:
        import verifiers
    except Exception as exc:  # pragma: no cover - import guard
        raise EnvLoadError(f"could not import the 'verifiers' library: {exc}") from exc

    try:
        env = verifiers.load_environment(env_id, **(env_args or {}))
    except Exception as exc:
        raise EnvLoadError(
            f"could not load environment '{env_id}': {exc}. "
            f"Is the environment package installed? Try `vf-install {env_id}` "
            f"(add -r for the verifiers example envs)."
        ) from exc

    return EnvHandle(env_id=env_id, env=env)


class EnvHandle:
    """Normalized handle over a loaded verifiers environment.

    The tools talk to this, not to verifiers. Holds a single persistent event
    loop so the synchronous ``score()`` can drive the async rubric repeatedly
    (the score tool runs dozens of completions in one session) without per-call
    loop churn, and so a rubric-owned ProcessPool survives across calls.
    """

    def __init__(self, env_id: str, env: Any):
        self.env_id = env_id
        self.env = env
        self.rubric = getattr(env, "rubric", None)
        self.parser = getattr(env, "parser", None)
        self._loop = asyncio.new_event_loop()

    # ------------------------------------------------------------------ rubric
    def reward_func_names(self) -> list[str]:
        """Names of the env's reward/metric functions (RubricGroup-aware)."""
        if self.rubric is None:
            return []
        try:
            return list(self.rubric._get_reward_func_names())
        except Exception:
            return [getattr(f, "__name__", repr(f)) for f in getattr(self.rubric, "funcs", [])]

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    def canonical_answer(self, answer: str) -> str | None:
        """Produce a completion string the env's parser extracts ``answer`` from.

        Prefers the parser's own ``format()`` (e.g. ``XMLParser`` wraps the
        ``answer_field`` in its tags); verifies by round-trip. Returns ``None``
        if the parser can't be coaxed into round-tripping — callers then fall
        back to generic format guesses. Lets a caller adapt to each env's answer
        format instead of assuming one.
        """
        parser = self.parser
        if parser is None:
            return None
        fmt = getattr(parser, "format", None)
        field = getattr(parser, "answer_field", None)
        if callable(fmt) and field:
            try:
                text = fmt(**{field: answer})
                got = parser.parse_answer([{"role": "assistant", "content": text}])
                if got is not None and str(got).strip() == str(answer).strip():
                    return text
            except Exception:
                pass
        return None

    def reward_sources(self, max_chars_per_func: int = 4000) -> dict[str, str]:
        """Source code of each reward function, best-effort (RubricGroup-aware).

        Surfaced by the inspect tool so the auditing agent can read what the
        verifier actually does. Functions whose source can't be retrieved
        (C extensions, lambdas defined interactively) are reported as unavailable
        rather than skipped silently.
        """
        import inspect
        import textwrap

        sources: dict[str, str] = {}
        if self.rubric is None:
            return sources
        try:
            funcs = self.rubric._get_reward_funcs()
        except Exception:
            funcs = list(getattr(self.rubric, "funcs", []))
        for func in funcs:
            name = getattr(func, "__name__", repr(func))
            try:
                src = textwrap.dedent(inspect.getsource(func))
                if len(src) > max_chars_per_func:
                    src = src[:max_chars_per_func] + "\n# ... truncated ..."
                sources[name] = src
            except Exception:
                sources[name] = "<source unavailable>"
        return sources

    def system_prompt(self) -> str | None:
        """The env's system prompt, from the env itself or the first dataset row."""
        sp = getattr(self.env, "system_prompt", None)
        if isinstance(sp, str) and sp.strip():
            return sp
        try:
            rows = self.dataset(n=1)
        except DatasetLoadError:
            return None
        if rows and isinstance(rows[0].get("prompt"), list):
            for msg in rows[0]["prompt"]:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    content = msg.get("content")
                    return content if isinstance(content, str) else None
        return None

    def module_file(self) -> str | None:
        """Path to the environment package's source file, for source-level review."""
        import importlib

        mod_name = self.env_id.replace("-", "_").split("/")[-1]
        try:
            return getattr(importlib.import_module(mod_name), "__file__", None)
        except Exception:
            return None

    def dataset_size(self) -> dict[str, int | None]:
        """Best-effort row counts for the train and eval splits."""
        out: dict[str, int | None] = {"train": None, "eval": None}
        for split, getter in (("train", "get_dataset"), ("eval", "get_eval_dataset")):
            try:
                ds = getattr(self.env, getter)(n=-1)
                out[split] = len(ds) if ds is not None else 0
            except Exception:
                out[split] = None
        return out

    # Dataset columns that verifiers already treats specially; everything else
    # in a row is exposed to reward functions as a named argument.
    _RESERVED_COLS = {"prompt", "answer", "info", "example_id"}

    def score(
        self,
        text: Any,
        prompt: Any,
        answer: str = "",
        info: dict[str, Any] | None = None,
        columns: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a completion through the env's rubric.

        ``text`` is either the assistant message content (a string, the common
        case) or a ready-made ``Messages`` list for multi-message completions.
        ``columns`` carries the rest of the dataset row so reward functions that
        read custom fields (test cases, oracles, …) get them — passed both via
        ``state["input"]`` (verifiers injects non-reserved input fields as named
        args) and at the top level of ``state`` (some rubrics read it directly).

        Returns ``(reward, metrics)``. Synchronous wrapper over the async
        verifiers path; works uniformly across ``Rubric``, ``MathRubric``, and
        ``RubricGroup``.
        """
        if self.rubric is None:
            raise ScoringError(f"environment '{self.env_id}' has no rubric")

        completion = text if isinstance(text, list) else [{"role": "assistant", "content": text}]
        cols = {k: v for k, v in (columns or {}).items() if k not in self._RESERVED_COLS}

        state: dict[str, Any] = {
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "info": info or {},
            "task": None,
            "input": {"prompt": prompt, "answer": answer, "info": info or {}, **cols},
            "trajectory": [],
        }
        for key, value in cols.items():
            state.setdefault(key, value)

        try:
            # score_rollout asserts there are no group-level reward funcs;
            # score_group handles both. Branch on what the rubric actually has.
            if getattr(self.rubric, "has_group_rewards", False):
                self._run(self.rubric.score_group([state]))
            else:
                self._run(self.rubric.score_rollout(state))
        except Exception as exc:
            raise ScoringError(f"failed to score completion: {exc}") from exc

        reward = float(state.get("reward", 0.0) or 0.0)
        metrics = {k: float(v) for k, v in (state.get("metrics") or {}).items()}
        return reward, metrics

    # ----------------------------------------------------------------- dataset
    def dataset(self, n: int = -1, split: str = "train") -> list[dict[str, Any]]:
        """Return normalized dataset rows: ``{prompt, answer, info, raw}``.

        Tries the requested split, then falls back to the other one — many Hub
        envs are *eval-only* (no train dataset) or train-only, and a check that
        needs "some tasks" shouldn't care which split they came from. Returns
        ``[]`` only if the env exposes no usable dataset at all.
        """

        errors: dict[str, str] = {}

        def _load(which: str):
            try:
                if which == "eval":
                    return self.env.get_eval_dataset(n=n)
                return self.env.get_dataset(n=n)
            except Exception as exc:
                errors[which] = f"{type(exc).__name__}: {exc}"
                return None

        order = ["eval", "train"] if split == "eval" else ["train", "eval"]
        ds = None
        for which in order:
            ds = _load(which)
            if ds is not None and len(ds) > 0:
                break
        if ds is None or len(ds) == 0:
            if errors:
                detail = "; ".join(f"{split}: {err}" for split, err in errors.items())
                raise DatasetLoadError(
                    f"could not load dataset for environment '{self.env_id}': {detail}"
                )
            return []

        rows: list[dict[str, Any]] = []
        for row in ds:
            rows.append(
                {
                    "prompt": row.get("prompt"),
                    "answer": row.get("answer", ""),
                    "info": row.get("info", {}) or {},
                    "raw": dict(row),
                }
            )
        return rows

    # ---------------------------------------------------------------- teardown
    def teardown(self) -> None:
        """Best-effort cleanup: tear down a rubric-owned ProcessPool, then close
        the loop. Never raises — cleanup failure must not fail an audit."""
        try:
            if self.rubric is not None and hasattr(self.rubric, "teardown"):
                self._run(self.rubric.teardown())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
