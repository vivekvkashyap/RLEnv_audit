"""Adapter: load a Prime Intellect Hub (`verifiers`) environment and normalize it
into an internal ``EnvHandle`` that the checks consume.

This is the *only* module that imports ``verifiers``. Everything verifiers-0.1.14
got surprising about is contained here (all verified against the installed
source — see DESIGN.md §3):

* ``verifiers.load_environment(env_id)`` is **synchronous** and imports an
  installed env package by module name.
* ``env.rubric`` is frequently a ``RubricGroup`` whose own ``.funcs`` is empty —
  the real reward functions surface via ``_get_reward_func_names()``.
* Reward scoring is **async** and mutates a ``state`` dict in place. We expose a
  synchronous ``score()`` so no check ever touches asyncio.
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

    Checks talk to this, not to verifiers. Holds a single persistent event loop
    so the synchronous ``score()`` can drive the async rubric repeatedly (the
    determinism check scores the same completion 5x) without per-call loop churn,
    and so a rubric-owned ProcessPool survives across calls.
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

    def score(
        self,
        text: str,
        prompt: Any,
        answer: str = "",
        info: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a single assistant completion through the env's rubric.

        ``text`` is the assistant message content (what a model would have
        produced). Returns ``(reward, metrics)`` where ``reward`` is the
        aggregate float and ``metrics`` maps each reward-function name to its
        score. Synchronous wrapper over the async verifiers path; works
        uniformly across plain ``Rubric``, ``MathRubric``, and ``RubricGroup``.
        """
        if self.rubric is None:
            raise ScoringError(f"environment '{self.env_id}' has no rubric")

        state: dict[str, Any] = {
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": text}],
            "answer": answer,
            "info": info or {},
            "task": None,
            "input": {"prompt": prompt, "answer": answer, "info": info or {}, **(extra or {})},
            "trajectory": [],
        }

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

        ``split="eval"`` reads the eval dataset (falls back to train in
        verifiers). Returns ``[]`` if the env exposes no dataset.
        """
        try:
            if split == "eval":
                ds = self.env.get_eval_dataset(n=n)
            else:
                ds = self.env.get_dataset(n=n)
        except Exception:
            return []
        if ds is None:
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
