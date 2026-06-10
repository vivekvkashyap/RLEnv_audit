"""The library entry point: ``audit()``.

This is where the real work is orchestrated; the CLI is a thin wrapper over it.
``audit()`` accepts either an env-id string (which it loads) or an already-loaded
verifiers ``Environment``, runs the selected checks, and returns a ``Scorecard``.
"""

from __future__ import annotations

from typing import Any

from rlenv_audit.adapters.verifiers import EnvHandle, load_handle
from rlenv_audit.checks import select
from rlenv_audit.checks.base import timed
from rlenv_audit.report import Scorecard


def _quiet_verifiers() -> None:
    """Silence verifiers' own logging so the scorecard is the only output."""
    try:
        import verifiers

        verifiers.quiet_verifiers()
    except Exception:
        pass


def _to_handle(env: Any, env_id: str | None) -> EnvHandle:
    """Coerce the ``env`` argument into an ``EnvHandle``.

    A string is treated as an env-id and loaded; anything else is assumed to be
    an already-loaded verifiers ``Environment`` and wrapped directly.
    """
    if isinstance(env, str):
        return load_handle(env)
    if isinstance(env, EnvHandle):
        return env
    resolved_id = env_id or getattr(env, "env_id", None) or "environment"
    return EnvHandle(env_id=resolved_id, env=env)


def audit(
    env: Any,
    *,
    env_id: str | None = None,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> Scorecard:
    """Audit a verifiers environment and return a ``Scorecard``.

    Parameters
    ----------
    env:
        An env-id string (loaded via verifiers) or a loaded ``Environment``.
    only / skip:
        Restrict or exclude checks by name (see ``rlenv_audit.checks.REGISTRY``).
    config:
        Per-check tuning knobs (e.g. ``determinism_repeats``, ``model``).

    Every check is isolated by ``timed()`` so one misbehaving check can never
    abort the audit. The env handle is always torn down.
    """
    _quiet_verifiers()
    config = dict(config or {})

    handle = _to_handle(env, env_id)
    try:
        specs = select(only=only, skip=skip)
        results = [timed(spec.name, lambda s=spec: s.func(handle, config)) for spec in specs]
        return Scorecard(env_id=handle.env_id, results=results)
    finally:
        handle.teardown()
