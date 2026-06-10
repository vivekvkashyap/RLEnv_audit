"""The check registry.

Each check is an independent function ``check_x(handle, config) -> CheckResult``.
A ``CheckSpec`` pairs that function with metadata (what hardware it needs, a
one-line description) so the CLI can list and filter checks, and so a check that
needs a GPU or Docker is discoverable up front.

Checks register themselves by being imported here and added to ``REGISTRY``. The
ordering of ``REGISTRY`` is the order checks run and appear in the scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rlenv_audit.checks.base import CheckResult

# A check: (handle: EnvHandle, config: dict) -> CheckResult
CheckFunc = Callable[..., CheckResult]


@dataclass(frozen=True)
class CheckSpec:
    name: str
    func: CheckFunc
    description: str
    needs_gpu: bool = False
    needs_docker: bool = False

    def needs(self) -> str:
        reqs = []
        if self.needs_gpu:
            reqs.append("GPU")
        if self.needs_docker:
            reqs.append("Docker")
        return ", ".join(reqs) if reqs else "—"


REGISTRY: dict[str, CheckSpec] = {}


def register(spec: CheckSpec) -> None:
    REGISTRY[spec.name] = spec


def _load_builtin_checks() -> None:
    """Import built-in check modules so they register themselves.

    Imported lazily (and tolerantly) so that a check module which can't import
    its optional deps doesn't break the whole registry.
    """
    from rlenv_audit.checks import determinism  # noqa: F401
    from rlenv_audit.checks import exploits  # noqa: F401
    from rlenv_audit.checks import parser  # noqa: F401
    from rlenv_audit.checks import contamination  # noqa: F401
    from rlenv_audit.checks import latency  # noqa: F401

    register(determinism.SPEC)
    register(exploits.SPEC)
    register(parser.SPEC)
    register(contamination.SPEC)
    register(latency.SPEC)


_load_builtin_checks()


def select(only: list[str] | None = None, skip: list[str] | None = None) -> list[CheckSpec]:
    """Resolve the ordered list of checks to run, honoring ``--only``/``--skip``.

    Unknown names raise ``KeyError`` with the offending name so the CLI can show
    a clear error rather than silently running the wrong set.
    """
    names = list(REGISTRY.keys())
    if only:
        unknown = [n for n in only if n not in REGISTRY]
        if unknown:
            raise KeyError(f"unknown check(s): {', '.join(unknown)}")
        names = [n for n in names if n in only]
    if skip:
        unknown = [n for n in skip if n not in REGISTRY]
        if unknown:
            raise KeyError(f"unknown check(s): {', '.join(unknown)}")
        names = [n for n in names if n not in skip]
    return [REGISTRY[n] for n in names]
