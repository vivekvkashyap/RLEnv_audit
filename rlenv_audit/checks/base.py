"""Core data model shared by every check.

A check is a function ``check_x(handle, config) -> CheckResult``. It must never
raise to the caller — every failure mode (env won't load, no GPU, Docker down)
becomes a ``CheckResult`` with status ``SKIP`` or ``FAIL``. The ``timed`` helper
wraps a check body so duration and uncaught-exception handling are uniform.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class CheckStatus(str, Enum):
    """Outcome of a single check.

    PASS  — the check ran and the environment passed it.
    FAIL  — the check ran and found a real problem.
    WARN  — the check ran and found something worth attention (not disqualifying).
    SKIP  — the check could not run here (no GPU, Docker down, N/A for this env).
            Deliberately distinct from FAIL: a skipped check is *not* a defect.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"

    def __str__(self) -> str:  # so f"{status}" renders "PASS", not "CheckStatus.PASS"
        return self.value


@dataclass
class CheckResult:
    """The structured result of one check.

    ``summary`` is the single human line shown in the terminal table; ``details``
    carries the full structured findings that land in ``report.json`` for the
    survey-stage analysis.
    """

    check_name: str
    status: CheckStatus
    summary: str
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": str(self.status),
            "score": self.score,
            "summary": self.summary,
            "details": self.details,
            "duration_s": round(self.duration_s, 4),
        }


# A check: (handle: EnvHandle, config: dict) -> CheckResult
CheckFunc = Callable[..., CheckResult]


@dataclass(frozen=True)
class CheckSpec:
    """A registered check: the function plus metadata the CLI needs to list and
    filter it (what hardware/services it requires, a one-line description)."""

    name: str
    func: CheckFunc
    description: str
    needs_gpu: bool = False
    needs_docker: bool = False
    needs_model: bool = False

    def needs(self) -> str:
        reqs = []
        if self.needs_gpu:
            reqs.append("GPU")
        if self.needs_docker:
            reqs.append("Docker")
        if self.needs_model:
            reqs.append("model endpoint")
        return ", ".join(reqs) if reqs else "—"


def timed(
    check_name: str,
    body: Callable[[], CheckResult],
) -> CheckResult:
    """Run a check body, stamping ``duration_s`` and converting any uncaught
    exception into a clean ``FAIL`` result instead of a traceback dump.

    Checks are expected to handle their own *expected* failure modes (returning
    SKIP/FAIL with a helpful summary). ``timed`` is the backstop for the
    *unexpected* — a check should never take down the whole run.
    """

    start = time.perf_counter()
    try:
        result = body()
    except Exception as exc:  # backstop only — checks should pre-empt this
        result = CheckResult(
            check_name=check_name,
            status=CheckStatus.FAIL,
            summary=f"check crashed: {type(exc).__name__}: {exc}",
            details={"error": str(exc), "traceback": traceback.format_exc()},
        )
    result.duration_s = time.perf_counter() - start
    return result
