"""Scorecard rendering — the pretty terminal table and the JSON report.

The ``Scorecard`` is the object the library's ``audit()`` returns. The CLI calls
``to_terminal()``; the survey pipeline calls ``to_json()``. Both are thin views
over the same list of ``CheckResult``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from rlenv_audit.checks.base import CheckResult, CheckStatus

# Terminal color + glyph per status. SKIP is dim, not alarming — a skipped check
# is not a defect.
_STATUS_STYLE: dict[CheckStatus, str] = {
    CheckStatus.PASS: "bold green",
    CheckStatus.FAIL: "bold red",
    CheckStatus.WARN: "bold yellow",
    CheckStatus.SKIP: "dim",
}

# Overall grade derived from the worst meaningful outcome present.
_GRADE_STYLE: dict[str, str] = {
    "PASS": "bold green",
    "WARN": "bold yellow",
    "FAIL": "bold red",
    "INCONCLUSIVE": "dim",
}


@dataclass
class Scorecard:
    """The audit result for one environment."""

    env_id: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Overall grade from the worst meaningful result.

        FAIL beats WARN beats PASS; if every check SKIP'd, the audit was
        INCONCLUSIVE (we learned nothing — distinct from a clean PASS).
        """
        statuses = {r.status for r in self.results}
        if CheckStatus.FAIL in statuses:
            return "FAIL"
        if CheckStatus.WARN in statuses:
            return "WARN"
        if CheckStatus.PASS in statuses:
            return "PASS"
        return "INCONCLUSIVE"

    def counts(self) -> dict[str, int]:
        """Tally of results by status, e.g. ``{"PASS": 3, "SKIP": 2}``."""
        out: dict[str, int] = {}
        for r in self.results:
            out[str(r.status)] = out.get(str(r.status), 0) + 1
        return out

    # ------------------------------------------------------------------ JSON
    def to_json(self) -> dict[str, Any]:
        """Full machine-readable report — carries every check's ``details``."""
        return {
            "env_id": self.env_id,
            "grade": self.grade,
            "counts": self.counts(),
            "checks": [r.to_dict() for r in self.results],
        }

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    # -------------------------------------------------------------- terminal
    def to_terminal(self, console: Console | None = None) -> None:
        """Render the scorecard as a color-coded table + overall grade."""
        console = console or Console()

        table = Table(title=f"RLEnv_audit · {self.env_id}", title_style="bold")
        table.add_column("check", style="bold", no_wrap=True)
        table.add_column("status", no_wrap=True)
        table.add_column("summary")

        for r in self.results:
            status = Text(str(r.status), style=_STATUS_STYLE.get(r.status, ""))
            table.add_row(r.check_name, status, r.summary)

        console.print(table)

        counts = self.counts()
        tally = "  ".join(f"{k} {v}" for k, v in counts.items())
        grade = Text(self.grade, style=_GRADE_STYLE.get(self.grade, "bold"))
        console.print(Text.assemble("overall grade: ", grade, f"   ({tally})"))
