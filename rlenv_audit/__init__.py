"""RLEnv_audit — pytest for RL environments.

Library-first; the CLI is a thin wrapper over ``audit()``:

    import rlenv_audit
    scorecard = rlenv_audit.audit("gsm8k")     # or pass a loaded Environment
    print(scorecard.grade)
    scorecard.write_json("report.json")
"""

from rlenv_audit.checks.base import CheckResult, CheckStatus
from rlenv_audit.core import audit
from rlenv_audit.report import Scorecard

__version__ = "0.1.0"

__all__ = ["audit", "CheckResult", "CheckStatus", "Scorecard", "__version__"]
