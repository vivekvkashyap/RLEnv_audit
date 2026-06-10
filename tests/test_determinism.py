"""Integration test: the determinism check passes on a deterministic env."""

from rlenv_audit.checks.base import CheckStatus
from rlenv_audit.checks.determinism import check_determinism


def test_determinism_passes_on_gsm8k(gsm8k_handle):
    result = check_determinism(gsm8k_handle, {"determinism_prompts": 2, "determinism_repeats": 3})
    assert result.status == CheckStatus.PASS
    assert result.score == 1.0
    # Every scored completion must be flagged deterministic.
    assert result.details["nondeterministic"] == []
    assert result.details["completions_scored"] > 0
