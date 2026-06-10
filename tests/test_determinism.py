"""Integration tests: determinism + integrity behave on a known-good env."""

from rlenv_audit.checks.base import CheckStatus
from rlenv_audit.checks.determinism import check_determinism
from rlenv_audit.checks.integrity import check_integrity


def test_determinism_passes_on_gsm8k(gsm8k_handle):
    result = check_determinism(gsm8k_handle, {"determinism_prompts": 2, "determinism_repeats": 3})
    assert result.status == CheckStatus.PASS
    assert result.score == 1.0
    # Every scored completion must be flagged deterministic.
    assert result.details["nondeterministic"] == []
    assert result.details["completions_scored"] > 0


def test_integrity_passes_on_gsm8k(gsm8k_handle):
    result = check_integrity(gsm8k_handle, {})
    assert result.status == CheckStatus.PASS
    d = result.details
    assert d["env_type"] == "SingleTurnEnv"
    assert "correct_answer" in d["reward_funcs"]
    assert d["system_prompt_present"] is True
    assert d["empty_answer_rate"] == 0.0


def test_reward_sources_retrievable(gsm8k_handle):
    sources = gsm8k_handle.reward_sources()
    assert "correct_answer" in sources
    assert "def" in sources["correct_answer"]
