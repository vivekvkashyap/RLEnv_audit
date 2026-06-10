"""Integration tests: determinism + integrity behave on a known-good env.

(The determinism happy-path, which needs a model endpoint, is covered with a
mocked client in test_skills.py.)"""

from rlenv_audit.checks.base import CheckStatus
from rlenv_audit.checks.determinism import check_determinism
from rlenv_audit.checks.integrity import check_integrity


def test_determinism_skips_without_endpoint(gsm8k_handle, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = check_determinism(gsm8k_handle, {})
    assert result.status == CheckStatus.SKIP  # no static fallback, by design


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
