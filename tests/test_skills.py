"""Tests for the skill-file framework and the skill-driven checks.

The model call is mocked, so these run with no endpoint and no network while
exercising the full path: skill load -> prompt build -> JSON parse -> check.
"""

import json

import pytest

from rlenv_audit import skills
from rlenv_audit.checks.base import CheckStatus
from rlenv_audit.checks.determinism import check_determinism
from rlenv_audit.skills import endpoint_configured, load_skill, validate_gold


class _FakeMessage:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return type("R", (), {"choices": [_FakeMessage(self._content)]})


class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})


def _mock_model(monkeypatch, content: str):
    """Force run_skill to use a fake client that returns `content`, with an
    endpoint considered configured."""
    monkeypatch.setattr(skills, "_client_and_model", lambda config: (_FakeClient(content), "mock-model"))
    monkeypatch.setattr(skills, "endpoint_configured", lambda config: True)
    # determinism imports endpoint_configured by name — patch there too.
    import rlenv_audit.checks.determinism as det
    monkeypatch.setattr(det, "endpoint_configured", lambda config: True)


def test_all_skill_files_present_and_nonempty():
    for name in ("determinism", "reward_design", "exploits", "parser"):
        text = load_skill(name)
        assert text and "JSON" in text, f"{name}.md missing or malformed"
    assert load_skill("does_not_exist") is None


def test_endpoint_detection(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert endpoint_configured({}) is False
    assert endpoint_configured({"api_key": "x"}) is True
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert endpoint_configured({}) is True


def test_determinism_skips_without_endpoint(gsm8k_handle, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = check_determinism(gsm8k_handle, {})
    assert result.status == CheckStatus.SKIP
    assert "endpoint" in result.summary


def test_determinism_runs_on_mocked_probes(gsm8k_handle, monkeypatch):
    row = gsm8k_handle.dataset(n=1)[0]
    ans = row["answer"]
    # The mocked model returns a gold + an edge probe for task 0.
    content = json.dumps({"probes": [
        {"task_index": 0, "kind": "gold", "label": "g1", "text": f"The answer is \\boxed{{{ans}}}"},
        {"task_index": 0, "kind": "edge", "label": "empty", "text": ""},
        {"task_index": 0, "kind": "wrong", "label": "w1", "text": "The answer is \\boxed{-1}"},
    ]})
    _mock_model(monkeypatch, content)

    result = check_determinism(gsm8k_handle, {"determinism_repeats": 3, "determinism_tasks": 1})
    assert result.status == CheckStatus.PASS
    assert result.details["probes_scored"] == 3
    assert result.details["nondeterministic"] == []
    assert result.details["model"] == "mock-model"


def test_validate_gold_round_trips(gsm8k_handle):
    ans = gsm8k_handle.dataset(n=1)[0]["answer"]
    assert validate_gold(gsm8k_handle, f"so the answer is \\boxed{{{ans}}}", ans) is True
    assert validate_gold(gsm8k_handle, "the answer is \\boxed{-99}", ans) is False
