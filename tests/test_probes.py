"""Unit tests for the model-generated probe layer (no endpoint needed)."""

from rlenv_audit.probes import _extract_json, _validate_gold, generate_probes


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"tasks": []}') == {"tasks": []}
    assert _extract_json('Sure!\n```json\n{"tasks": [1]}\n```\nDone.') == {"tasks": [1]}
    assert _extract_json("not json at all") is None


def test_generate_probes_without_endpoint_falls_back(gsm8k_handle, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config: dict = {}
    rows = gsm8k_handle.dataset(n=1)
    probes = generate_probes(gsm8k_handle, config, rows)
    assert probes == {}              # no endpoint -> static-battery fallback
    assert "_generated_probes" in config  # and the result is cached


def test_validate_gold_round_trips_through_parser(gsm8k_handle):
    row = gsm8k_handle.dataset(n=1)[0]
    ans = row["answer"]
    good = f"Reasoning here. The answer is \\boxed{{{ans}}}"
    bad = "Reasoning here. The answer is \\boxed{-1}"
    assert _validate_gold(gsm8k_handle, good, ans) is True
    assert _validate_gold(gsm8k_handle, bad, ans) is False
