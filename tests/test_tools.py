"""Tests for the env_audit tool layer and the skill files."""

from pathlib import Path

from rlenv_audit.tools import build_scorecard, inspect_env, score_completions

REPO = Path(__file__).resolve().parents[1]


def test_inspect_gsm8k():
    info = inspect_env("gsm8k", n=3)
    if not info.get("loaded"):
        import pytest
        pytest.skip(f"gsm8k not installed: {info.get('error')}")
    if info.get("dataset_error"):
        import pytest
        pytest.skip(f"gsm8k dataset unavailable: {info['dataset_error']}")
    assert info["env_type"] == "SingleTurnEnv"
    assert any(f["name"] == "correct_answer" and "def" in f["source"] for f in info["reward_funcs"])
    assert info["dataset_size"]["train"]
    assert len(info["sample"]) == 3
    assert info["sample"][0]["answer"]


def test_score_discriminates():
    info = inspect_env("gsm8k", n=1)
    if not info.get("loaded"):
        import pytest
        pytest.skip("gsm8k not installed")
    if info.get("dataset_error"):
        import pytest
        pytest.skip(f"gsm8k dataset unavailable: {info['dataset_error']}")
    ans = info["sample"][0]["answer"]
    out = score_completions("gsm8k", [
        {"prompt_index": 0, "label": "correct", "text": f"\\boxed{{{ans}}}"},
        {"prompt_index": 0, "label": "wrong", "text": "\\boxed{-1}"},
        {"prompt_index": 0, "label": "empty", "text": ""},
    ])
    by = {r["label"]: r["reward"] for r in out["results"]}
    assert by["correct"] > 0
    assert by["wrong"] == 0
    assert by["empty"] == 0


FAKE_ROWS = [
    {"prompt": "p0", "answer": "a0", "info": {}, "raw": {}},
    {"prompt": "p1", "answer": "a1", "info": {}, "raw": {}},
]


def test_score_in_sandbox_batches_orders_and_tags(monkeypatch):
    """One run_scoring call for all rows; results in input order; errors tagged."""
    from rlenv_audit import tools

    completions = [
        {"prompt_index": 0, "label": "correct", "text": "x"},
        {"prompt_index": 1, "label": "wrong", "text": "y"},
        {"prompt_index": 0, "label": "correct", "text": "z"},  # duplicate label
    ]
    calls = []

    def fake_run_scoring(env_id, items, **kw):
        calls.append(items)
        assert {i["task"]["prompt"] for i in items} == {"p0", "p1"}
        return {"0": {"reward": 1.0, "metrics": {"m": 1.0}},
                "1": {"error": "boom"}}  # "2" deliberately missing

    monkeypatch.setattr("rlenv_audit.sandbox.run_scoring", fake_run_scoring)
    out = tools._score_in_sandbox("env", completions, FAKE_ROWS)

    assert len(calls) == 1                      # a single container for everything
    assert [e["label"] for e in out] == ["correct", "wrong", "correct"]
    assert out[0]["reward"] == 1.0 and out[0]["metrics"] == {"m": 1.0}
    assert out[1]["error"] == "boom" and out[1]["error_kind"] == "reward"
    assert out[2]["error_kind"] == "infra"      # no result returned for it


def test_score_in_sandbox_infra_failure_tags_all_entries(monkeypatch):
    from rlenv_audit import tools
    from rlenv_audit.sandbox import SandboxError

    def boom(env_id, items, **kw):
        raise SandboxError("docker died")

    monkeypatch.setattr("rlenv_audit.sandbox.run_scoring", boom)
    out = tools._score_in_sandbox(
        "env", [{"prompt_index": 0, "label": "c", "text": "t"}], FAKE_ROWS
    )
    assert out[0]["error_kind"] == "infra"
    assert "docker died" in out[0]["error"]


def test_build_scorecard_excludes_na_from_rating():
    card = build_scorecard({"env_id": "e", "checks": [
        {"name": "integrity", "status": "PASS", "score": 9, "justification": "x"},
        {"name": "reward_design", "status": "WARN", "score": 6, "justification": "y"},
        {"name": "latency", "status": "N/A", "score": None, "justification": "no endpoint"},
    ]})
    assert card["rating"] == 7.5         # (9 + 6) / 2, N/A excluded
    assert card["grade"] == "WARN"


def test_build_scorecard_weights_latency_and_contamination_half():
    card = build_scorecard({"env_id": "e", "checks": [
        {"name": "integrity", "status": "PASS", "score": 10, "justification": "x"},
        {"name": "contamination", "status": "FAIL", "score": 2, "justification": "y"},
    ]})
    assert card["rating"] == 7.3         # (10*1 + 2*0.5) / 1.5


def test_build_scorecard_error_grades_fail():
    card = build_scorecard({"env_id": "e", "checks": [
        {"name": "integrity", "status": "ERROR", "score": None, "justification": "crashed"},
        {"name": "contamination", "status": "PASS", "score": 10, "justification": "clean"},
    ], "feedback": "fine env, scoring crashed"})
    assert card["grade"] == "FAIL"
    assert card["rating"] == 10          # only the check that produced a score
    assert card["feedback"] == "fine env, scoring crashed"


def test_install_skills_copies_bundled_skills(tmp_path):
    from click.testing import CliRunner

    from rlenv_audit.cli import main

    result = CliRunner().invoke(main, ["install-skills", "--target", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for name in ("env-audit", "integrity", "problem-alignment", "reward-design",
                 "latency", "rollout-quality", "contamination"):
        assert (tmp_path / name / "SKILL.md").exists(), f"{name} not installed"


def test_all_six_check_skills_present_and_valid():
    expected = {"env-audit", "integrity", "problem-alignment", "reward-design",
                "latency", "rollout-quality", "contamination"}
    found = set()
    for skill_md in (REPO / "skills").glob("*/SKILL.md"):
        text = skill_md.read_text()
        assert text.startswith("---"), f"{skill_md} missing frontmatter"
        assert "name:" in text and "description:" in text, f"{skill_md} missing name/description"
        found.add(skill_md.parent.name)
    assert expected <= found, f"missing skills: {expected - found}"
