"""Tests for the env_audit tool layer and the skill files."""

from pathlib import Path

from rlenv_audit.tools import build_scorecard, inspect_env, score_completions

REPO = Path(__file__).resolve().parents[1]


def test_inspect_gsm8k():
    info = inspect_env("gsm8k", n=3)
    if not info.get("loaded"):
        import pytest
        pytest.skip(f"gsm8k not installed: {info.get('error')}")
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


def test_build_scorecard_excludes_na_from_rating():
    card = build_scorecard({"env_id": "e", "checks": [
        {"name": "integrity", "status": "PASS", "score": 9, "justification": "x"},
        {"name": "reward_design", "status": "WARN", "score": 6, "justification": "y"},
        {"name": "latency", "status": "N/A", "score": None, "justification": "no endpoint"},
    ]})
    assert card["rating"] == 7.5         # (9 + 6) / 2, N/A excluded
    assert card["letter"] == "B"
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
