"""Unit tests for the data model + scorecard rendering (no env needed)."""

import json

from rlenv_audit import CheckResult, CheckStatus, Scorecard


def test_grade_derivation():
    fail = Scorecard("e", [CheckResult("a", CheckStatus.PASS, "ok"),
                           CheckResult("b", CheckStatus.FAIL, "bad")])
    assert fail.grade == "FAIL"

    warn = Scorecard("e", [CheckResult("a", CheckStatus.PASS, "ok"),
                           CheckResult("b", CheckStatus.WARN, "meh")])
    assert warn.grade == "WARN"

    ok = Scorecard("e", [CheckResult("a", CheckStatus.PASS, "ok"),
                         CheckResult("b", CheckStatus.SKIP, "n/a")])
    assert ok.grade == "PASS"

    inconclusive = Scorecard("e", [CheckResult("a", CheckStatus.SKIP, "n/a")])
    assert inconclusive.grade == "INCONCLUSIVE"


def test_json_round_trip_preserves_details():
    sc = Scorecard("demo", [
        CheckResult("exploits", CheckStatus.FAIL, "2 cheats",
                    score=0.5, details={"succeeded": ["sys_exit", "empty"]}, duration_s=1.2),
    ])
    blob = json.loads(json.dumps(sc.to_json()))
    assert blob["env_id"] == "demo"
    assert blob["grade"] == "FAIL"
    assert blob["counts"] == {"FAIL": 1}
    assert blob["checks"][0]["details"]["succeeded"] == ["sys_exit", "empty"]
    assert blob["checks"][0]["status"] == "FAIL"


def test_status_str_is_clean():
    assert str(CheckStatus.PASS) == "PASS"
    assert f"{CheckStatus.SKIP}" == "SKIP"
