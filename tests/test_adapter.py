"""Integration tests for the verifiers EnvHandle against a real Hub env."""

import pytest

from rlenv_audit.adapters.verifiers import DatasetLoadError, EnvLoadError, load_handle


def test_bad_env_id_raises_clean_error():
    try:
        load_handle("definitely_not_a_real_env_zzz")
    except EnvLoadError as exc:
        assert "definitely_not_a_real_env_zzz" in str(exc)
    else:
        raise AssertionError("expected EnvLoadError for a bogus env id")


def test_handle_exposes_rubric_parser_dataset(gsm8k_handle):
    assert gsm8k_handle.parser is not None
    assert gsm8k_handle.rubric is not None
    # gsm8k's rubric is a RubricGroup — names must surface despite empty .funcs.
    assert "correct_answer" in gsm8k_handle.reward_func_names()
    try:
        rows = gsm8k_handle.dataset(n=2)
    except DatasetLoadError as exc:
        pytest.skip(f"gsm8k dataset unavailable: {exc}")
    assert len(rows) == 2
    assert set(rows[0]) >= {"prompt", "answer", "info", "raw"}


def test_score_discriminates_correct_from_cheat(gsm8k_handle):
    try:
        row = gsm8k_handle.dataset(n=1)[0]
    except DatasetLoadError as exc:
        pytest.skip(f"gsm8k dataset unavailable: {exc}")
    prompt, answer, info = row["prompt"], row["answer"], row["info"]

    correct_reward, _ = gsm8k_handle.score(f"\\boxed{{{answer}}}", prompt, answer, info)
    cheat_reward, _ = gsm8k_handle.score("\\boxed{-999999}", prompt, answer, info)

    assert correct_reward > 0
    assert cheat_reward == 0
