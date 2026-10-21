"""Required judge outages must not become accepted successes."""

import pytest

from chi_bench.verifier.task_runtime import apply_judge_validity, build_scorecard


@pytest.mark.parametrize("rubric", [
    {"rubrics": [{"rubric_id": "required"}]},
    {"judge_unavailable_reason": "timeout"},
    {"degraded_rubric_ids": ["required"]},
])
def test_missing_or_degraded_judge_preserves_raw_reward_and_invalidates(rubric):
    score = build_scorecard({"state.correct": True, "judge.required": "not_applicable"})
    apply_judge_validity(score, rubric)
    assert score["upstream_binary_reward"] == 1.0
    assert score["binary_reward"] == 0.0
    assert score["evaluation_validity"] == "invalid"


def test_legitimate_no_submit_skip_is_not_an_outage():
    score = build_scorecard({"state.correct": True, "judge.payer": "not_applicable"})
    apply_judge_validity(score, {
        "rubrics": [{"rubric_id": "payer"}],
        "session_metadata": {"skipped": "no_submit_e2e_path"},
        "judge_unavailable_reason": "agent took no_submit path",
    })
    assert score["evaluation_validity"] == "valid"
    assert score["binary_reward"] == 1.0


def test_judge_said_no_is_a_valid_agent_failure():
    score = build_scorecard({"state.correct": True, "judge.required": False})
    apply_judge_validity(score, {"rubrics": [{"rubric_id": "required"}]})
    assert score["evaluation_validity"] == "valid"
    assert score["binary_reward"] == 0.0


def test_aggregation_refuses_invalid_judge_sentinel(tmp_path):
    import json
    from chi_bench.aggregator import _parse_trial

    (tmp_path / "result.json").write_text(json.dumps({"verifier_result": {"reward": 0}}))
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "scorecard.json").write_text(
        json.dumps({"evaluation_validity": "invalid"})
    )
    with pytest.raises(ValueError, match="rejudge"):
        _parse_trial(tmp_path / "result.json")


def test_unknown_usage_retains_outcome_but_never_free_cost(tmp_path):
    import json
    from chi_bench.aggregator import aggregate_to_rows

    (tmp_path / "result.json").write_text(json.dumps({
        "verifier_result": {"reward": 1},
        "agent_result": {"input_tokens": 0, "output_tokens": 0},
    }))
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "run_result.json").write_text(
        json.dumps({"usage_status": "unknown"})
    )
    rows = aggregate_to_rows(tmp_path, bootstrap_iters=10)
    assert rows[0]["pass_at_1"] == 1.0
    assert rows[0]["mean_cost_usd"] is None
    assert rows[0]["n_unknown_cost_trials"] == 1
