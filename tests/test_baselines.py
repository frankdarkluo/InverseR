import math
import sqlite3

import numpy as np
import pytest

from inverser.baselines import (
    SearchConfig,
    action_unit,
    geometry_actions,
    make_action_bank,
    sampled_actions,
    search_goal,
    unit_action,
)
from inverser.evaluation import (
    BudgetExhausted,
    BudgetLimits,
    TrialLedger,
    UnresolvedRequest,
    recorded_replay,
)
from inverser.physics import SimulationFailure
from inverser.study import make_study_case


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_proposals_are_reproducible_legal_and_do_not_need_a_goal(method):
    config = SearchConfig(method=method, seed=17, max_objective_calls=8)
    proposals = sampled_actions(config, 8)
    assert proposals == sampled_actions(config, 8)
    assert len(proposals) == 8
    for action in proposals:
        assert math.hypot(action.tip_side, action.tip_vertical) <= 0.5 + 1e-12
        assert config.min_cue_speed_mps <= action.cue_speed_mps <= config.max_cue_speed_mps
        assert unit_action(action_unit(action, config), config).model_dump() == pytest.approx(
            action.model_dump()
        )
    with pytest.raises(ValueError, match="cube"):
        unit_action(np.asarray([1, 0, -1, 0], dtype=np.float64), config)


def test_bank_contains_only_goal_independent_public_actions():
    first = make_action_bank("case-one", count=8, seed=7)
    second = make_action_bank("case-two", count=8, seed=7)
    assert first.candidates == second.candidates
    assert len({candidate.candidate_id for candidate in first.candidates}) == 8
    assert {"goal", "witness_action", "truth", "events"}.isdisjoint(first.model_dump())


@pytest.mark.parametrize("method", ["random", "sobol", "geometry_powell"])
def test_search_has_bounded_objective_and_real_evaluation_counts(method, tmp_path, study_fixture):
    simulator, _, case = study_fixture
    ledger = TrialLedger(
        tmp_path / f"{method}.sqlite",
        BudgetLimits(max_model_requests=0, max_physics_evaluations=4),
    )
    result = search_goal(
        case.goal_input(),
        SearchConfig(method=method, seed=3, max_objective_calls=4),
        simulator,
        ledger,
        namespace="engineering",
    )
    assert len(result.attempts) <= 4
    assert ledger.counts()["model_requests"] == 0
    assert result.actual_new_physics_evaluations == ledger.counts()["physics_evaluations"]
    assert result.actual_new_physics_evaluations <= 4
    if result.selected_action is not None:
        assert result.selected_action in [attempt.action for attempt in result.attempts]
    ledger.close()


def test_geometry_seed_solves_analytical_fixture_without_privileged_action(study_fixture):
    _, record, case = study_fixture
    config = SearchConfig(method="geometry_powell", seed=3, max_objective_calls=8)
    seed = geometry_actions(case.goal_input(), config)[0]
    assert seed.azimuth_deg == pytest.approx(90)
    assert seed.cue_speed_mps == pytest.approx(record.witness_action.cue_speed_mps, rel=1e-8)
    assert seed.tip_side == seed.tip_vertical == 0


def test_zero_budget_does_not_invent_a_selected_action(tmp_path, study_fixture):
    simulator, _, case = study_fixture
    ledger = TrialLedger(
        tmp_path / "zero.sqlite", BudgetLimits(max_model_requests=0, max_physics_evaluations=0)
    )
    result = search_goal(
        case.goal_input(),
        SearchConfig(method="sobol", seed=0, max_objective_calls=0),
        simulator,
        ledger,
        namespace="zero",
    )
    assert result.stop_reason == "zero_budget"
    assert result.selected_action is None and not result.selected_goal_success
    assert not result.attempts and ledger.counts()["physics_evaluations"] == 0
    ledger.close()


def test_hidden_physics_cannot_be_queried_as_ordinary_adaptation(tmp_path, study_fixture):
    simulator, record, _ = study_fixture
    case = make_study_case(record, simulator, withheld_parameters=("rolling_friction",))
    ledger = TrialLedger(
        tmp_path / "hidden.sqlite", BudgetLimits(max_model_requests=0, max_physics_evaluations=1)
    )
    with pytest.raises(ValueError, match="declared public"):
        search_goal(
            case.goal_input(),
            SearchConfig(method="random", seed=0, max_objective_calls=1),
            simulator,
            ledger,
            namespace="hidden",
        )
    assert ledger.counts()["physics_attempts"] == 0
    ledger.close()


def test_search_restarts_from_cached_feedback_without_duplicate_rollouts(tmp_path, study_fixture):
    simulator, _, case = study_fixture
    path = tmp_path / "resume.sqlite"
    limits = BudgetLimits(max_model_requests=0, max_physics_evaluations=4)
    config = SearchConfig(method="sobol", seed=7, max_objective_calls=4)
    ledger = TrialLedger(path, limits)
    first = search_goal(case.goal_input(), config, simulator, ledger, namespace="same-job")
    count = simulator.rollouts
    ledger.close()
    ledger = TrialLedger(path, limits)
    second = search_goal(case.goal_input(), config, simulator, ledger, namespace="same-job")
    assert second.selected_action == first.selected_action
    assert simulator.rollouts == count
    assert second.actual_new_physics_evaluations == 0
    assert all(attempt.cache_hit for attempt in second.attempts)
    ledger.close()


def test_search_and_hidden_judge_have_separate_cache_namespaces(tmp_path, study_fixture):
    simulator, record, _ = study_fixture
    ledger = TrialLedger(
        tmp_path / "roles.sqlite", BudgetLimits(max_model_requests=0, max_physics_evaluations=2)
    )
    replay, cached = recorded_replay(
        record.initial,
        record.witness_action,
        simulator,
        ledger,
        namespace="search",
        role="agent_search",
    )
    assert replay.goal == record.goal and not cached
    _, cached = recorded_replay(
        record.initial,
        record.witness_action,
        simulator,
        ledger,
        namespace="judge",
        role="hidden_judge",
    )
    assert not cached and ledger.counts()["physics_evaluations"] == 2
    assert ledger.physics_usage_by_role() == {
        "agent_search": {"attempts": 1, "evaluations": 1, "unresolved": 0},
        "hidden_judge": {"attempts": 1, "evaluations": 1, "unresolved": 0},
    }
    ledger.close()


def test_reserved_physics_is_not_automatically_reexecuted(tmp_path):
    ledger = TrialLedger(
        tmp_path / "interrupted.sqlite",
        BudgetLimits(max_model_requests=0, max_physics_evaluations=1),
    )
    ledger.reserve_cached_physics("interrupted-key")
    with pytest.raises(UnresolvedRequest, match="Interrupted"):
        ledger.reserve_cached_physics("interrupted-key")
    with pytest.raises(BudgetExhausted):
        ledger.reserve_cached_physics("new-key")
    ledger.close()


def test_simulation_failures_remain_failures_after_resume(tmp_path, study_fixture, monkeypatch):
    simulator, record, _ = study_fixture

    def fail(initial, action):
        simulator.rollouts += 1
        raise SimulationFailure("test event cap")

    monkeypatch.setattr(simulator, "replay", fail)
    ledger = TrialLedger(
        tmp_path / "failure.sqlite", BudgetLimits(max_model_requests=0, max_physics_evaluations=1)
    )
    for _ in range(2):
        with pytest.raises(SimulationFailure):
            recorded_replay(
                record.initial, record.witness_action, simulator, ledger, namespace="failure"
            )
    assert ledger.counts()["physics_evaluations"] == 1
    ledger.close()


def test_reserved_oracle_evidence_cannot_be_reclassified_as_search(tmp_path):
    ledger = TrialLedger(
        tmp_path / "role.sqlite",
        BudgetLimits(max_model_requests=0, max_physics_evaluations=1),
    )
    ledger.reserve_cached_physics("same-key", role="hidden_judge")
    with pytest.raises(ValueError, match="another or legacy role"):
        ledger.reserve_cached_physics("same-key", role="agent_search")
    assert ledger.physics_usage_by_role()["hidden_judge"]["unresolved"] == 1
    ledger.close()


def test_legacy_cache_migration_preserves_unknown_cost_role(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE physics (id INTEGER PRIMARY KEY,status TEXT,evaluations INTEGER);
            CREATE TABLE physics_cache (
                cache_key TEXT PRIMARY KEY,reservation INTEGER UNIQUE,
                replay_json TEXT,error_code TEXT
            );
            INSERT INTO physics VALUES (1,'complete',1);
            INSERT INTO physics_cache VALUES ('legacy-key',1,NULL,'simulation_failure');
            """
        )
    ledger = TrialLedger(path, BudgetLimits(max_model_requests=0, max_physics_evaluations=2))
    assert ledger.physics_usage_by_role() == {
        "legacy_unclassified": {"attempts": 1, "evaluations": 1, "unresolved": 0}
    }
    with pytest.raises(ValueError, match="another or legacy role"):
        ledger.reserve_cached_physics("legacy-key", role="agent_search")
    ledger.close()
