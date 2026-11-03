import json

import pytest

from inverser import study_calls
from inverser.formal_panel import (
    FormalTestConfig,
    formal_requests,
    generate_test_records,
    load_formal_call,
    prepare_formal_test,
    validate_formal_test,
)
from inverser.runtime import write_json

SMALL = FormalTestConfig(per_family=1, bev_per_family=1, feedback_per_family=1, candidates=2)


def formal_panel(tmp_path, monkeypatch, excluded=()):
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(study_calls, "check_storage", lambda reserve: None)
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps({"group_id": group}) + "\n" for group in excluded))
    output = tmp_path / "data/formal"
    return dev, output, prepare_formal_test(dev, output, SMALL)


def test_formal_test_is_fresh_balanced_frozen_and_hides_goals(tmp_path, monkeypatch):
    dev, output, report = formal_panel(tmp_path, monkeypatch)
    assert report["layouts"] == 4 and report["request_records"] == 4 * 8 + 4 * 5
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["partition"] == "formal_test" and manifest["dev_overlapping_groups"] == 0
    assert set(manifest["family_counts"]) == {"free_roll", "center", "vertical", "side"}
    for path in (output / "private").glob("case-*.json"):
        labels = json.loads(path.read_text())
        assert labels["source_task_id"].startswith("formal-test-20261021-")
        outcomes = {o["candidate_id"]: o for o in labels["bank"]["outcomes"]}
        assert outcomes[labels["bank"]["witness_candidate_id"]]["goal_success"]
        effect = labels["intervention"]["effect_radii"]
        assert labels["intervention"]["weak"] or effect is None or effect >= 2.0
    for request in formal_requests(output):
        task = json.loads(load_formal_call(output, request).prompt())["task"]
        if request.kind in ("forward", "bank_forecast"):
            assert "goal" not in task
        if request.view == "bev":
            assert "initial" not in task
    assert prepare_formal_test(dev, output, SMALL)["status"] == "frozen_formal_test_validated"
    with pytest.raises(ValueError, match="new formal-test version"):
        prepare_formal_test(dev, output, SMALL.model_copy(update={"seed": 1}))
    ranking = next((output / "public").rglob("ranking.json"))
    value = json.loads(ranking.read_text())
    value["goal"]["tolerance_radii"] = 2.0
    write_json(ranking, value)
    with pytest.raises(ValueError, match="artifact changed"):
        validate_formal_test(output)


def test_layouts_already_used_in_development_are_skipped():
    records, stats = generate_test_records(SMALL, set())
    again, stats_again = generate_test_records(SMALL, {records[0].group_id})
    assert records[0].group_id not in {record.group_id for record in again}
    assert stats_again["skipped_overlap"] == 1 and "skipped_overlap" not in stats
