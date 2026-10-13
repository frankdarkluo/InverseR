import json

import pytest

import inverser.corpus_catalog as catalog


def test_catalog_reference_does_not_promote_missing_real_actions():
    record = catalog.CorpusReference(
        sample_id="real",
        role="real_main_observation",
        source="author_dataset",
        game="snooker",
        source_group="shot-group",
        payload_file="record.json",
        quality_status="automatic_observations_unverified",
        manual_path_files=("manual_track.png",),
    )
    assert not record.measured_real_cue_action_available
    assert not record.frame_timed_manual_trajectory_available
    assert not record.independently_accepted_benchmark_task
    assert not record.simulated_cue_action_available
    with pytest.raises(ValueError):
        catalog.CorpusReference.model_validate(
            record.model_dump() | {"measured_real_cue_action_available": True}
        )
    with pytest.raises(ValueError, match="synthetic"):
        catalog.CorpusReference.model_validate(
            record.model_dump() | {"simulated_cue_action_available": True}
        )


def test_catalog_requires_complete_author_ingestion_not_a_planned_count(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(catalog, "check_storage", lambda *args: None)
    author = tmp_path / "data" / "author"
    author.mkdir(parents=True)
    (author / "latest.json").write_text(
        json.dumps(
            {
                "failed_this_run": 0,
                "ready_not_processed_due_limit": 10,
                "deferred_content_groups": 0,
            }
        )
    )
    with pytest.raises(ValueError, match="all author snooker groups"):
        catalog.build_corpus_catalog(
            author,
            tmp_path / "data" / "inventory.json",
            tmp_path / "data" / "matches",
            tmp_path / "data" / "screening",
            tmp_path / "data" / "synthetic",
            tmp_path / "data" / "catalog",
        )


def test_data_references_cannot_follow_symlinks_or_escape(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    target = root / "record"
    target.write_text("fixture")
    link = root / "linked"
    link.symlink_to(target)
    assert catalog._check_reference(target, root) == target
    with pytest.raises(ValueError, match="out-of-scope"):
        catalog._check_reference(link, root)
    outside = tmp_path / "other"
    outside.write_text("fixture")
    with pytest.raises(ValueError, match="out-of-scope"):
        catalog._check_reference(outside, root)
