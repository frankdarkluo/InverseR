import json

import pytest

import inverser.pilot as pilot


def test_pilot_roundtrip_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "check_storage", lambda reserve: None)
    output = tmp_path / "pilot"
    first = pilot.generate_pilot(output, count=4, seed=42)
    second = pilot.generate_pilot(output, count=4, seed=42)
    assert first == second
    assert first["accepted_tasks"] == 4
    report = pilot.audit_pilot(output, replay_count=4)
    assert report["tasks_validated"] == 4
    assert report["witnesses_replayed"] == 4
    for line in (output / "public.jsonl").read_text().splitlines():
        assert "witness_action" not in json.loads(line)
        assert "events" not in json.loads(line)
    with pytest.raises(ValueError, match="Resume refused"):
        pilot.generate_pilot(output, count=5, seed=42)


def test_interrupted_generation_resumes_without_duplicate_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "check_storage", lambda reserve: None)
    original = pilot.sample_case

    def interrupt(seed, index, candidate, config):
        if index == 1:
            raise RuntimeError("test interruption")
        return original(seed, index, candidate, config)

    monkeypatch.setattr(pilot, "sample_case", interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        pilot.generate_pilot(tmp_path / "resume", count=3, seed=8)
    manifest = json.loads((tmp_path / "resume" / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["accepted_tasks"] == 1
    monkeypatch.setattr(pilot, "sample_case", original)
    resumed = pilot.generate_pilot(tmp_path / "resume", count=3, seed=8)
    clean = pilot.generate_pilot(tmp_path / "clean", count=3, seed=8)
    assert resumed["exports"] == clean["exports"]


def test_pre_gate_size_limit_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="pre-G1"):
        pilot.generate_pilot(tmp_path, count=1_000_000, seed=0)
