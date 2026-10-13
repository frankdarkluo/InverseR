import csv
import json
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from inverser.author_uploads import UploadedAsset
from inverser.quality_screening import ScreeningDecision
from inverser.real_data import SnookerEpisode
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, write_json
from inverser.schemas import Record, ShotRecord


class CorpusReference(Record):
    sample_id: str
    role: Literal["real_main_observation", "auxiliary_capture", "synthetic_shot"]
    source: Literal["author_dataset", "match_videos", "pooltool_pilot"]
    game: Literal["snooker", "blackball", "carom"]
    source_group: str
    payload_file: str
    payload_line_1based: int | None = None
    original_file: str | None = None
    raw_capture_file: str | None = None
    endpoint_input_file: str | None = None
    manual_path_files: tuple[str, ...] = ()
    quality_status: str
    assistant_quality_hold: bool = False
    measured_real_cue_action_available: Literal[False] = False
    simulated_cue_action_available: bool = False
    frame_timed_manual_trajectory_available: Literal[False] = False
    independently_accepted_benchmark_task: Literal[False] = False

    @model_validator(mode="after")
    def check_label_role(self) -> Self:
        if self.simulated_cue_action_available != (self.role == "synthetic_shot"):
            raise ValueError("Exact simulator actions are available only in the synthetic stratum")
        if self.role == "auxiliary_capture" and self.game == "snooker":
            raise ValueError("The main snooker source must not be silently classified as auxiliary")
        if self.role != "auxiliary_capture" and self.game != "snooker":
            raise ValueError("Other game domains do not count toward the main snooker benchmark")
        return self


def _check_reference(path: str | Path, data_root: Path) -> Path:
    result = Path(path)
    if (
        result.is_symlink()
        or not result.is_file()
        or not result.resolve().is_relative_to(data_root)
    ):
        raise ValueError(f"Missing or out-of-scope data reference: {result}")
    return result


def build_corpus_catalog(
    author: Path,
    author_inventory: Path,
    matches: Path,
    match_screening: Path,
    synthetic: Path,
    output: Path,
) -> dict[str, object]:
    data_root = (PROJECT_ROOT / "data").resolve()
    output = output.resolve()
    if not output.is_relative_to(data_root):
        raise ValueError("Research catalog must remain under project data")
    check_storage(100_000_000)
    output.mkdir(parents=True, exist_ok=False)
    references = []
    author_latest = json.loads(_check_reference(author / "latest.json", data_root).read_text())
    if (
        author_latest["failed_this_run"]
        or author_latest["ready_not_processed_due_limit"]
        or author_latest["deferred_content_groups"]
    ):
        raise ValueError(
            "Completed-upload catalog requires all author snooker groups ready and converted"
        )
    author_payload = _check_reference(author_latest["episodes_jsonl"], data_root)
    author_groups: set[str] = set()
    for line in author_payload.open():
        episode = SnookerEpisode.model_validate_json(line)
        if episode.source.group_id in author_groups:
            raise ValueError("Author aliases were counted as multiple recording groups")
        author_groups.add(episode.source.group_id)
        directory = Path(episode.media.initial_source).parent
        payload = _check_reference(directory / "structured.json", data_root)
        endpoints = _check_reference(directory / "endpoints.json", data_root)
        if SnookerEpisode.model_validate_json(payload.read_text()) != episode:
            raise ValueError("Author index payload and per-record observations disagree")
        manual = tuple(
            annotation.path
            for annotation in episode.source_annotations
            if annotation.kind == "manual_path_image"
        )
        for path in manual:
            _check_reference(path, data_root)
        references.append(
            CorpusReference(
                sample_id=f"author:{episode.episode_id}",
                role="real_main_observation",
                source="author_dataset",
                game="snooker",
                source_group=episode.source.group_id,
                payload_file=str(payload),
                original_file=episode.source.original_file,
                endpoint_input_file=str(endpoints),
                manual_path_files=manual,
                quality_status="automatic_observations_unverified",
            )
        )
    snapshot = json.loads(_check_reference(author_inventory, data_root).read_text())
    upload_root = Path(snapshot["source"])
    assets = {row["path"]: UploadedAsset.model_validate(row) for row in snapshot["assets"]}
    auxiliary: dict[tuple[Literal["blackball", "carom"], str], list[UploadedAsset]] = {}
    for asset in assets.values():
        game = asset.game
        if asset.role == "original_video" and game in ("blackball", "carom") and asset.stable:
            if asset.sha256 is None:
                raise ValueError("Stable auxiliary video has no complete hash")
            auxiliary.setdefault((game, asset.sha256), []).append(asset)
    for (game, digest), aliases in auxiliary.items():
        aliases.sort(key=lambda asset: ("manual" not in asset.path.lower(), asset.path))
        original = aliases[0]
        raw = assets.get(str(Path(original.path).with_suffix(".dat")))
        if raw is None or not raw.stable:
            raise ValueError("Completed auxiliary capture has no stable raw companion")
        original_path = _check_reference(upload_root / original.path, data_root)
        raw_path = _check_reference(upload_root / raw.path, data_root)
        manual = tuple(
            sorted(
                {
                    candidate.path
                    for alias in aliases
                    for candidate in assets.values()
                    if candidate.stable
                    and candidate.role == "manual_path_image"
                    and candidate.path.startswith(str(Path(alias.path).parent) + "/")
                }
            )
        )
        manual_paths = tuple(
            str(_check_reference(upload_root / path, data_root)) for path in manual
        )
        references.append(
            CorpusReference(
                sample_id=f"{game}:{digest}",
                role="auxiliary_capture",
                source="author_dataset",
                game=game,
                source_group=f"{game}:{raw.sha256}",
                payload_file=str(author_inventory.resolve()),
                original_file=str(original_path),
                raw_capture_file=str(raw_path),
                manual_path_files=manual_paths,
                quality_status="source_capture_not_converted_for_main_snooker",
            )
        )
    state = json.loads(_check_reference(matches / "collection_state.json", data_root).read_text())
    if state["status"] != "complete" or state["runner_active"]:
        raise ValueError("Match catalog requires complete raw collection")
    decisions = {
        row.episode_id: row
        for row in (
            ScreeningDecision.model_validate_json(line)
            for line in _check_reference(match_screening / "decisions.jsonl", data_root).open()
        )
    }
    triage = json.loads(
        _check_reference(match_screening / "assistant_triage.json", data_root).read_text()
    )
    held = set(triage["held_episode_ids"])
    match_payload = _check_reference(matches / "episodes.jsonl", data_root)
    match_count = 0
    for index, line in enumerate(match_payload.open(), 1):
        episode = SnookerEpisode.model_validate_json(line)
        decision = decisions.get(episode.episode_id)
        if decision is None:
            raise ValueError("Match record is not covered by the declared quality screen")
        directory = Path(episode.media.initial_source).parent
        endpoints = _check_reference(directory / "endpoints.json", data_root)
        references.append(
            CorpusReference(
                sample_id=f"match:{episode.episode_id}",
                role="real_main_observation",
                source="match_videos",
                game="snooker",
                source_group=episode.source.group_id,
                payload_file=str(match_payload),
                payload_line_1based=index,
                original_file=episode.source.original_file,
                endpoint_input_file=str(endpoints),
                quality_status=decision.disposition,
                assistant_quality_hold=episode.episode_id in held,
            )
        )
        match_count += 1
    if match_count != state["raw_candidates"] or match_count != len(decisions):
        raise ValueError("Match record and quality-decision counts disagree")
    synthetic_manifest = json.loads(
        _check_reference(synthetic / "manifest.json", data_root).read_text()
    )
    audit = json.loads(_check_reference(synthetic / "audit.json", data_root).read_text())
    if synthetic_manifest["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Synthetic source has no completed validation")
    private = _check_reference(synthetic / "privileged.jsonl", data_root)
    public = _check_reference(synthetic / "public.jsonl", data_root)
    for export_path in (private, public):
        if file_sha256(export_path) != synthetic_manifest["exports"][export_path.name]:
            raise ValueError("Synthetic export changed since its replay audit")
    synthetic_count = 0
    for index, line in enumerate(private.open(), 1):
        record = ShotRecord.model_validate_json(line)
        references.append(
            CorpusReference(
                sample_id=f"simulated:{record.task_id}",
                role="synthetic_shot",
                source="pooltool_pilot",
                game="snooker",
                source_group=record.group_id,
                payload_file=str(private),
                payload_line_1based=index,
                endpoint_input_file=str(public),
                quality_status="simulator_truth_development_pilot",
                simulated_cue_action_available=True,
            )
        )
        synthetic_count += 1
    if synthetic_count != synthetic_manifest["accepted_tasks"]:
        raise ValueError("Synthetic manifest and record counts disagree")
    if len({row.sample_id for row in references}) != len(references):
        raise ValueError("Research catalog contains repeated sample IDs")
    for role in ("real_main_observation", "auxiliary_capture", "synthetic_shot"):
        with (output / f"{role}.jsonl").open("w") as stream:
            for row in references:
                if row.role == role:
                    stream.write(row.model_dump_json() + "\n")
    with (output / "index.csv").open("w", newline="") as stream:
        fields = [
            "sample_id",
            "role",
            "source",
            "game",
            "source_group",
            "quality_status",
            "payload_file",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in references:
            values = row.model_dump()
            writer.writerow({key: values[key] for key in fields})
    summary = {
        "status": "cataloged_with_label_and_quality_boundaries",
        "total_index_references": len(references),
        "role_counts": dict(Counter(row.role for row in references)),
        "source_counts": dict(Counter(row.source for row in references)),
        "game_counts": dict(Counter(row.game for row in references)),
        "quality_counts": dict(Counter(row.quality_status for row in references)),
        "unique_author_main_recordings": len(author_groups),
        "author_main_recordings_with_manual_raster": sum(
            bool(row.manual_path_files)
            for row in references
            if row.source == "author_dataset" and row.role == "real_main_observation"
        ),
        "auxiliary_recordings_with_manual_raster_by_game": dict(
            Counter(
                row.game
                for row in references
                if row.role == "auxiliary_capture" and row.manual_path_files
            )
        ),
        "measured_real_action_records": 0,
        "frame_timed_manual_trajectory_records": 0,
        "known_simulator_action_records": synthetic_count,
        "independently_accepted_real_benchmark_records": 0,
        "model_training_or_api_calls": 0,
        "data_payloads_copied_or_modified": False,
        "source_manifests": {
            str(path.resolve()): file_sha256(path)
            for path in (
                author / "latest.json",
                author_inventory,
                matches / "collection_manifest.json",
                match_screening / "summary.json",
                synthetic / "manifest.json",
            )
        },
        "interpretation": (
            "Index rows are not a million-task dataset or independent physical GT. "
            "Auxiliary domains, weak real observations and known simulator controls "
            "remain separate."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary
