import base64
import json
import logging
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from inverser.metrics import score_endpoint
from inverser.model_adapters import ModelCall, RenderedAsset, rendered_asset
from inverser.physics import SnookerSimulator
from inverser.preview import render_layout
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Record, ShotRecord
from inverser.study import (
    ForwardInput,
    GoalInput,
    make_study_case,
    replay_forward_truth,
    translating_ball_ids,
)
from inverser.world_model_data import FactualForecastInput, PrefixPoint

LOGGER = logging.getLogger(__name__)
FAMILIES = ("free_roll", "center", "vertical", "side")


class DiagnosticPanelConfig(Record):
    schema_version: Literal["development-diagnostic-panel-v1"] = "development-diagnostic-panel-v1"
    seed: int = Field(default=20261014, ge=0)
    synthetic_layouts: int = Field(default=64, ge=4, le=64)
    real_round_cases: int = Field(default=64, ge=4, le=64)
    sample_times_s: tuple[float, float, float] = (0.25, 0.5, 1.0)
    tolerance_radii: float = Field(default=1.0, ge=1.0, le=1.0)
    native_tool_budget: Literal[0] = 0

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        if self.synthetic_layouts % 4:
            raise ValueError("Synthetic development layouts must be balanced across four families")
        if not 0 <= self.sample_times_s[0] < self.sample_times_s[1] < self.sample_times_s[2]:
            raise ValueError("Panel sampling times must be nonnegative and strictly increasing")
        return self


class RealForecastLabels(Record):
    case_id: str
    source_file: str
    source_sha256: str
    source_group: str
    source_label: str
    future: tuple[PrefixPoint, ...] = Field(min_length=3, max_length=3)
    measured_cue_action_available: Literal[False] = False
    source_calibration_independently_verified: Literal[False] = False


class PanelRequest(Record):
    request_id: str
    case_id: str
    task: Literal["synthetic_forward", "synthetic_goal", "real_factual_forecast"]
    condition: Literal["base_action", "intervened_action", "goal", "prefix"]
    view: Literal["structured", "bev", "prefix"]
    payload_file: str
    image_files: tuple[str, ...] = ()
    local_only: bool
    model_adapter_ready: bool
    api_common_subset: bool

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        paths = (self.payload_file, *self.image_files)
        if any(
            not path.startswith("public/") or ".." in Path(path).parts or Path(path).is_absolute()
            for path in paths
        ):
            raise ValueError("Model requests may reference only contained public panel artifacts")
        if self.task == "real_factual_forecast" and (
            not self.local_only or self.api_common_subset or self.model_adapter_ready
        ):
            raise ValueError(
                "Real-prefix requests remain local-only and need a future local adapter"
            )
        return self


def select_synthetic_records(source: Path, config: DiagnosticPanelConfig) -> tuple[ShotRecord, ...]:
    pools: dict[str, list[ShotRecord]] = {family: [] for family in FAMILIES}
    unique: dict[str, ShotRecord] = {}
    for line in source.open():
        record = ShotRecord.model_validate_json(line)
        key = fingerprint(record.initial.endpoint_view().model_dump(mode="json"))
        old = unique.get(key)
        priority = FAMILIES.index(record.family), record.group_id, record.task_id
        if old is None or priority < (FAMILIES.index(old.family), old.group_id, old.task_id):
            unique[key] = record
    for record in unique.values():
        pools[record.family].append(record)
    rng = np.random.default_rng(config.seed)
    chosen: list[ShotRecord] = []
    per_family = config.synthetic_layouts // 4
    for family in FAMILIES:
        pool = sorted(pools[family], key=lambda record: (record.group_id, record.task_id))
        if len(pool) < per_family:
            raise ValueError(f"Source has too few distinct {family} layouts for the fixed panel")
        indices = rng.permutation(len(pool))[:per_family]
        chosen.extend(pool[int(index)] for index in indices)
    if len({record.group_id for record in chosen}) != len(chosen):
        raise ValueError(
            "Distinct selected layouts must also have distinct development source groups"
        )
    return tuple(chosen)


def load_synthetic_panel_call(output: Path, request: PanelRequest) -> ModelCall:
    if request.task == "real_factual_forecast" or request.view == "prefix":
        raise ValueError("Real factual prefix needs a separately approved local-only adapter")
    payload_path = output / request.payload_file
    structured_path = payload_path.with_name(
        payload_path.name.removesuffix(f"_{request.view}.json") + "_structured.json"
    )
    structured = json.loads(structured_path.read_text())
    data = (
        ForwardInput.model_validate(structured["task"])
        if request.task == "synthetic_forward"
        else GoalInput.model_validate(structured["task"])
    )
    assets = []
    for name in request.image_files:
        path = output / name
        with Image.open(path) as image:
            width, height = image.size
        assets.append(
            RenderedAsset.model_validate(
                {
                    "case_id": request.case_id,
                    "role": path.stem,
                    "png_base64": base64.b64encode(path.read_bytes()).decode(),
                    "sha256": file_sha256(path),
                    "width_px": width,
                    "height_px": height,
                }
            )
        )
    if request.view not in ("structured", "bev"):
        raise ValueError("Synthetic model calls require a structured or BEV view")
    call = ModelCall(view=request.view, input=data, images=tuple(assets))
    if json.loads(call.prompt()) != json.loads(payload_path.read_text()):
        raise ValueError("Loaded model call differs from the frozen visible prompt")
    return call


def intervention_action(record: ShotRecord) -> tuple[CueAction, str]:
    values = record.witness_action.model_dump(mode="json")
    if record.family == "free_roll":
        name = "cue_speed_mps"
        values[name] = record.witness_action.cue_speed_mps * 1.25
    elif record.family == "center":
        name = "azimuth_deg"
        values[name] = (record.witness_action.azimuth_deg + 15) % 360
    else:
        name = "tip_vertical" if record.family == "vertical" else "tip_side"
        original = getattr(record.witness_action, name)
        values[name] = original + (0.05 if original >= 0 else -0.05)
    return CueAction.model_validate(values), name


def select_real_cases(
    public_source: Path, private_source: Path, config: DiagnosticPanelConfig
) -> tuple[tuple[FactualForecastInput, RealForecastLabels], ...]:
    public = {
        value.case_id: value
        for value in (
            FactualForecastInput.model_validate_json(line) for line in public_source.open()
        )
    }
    private = {
        value.case_id: value
        for value in (
            RealForecastLabels.model_validate_json(line) for line in private_source.open()
        )
    }
    if set(public) != set(private):
        raise ValueError("Public factual prefixes and private future labels must have matching IDs")
    grouped: dict[str, list[tuple[FactualForecastInput, RealForecastLabels]]] = {}
    for case_id in sorted(public):
        data, labels = public[case_id], private[case_id]
        if tuple(point.time_s for point in labels.future) != data.query_times_s:
            raise ValueError("Private future annotations disagree with the public query times")
        grouped.setdefault(labels.source_group, []).append((data, labels))
    if len(grouped) < config.real_round_cases:
        raise ValueError("Too few distinct source-round groups for the fixed real-prefix panel")
    rng = np.random.default_rng(config.seed + 1)
    groups = sorted(grouped)
    chosen = []
    for index in rng.permutation(len(groups))[: config.real_round_cases]:
        values = grouped[groups[int(index)]]
        chosen.append(values[int(rng.integers(0, len(values)))])
    return tuple(chosen)


def validate_diagnostic_panel(output: Path) -> dict[str, object]:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["artifact_hashes"].items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Frozen panel contains an out-of-scope artifact")
        if file_sha256(path) != checksum:
            raise ValueError(f"Frozen panel artifact changed: {name}")
    requests = [
        PanelRequest.model_validate_json(line) for line in (output / "request_index.jsonl").open()
    ]
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Panel request IDs must be unique")
    for request in requests:
        payload = json.loads((output / request.payload_file).read_text())
        if request.task == "real_factual_forecast":
            FactualForecastInput.model_validate(payload)
        else:
            load_synthetic_panel_call(output, request)
            if request.task == "synthetic_forward" and "goal" in payload["task"]:
                raise ValueError("Blind forward prompt contains a hidden goal")
        if request.task == "synthetic_forward" and len(request.image_files) > 1:
            raise ValueError("Blind forward request cannot include goal images")
        if any(name in str(payload) for name in ("witness_action", "private_record")):
            raise ValueError("A model-visible panel payload contains privileged data")
    expected = manifest["config"]["synthetic_layouts"] * 6 + manifest["config"]["real_round_cases"]
    if len(requests) != expected:
        raise ValueError("Panel does not have the declared paired-view/request shape")
    return {
        "status": "frozen_development_panel_validated",
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "synthetic_layouts": manifest["config"]["synthetic_layouts"],
        "real_source_round_cases": manifest["config"]["real_round_cases"],
        "request_records": len(requests),
        "synthetic_request_records": sum(
            request.task != "real_factual_forecast" for request in requests
        ),
        "real_local_only_request_records": sum(request.local_only for request in requests),
        "api_common_subset_request_records": sum(request.api_common_subset for request in requests),
        "actual_model_calls": 0,
        "formal_test_freeze": False,
        "real_measured_control_gt_added": False,
    }


def prepare_diagnostic_panel(
    synthetic: Path,
    real_public: Path,
    real_private: Path,
    output: Path,
    config: DiagnosticPanelConfig,
) -> dict[str, object]:
    sources = {
        "synthetic": file_sha256(synthetic),
        "real_public": file_sha256(real_public),
        "real_private": file_sha256(real_private),
    }
    protocol = {
        "config": config.model_dump(mode="json"),
        "source_hashes": sources,
        "producer_hashes": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "diagnostic_panel.py",
                "physics.py",
                "study.py",
                "schemas.py",
                "model_adapters.py",
                "preview.py",
                "world_model_data.py",
                "runtime.py",
            )
        },
        "runtime_versions": {
            name: version(name) for name in ("numpy", "pydantic", "pooltool-billiards", "Pillow")
        },
        "partition": "development",
        "new_independent_physical_tasks_claimed": False,
        "actual_model_inference": False,
        "gpu_api_allocation_approved": False,
        "formal_g2_test_freeze": False,
        "model_generation_seed": 0,
    }
    protocol_hash = fingerprint(protocol)
    if output.is_symlink() or not output.resolve().is_relative_to(
        (PROJECT_ROOT / "data").resolve()
    ):
        raise ValueError("Development panels must remain inside project data")
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text())
        if existing["protocol_hash"] != protocol_hash:
            raise ValueError("Changed sources/code/protocol require a new diagnostic-panel version")
        return validate_diagnostic_panel(output)
    records = select_synthetic_records(synthetic, config)
    real = select_real_cases(real_public, real_private, config)
    check_storage(200_000_000)
    output.mkdir(parents=True, exist_ok=False)
    (output / "public").mkdir()
    (output / "private").mkdir()
    requests = []
    private_selection = []
    simulators: dict[str, SnookerSimulator] = {}
    per_family = config.synthetic_layouts // 4
    api_per_family = min(2, per_family)
    views: tuple[Literal["structured", "bev"], ...] = ("structured", "bev")
    for index, record in enumerate(records):
        physics_key = fingerprint(record.physics.model_dump(mode="json"))
        if physics_key not in simulators:
            simulators[physics_key] = SnookerSimulator(record.physics)
        simulator = simulators[physics_key]
        case = make_study_case(record, simulator)
        goal = case.goal_input()
        case_dir = output / "public" / case.case_id
        case_dir.mkdir()
        image_paths = {}
        for role, layout in (("initial", goal.initial), ("goal", goal.goal)):
            image, _ = render_layout(layout, goal.physics.table)
            path = case_dir / f"{role}.png"
            image.save(path)
            image_paths[role] = str(path.relative_to(output))
        original = simulator.replay(record.initial, record.witness_action)
        if not score_endpoint(original.goal, record.goal, 1e-8).success:
            raise ValueError("Selected development witness no longer reproduces its endpoint")
        if original.events != record.events:
            raise ValueError("Selected development witness no longer reproduces its events")
        altered, parameter = intervention_action(record)
        intervention = simulator.replay(record.initial, altered)
        labels = {}
        api_subset = index % per_family < api_per_family
        for condition, action, replay in (
            ("base_action", record.witness_action, original),
            ("intervened_action", altered, intervention),
        ):
            data = case.forward_input(action, config.sample_times_s)
            labels[condition] = {
                "action": action.model_dump(mode="json"),
                "truth": replay_forward_truth(
                    record.initial, replay, record.physics, config.sample_times_s
                ).model_dump(mode="json"),
                "moving_ball_ids": translating_ball_ids(record.initial, replay.motion),
            }
            for view in views:
                call = ModelCall(
                    view=view,
                    input=data,
                    images=(rendered_asset(data, "initial", data.physics.table),)
                    if view == "bev"
                    else (),
                )
                payload = case_dir / f"{condition}_{view}.json"
                write_json(payload, json.loads(call.prompt()))
                requests.append(
                    PanelRequest.model_validate(
                        {
                            "request_id": fingerprint((case.case_id, condition, view))[:24],
                            "case_id": case.case_id,
                            "task": "synthetic_forward",
                            "condition": condition,
                            "view": view,
                            "payload_file": str(payload.relative_to(output)),
                            "image_files": (image_paths["initial"],) if view == "bev" else (),
                            "local_only": False,
                            "model_adapter_ready": True,
                            "api_common_subset": api_subset,
                        }
                    )
                )
        for view in views:
            call = ModelCall(
                view=view,
                input=goal,
                images=(
                    rendered_asset(goal, "initial", goal.physics.table),
                    rendered_asset(goal, "goal", goal.physics.table),
                )
                if view == "bev"
                else (),
            )
            payload = case_dir / f"goal_{view}.json"
            write_json(payload, json.loads(call.prompt()))
            requests.append(
                PanelRequest.model_validate(
                    {
                        "request_id": fingerprint((case.case_id, "goal", view))[:24],
                        "case_id": case.case_id,
                        "task": "synthetic_goal",
                        "condition": "goal",
                        "view": view,
                        "payload_file": str(payload.relative_to(output)),
                        "image_files": (image_paths["initial"], image_paths["goal"])
                        if view == "bev"
                        else (),
                        "local_only": False,
                        "model_adapter_ready": True,
                        "api_common_subset": api_subset,
                    }
                )
            )
        write_json(
            output / "private" / f"{case.case_id}.json",
            {
                "source_task_id": record.task_id,
                "source_group": record.group_id,
                "family": record.family,
                "record": record.model_dump(mode="json"),
                "intervened_parameter": parameter,
                "forward_labels": labels,
            },
        )
        private_selection.append(
            {
                "case_id": case.case_id,
                "source_group": record.group_id,
                "family": record.family,
                "initial_hash": fingerprint(record.initial.endpoint_view().model_dump(mode="json")),
            }
        )
    for real_input, real_labels in real:
        payload = output / "public" / f"{real_input.case_id}.json"
        write_json(payload, real_input.model_dump(mode="json"))
        write_json(
            output / "private" / f"{real_input.case_id}.json", real_labels.model_dump(mode="json")
        )
        requests.append(
            PanelRequest(
                request_id=fingerprint((real_input.case_id, "prefix"))[:24],
                case_id=real_input.case_id,
                task="real_factual_forecast",
                condition="prefix",
                view="prefix",
                payload_file=str(payload.relative_to(output)),
                local_only=True,
                model_adapter_ready=False,
                api_common_subset=False,
            )
        )
    with (output / "request_index.jsonl").open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(request.model_dump_json() + "\n")
    write_json(
        output / "private/selection.json",
        {
            "synthetic": private_selection,
            "real": [labels.model_dump(mode="json") for _, labels in real],
        },
    )
    manifest = {
        **protocol,
        "protocol_hash": protocol_hash,
        "synthetic_family_counts": dict(Counter(record.family for record in records)),
        "physics_replays": sum(simulator.rollouts for simulator in simulators.values()),
        "resolved_physics": [simulator.metadata() for simulator in simulators.values()],
        "artifact_hashes": {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    report = validate_diagnostic_panel(output)
    write_json(output / "validation.json", report)
    LOGGER.info(
        "Frozen development panel: %d simulation layouts, %d local-only real cases",
        len(records),
        len(real),
    )
    return report
