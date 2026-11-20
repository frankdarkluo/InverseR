"""Verify an explicit, pre-inference freeze receipt for InverseR-VLM test evaluation."""

import json
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.study_calls import StudyBackendConfig, producer_hashes, require_data_path


class FrozenArm(Record):
    arm: str = Field(min_length=1)
    seed: int = Field(ge=0)
    backend_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_summary: Path
    selection_summary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class BlindProtocol(Record):
    test_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    valid_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    producers: dict[str, str]
    arms: tuple[FrozenArm, ...] = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        body = self.model_dump(mode="json", exclude={"protocol_sha256"})
        if fingerprint(body) != self.protocol_sha256:
            raise ValueError("Blind protocol changed after freeze")
        if len({(arm.arm, arm.seed) for arm in self.arms}) != len(self.arms):
            raise ValueError("Freeze receipt has duplicate arm/seed entries")
        return self


def freeze_blind_protocol(
    path: Path, test_manifest: Path, valid_manifest: Path, arms_path: Path
) -> BlindProtocol:
    require_data_path(path)
    if path.exists():
        raise ValueError("Existing blind protocols are immutable")
    for manifest, split in ((test_manifest, "test"), (valid_manifest, "valid")):
        require_data_path(manifest)
        if json.loads(manifest.read_text()).get("split") != split:
            raise ValueError("Wrong partition manifest in freeze")
    arms = [FrozenArm.model_validate(item) for item in json.loads(arms_path.read_text())]
    valid_hash = file_sha256(valid_manifest)
    for arm in arms:
        verify_arm_evidence(arm, valid_hash)
    body = {
        "test_manifest_sha256": file_sha256(test_manifest),
        "valid_manifest_sha256": valid_hash,
        "producers": evaluation_producers(),
        "arms": [arm.model_dump(mode="json") for arm in arms],
    }
    receipt = BlindProtocol.model_validate(body | {"protocol_sha256": fingerprint(body)})
    write_json(path, receipt.model_dump(mode="json"))
    return receipt


def evaluation_producers() -> dict[str, str]:
    return producer_hashes(
        "blind_protocol.py", "vlm_eval.py", "vlm_suite.py", "capo_env.py", "study_calls.py"
    )


def verify_blind_protocol(
    path: Path, test_manifest: Path, backend: StudyBackendConfig, arm: str, seed: int
) -> str:
    require_data_path(path)
    receipt = BlindProtocol.model_validate_json(path.read_text())
    if file_sha256(test_manifest) != receipt.test_manifest_sha256:
        raise ValueError("The test manifest is not the frozen dataset")
    if receipt.producers != evaluation_producers():
        raise ValueError("Evaluation source differs from the frozen protocol")
    chosen = next((item for item in receipt.arms if (item.arm, item.seed) == (arm, seed)), None)
    if chosen is None:
        raise ValueError("Requested model arm/seed was not frozen")
    if chosen.backend_sha256 != fingerprint(backend.model_dump(mode="json")):
        raise ValueError("Requested backend config differs from the frozen arm")
    if backend.model_revision != "checkpoint-manifest-sha256:" + (
        chosen.checkpoint_manifest_sha256
    ):
        raise ValueError("The served backend must attest its frozen checkpoint manifest")
    verify_arm_evidence(chosen, receipt.valid_manifest_sha256)
    return receipt.protocol_sha256


def verify_arm_evidence(chosen: FrozenArm, valid_hash: str) -> None:
    for candidate, digest in (
        (chosen.checkpoint_manifest, chosen.checkpoint_manifest_sha256),
        (chosen.selection_summary, chosen.selection_summary_sha256),
    ):
        require_data_path(candidate)
        if file_sha256(candidate) != digest:
            raise ValueError("Checkpoint or selection receipt changed after freeze")
    checkpoint = json.loads(chosen.checkpoint_manifest.read_text())
    assets = checkpoint.get("files", {})
    if not assets or not any(name.endswith(".safetensors") for name in assets):
        raise ValueError("Checkpoint manifest must pin weight files")
    for filename, digest in assets.items():
        candidate = chosen.checkpoint_manifest.parent / filename
        if not candidate.resolve().is_relative_to(chosen.checkpoint_manifest.parent.resolve()):
            raise ValueError("Checkpoint asset points outside its manifest directory")
        if file_sha256(candidate) != digest:
            raise ValueError("A frozen checkpoint asset changed")
    selected = json.loads(chosen.selection_summary.read_text())
    if (
        selected.get("status") != "complete"
        or selected.get("valid_manifest_sha256") != valid_hash
        or selected.get("checkpoint_manifest_sha256") != chosen.checkpoint_manifest_sha256
    ):
        raise ValueError("Checkpoint selection does not attest a complete valid evaluation")
