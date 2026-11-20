"""Export and validate a full PEFT LoRA adapter from FSDP2's collective state."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from inverser.runtime import file_sha256, write_json
from inverser.study_calls import require_data_path


def canonical_lora_state(state: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, tensor in state.items():
        name = key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        if name in result:
            raise ValueError(f"Duplicate PEFT parameter: {name}")
        result[name] = tensor
    if result.keys() != reference.keys() or any(
        result[key].shape != tensor.shape for key, tensor in reference.items()
    ):
        raise ValueError("Adapter tensor inventory or shapes do not match the initial LoRA")
    return result


def export_fsdp_adapter(checkpoint: Path, reference: Path, output: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    for path in (checkpoint, reference, output):
        require_data_path(path)
    if output.exists():
        raise ValueError("Adapter export may not overwrite a checkpoint")
    weights = checkpoint / "pytorch_model_fsdp.bin"
    initial = load_file(reference / "adapter_model.safetensors")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    normalized = canonical_lora_state(state, initial)
    if not all(torch.isfinite(tensor).all() for tensor in normalized.values()):
        raise ValueError("Adapter contains non-finite parameters")
    params = json.loads((checkpoint / "adapter_config.json").read_text())
    base_params = json.loads((reference / "adapter_config.json").read_text())
    for key in ("base_model_name_or_path", "r", "lora_alpha", "target_modules"):
        if params[key] != base_params[key]:
            raise ValueError(f"Adapter configuration changed: {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".adapter-export-", dir=output.parent) as temp:
        staging = Path(temp) / "adapter"
        staging.mkdir()
        shutil.copyfile(checkpoint / "adapter_config.json", staging / "adapter_config.json")
        artifact = staging / "adapter_model.safetensors"
        save_file({key: value.contiguous() for key, value in normalized.items()}, artifact)
        report = {
            "status": "complete",
            "tensors": len(normalized),
            "changed_tensors": sum(not torch.equal(t, initial[k]) for k, t in normalized.items()),
            "collective_sha256": file_sha256(weights),
            "initial_adapter_sha256": file_sha256(reference / "adapter_model.safetensors"),
            "files": {
                name: file_sha256(staging / name)
                for name in ("adapter_model.safetensors", "adapter_config.json")
            },
        }
        write_json(staging / "manifest.json", report)
        staging.rename(output)
    return report


def merge_adapter(base: Path, adapter: Path, output: Path, *, device: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from inverser.runtime import check_storage

    for path in (adapter, output):
        require_data_path(path)
    if output.exists() or device not in ("cpu", "cuda"):
        raise ValueError("Merge requires a new checkpoint directory and a supported device")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA merge requested without a visible GPU")
    weights = adapter / "adapter_model.safetensors"
    base_weights = sorted(base.glob("model*.safetensors"))
    if not base_weights:
        raise ValueError("Base checkpoint has no safetensors files")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if Path(config["base_model_name_or_path"]).resolve() != base.resolve():
        raise ValueError("Adapter is not declared for the given base")
    total = sum(path.stat().st_size for path in base_weights)
    check_storage(total + 1_000_000_000)
    policy = AutoModelForImageTextToText.from_pretrained(
        base, local_files_only=True, dtype=torch.bfloat16, device_map={"": device}
    )
    peft = PeftModel.from_pretrained(policy, adapter, local_files_only=True)
    # PEFT can silently omit missing adapter tensors when loading an incomplete rank save.
    from peft import get_peft_model_state_dict

    canonical_lora_state(load_file(weights), get_peft_model_state_dict(peft))
    merged = peft.merge_and_unload(safe_merge=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".merged-export-", dir=output.parent) as temp:
        staging = Path(temp) / "model"
        merged.save_pretrained(staging)
        AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(staging)
        report = {
            "status": "complete",
            "adapter_sha256": file_sha256(weights),
            "dtype": "bfloat16",
            "merge_device": device,
            "base_config_sha256": file_sha256(base / "config.json"),
            "base_weight_sha256": {path.name: file_sha256(path) for path in base_weights},
            "files": {p.name: file_sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
        }
        write_json(staging / "manifest.json", report)
        staging.rename(output)
    return report
