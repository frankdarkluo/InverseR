from types import SimpleNamespace

import pytest

from inverser.capo_checkpoint import canonical_lora_state


def test_collective_adapter_export_refuses_partial_rank_artifact():
    a = SimpleNamespace(shape=(64, 4096))
    b = SimpleNamespace(shape=(1024, 64))
    reference = {"decoder.lora_A.weight": a, "decoder.lora_B.weight": b}
    collective = {"decoder.lora_A.default.weight": a, "decoder.lora_B.default.weight": b}
    assert canonical_lora_state(collective, reference) == reference
    with pytest.raises(ValueError, match="inventory or shapes"):
        canonical_lora_state({"decoder.lora_A.default.weight": a}, reference)
    with pytest.raises(ValueError, match="inventory or shapes"):
        canonical_lora_state(collective | {"decoder.lora_B.default.weight": a}, reference)
