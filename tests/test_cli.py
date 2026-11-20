import pytest

from inverser import cli


def test_cli_exposes_core_vlm_and_capo_commands(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["inverser", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for cmd in ("prepare-vlm-suite", "run-vlm-eval", "train-capo-sft", "train-capo-grpo"):
        assert cmd in help_text
    assert "formal-suite" not in help_text

