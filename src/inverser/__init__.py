"""Provenance-aware inverse snooker experiments."""

import importlib
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

__version__ = "0.1.0"
_PKG_DIR = Path(__file__).resolve().parent

# ponytail: flat module aliases keep existing tests and frozen runner scripts working unchanged;
# remove once external callers and tests migrate to explicit subpackage imports.
_SUBPACKAGES: dict[str, str] = {
    mod: pkg
    for pkg, mods in {
        "core": ("archive", "metrics", "physics", "pilot", "preview", "schemas"),
        "benchmark": ("blind_protocol", "external_benchmarks", "study", "vlm_suite"),
        "capo": (
            "capo_checkpoint",
            "capo_env",
            "capo_grpo",
            "capo_selection",
            "capo_trainer",
            "training_env",
        ),
        "eval": (
            "agentic_baselines",
            "capo_figures",
            "diagnostics",
            "evaluation",
            "iphyre_eval",
            "learned_wm",
            "model_adapters",
            "paper_analysis",
            "public_eval",
            "retention_eval",
            "self_verify",
            "study_calls",
            "vlm_eval",
            "wandb_tracker",
        ),
    }.items()
    for mod in mods
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        mod = importlib.import_module(self._target)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module: ModuleType) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        prefix, dot, name = fullname.rpartition(".")
        if (
            prefix == "inverser"
            and dot
            and name in _SUBPACKAGES
            and (_PKG_DIR / _SUBPACKAGES[name] / f"{name}.py").is_file()
        ):
            canonical = f"inverser.{_SUBPACKAGES[name]}.{name}"
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(canonical))
        return None


if not any(type(finder).__name__ == "_AliasFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> ModuleType:
    if name in _SUBPACKAGES and (_PKG_DIR / _SUBPACKAGES[name] / f"{name}.py").is_file():
        mod = importlib.import_module(f"inverser.{_SUBPACKAGES[name]}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'inverser' has no attribute {name!r}")

