# Local Data & Reproducible Manifests

This directory stores generated `InverseR-VLM` splits, external benchmark snapshots, and post-training/evaluation outputs. Large binary payloads (`.png`, `.safetensors`, `.sqlite`, and private blind-test labels) remain gitignored, while lightweight SHA-256 manifests and statistical analysis tables are version-controlled for reproducibility:

- `research/inverser/{sft,rl,valid,test}/manifest.json` — Current InverseR splits (`research/inverser-v3/` is a transitional copy used by an in-flight RL run) (layout-group 80:10:10, SFT:RL 80:20, ~2/3 contact goals, T5 capped at 3 attempts) with SHA-256 digests of every file; see `docs/research/inverser_v3.md`. The legacy `inverser-vlm-v1` splits were deleted.
- `research/capo-sft-tag-400step-bs32-seed20261119-v1/{manifest.json,summary.json}` — 4-GPU 400-step non-leaking SFT training receipt.
- `research/capo-valid-base-vs-sft400-analysis-v1/` — Paired bootstrap evaluation (`analysis.json`, `table.csv`, and SVG plots) comparing Base `Qwen3.5-9B` vs. 400-step SFT on `valid`.
- `external/` — Local snapshots of `PhysBench`, `DeepPHY`, `phyre`, `cuetip`, and `dino_wm` verified via `python -m inverser verify-external-suites`.
