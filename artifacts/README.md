# Experiment Logs & W&B Storage

Runtime logs and experiment telemetry are organized into three subdirectories (ignored by Git):

- `sft/` — Supervised fine-tuning (`train-capo-sft` & adapter merge) logs.
- `capo/` — `CAPO` / `Outcome` / `Uncoupled` GRPO training, vLLM rollout server, and validation queue logs.
- `wandb/` — Local Weights & Biases run cache synced to `frankdarkluo/inverser` via live callbacks or `python -m inverser sync-wandb`.
