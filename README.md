# InverseR: 2D-to-3D Physical World Modeling & Causally-Grounded Agentic Post-Training (CAPO)

**InverseR** studies whether multimodal foundation models can bridge the gap between **passive visual physics prediction** ("knowing what will happen") and **continuous 3D physical control & online adaptation** ("knowing how to act") directly from 2D camera observations.

Built on deterministic rigid-body contact, spin, and Coulomb friction dynamics (`Pooltool 0.6.0`), this repository provides:
1. **`InverseR-VLM v1`**: A leak-free 2D-to-3D visual physics benchmark featuring **Counterfactual Twin** scenes, forward causal world-modeling tasks (`T1–T3`), single-shot continuous inverse control (`T4`), and interactive unknown-friction system identification (`T5`).
2. **`CAPO` (Causally-Grounded Agentic Post-Training)**: A multimodal SFT + 3-stage curriculum GRPO/DAPO post-training framework that eliminates reward hacking ("lucky shots with hallucinated physics") via a **Multiplicative Causal Gate** and multi-turn `try_shot` probing, together with a **Skill-Invoking Recursive Self-Improvement (`CAPO-RSI`)** roadmap ([`docs/research/rsi_skill_post_training_plan.md`](docs/research/rsi_skill_post_training_plan.md)).

---

## 1. Motivation

Existing Vision-Language Models (VLMs) and embodied post-training pipelines face four structural bottlenecks on continuous physical reasoning:

1. **The 2D-to-3D Grounding Gap**: Real cameras provide 2D images (angled + overhead views), not privileged 3D simulator coordinates. Off-the-shelf VLMs fail to map pixel geometry to metric contact angles, cue-ball throw, and post-impact spin deflection (`Qwen3.5-9B` achieves `< 1%` 1-ball-radius coordinate accuracy zero-shot).
2. **Inference-Time Scaffolding Fails When the Internal World Model Hallucinates**: Prior agentic wrappers (e.g., `CueTip`, `DeepPHY WM-Prompting`, `Self-Verify`) prompt frozen VLMs to predict outcomes before acting without updating model weights—often degrading control accuracy because hallucinated forecasts mislead action selection.
3. **Pure Outcome RL Encourages Causal Hallucination**: Standard outcome-only GRPO rewards potting or reaching target zones regardless of *how* the shot succeeded, reinforcing physically brittle flukes that collapse under distribution shifts.
4. **Unobservable Physical Regimes Require Active Probing**: Surface sliding/rolling friction and rail restitution cannot be inferred from a single static image; an embodied agent must execute diagnostic probes (`try_shot`), observe visual feedback, and adapt online (`T5`).

**Our Solution (`InverseR-VLM` + `CAPO` / `CAPO-RSI`)**:
- **Flat `<think>...</think><answer>...</answer>` Slot-Tag Contract**: Eliminates nested JSON-schema decoding collapse while bounding reasoning within `1024` tokens.
- **Cross-Modal Non-Leaking SFT Warmup**: Jointly adapts the visual-language projector (`visual.merger.linear_fc[12]`) and LLM attention layers (`q/k/v/o_proj`) using pure ghost-ball geometric derivations with zero ground-truth action leakage inside `<think>`.
- **Multiplicative Causal Gate & 3-Stage Curriculum RL**: Coupling inverse control reward with forward contact/trajectory fidelity ($R_{\text{CAPO}} = R_{\text{goal}} \times G_{\text{causal}}$) across a progressive physics curriculum (`Stage 1: free_roll` $\rightarrow$ `Stage 2: +center` $\rightarrow$ `Stage 3: +vertical/side + T5 interactive adaptation`), extensible to skill-invoking recursive self-improvement (`CAPO-RSI`).

---

## 2. Benchmark: `InverseR-VLM v1`

All splits are generated via forward witness simulation (`prepare-vlm-suite`), verified for zero layout-group overlap across splits and historical panels, and pinned by SHA-256 manifests under `data/research/inverser-vlm-v1/`:

| Split | Standard Scenes (`Anchor + Twin`) | OOD Unknown-Friction Scenes | Total Scenes | Families (`roll / center / vert / side`) | Primary Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`train`** | `24,000` (`12,000` pairs) | `0` | `24,000` | `6,000` per family | SFT warmup & 3-stage CAPO RL |
| **`valid`** | `240` (`120` pairs) | `0` | `240` | `60` per family | Hyperparameter & checkpoint selection |
| **`test`** | `600` (`300` pairs) | `120` (`slow` & `fast` friction) | `720` | Balanced across all families | Pre-frozen single-pass blind evaluation |

### Task Suite (`T1–T5`)
- **Forward World Modeling** (`INITIAL` + action ray/tip overlay $\rightarrow$ `<answer>`):
  - **`T1`**: Ball-to-ball collision mask & exact first cue-object contact.
  - **`T2`**: Ordered cushion bounce sequence (`LEFT`, `RIGHT`, `BOTTOM`, `TOP`) & cushion set.
  - **`T3`**: Dense 2D ball positions at `1s..5s` and stationary resting state (`1r = 2.62 cm` strict & `3r` normalized tolerances; composite score `S_total = 0.3 T1 + 0.3 T2 + 0.4 T3`).
- **Inverse Control & Online Adaptation** (`INITIAL` + `GOAL` $\rightarrow$ continuous 4D `CueAction`: `azimuth_deg`, `cue_speed_mps`, `tip_side`, `tip_vertical`):
  - **`T4` (Visual Inverse Control)**: Single-shot continuous control under known table parameters.
  - **`T5` (Interactive Adaptation)**: Up to `6` closed-loop `try_shot` interactions, including OOD tables with withheld `sliding_friction`, `rolling_friction`, and `cushion_restitution`.

---

## 3. Core Experimental Configuration

| Component | Configuration |
| :--- | :--- |
| **Backbone** | `Qwen3.5-9B` (`bfloat16`, `--reasoning-parser qwen3`, `thinking_token_budget=1024`) |
| **Hardware Topology** | **4× NVIDIA A100 80GB**: SFT uses 4-GPU DDP; GRPO/CAPO uses **2+2 split** (`GPUs 0,1`: `TP=2` vLLM rollout server on `:18021`; `GPUs 2,3`: 2-rank `FSDP2` trainer) |
| **LoRA Architecture** | `r=64`, `lora_alpha=128`, targets `language_model.*.(q_proj\|k_proj\|v_proj\|o_proj)` + `visual.merger.linear_fc[12]` (68 trainable tensors) |
| **Stage 0: SFT Warmup** | `400` optimizer steps, `per_device_bs=4`, `grad_accum=2`, `effective_batch_size=32` (`12,800` completion-only `<think>+<answer>` samples), `lr=2e-5` |
| **Stage 1: 3-Arm RL (`Outcome` / `Uncoupled` / `CAPO`)** | `200` steps (`stage_steps = 40 / 60 / 100`), `G=8` rollouts, `temp=0.8`, `lr=5e-6`, **`beta=0.0` (Zero-KL RLVR)**, `loss_type="dapo"`, `vllm_importance_sampling_mode="token_truncate"`, DAPO dynamic zero-variance pair resampling |
| **Experiment Tracking** | Weights & Biases (`frankdarkluo/inverser`), EMA curves + `log_metric` format/hit diagnostics + progressive `metrics.csv`, local cache under `artifacts/wandb/` |

### Validation Snapshot: Base vs. 400-Step Non-Leaking SFT (`inverser-vlm-v1/valid`, `N=240`)

| Arm | `T1` Exact | `T2` Set | `T3_1r` (`2.6cm`) | `T3_3r` | `S_total` / `S_total_3r` | `T4` Valid Action | `T5` (`<=5r` / `<=1r`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-9B` (Base)** | `48.3%` | `16.7%` | `0.95%` | `5.01%` | `18.8%` / `24.4%` | `60.0%` | `0.4%` / `0.0%` |
| **`Qwen3.5-9B` (400-Step SFT)** | **`56.7%`** | **`22.5%`** | **`14.88%` (`15.6×`)** | **`23.22%` (`4.6×`)** | **`28.1%` / `35.9%`** | **`91.2%`** | **`5.4%` / `0.83%`** |

*(Full paired bootstrap confidence intervals and SVG plots are archived in [`data/research/capo-valid-base-vs-sft400-analysis-v1/`](data/research/capo-valid-base-vs-sft400-analysis-v1/).)*

---

## 4. End-to-End Experimental Pipeline

### 4.1 Environment & W&B Setup

```bash
# 1. Core physics, dataset, evaluation, and test environment (.venv, Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install --index-url https://archive.panda3d.org/simple/ panda3d==1.11.0.dev3702
.venv/bin/pip install -e '.[dev,data,processing,inference]'

# 2. Training & vLLM serving environment (.venv-train: PyTorch + FSDP2 + TRL + PEFT + vLLM + wandb)
cp .env.example .env   # Fill in WANDB_API_KEY (gitignored)

# 3. Code quality & unit test suite
.venv/bin/ruff check src tests && .venv/bin/mypy && .venv/bin/python -m pytest -q
```

### 4.2 Dataset Generation & Integrity Verification

```bash
# Verify frozen train / valid / test splits against SHA-256 manifests (0 model calls)
.venv/bin/python -m inverser prepare-vlm-suite --output-root data/research --verify-only

# Verify external public benchmark assets (PhysBench, DeepPHY, I-PHYRE, Pooltool)
.venv/bin/python -m inverser verify-external-suites
```

### 4.3 Stage 0: 4-GPU Non-Leaking SFT Warmup & Adapter Merge

```bash
# 4-GPU DDP SFT (effective_batch_size=32, 400 steps; auto-streams to W&B as sft-qwen3.5-9b-...)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache" PYTHONPATH="$PWD/src" \
.venv-train/bin/torchrun --standalone --nproc-per-node=4 --module inverser train-capo-sft \
  --split data/research/inverser-vlm-v1/train \
  --model-path /data/v-guoqingluo/models/Qwen3.5-9B \
  --output data/research/capo-sft-tag-400step-bs32-seed20261119-v1 \
  --steps 400 --seed 20261119 \
  > artifacts/sft/capo-sft-tag-400step-bs32-seed20261119-v1.log 2>&1

# Merge verified 68-tensor LoRA adapter into a standalone bf16 checkpoint for vLLM serving
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" \
.venv-train/bin/python -m inverser merge-capo-adapter \
  --model-path /data/v-guoqingluo/models/Qwen3.5-9B \
  --adapter data/research/capo-sft-tag-400step-bs32-seed20261119-v1/checkpoint \
  --output data/research/capo-sft-tag-400step-merged-seed20261119-v1 \
  --device cuda
```

### 4.4 Stage 1: 4-GPU (`2+2`) CAPO & Ablation GRPO Training Queue

```bash
# Terminal 1 — Dedicated TP=2 vLLM rollout server on GPUs 0,1 (port 18021)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
VLLM_CACHE_ROOT="$PWD/.cache/vllm" HF_HOME="$PWD/.cache/huggingface" \
.venv-train/bin/python -m trl.scripts.vllm_serve \
  --model /data/v-guoqingluo/models/Qwen3.5-9B \
  --host 127.0.0.1 --port 18021 \
  --tensor_parallel_size 2 --gpu_memory_utilization 0.75 \
  --dtype bfloat16 --max_model_len 8192 --enforce_eager True \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 8}' \
  > artifacts/capo/vllm-rollout-18021.log 2>&1

# Terminal 2 — 2-rank FSDP2 3-Arm GRPO queue (outcome / uncoupled / capo) on GPUs 2,3
# Logs write to artifacts/capo/<RUN_ID>/; metrics stream live to W&B as capo-<mode>-...
CAPO_SEEDS="20261119" CAPO_FSDP_SMOKE="capo-grpo-tag-capo-4gpu-smoke-v1" \
bash scripts/capo_train_queue.sh \
  capo-stage1-seed20261119-v1 \
  data/research/capo-sft-tag-400step-bs32-seed20261119-v1/checkpoint \
  40 60 100
```

### 4.5 Validation, Agentic Baselines, Paired Analysis & W&B Cloud Sync

```bash
# Parallel single-GPU validation + Self-Verify evaluation (GPU_ID 0..3 -> ports 18010..18013)
bash scripts/capo_valid_queue.sh \
  sft400-seed20261119 \
  data/research/capo-sft-tag-400step-merged-seed20261119-v1 \
  Qwen3.5-9B-SFT-400 0

# Matched-budget agentic baselines (CueTip / DeepPHY WM-Reflection / Self-Verify) & public evals
.venv/bin/python -m inverser run-agentic-baseline --help
.venv/bin/python -m inverser run-public-eval --help

# Offline paired bootstrap statistical analysis & SVG figure generation
.venv/bin/python -m inverser analyze-capo-runs \
  --run base data/research/capo-valid-qwen35-base-tag-v1 \
  --run sft400 data/research/capo-valid-qwen35-sft400-tag-v1 \
  --output data/research/capo-valid-base-vs-sft400-analysis-v1

# Sync any completed SFT, CAPO, or evaluation run to W&B Cloud (stored locally in artifacts/wandb/)
.venv/bin/python -m inverser sync-wandb \
  --run-dir data/research/capo-valid-base-vs-sft400-analysis-v1 \
  --split data/research/inverser-vlm-v1/valid
```

---

## 5. Repository Structure

```text
InverseR/
├── configs/
│   └── proxy_model_aliases.json      # Model alias registry for external API smoke checks
├── docs/                             # Protocols, literature surveys & CAPO-RSI design docs
│   ├── README.md                     # Documentation index
│   ├── benchmark/                    # Simulation & study preparation specifications
│   └── research/
│       ├── rsi_skill_post_training_plan.md  # CAPO-RSI: Skill-invoking RL & recursive self-improvement
│       ├── capo_execution.md                # End-to-end CAPO execution & gate protocol
│       ├── capo_vlm_protocol.md             # InverseR-VLM v1 dataset & scoring contract
│       └── paper_framing.md                 # Paper thesis & experimental framing
├── scripts/
│   ├── capo_train_queue.sh           # 2-rank FSDP2 + TP=2 vLLM 3-arm GRPO training queue
│   ├── capo_valid_queue.sh           # Multi-GPU parallel validation & Self-Verify runner
│   └── {iphyre,phyre,pooltool}_runner.py    # Isolated external physics benchmark runners
├── src/inverser/
│   ├── cli.py                        # Unified CLI entry point (`python -m inverser ...`)
│   ├── runtime.py                    # Deterministic runtime, SHA-256 hashing & storage guards
│   ├── core/                         # Rigid-body physics, rendering, schemas & metrics
│   │   ├── schemas.py, physics.py, preview.py, metrics.py, pilot.py, archive.py
│   ├── benchmark/                    # Dataset generation, <answer> parsers & blind-test locks
│   │   ├── vlm_suite.py, study.py, blind_protocol.py, external_benchmarks.py
│   ├── capo/                         # Post-training: SFT, interactive env, GRPO/CAPO & merges
│   │   ├── capo_env.py, capo_trainer.py, capo_grpo.py, capo_checkpoint.py, capo_selection.py
│   └── eval/                         # Ledgered VLM/public evaluators, baselines, plots & W&B
│       ├── vlm_eval.py, agentic_baselines.py, self_verify.py, learned_wm.py, public_eval.py
│       ├── iphyre_eval.py, retention_eval.py, paper_analysis.py, capo_figures.py
│       └── wandb_tracker.py, model_adapters.py, study_calls.py, evaluation.py, diagnostics.py
├── tests/                            # Hermetic unit & regression test suite
├── data/                             # Local datasets, manifests & analysis receipts (weights gitignored)
└── artifacts/                        # Organized logs (`sft/`, `capo/`) & local W&B cache (`wandb/`)
```

---

## License

Original source code and documentation are released under the [MIT License](LICENSE). Third-party benchmarks and pretrained model weights remain subject to their respective upstream licenses.

