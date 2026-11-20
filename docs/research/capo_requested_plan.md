这一切计划都是针对InverseR/里面，模型反代先只需要试一下GPT 5.6系列的，如果消耗额度太大的话可以先不做

更新了一下plan，更全面了：# Next-Stage Research & Engineering Plan: 2D-to-3D Multimodal World Modeling & Causally-Grounded Agentic Post-Training

## 1. Problem Statement & Core ICML Storytelling

### 1.1 Limitations of Prior Work
1. **Passive & Simplified 2D Physics Diagnostics Without Solutions**: Recent multimodal physics benchmarks (such as BilliardPhys-Bench and PhysBench) evaluate passive forward question answering under simplified 2D assumptions (constant friction, purely elastic disks, zero 3D spin, no sliding-to-rolling transition). Crucially, frontier MLLMs exhibit two severe structural pathologies:
   - **Stasis Bias**: Under multi-body or long-horizon uncertainty, models default to conservative "no collision / no change" predictions.
   - **Event–State Causal Decoupling**: Some frontier models extrapolate plausible terminal coordinates without identifying the collisions that caused them, while others classify wall/ball events accurately without calibrated continuous coordinate rollouts. Prior work stops at reporting leaderboards (~40%–73% weighted accuracy) and provides no post-training algorithm to resolve these pathologies.
2. **Discretized Shortcuts & Missing Forward-to-Inverse Bridge**: Interactive agentic benchmarks (such as DeepPHY) discretize actions into coarse named menus and test potting-only objectives, allowing models to exploit brute-force max-power heuristics while ignoring precise resting-position control, continuous speed calibration, and 2D-observed 3D spin dynamics.
3. **Failure of Naive Post-Training (SFT & Outcome-Only RLVR)**: Supervised fine-tuning memorizes training layouts and degrades under goal or physics shifts, whereas naive outcome-only GRPO rewards lucky coordinate hits without enforcing intermediate collision causality and collapses under zero-variance sparse rewards in inverse control.

### 1.2 Proposed Approach & Novel Algorithm (`CAPO`: Causally-Grounded Agentic Physics Optimization)
We establish a **2D-to-3D Multimodal Physical World-Model Benchmark & Agentic Post-Training Framework** centered on `Qwen3.5-9B` (natively multimodal VLM with vision encoder and `<think>` reasoning):
1. **Unified Benchmark Design (Public External Benchmarks + `InverseR-VLM` 2D-to-3D Suite)**:
   - **External Public Benchmarks (`DeepPHY` + `PhysBench` Dual Suite)**: Evaluate frontier MLLMs and open VLMs on both **`DeepPHY`** (AAAI 2026 open-source interactive physics suite: `Pooltool`, `PHYRE`, `I-PHYRE`) and **`PhysBench`** (ICLR 2025 physical dynamics subset, `USC-PSI-Lab/PhysBench`) to ground claims on recognized interactive and general physical VLM benchmarks.
   - **Confirmed Frontier MLLM Lineup (via OpenAI-Compatible Reverse Proxy)**: Benchmark **`GPT-5.6 Sol`**, **`GPT-5.6 Terra`**, **`GPT-6.1 Sol`**, **`GPT-6.1 Terra`**, **`GPT-6.2 Astra`**, **`Gemini-3.8-Flash`**, and **`Qwen3.6-Plus`**, alongside local open-weight VLMs **`Qwen3.5-9B`** and **`Qwen3-VL-32B-Instruct`**.
   - **`InverseR-VLM` 2D-to-3D Visual Physics Suite (Native Multimodal BEV Image + Physical Prompt)**: Built on `Pooltool` 3D rigid-body/friction/spin physics and isotropic BEV rendering (`preview.render_layout`), strictly subsuming the three visual forward tasks (**Task 1: Ball-Ball Collision Detection**, **Task 2: Cushion Reflection Reasoning**, **Task 3: Multi-Horizon & Resting 2D Coordinate Estimation**, plus weighted composite score $S_{\text{total}} = 0.3\,\text{Acc}_{T1} + 0.3\,\text{Acc}_{T2} + 0.4\,\text{Acc}_{T3}$) while deepening into **2D-Observed 3D Spin Dynamics** (topspin/backspin/sidespin throw and curve), **Task 4: Visual Single-Shot Inverse Control**, and **Task 5: Multi-Turn Interactive Adaptation under Unknown Table Friction (`withheld_parameters`)**.
2. **Algorithmic Innovation (`CAPO` on `Qwen3.5-9B`)**:
   - **Causal Event-Gated Physics Reward & Counterfactual Twin Contrast**: Constrain `<think>` to generate an explicit causal physical trace (contact topology $\to$ 2D-to-3D spin/sliding-rolling transition $\to$ coordinates/action). Gate continuous coordinate rewards (Task 3) on discrete causal event correctness (Tasks 1 & 2) so lucky endpoint guesses without correct collision paths are discounted, and pair training rollouts on **counterfactual action twins** (rewarding accurate prediction of outcome shift $\Delta S$ when speed or 3D tip offset changes) to directly eliminate Stasis Bias.
   - **Bidirectional Forward–Inverse Agentic Co-Training with 2D-to-3D Curriculum**: Jointly train `Qwen3.5-9B` across visual forward world modeling (Tasks 1–3), inverse action generation (Task 4), and multi-turn simulator residual refinement (Task 5) using GRPO with DAPO dynamic zero-variance group filtering across a three-stage physics curriculum (*Stage 1: 2D Planar Sliding/Rolling $\to$ Stage 2: Multi-Ball Elastic Collisions & Cushion Bounces $\to$ Stage 3: 3D Tip-Spin Coupling*).
   - **Test-Time Internal World-Model Self-Verification**: Leverage the unified model's causally aligned forward head to score and select among $N$ self-proposed inverse actions at test time without external simulator queries.
3. **Target Empirical Validation & Central Scientific Finding**:
   - Lift multimodal `Qwen3.5-9B` on visual forward Tasks 1, 2, and 3 from base ~40% into the **$\ge 70\%$ accuracy tier**—surpassing mid/high-tier proprietary MLLMs and rivaling or exceeding flagship models at substantially lower reasoning-token cost—while simultaneously unlocking open-loop inverse 3D-spin control (Task 4), unknown-friction adaptation (Task 5), and zero-shot transfer to public `DeepPHY` tasks.
   - **Core Mechanistic Finding**: Demonstrate via joint causal consistency analysis (`Acc(T1 & T2 & T3)` vs. `P(T3 correct | T1/T2 wrong)`) that closing the **Event–State Causal Split** and eliminating **Stasis Bias** is the decisive mechanism enabling both accurate 2D-to-3D forward rollouts and downstream inverse physical planning.

---

## 2. External Datasets, Public Repositories & Concrete Execution Todos (GPU & API Only)

### 2.1 Datasets & Open-Source Repositories to Download (`data/external/`, Git-Ignored)

| Asset / Benchmark | Source URL / HF Repo | License | Target Path & Scope | Purpose in Paper |
| --- | --- | --- | --- | --- |
| **`DeepPHY` Benchmark & Harness** | `https://github.com/XinrunXu/DeepPHY.git` + PyPI `iphyre`, `https://github.com/facebookresearch/phyre.git` | MIT | `data/external/DeepPHY/` (Linux-compatible suites: `Pooltool`, `I-PHYRE`, `PHYRE`, `Kinetix`; skip macOS/iOS-only `Angry Birds` & `Cut the Rope`) | External interactive physical reasoning benchmark (baseline frontier MLLM sweep + zero-shot transfer of post-trained `Qwen3.5-9B` + `WM-Prompt` baseline) |
| **`PhysBench` Benchmark** | HF Dataset `USC-GVL/PhysBench` + `https://github.com/USC-GVL/PhysBench.git` | Apache-2.0 | `data/external/PhysBench/` (`image.zip`, `video.zip`, labeled validation split + `physics-based dynamics` / `collision` test subset for local scoring & EvalAI export) | External general physical dynamics VQA benchmark to compare frontier MLLMs vs. `CAPO`-trained `Qwen3.5-9B` |
| **`CueTip` Pool Baseline** | `https://github.com/SeanMemery/cuetip.git` | MIT | `data/external/cuetip/` (adapt NL event proposal + simulated annealing optimizer to `Pooltool 0.6.0` `endpoint_loss`) | Closest domain-specific agentic + physics-optimizer public baseline under matched simulator query budget ($\le 8$) |
| **Visual Encoder for `DINO-WM` Baseline** | HF / TorchHub `facebook/dinov2-small` (`ViT-S/14`) + reference `https://github.com/gaoyuezhou/dino_wm.git` | Apache-2.0 / MIT | `/data/v-guoqingluo/models/dinov2-small/` + `data/external/dino_wm/` | Frozen visual patch encoder to train a same-data latent world model + CEM planner baseline on `InverseR-VLM` |
| **General VLM Retention Benchmarks** | HF Datasets `MMMU/MMMU` (`validation`, 900 items) & `xai-org/RealWorldQA` (`test`, 765 items) | Apache-2.0 / CC-BY-4.0 | `data/external/general_vlm_eval/` | Verify zero general multimodal reasoning degradation before vs. after `CAPO` RL post-training |
| **Local Base VLMs (Already Downloaded)** | `/data/v-guoqingluo/models/Qwen3.5-9B` & `/data/v-guoqingluo/models/Qwen3-VL-32B-Instruct` | Open-weight | Already on disk (`Qwen3_5ForConditionalGeneration` with 27-layer vision tower + native `<think>`) | `Qwen3.5-9B` is the primary post-training backbone; `Qwen3-VL-32B-Instruct` serves as larger open-weight VLM reference |

---

### 2.2 Detailed Execution Todos (File-Level Code, Data Generation & 4× A100 Execution)

#### Todo 1: `acquire-external-benchmarks` — Downloading & Verifying External Datasets and Baseline Repos
- **Download & Environment Setup**:
  - Clone `XinrunXu/DeepPHY`, `USC-GVL/PhysBench`, `SeanMemery/cuetip`, and `gaoyuezhou/dino_wm` into `data/external/` with pinned commit hashes recorded in `data/external/manifest.json`.
  - Use `.venv-train/bin/huggingface-cli download` to fetch `USC-GVL/PhysBench` (extracting `image.zip` and filtering dynamics/collision entries), `facebook/dinov2-small`, `MMMU/MMMU` (`validation`), and `xai-org/RealWorldQA`.
  - Install `iphyre` (applying `DeepPHY/res/02_iphyre/simulator.py` annotated-render patch) and `phyre` (in an isolated Python 3.9/3.10 runner venv `.venv-phyre` if C++/Boost wheels require Python <3.12, invoked cleanly via subprocess JSON IPC).
- **Code Deliverable**:
  - Add `src/inverser/external_benchmarks.py` and CLI subcommand `inverser verify-external-suites` to check file existence, SHA-256 manifests, image decoding, and sample task loading across `DeepPHY` (`Pooltool`, `I-PHYRE`, `PHYRE`) and `PhysBench`.
- **Verification**: `verify-external-suites` passes with zero missing assets and deterministic sample rendering.

#### Todo 2: `proxy-and-vllm-adapters` — Implementing OpenAI Reverse-Proxy & 4× A100 Multimodal vLLM Adapters
- **Files to Modify / Create**:
  - Modify `src/inverser/model_adapters.py`:
    - Relax strict exact-string `body.get("model") == self.config.model_id` check under a new `allow_routed_model_id: bool = False` field on `BackendConfig` (recording both requested alias and provider-returned `model_id` in `ModelReply`), since OpenAI-compatible reverse proxies rewrite model IDs with date/provider suffixes.
    - Support `<think>...</think>` stripping before JSON schema validation in `ModelAdapter.validate_submission`, preserving the raw `<think>` trace and reasoning token count (`usage.completion_tokens_details.reasoning_tokens` + parsed `<think>` character/token length) inside `ModelReply`.
    - Support `reasoning_effort` / `chat_template_kwargs={"enable_thinking": ...}` and configurable `temperature` for both OpenAI-compatible proxy endpoints and local vLLM endpoints.
  - Modify `src/inverser/study_calls.py` & `src/inverser/cli.py`:
    - Add `openai_proxy_backends(config_path, models, ...)` reading `INVERSE_PROXY_BASE_URL` and `INVERSE_PROXY_API_KEY` from environment variables (never hardcoded or committed), configured for the 7 target models: `GPT-5.6 Sol`, `GPT-5.6 Terra`, `GPT-6.1 Sol`, `GPT-6.1 Terra`, `GPT-6.2 Astra`, `Gemini-3.8-Flash`, and `Qwen3.6-Plus`.
    - Add multi-GPU vLLM launcher helper scripts/commands serving `Qwen3.5-9B` (data-parallel 4 replicas across GPUs `0,1,2,3` for evaluation) and `Qwen3-VL-32B-Instruct` (`--tensor-parallel-size 2` across GPUs `0,1` and `2,3`, `--limit-mm-per-prompt '{"image": 2}'`).
  - Add unit tests in `tests/test_proxy_and_multimodal_adapter.py` using `httpx.MockTransport`.
- **Verification**: `pytest`, `ruff`, and `mypy` pass; 8-sample live smoke test succeeds across both the OpenAI reverse proxy (all 7 model IDs) and local vLLM (`Qwen3.5-9B` with BEV PNG inputs + `<think>` JSON parsing).

#### Todo 3: `public-benchmark-sweep` — Benchmarking Frontier & Base Open VLMs on `DeepPHY` & `PhysBench`
- **Files to Create**:
  - Create `src/inverser/public_eval.py` and CLI subcommand `inverser run-public-eval` wrapping `DeepPHY` (`Pooltool`, `I-PHYRE`, `PHYRE`) and `PhysBench` (dynamics/collision subset) into `InverseR`'s resumable `TrialLedger` (`JobEnvelope` per episode, token/cost accounting, no silent retries).
- **Experiments to Run (GPU + Proxy API)**:
  - Evaluate all 7 reverse-proxy frontier MLLMs + local `Qwen3.5-9B` (`thinking` on vs. off) + `Qwen3-VL-32B-Instruct` on `DeepPHY` (`Pooltool`, `I-PHYRE`, `PHYRE`) and `PhysBench` dynamics.
  - Log both aggregate task accuracy and diagnostic failure modes (e.g., max-power brute-force potting rate without cue-ball control in `DeepPHY-Pooltool`, and collision-event vs. trajectory errors in `PhysBench`).
- **Verification**: Complete resumable job ledgers written under `data/research/public-eval-v1/` with zero unresolved reservations.

#### Todo 4: `vlm-2d3d-benchmark-freeze` — Building & Freezing the `InverseR-VLM` 2D-to-3D Multimodal Dataset
- **Files to Modify / Create**:
  - Modify `src/inverser/preview.py`:
    - Add `render_vlm_forward_scene(...)` and `render_vlm_goal_pair(...)`: high-contrast $1024\text{ px}$ isotropic BEV table renders featuring numbered ball badges (`0` for cue ball, `1..K` for object balls), explicit table coordinate ticks (`x`/cushion labels `TOP`/`BOTTOM`/`LEFT`/`RIGHT`), a calibrated cue-direction ray + speed/tip-spin visual inset on the cue ball for Forward scenes, and side-by-side/paired Initial + Goal BEV renders for Inverse scenes.
  - Create `src/inverser/vlm_suite.py` and CLI subcommand `inverser prepare-vlm-suite`:
    - Define unified multimodal schemas (`VLMForwardInput`, `VLMForwardOutput`, `VLMGoalInput`, `VLMGoalOutput`) that directly evaluate:
      - **Task 1 (`T1_ball_collision`)**: Per-object-ball collision binary vector (`T/F`) + first-contact identity.
      - **Task 2 (`T2_cushion_bounce`)**: Exact cushion/boundary reflection set + ordered cushion sequence.
      - **Task 3 (`T3_coordinates`)**: 2D coordinates `[x, y]` (or `pocketed`) of all balls at multi-horizon checkpoints ($t \in \{1, 2, 3, 4, 5\}\text{ s}$ and `resting_state`), scored at $1r$ Euclidean tolerance, plus composite score $S_{\text{total}} = 0.3\,\text{Acc}_{T1} + 0.3\,\text{Acc}_{T2} + 0.4\,\text{Acc}_{T3}$.
      - **Task 4 (`T4_visual_inverse`)**: Single-shot continuous 3D-spin cue action `(azimuth_deg, cue_speed_mps, tip_side, tip_vertical)` from `(Initial BEV, Goal BEV)`, judged by hidden `Pooltool` execution.
      - **Task 5 (`T5_interactive_adaptation`)**: Multi-turn ($\le 6$ attempts) visual + structured residual feedback episodes under both standard physics and **unknown table physics** (`withheld_parameters`: sliding/rolling friction $\times 0.5, \times 2.0$ and cushion restitution shifts).
    - Generate and freeze three strictly disjoint (`group_id` overlap = 0, also excluding legacy `pilot-v0` and `formal-test-v1`) partitions balanced across **2D Planar Regimes** (`free_roll`, `center` 1–4 balls) and **2D-Observed 3D Spin Regimes** (`vertical` follow/draw, `side` English/throw):
      - `data/research/inverser-vlm-v1/train`: **12,000 multimodal scenes** (3,000 per family, each paired with a counterfactual action twin where $\Delta\text{outcome} \ge 2r$).
      - `data/research/inverser-vlm-v1/valid`: **240 multimodal scenes** (60 per family) exclusively for RL checkpoint selection and reward-gate tuning.
      - `data/research/inverser-vlm-v1/test`: **600 frozen multimodal test scenes** (150 per family, multi-horizon + resting + twins + inverse goals) + **120 frozen unknown-friction OOD episodes**.
  - Add unit tests in `tests/test_vlm_suite.py` verifying zero `group_id` leakage, witness replay pass rate = 100%, and exact `T1`/`T2`/`T3`/`S_total` metric parity.
- **Verification**: Frozen manifests and SHA-256 hashes generated; 100% of sampled witnesses replay cleanly in `Pooltool`.

#### Todo 5: `frontier-mllm-leaderboard` — Running Frontier MLLMs & Public Method Baselines on `InverseR-VLM`
- **Files to Create**:
  - Create `src/inverser/learned_wm.py`: PyTorch implementation of two non-LLM learned world-model baselines trained on `inverser-vlm-v1/train` on A100 GPUs:
    1. **`DINO-WM + CEM`**: Frozen `dinov2-small` patch features + 6-layer action-conditioned ViT transition predictor + Cross-Entropy Method (CEM) action optimizer for Task 4.
    2. **`Object-State Transformer WM + CEM`**: 2D ball-state + 3D-cue-action transition Transformer predicting next states/events + CEM planner.
  - Create `src/inverser/agentic_baselines.py`: Adapters for **`CueTip`-style Event-Guided Simulated Annealing** (adapted from `data/external/cuetip`) and **`DeepPHY` WM-Reflection Prompting** under matched simulator query budgets.
- **Experiments to Run (4× A100 + Proxy API)**:
  - Run the 7 reverse-proxy frontier MLLMs (`GPT-5.6 Sol/Terra`, `GPT-6.1 Sol/Terra`, `GPT-6.2 Astra`, `Gemini-3.8-Flash`, `Qwen3.6-Plus`) + local base `Qwen3.5-9B` & `Qwen3-VL-32B-Instruct` + `DINO-WM` + `State-WM` + `CueTip` on `inverser-vlm-v1/test`.
  - Compute baseline leaderboards across `T1`, `T2`, `T3`, $S_{\text{total}}$, `T4`, `T5`, **Stasis Bias Rate** (false-negative no-collision rate on true-collision scenes), and **Event–State Causal Split** (`P(T3 correct | T1 or T2 wrong)`).
- **Verification**: Complete baseline summary JSON produced with paired bootstrap 95% CIs.

#### Todo 6: `capo-reward-and-env` — Implementing the `CAPO` Multimodal RL Environment & Causal Rewards
- **Files to Create / Modify**:
  - Create `src/inverser/capo_env.py` and `src/inverser/capo_trainer.py` (building on `TRL` `GRPOTrainer`, `PEFT` LoRA, `vLLM` colocated/server rollout engine in `.venv-train`, and `src/inverser/training_env.py`):
    - **Structured Causal `<think>` Protocol**: Prompt `Qwen3.5-9B` with BEV image(s) to reason inside `<think>` (bounded to $\le 1024$ tokens with soft length overrun penalty) over *(a) geometric aiming & contact sequence $\to$ (b) 2D sliding/rolling vs. 3D spin deflection/curve $\to$ (c) numerical coordinate/action calibration*, followed by strict JSON output.
    - **Causal Event-Gated Forward Reward ($R_{\text{fwd}}$)**:
      $$R_{\text{event}} = 0.5\,F_1(\hat{E}_{\text{ball}}, E_{\text{ball}}) + 0.5\,\mathbb{I}[\hat{E}_{\text{wall}} = E_{\text{wall}}], \qquad R_{\text{coord}} = \text{Acc}_{1r}(\hat{Y}, Y) + 0.5\max(0, 1 - \bar{d}/15r)$$
      $$R_{\text{fwd}} = R_{\text{event}} + \bigl(\alpha + (1 - \alpha)\,R_{\text{event}}\bigr)\,R_{\text{coord}} \quad (\alpha = 0.25)$$
      so coordinate guesses with wrong collision topology ($R_{\text{event}} = 0$) have their coordinate reward suppressed by $75\%$, forcing causal alignment between events (`T1`/`T2`) and states (`T3`).
    - **Counterfactual Twin Contrast Bonus ($R_{\text{twin}}$)**: On paired twin scenes $(x, x')$ differing by a single speed/angle/spin intervention, reward cosine/rank agreement between predicted shift $\Delta\hat{Y} = \hat{Y}' - \hat{Y}$ and true shift $\Delta Y = Y' - Y$, directly penalizing zero-shift Stasis Bias ($\Delta\hat{Y} \approx 0$).
    - **Bidirectional Inverse & Multi-Turn Residual Reward ($R_{\text{inv}}, R_{\text{loop}}$)**: Combine `Pooltool` execution reward (`goal_reward` from `training_env.py`) on single-shot `T4` and multi-turn ($\le 6$ turns) residual refinement `T5`, plus **DAPO Dynamic Group Filtering** (discarding rollout groups of size $G=8$ with zero reward variance $\text{std}(R)=0$).
    - **Test-Time Internal World-Model Self-Verification (`src/inverser/self_verify.py`)**: Sample $N=8$ candidate actions from `Qwen3.5-9B`'s inverse head and score each candidate `(Initial BEV, a_i)` using `Qwen3.5-9B`'s own `CAPO`-trained forward head (0 simulator queries at test time).
  - Add unit tests in `tests/test_capo_env.py` covering causal reward gating, twin-shift anti-stasis scoring, DAPO filtering, multi-turn context assembly, and self-verification ranking.
- **Verification**: All reward invariants, gradient shapes, and 2-step mini-rollout smoke test on A100 pass cleanly.

#### Todo 7: `post-training-ablations` — Training Multimodal `Qwen3.5-9B` Across Baselines & `CAPO` on 4× A100
- **4× A100 80GB (`0,1,2,3`) Hardware Allocation**:
  - GPUs `0,1`: Dedicated asynchronous/colocated `vLLM` multimodal rollout engines (`--enable-prefix-caching`, `bfloat16`) + parallel `Pooltool` verifier workers for `T4`/`T5` rollouts.
  - GPUs `2,3`: PyTorch FSDP / DeepSpeed ZeRO-2 policy trainer (`Qwen3.5-9B`, LoRA rank $r=64, \alpha=128$ targeting language + projector/vision cross-layers, gradient checkpointing, FlashAttention, group size $G=8$).
- **Controlled Post-Training Arms (All Evaluated Strictly on `inverser-vlm-v1/valid` During Training)**:
  1. **`Arm-0 (Base)`**: Zero-shot multimodal `Qwen3.5-9B` (`thinking` on vs. `thinking` off).
  2. **`Arm-1 (Multimodal SFT)`**: Supervised fine-tuning on ground-truth causal traces + `T1–T4` targets from `inverser-vlm-v1/train`.
  3. **`Arm-2 (Outcome-Only GRPO)`**: Standard RLVR optimizing `T3` coordinates and `T4` endpoint reward *without* causal event gating or counterfactual twin contrast.
  4. **`Arm-3 (RWML-Style Uncoupled Two-Stage RL)`**: Plain forward next-state RL followed by separate inverse policy RL without event gating or multi-turn co-training.
  5. **`Arm-4 (CAPO — Ours)`**: Full Causally-Grounded Agentic Post-Training across the **3-Stage 2D $\to$ 3D Physics Curriculum** (*Stage 1: 2D `free_roll` $\to$ Stage 2: `+ center` collisions/bounces $\to$ Stage 3: `+ vertical/side` 3D spin + multi-turn adaptation*), plus `CAPO + Self-Verify (N=8)` at inference.
- **Verification**: Valid curves confirm `CAPO` eliminates Stasis Bias, lifts valid `T1`/`T2`/`T3` into the $\ge 70\%$ range, and substantially improves `T4`/`T5` over `Base`, `SFT`, and `Outcome-Only GRPO`.

#### Todo 8: `unified-blind-eval-and-analysis` — Blind Test Evaluation, External Transfer & Mechanistic Plots
- **Blind Evaluations (Multi-Seed Checkpoints Locked Before Touching Test Splits)**:
  1. **`InverseR-VLM` Frozen Test (`600` scenes + `120` Unknown-Friction OOD)**: Full comparison of `CAPO (Qwen3.5-9B)` vs. `Arm 0–3`, `DINO-WM`, `CueTip`, and the 7 frontier MLLMs (`GPT-6.2 Astra`, `GPT-6.1 Sol/Terra`, `GPT-5.6 Sol/Terra`, `Gemini-3.8-Flash`, `Qwen3.6-Plus`) across `T1`, `T2`, `T3`, $S_{\text{total}}$, `T4`, and `T5` with 10,000-sample layout-grouped bootstrap 95% CIs.
  2. **Zero-Shot External Benchmark Transfer**: Evaluate `CAPO (Qwen3.5-9B)` zero-shot on **`DeepPHY`** (`Pooltool`, `I-PHYRE`, `PHYRE`) and **`PhysBench`** dynamics subset without any benchmark-specific fine-tuning.
  3. **General VLM Capability Retention**: Paired evaluation of Base vs. `CAPO` `Qwen3.5-9B` on `MMMU` (`validation`) and `RealWorldQA` to confirm no general vision/reasoning regression.
- **Mechanistic & Diagnostic Artifacts (`src/inverser/paper_analysis.py`)**:
  - Export paper-ready JSON/CSV tables and figures: (i) **Causal Alignment Breakdown** (`Acc(T1 ∧ T2 ∧ T3)` vs. spurious `P(T3 correct | T1/T2 wrong)` and Stasis Bias reduction); (ii) **2D Planar vs. 3D Spin Transfer Matrix** (recording zero-shot 3D spin emergence right after Stage 2 before Stage 3 training); (iii) **Accuracy vs. Reasoning-Token Pareto Frontier** (`CAPO-9B` vs. `GPT-6.2 Astra` / `GPT-6.1` / `GPT-5.6`); (iv) **In-Context Friction System Identification Curves** across turns $1..6$ on unknown tables.
- **Verification**: Offline rescore with network and simulator disabled completes with `0` new calls and exact hash reproducibility.

---

## 3. Key Considerations & Guardrails

- **No CPU-Only Classical Sweeps**: All engineering and compute focus strictly on GPU VLM inference/post-training and API MLLM benchmarking; `Pooltool` executes strictly as the ground-truth environment and reward verifier.
- **Strict Partition Hygiene**: `train`, `valid`, and `frozen-test` splits enforce zero `layout_sha256` layout overlap with SHA-256 manifest auditing; hyperparameter and reward-scale tuning touch only `valid`.
- **Preventing Format/CoT Collapse**: Unlike BilliardPhys-Bench (which disabled CoT on Task 3 due to JSON parsing failures), `CAPO` isolates `<think>...</think>` reasoning from strict schema-validated JSON answers and penalizes malformed outputs directly during RL.
- **2D-to-3D Physical Bridge**: Every evaluation explicitly contrasts *2D Planar Mechanics* (`free_roll`, `center` zero-spin) against *2D-Observed 3D Contact-Spin Mechanics* (`vertical` follow/draw, `side` English/cushion throw), demonstrating how a VLM learns latent 3D physical laws from 2D visual observations.