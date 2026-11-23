# InverseR：面向二维物理世界建模与因果对齐的后训练研究

**InverseR** 关注一个基础而核心的问题：多模态大语言模型（VLM）能否仅凭二维相机图像，在没有真实三维坐标输入的前提下，同时完成**正向物理后果预测**（“判断将会发生什么”）与**逆向连续物理控制**（“规划应当如何出杆”），并在环境物理参数未知时通过交互试探实现在线适应。

项目基于确定性刚体碰撞、自旋与库仑摩擦仿真引擎（`Pooltool 0.6.0`）构建，主要包含两部分工作：
1. **`InverseR-VLM v1`**：无信息泄漏的二维到三维视觉物理基准，涵盖 **Counterfactual Twin**（反事实双胞胎对比场景）、正向因果世界模型任务（`T1–T3`）、单杆连续逆向控制任务（`T4`）以及未知摩擦参数下的多轮交互适应任务（`T5`）。
2. **`CAPO`（Causally-Grounded Agentic Post-Training）**：多模态监督微调（SFT）与三阶段课程强化学习（GRPO / DAPO）后训练框架，通过**乘法因果门控奖励**（$R_{\text{CAPO}} = R_{\text{goal}} \times G_{\text{causal}}$）与多轮 `try_shot` 试探机制，约束模型在物理因果一致的前提下优化控制策略；相关技能调用与递归自进化扩展方案见 [`docs/research/rsi_skill_post_training_plan.md`](docs/research/rsi_skill_post_training_plan.md)。

---

## 1. 研究背景与动机

将视觉语言模型应用于连续物理交互控制时，现有方法普遍面临以下四方面限制：

1. **从二维观测到三维物理量的映射困难**：真实视觉系统只能获取二维图像（斜视与俯视双视角），无法直接读取仿真器内部的精确三维坐标与速度。原生 VLM 难以将像素空间中的球位关系准确转化为击球方位角、偏转（throw）及旋转效应（例如 `Qwen3.5-9B` 在零样本条件下，1 倍球半径误差内的坐标预测准确率不足 `1%`）。
2. **仅依赖推理阶段提示难以弥补内部物理模型的偏差**：诸如 `CueTip`、`DeepPHY WM-Prompting` 与 `Self-Verify` 等方法在冻结模型权重的前提下，通过提示词引导模型先预测后果再决策；然而当模型自身的物理动力学表征不准确时，推理期预测反而容易误导动作选择。
3. **单纯基于结果的强化学习容易诱发“过程与结果脱节”**：传统的 Outcome GRPO 仅依据最终是否到达目标位置给予奖励，模型可能学到依赖巧合或缺乏物理依据的动作策略，在任务分布或物理条件变化时表现退化。
4. **不可观测的物理参数需要主动交互辨识**：台面的滑动/滚动摩擦系数与库边弹性系数无法从单张静态图像中直接观测，智能体需要通过可控的诊断试打（`try_shot`）并结合视觉反馈进行在线系统辨识（`T5`）。

针对上述问题，本项目采用三项简洁而直接的设计：
- **扁平化标签输出协议（`<think>...</think><answer>...</answer>`）**：替代容易在长推理中发生格式截断的嵌套 JSON Schema，并将推理长度控制在 `1024` tokens 以内。
- **无动作真值泄漏的跨模态 SFT 热身**：联合微调视觉投影层（`visual.merger.linear_fc[12]`）与语言模型注意力层（`q/k/v/o_proj`），在 `<think>` 过程仅保留基于“假想球（Ghost-Ball）”的几何推导，不泄漏真值动作参数。
- **因果门控与三阶段物理课程强化学习**：按“自由滚动（`Stage 1`）$\rightarrow$ 直线碰撞（`Stage 2`）$\rightarrow$ 旋转反弹与未知摩擦交互（`Stage 3`）”循序推进训练，并将控制得分与前向物理因果预测准确度相乘，确保动作优化建立在正确的物理预测之上。

---

## 2. 基准数据集：`InverseR-VLM v1`

> 本节描述的 v1 分区已删除，当前数据位于 `data/research/inverser/`，说明见 [InverseR](docs/research/inverser_v3.md)；本节和第 4 节的命令待数据目录统一改名后一并更新。

v1 的冻结分区继续用于复现。新的 SFT / RL / valid / test 四分区（按布局组 80:10:10，train 内 SFT:RL = 80:20，约 2/3 的题需要碰球，T5 最多 3 次，另有 1,234 组 OOD 盲测）见 [InverseR v3](docs/research/inverser_v3.md)；下表仍描述 v1。

`train / valid / test` 三个切分统一位于 `data/research/inverser-vlm-v1/{train,valid,test}/`，均由正向物理推演生成（`prepare-vlm-suite`），通过后台 `layout_sha256` 保证跨分区与历史测试集零球局重叠，并由各分区的 SHA-256 `manifest.json` 锁定。公开图像按物理机制分一层子目录存放并使用简短可读命名：
- **标准场景（每组 5 张 PNG）**：`public/images/{roll,center,vert,side}/{family}-{1..4}ball-{0001}-{init,anchor-fwd,anchor-goal,twin-fwd,twin-goal}.png`
- **OOD 未知摩擦场景（每组 2 张 PNG）**：`test/public/images/ood/{fast,slow}-{family}-{1..4}ball-{0001}-{init,goal}.png`

| Split | 标准场景（`Anchor + Twin`） | OOD 未知摩擦场景 | 总场景数 / 图片数 | 物理族分布（`roll / center / vert / side`） | 用途 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`train`** | `24,000`（`12,000` 对） | `0` | `24,000` / `60,000` | 每族 `6,000` | SFT 热身与三阶段 CAPO RL |
| **`valid`** | `240`（`120` 对） | `0` | `240` / `600` | 每族 `60` | 超参调优与检查点选择 |
| **`test`** | `600`（`300` 对） | `120`（`slow` / `fast` 摩擦） | `720` / `1,740` | 四族均衡覆盖 | 盲测协议锁定后的单次评测 |

### 任务体系（`T1–T5`）
- **正向物理预测**（`INITIAL` + 出杆指示图 $\rightarrow$ `<answer>`）：
  - **`T1`**：球-球碰撞掩码与精确首碰球编号。
  - **`T2`**：有序库边反弹序列（`LEFT`、`RIGHT`、`BOTTOM`、`TOP`）及库边集合。
  - **`T3`**：`1s..5s` 轨迹与静止终态坐标（`1r = 2.62 cm` 严格阈值及 `3r` 归一化阈值；综合分 `S_total = 0.3 T1 + 0.3 T2 + 0.4 T3`）。
- **逆向控制与适应**（`INITIAL` + `GOAL` $\rightarrow$ 连续四维 `CueAction`：`azimuth_deg`, `cue_speed_mps`, `tip_side`, `tip_vertical`）：
  - **`T4`**：已知台面参数下的单杆连续控制。
  - **`T5`**：隐藏摩擦与库边弹性参数下，最多 `6` 轮闭环 `try_shot` 交互试探与修正。

---

## 3. 实验配置与基线验证结果

| 模块 | 实验设置 |
| :--- | :--- |
| **基础模型** | `Qwen3.5-9B`（`bfloat16`，`--reasoning-parser qwen3`，`thinking_token_budget=1024`） |
| **4× A100 硬件分配** | SFT 采用 4 卡 DDP 并行；GRPO / CAPO 采用 **2+2 异步分离架构**（`GPUs 0,1` 运行 `TP=2` vLLM 采样服务 `:18021`；`GPUs 2,3` 运行 2-rank `FSDP2` 训练器） |
| **LoRA 参数配置** | `r=64`，`lora_alpha=128`，微调 `language_model.*.(q\|k\|v\|o_proj)` 与 `visual.merger.linear_fc[12]`（共 `68` 个可训练张量） |
| **Stage 0: SFT** | `400` 步，`effective_batch_size=32`（`per_device_bs=4`, `grad_accum=2`），`lr=2e-5`，累计 `12,800` 条 completion-only 样本 |
| **Stage 1: RL 对照组** | `Outcome` / `Uncoupled` / `CAPO` 各训练 `200` 步（`stage_steps = 40 / 60 / 100`），`G=8`，`temp=0.8`，`lr=5e-6`，**`beta=0.0`**，`loss_type="dapo"`，`vllm_importance_sampling_mode="token_truncate"` |
| **过程记录与监控** | Weights & Biases（`frankdarkluo/inverser`），同步记录 `EMA` 平滑曲线、`log_metric` 格式合规率/命中率及本地 `metrics.csv`（缓存于 `artifacts/wandb/`） |

### `valid` 验证集对比：Base vs. 400-Step SFT（`N=240`）

| 实验组 | `T1` 精确率 | `T2` 库边集合 | `T3_1r` (`2.6cm`) | `T3_3r` | `S_total` / `S_total_3r` | `T4` 格式合法率 | `T5` (`<=5r` / `<=1r`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-9B` (Base)** | `48.3%` | `16.7%` | `0.95%` | `5.01%` | `18.8%` / `24.4%` | `60.0%` | `0.4%` / `0.0%` |
| **`Qwen3.5-9B` (400-Step SFT)** | **`56.7%`** | **`22.5%`** | **`14.88%`** | **`23.22%`** | **`28.1%` / `35.9%`** | **`91.2%`** | **`5.4%` / `0.83%`** |

*(由上可见，400 步 SFT 显著提升了模型的二维空间定位与动作格式输出能力，但静态模仿学习在 `1r` 严格控制精度上仍接近零，说明引入闭环物理强化学习十分必要。完整配对 Bootstrap 置信区间与对比图见 [`data/research/capo-valid-base-vs-sft400-analysis-v1/`](data/research/capo-valid-base-vs-sft400-analysis-v1/)。)*

---

## 4. 实验运行指南

所有命令均在仓库根目录下执行：
- **主力全量环境 `envs/main`**：统一承担 `Pooltool` 物理仿真、代码检查、单测、VLM 评测及 `PyTorch / FSDP2 / TRL / PEFT / vLLM` 训练推理；
- **外部基准隔离环境 `envs/iphyre`、`envs/phyre`**：由 `run-public-eval` 自动按子进程调用，无需手动切换；
- **本地运行缓存 `cache/`**：统一存放 `vllm`、`huggingface`、`torchinductor`、`numba` 与 `pooltool-home` 缓存（首次运行前请先 `cp .env.example .env` 并填入 `WANDB_API_KEY`）。

### 4.1 代码质量检查与数据集校验

```bash
# 静态检查、类型检查与全量单元测试
envs/main/bin/ruff check src tests && envs/main/bin/mypy && envs/main/bin/python -m pytest -q

# 离线核验 train / valid / test 全量 SHA-256 与零重叠约束（若已清理 data/pilot-v0，可先用固定种子恢复排除集后再核验）
envs/main/bin/python -m inverser prepare-vlm-suite --output-root data/research --verify-only
```

### 4.2 Stage 0：4 卡多模态 SFT 热身与权重合并

```bash
# 4 卡 DDP SFT（effective_batch_size=32, 400 步；自动同步至 W&B）
mkdir -p artifacts/sft
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache" PYTHONPATH="$PWD/src" \
envs/main/bin/torchrun --standalone --nproc-per-node=4 --module inverser train-capo-sft \
  --split data/research/inverser-vlm-v1/train \
  --model-path /data/v-guoqingluo/models/Qwen3.5-9B \
  --output data/research/capo-sft-tag-400step-bs32-seed20261119-v1 \
  --steps 400 --seed 20261119 \
  > artifacts/sft/capo-sft-tag-400step-bs32-seed20261119-v1.log 2>&1

# 将验签通过的 68 张量 LoRA 适配器合并为独立 bf16 权重目录（供 vLLM 推理服务加载）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" \
envs/main/bin/python -m inverser merge-capo-adapter \
  --model-path /data/v-guoqingluo/models/Qwen3.5-9B \
  --adapter data/research/capo-sft-tag-400step-bs32-seed20261119-v1/checkpoint \
  --output data/research/capo-sft-tag-400step-merged-seed20261119-v1 \
  --device cuda
```

### 4.3 Stage 1：4 卡（`2+2`）CAPO 与消融对照组 GRPO 训练

```bash
# 终端 1 — GPUs 0,1 启动 TP=2 vLLM 采样服务（端口 18021；无 nvcc 环境需保持 VLLM_USE_FLASHINFER_SAMPLER=0）
mkdir -p artifacts/capo
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
VLLM_CACHE_ROOT="$PWD/cache/vllm" HF_HOME="$PWD/cache/huggingface" \
XDG_CACHE_HOME="$PWD/cache" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" \
envs/main/bin/python -m trl.scripts.vllm_serve \
  --model /data/v-guoqingluo/models/Qwen3.5-9B \
  --host 127.0.0.1 --port 18021 \
  --tensor_parallel_size 2 --gpu_memory_utilization 0.75 \
  --dtype bfloat16 --max_model_len 8192 --enforce_eager True \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 8}' \
  > artifacts/capo/vllm-rollout-stage1-18021.log 2>&1 &

# 终端 2 — GPUs 2,3 依次运行三臂（outcome / uncoupled / capo）2-rank FSDP2 课程训练队列
CAPO_SEEDS="20261119" CAPO_FSDP_SMOKE="capo-grpo-tag-capo-4gpu-smoke-v1" \
bash scripts/capo_train_queue.sh \
  capo-stage1-seed20261119-v1 \
  data/research/capo-sft-tag-400step-bs32-seed20261119-v1/checkpoint \
  40 60 100

# 合并任意已完成的 RL 检查点（full_adapter）为可部署 bf16 模型
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" \
envs/main/bin/python -m inverser merge-capo-adapter \
  --model-path /data/v-guoqingluo/models/Qwen3.5-9B \
  --adapter data/research/capo-stage1-seed20261119-v1/capo-20261119/checkpoint/full_adapter \
  --output data/research/capo-stage1-capo-merged-seed20261119-v1 \
  --device cuda
```

### 4.4 `valid` 并发评测、对比基线与检查点选优

```bash
# 1) 单卡独立部署合并模型并评测 valid 全集（forward + T4 + T5）与 Self-Verify（GPU 0..3 对应端口 18010..18013，可四卡并行）
bash scripts/capo_valid_queue.sh \
  sft400-seed20261119 \
  data/research/capo-sft-tag-400step-merged-seed20261119-v1 \
  Qwen3.5-9B-SFT-400 0

# 2) 非 LLM 世界模型基线（DINO-WM + CEM / Object-State WM + CEM）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" \
envs/main/bin/python -m inverser train-learned-wm \
  --method dino \
  --train data/research/inverser-vlm-v1/train \
  --valid data/research/inverser-vlm-v1/valid \
  --dino-path /data/v-guoqingluo/models/dinov2-small \
  --output data/research/capo-dino-wm-valid-seed20261119-v1 \
  --epochs 20 --seed 20261119

# 3) 同等仿真预算 Agentic 基线（cuetip / wm_reflection / self_verify，针对已启动的本地 vLLM 服务）
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser run-agentic-baseline \
  --split data/research/inverser-vlm-v1/valid \
  --output data/research/capo-cuetip-qwen35-valid-v1 \
  --local-vllm-url http://127.0.0.1:18010/v1 \
  --model Qwen3.5-9B-SFT-400 --revision "<checkpoint-manifest-sha256>" \
  --method cuetip --thinking --thinking-token-budget 1024 \
  --workers 16 --max-total-tokens 31457280 --execute

# 4) 基于完整 valid（forward + T4 + T5）均分锁定最优检查点
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser select-capo-checkpoint \
  --valid-manifest data/research/inverser-vlm-v1/valid/manifest.json \
  --candidate data/research/capo-stage1-capo-merged-seed20261119-v1/manifest.json \
              data/research/capo-stage1-seed20261119-v1-valid \
  --output data/research/capo-selected-seed20261119-v1.json
```

### 4.5 配对统计分析、外部泛化评测、盲测锁定与 W&B 同步

```bash
# 离线配对 Bootstrap（10,000 次重采样）显著性分析与 SVG 图表导出（0 模型/仿真调用）
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser analyze-capo-runs \
  --run base data/research/capo-valid-qwen35-base-tag-v1 \
  --run sft400 data/research/capo-valid-qwen35-sft400-tag-v1 \
  --output data/research/capo-valid-base-vs-sft400-analysis-v1

# 外部公开物理与通用保留度评测（suite 可选：physbench / iphyre / phyre / pooltool / MMMU / RealWorldQA）
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser run-public-eval \
  --suite iphyre \
  --output data/research/public-eval-v1/iphyre-sft400-v1 \
  --local-vllm-url http://127.0.0.1:18010/v1 \
  --model Qwen3.5-9B-SFT-400 --revision "<checkpoint-manifest-sha256>" \
  --thinking --max-total-tokens 16777216 --execute

# 锁定最终盲测协议（签发不可变回执后，方可在 inverser-vlm-v1/test 上执行单次盲测）
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser freeze-vlm-blind-protocol \
  --test-manifest data/research/inverser-vlm-v1/test/manifest.json \
  --valid-manifest data/research/inverser-vlm-v1/valid/manifest.json \
  --arms configs/frozen_arms.json \
  --output data/research/blind-protocol-v1.json

# 将任意已完成的训练、评测或配对分析目录同步上传至 W&B Cloud
PYTHONPATH="$PWD/src" envs/main/bin/python -m inverser sync-wandb \
  --run-dir data/research/capo-valid-base-vs-sft400-analysis-v1 \
  --split data/research/inverser-vlm-v1/valid
```

> 完整参数说明可通过 `envs/main/bin/python -m inverser --help`（或 `<subcommand> --help`）查看；执行细节与门控约束参见 [`docs/research/capo_execution.md`](docs/research/capo_execution.md) 与 [`docs/research/capo_vlm_protocol.md`](docs/research/capo_vlm_protocol.md)。

---

## 5. 项目结构

```text
InverseR/
├── configs/proxy_model_aliases.json  # 外部 API 模型别名配置
├── docs/                             # 研究协议、文献综述与 CAPO-RSI 方案（见 docs/README.md）
├── scripts/                          # 队列调度（capo_train_queue.sh, capo_valid_queue.sh）与外部 runner
├── src/inverser/
│   ├── cli.py / runtime.py           # 统一 CLI 入口与 SHA-256 / 磁盘安全守卫
│   ├── core/                         # 刚体物理、渲染、Schema 与基础指标
│   ├── benchmark/                    # InverseR-VLM 数据构建、<answer> 解析与盲测锁
│   ├── capo/                         # SFT、交互环境、FSDP2/vLLM GRPO/CAPO 与权重合并
│   └── eval/                         # 评测器、Agentic 基线、配对分析、画图与 W&B 追踪
├── tests/                            # 单元与回归测试集
├── envs/                             # 本地 Python 运行环境（main/, iphyre/, phyre/，gitignored）
├── cache/                            # 本地运行时缓存（vllm/, huggingface/, numba/, uv-python/，gitignored）
├── data/                             # 本地数据集（inverser-vlm-v1/{train,valid,test}）、Manifest 与分析回执
└── artifacts/                        # 运行日志（sft/, capo/）与本地 W&B 缓存（wandb/）
```

---

## License

原创代码与文档采用 [MIT License](LICENSE)；第三方基准数据集与预训练权重遵循各自上游许可协议。
