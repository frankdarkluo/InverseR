# InverseR-VLM / CAPO：当前执行计划

更新：2026-11-19（Asia/Shanghai）。范围仅限 `/data/v-guoqingluo/InverseR/`。

研究者原始计划完整保存在 [capo_requested_plan.md](capo_requested_plan.md)。
本文件记录实际验收门槛和进度；原始计划里的预期成绩、机制解释、模型目录与文献定位
不是已经验证的结果。当前 goal 覆盖全部八阶段，不因完成适配器或启动训练就判为完成。

## 当前授权与不变边界

- 按新请求推进 GPU VLM 推理/后训练与 API 多模态评测；GPU 范围为 0、1、2、3。
  每次启动检查实际占用，不终止其他项目进程。不开 CPU-only classical sweep。
- 最新覆盖指示：反代先只小规模试 GPT-5.6 系列，若额度消耗大可以不做。
  原计划中的其他模型 API sweep 当前不执行；目录查询不等于付费推理。
- 渲染、Pooltool 标签/奖励验证、数据划分及软件测试属于模型实验的必要支持，
  不构成独立 CPU 基线实验。已有 classical artifacts 保留，不追加同类 sweep。
- 既有 formal-test-v1、既有研究结果、现有未提交修改不得覆盖或混作新训练数据。
  新实验使用独立版本、训练/holdout/frozen-test 分组清单和 SHA-256 审计。
- API 仅接收自生成合成状态/图像；真实录像和人工标注继续保持本地。
  密钥从运行环境注入，不进入代码、日志、prompt 或 manifest。
- 不承诺 ≥70%、优于旗舰模型或特定机制成立。奖励诱导的 event/state 一致性不是
  自然因果推理的证明；需 held-out 干预、配对反事实和消融支持。
- 不静默替代不可用模型，不把 state-dynamics+CEM 叫作 DINO-WM 复现。
  不提交 Git commit，不发布/上传数据或模型。

## 八阶段与验收

| 阶段 | 当前状态 | 必须留下的证据 |
| --- | --- | --- |
| acquire-external-benchmarks | 资产校验及三个隔离/适配 runtime 通过 | `verification-v5.json`：1,690 文件、三份确定性渲染回执；九球为 Pooltool 0.6 适配，不称上游 0.4.4 复现 |
| proxy-and-vllm-adapters | 本地 9B on/off 各 8/8；32B TP=2 已实测 | 精确 ID、BEV、think/JSON、usage、缓存已测；API 单次 HTTP 400 后停止；DP=4 与第二个 32B 副本未验收 |
| public-benchmark-sweep | PhysBench validation 三臂；I-PHYRE 全部 40 题；PHYRE/九球 smoke | 适配任务保持独立账本，不称官方榜单；其他模型和完整正式 public sweep 未完成 |
| vlm-2d3d-benchmark-freeze | **完整三分区已冻结且复核通过** | [T1–T5 合同、分区与评分](capo_vlm_protocol.md)，24,960 见证、62,340 张 PNG、跨组交集 0 |
| frontier-mllm-leaderboard | 两条学习基线 holdout 全量完成；两种 agentic 适配 smoke 完成 | [实测结果与预算](capo_learned_and_rl.md)；正式盲测/排行榜未运行 |
| capo-reward-and-env | T4、twin+DAPO、T5 各两步 GPU 实测；联合轮转正在验收 | T5 54 次真仿真与图像反馈，第二步梯度近零；四卡联合训练及实际模型自验证仍未跑完 |
| post-training-ablations | 单 rank 与 2+2 四卡七步联调均完成，Outcome 对照联调进行中 | 从阶段二恢复后，两 trainer rank 的 T5 非零梯度及完整 LoRA；正式多种子与 checkpoint 选择未完成 |
| unified-blind-eval-and-analysis | 冻结授权门已加入；真正盲测尚未授权/执行 | 锁定 checkpoint 与协议后盲测、多 seed/cluster bootstrap、迁移/保持性、机制联合指标、真实 token 成本 |

新版指定规模为 train 12,000、holdout 240、test 600 + OOD 120。
生成前 manifest 需分别列出 anchor、twin 和独立 layout group 数；twins 不得当成独立布局
跨分区，也不得通过混淆 pair/scene 数量缩减研究者要求的样本量。
GPUs 0/1 用 rollout，2/3 用 trainer；评测阶段可按新版 data/tensor parallel 分配。
进入正式 sweep 前需把每模型最大请求数/总 token、训练步数与 seed、
early stop、bootstrap unit、主次指标固定在 run manifest。先做有界 smoke；
“GPU/API 已授权”不等于任意消耗或将预期结果当验收条件。

## 2026-11-19 基础检查

- 宿主机四张 NVIDIA A100 80GB PCIe 均显示 0 MiB 已用、0% utilization（一次性快照）。
- 本地模型目录含 Qwen3.5-9B、Qwen3-VL-32B-Instruct；目录存在不等于 GPU 加载成功。
- 已有 `.venv-train` 报告 torch 2.14.0、vLLM 0.33.0、Transformers 5.20.2、
  TRL 1.17.1、PEFT 0.21.2、Accelerate 1.15.0；以实际 smoke 为准。
- 已运行代理目录包含 `gpt-5.6-sol`、`gpt-6.1-sol`、`gpt-6.1-terra`、
  `gpt-6.2-astra`、`gemini-3.8-flash`。
  此次目录未见 `gpt-5.6-terra`、`qwen3.6-plus`，两项保持 unavailable，
  不用 `gpt-5.6-luna` 或本地 Qwen 代替。
- 代理列出模型不证明该路由支持图像或完整 token usage，也不证明其供应商身份；
  实际请求结果须单独记录。
- 原训练文档中的“11 月 15 日”决策点已在过去，不应继续当成未来里程碑；
  新阶段按验收门槛而非过期日期推进。

## 原计划中需要核查的研究表述

1. 不能将 PhysBench 的全部任务概括为简化 2D 弹性球物理，也不能未核查就声称
   其没有方法改进。需按原论文与 physical-dynamics 子集精确比较。
2. DeepPHY 的版本、会议、动作空间和目标类型需逐环境核查；
   potting/max-power shortcut 是待测假设，不是预定失败归因。
3. BilliardPhys-Bench 的分数区间、CoT/JSON 设置需原始来源支持，
   不将别人的不同加权准确率当作 InverseR 的 Base 起点。
4. RWML、CueTip、DINO-WM 的目标/输入/训练数据差异需明示；
   对照实现与真正复现分开报告。
5. 不通过弱化 frozen test 或调整加权指标达成 ≥70%。

## 当前执行 checkpoint

- 研究者最新原文已完整落盘，包含“反代先只试 GPT-5.6，额度高可不做”的覆盖指示。
- 实际调用：GPT-5.6 Sol 1 次，HTTP 400、usage unknown，停止且未重试；
  其余反代模型调用为 0。
- A100 本地 Qwen3.5-9B：thinking-off 修正指令后 8/8 schema 通过；
  thinking-on 加 1024-token 原生预算与最终答案 JSON mode 后 8/8 通过。
  合计分别为 33,611 / 41,787 token，后者含 8,184 provider reasoning tokens。
  这是传输与格式验收，不是物理准确率或 CAPO 训练结果。
- 三个更早的本地失败协议分别暴露 schema echo、推理耗尽输出、Markdown 围栏；
  原始 ledger/reply 和失败结果均保留，不从报告里删除。
- 外部资产 hash、PhysBench 问题/标签关联与 PNG 样本可读；
  MMMU validation 的 Parquet 行数为 900，RealWorldQA test 行数为 765。
- PhysBench 官方说明包含 PhysAgent，且覆盖 property/relationships/scene/dynamics，
  因而原计划“PhysBench 仅简化二维且没有方法”的概括不能作为论文事实。
  证据为固定仓库 `data/external/PhysBench/README.md` 与
  `eval/physagent/`，revision 见外部 manifest。
- 代码验收与详细命令见 [capo_runtime.md](capo_runtime.md)。
  world-model baseline、CAPO RL trainer、正式多臂消融和盲测仍未完成；本 goal 保持 active。

## 第二轮验收（2026-11-19，Asia/Shanghai）

- `artifacts/capo-vlm-verification-v1.json`：train 24,000 行（12,000 对）、holdout
  240 行（120 对）、test 600 行（300 对）与另 120 个 OOD T5；共 62,340 张 PNG。
  物理见证全部重复执行，脚本独立读回哈希/组/配额/标签与图像，无 API 调用。
- `capo-vlm-qwen32-tp2-dev-smoke-v2`：16/16 请求账本完成；场景专属 schema 后
  T1–T3 有效 6/8，剩余错误保留。`capo-vlm-qwen35-dev-smoke-v2`：32/32 请求完成，
  T5 跑足预算且没有三图 HTTP 400；本次使用旧的通用球数 schema，T1–T3 有效 0/4，
  不作为最终约束协议的性能估计。
- `capo-sft-gpu-smoke-v1`：A100 单卡完成 2 步，梯度 norm 为 2.086/9.617，
  2 个视觉投影和 32 个语言 LoRA-B 张量有非零更新。该运行是 SFT，不是 GRPO/CAPO。
- `.venv-iphyre` 独立安装，使用 DeepPHY 固定提交的渲染补丁，不更改外部仓库；
  40 个任务可列出，`support` 两次 600×600 PNG 同 hash，空动作负对照未解。
  `data/external/verification-iphyre-v1.json` 记录依赖与源码哈希。
- `public-eval-v1`：32B 44/71，9B thinking-off 38/71，9B thinking-on 49/71。均为本地 image-panel
  adaptation 的验证集，非官方榜单或盲测。公开原始标签无手动修正；
  字面 exact 与上游 prefix 各自记录。反代无新增请求。
- `capo-sft-warmup-seed20261119-v1`：256 步完成，trainer 记录训练 loss 0.9124；
  checkpoint 未经 holdout 选优，不是正式 SFT 结论。合并视觉与语言适配器，
  保存模型/processor 文件 SHA-256，保留原始 adapter。
- `analyze-capo-runs` 已对两个完成的 PhysBench validation 臂做两次独立离线重算，
  `analysis.json` 与 `table.csv` 逐字节一致（0 模型、0 simulator）。另有三臂 10,000 次
  成对 bootstrap `public-eval-v1/physbench-val-paired-analysis-v2/`。
- 软件验收：全量 274 passed，既有数值 warning 1 条，Ruff 与 Mypy 通过。
