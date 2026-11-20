# 训练路线协议：后训练阶梯 × 能力链

写定日期：2026-11-03，在任何训练之前。研究者选定主攻 ICML 2027（截稿未公布，按上届推算
1 月下旬），IJCAI 2027（全文 1 月 11 日）作为 benchmark 版本的备选。本文件是预注册：训练集、
奖励、臂、假设和评测在看到任何训练结果之前写定；之后的改动要有新版本号，不回头改阈值。
主线和已有证据见[论文主线](paper_framing.md)与[五模型结果](formal_model_sweep.md)。

## 1. 问题

正式测试的事实：五个模型直接出杆 ≤2/200（几何直接 77/200）；后果预测和候选排序最好也只到
直线近似的水平；只有 GPT-5.5/5.4 能利用真实反馈（36/64、29/64，打乱后 0）。事后分析指向一个
局部解释：瞄准基本正确（方位角偏差中位 0.003°），杆速没有标定（偏差中位 31%）。

训练路线问两件事：**哪种监督信号能把"预测"变成"规划"**；**训练好一环，会不会迁移到另一环**。
不主张新算法：每一臂都是标准后训练配方（SFT、GRPO、多轮 RL 环境），贡献在受控的迁移测量。
最近邻：RWML（2026.02，先用 next-state 奖励再做任务 RL）、Dyna-Think/Dyna-Mind（世界模型模拟与
策略同训）、RLVP（2026.07，数值求解器做物理奖励）。与它们的差别轴：连续带旋转的动作、精确仿真器
给每个环节判分、同一批布局上的五环能力链、几何求解器作天花板。

## 2. 训练数据（`prepare-training-set`）

- 同一采样器 `pilot.sample_case`，新种子；拒绝母球落袋与仿真失败的见证动作；每个布局按 group
  指纹（初始布局＋物理）与 2,000 条开发记录和 200 个正式测试布局逐一核对，重叠必须为 0，命令
  会自行断言。
- `training-set-v1`：种子 20261103，每 family 1,000 个布局，共 4,000。
- `training-holdout-v1`：种子 20261104，每 family 50 个，共 200；另外排除 training-set-v1。只用于
  checkpoint 选择和早停，**正式测试集在训练期间不读、不评**。
- 产物：`privileged.jsonl`（完整 ShotRecord：见证动作、事件、轨迹；供 T1 的 SFT 目标和 T3 的
  预测标签）、`goals.jsonl`（环境读取的 GoalInput 与见证）、`manifest.json`（种子、排除源 hash、
  producer hash、物理、family 计数、见证回放核查、产物 hash）。
- 物理与正式测试相同（默认 `PhysicsConfig`）。物理参数漂移作为 OOD 评测另行冻结，不进训练。

## 3. 单轮环境与奖励（`training_env.GoalEnvironment`）

- 提示与正式套件的直接出杆请求逐字相同：system `Return only the requested numerical JSON.`，
  user 为同一个 JSON 任务（初始、目标、公开物理、1r 容差、no-scratch）加 `GoalSubmission` 的
  output schema。测试 `test_environment_prompt_matches_the_study_adapter_request` 守住这一点。
- 解析：回复必须是 `GoalSubmission` 纯 JSON；只允许思考模型在前面带一个 `<think>…</think>` 块。
  不修格式、不重试。
- 奖励 r = 1[1r 内达标] + max(0, 1 − loss/20)。loss 是 `endpoint_loss`（球半径单位，含球状态、
  落袋和母球入袋罚分）。非法 JSON、仿真失败 = 0。GRPO 用组内相对优势，奖励只需单调。
  **唯一预注册的调整**：若 T2 首轮超过一半的提示组内奖励全为 0，把尺度从 20r 放大到 60r，并记入
  版本号。
- 隐藏判分与训练奖励用同一仿真器、同一判分代码；每次 rollout 计数。

## 4. 臂

训练全部用结构化输入；基座 Qwen3.5-9B（dense，与本地 Qwen3.6-27B 同架构家族，便于规模点）。
LoRA；单卡 A100 80GB 一条 run 的预计耗时在首轮实测后填入。

| 臂 | 监督信号 | 训练哪一环 | 回答什么 |
| --- | --- | --- | --- |
| T0 | 无 | 基座零样本（结构化，thinking 关） | 与 5 个 API 模型对齐的起点 |
| T1 | SFT：几何求解器的解；蒸馏：GPT-5.5 成功的反馈轨迹 | L4 / L5 | 模仿能到哪 |
| T2 | GRPO，第 3 节奖励 | L4 | 不懂后果能不能学会做（反向 gap） |
| T3 | GRPO，预测精度奖励（静止终态误差与接触 F1） | L1/L2 | 训练内部世界模型 |
| T4 | T3 → T2 | 先 L1 再 L4 | **预测→规划是否迁移（主问题）** |
| T5 | 多轮 RL：仿真反馈环境，≤8 杆 | L5 | 能否学会"会做实验" |
| 推理期 | T2 提案 × T3 自验证 best-of-N，对照真仿真器验证 | L3 | 自己的世界模型能不能当 verifier |

T1 中几何求解器的解只覆盖它能解的布局（正式测试上 77/200，free_roll 全解）；其余布局用 128 次
几何＋Powell 的成功解，仍无解的布局不进 SFT。

## 5. 预注册假设与判定

指标全部在冻结的 formal-test-v1 上计算；配对差按布局 bootstrap，95% 区间；每个主格 3 个种子。

- **H1**：T2 提升 L4（1r 成功率）但不提升 L1（优于直线近似的比例）。
- **H2**：在 center/vertical/side family 上 T4 > T2；在 free_roll 上两者相当。
- **H3**：T5 的策略在**无反馈**的 L4 测试上也优于 T2。
- **H4**：自验证 best-of-N 的 L4 成功率低于真仿真器验证，高于不验证。

每条都允许为负；负结果照报。判定用区间是否跨 0，不用 p 值单独下结论。L4 的参照是几何直接
77/200 与几何＋Powell 128 次 81/200；超过与否分别报告，不作为假设成立的条件。

## 6. 评测

- `formal-suite --local-vllm-url http://127.0.0.1:<port>/v1 --local-model-revision <rev>
  --reasoning-effort none`：本地 vLLM 的 OpenAI 兼容端口，JSON 模式，`enable_thinking=false`，与
  Azure 上 Kimi/DeepSeek 的 `none` 档对应；无凭据、无费用，请求/物理账本照记。
- 服务端（分配到卡后再起，只绑回环地址；BEV 的目标请求带两张图）：

  ```bash
  CUDA_VISIBLE_DEVICES=<idx> .venv-train/bin/vllm serve /data/v-guoqingluo/models/Qwen3.5-9B \
    --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 8000 \
    --max-model-len 40960 --limit-mm-per-prompt '{"image": 2}'
  ```

  先用 `--limit-jobs 8` 做 P2 冒烟（模型身份、JSON、usage、图像输入、错误路径），再跑完整套件。
- 训练期间不触碰 formal-test-v1；每个 checkpoint 只在 training-holdout-v1 上选。
- L5 评测复用 feedback-study 的 genuine/shuffled/reflection 三条件。
- OOD（物理参数漂移、held-out family）在训练结果出来前另行冻结，不在本文件内。

## 7. 资源与决策点

- GPU：四张 A100 80GB 当前都有他人任务；训练需要研究者明确分配 1–2 张（索引/UUID）。没有分配前
  只做 CPU 工作。
- **11 月 15 日**：有卡且 T2 在 holdout 上有信号（L4 成功率离开 0）→ 继续；否则切回 benchmark 论文
  路线（IJCAI 1/11）。
- **12 月 1 日**：至少一条迁移结果（正或负）带区间。
- **12 月 15 日**：定 headline；决定一篇 ICML 还是拆成 benchmark→IJCAI＋训练→ICML。

## 8. 本轮状态（2026-11-03）

- 训练环境 `.venv-train`（与物理 `.venv` 分开）：torch 2.14.0、vLLM 0.33.0、TRL 1.17.1、
  PEFT 0.21.2、transformers 5.20.2；CUDA 可见 4 张卡；pooltool 0.6.0 与本包可导入。
- 基座 Qwen3.5-9B 已下载到 `/data/v-guoqingluo/models/Qwen3.5-9B`（commit
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，19 GB）。
- `training-set-v1`：4,000 布局（每 family 1,000），4,107 次回放，拒绝 107 次母球落袋；排除 2,200 个
  组、重叠 0；见证回放核查 20/20；38 s、52 MB。`training-holdout-v1`：200 布局，206 次回放，拒绝 6；
  排除 6,200 个组、重叠 0；10 s。
- T0 计划（未执行）：`data/research/formal-suite-qwen35-9b-local-v1`，3,328 请求，run hash
  `f16ebc3f…`。
- 代码：`training_env.py`（训练集、环境、奖励）、`formal-suite` 的本地 vLLM 路线、
  `prepare-training-set` 命令；234 测试、Ruff、Mypy 通过。模型调用 0，训练 0。
