# CAPO-RSI Stage 2 技能规格（v0.2 草案）

状态：草案，未实现。不改动正在运行的 Stage 1。第 5 节还有待拍板的项，拍板前不写代码。
方案背景见 [rsi_skill_post_training_plan.md](rsi_skill_post_training_plan.md)；本文件取代其中的技能池和奖励设计。

## 0. 依据（2026-11-20 核查）

| 事实 | 来源 |
|---|---|
| SFT 读位置中位误差 0.75r；真正动了的球，停球误差中位 31r。用"模型读到的位置 + 真物理"重放，T2 从 20.5% 升到 76%，T3@3r 从 27.9% 升到 68% | SFT valid forward 200 条有效答案，CPU 重放 |
| 即使用**真实**球位，直线射线判断首碰球也只有 73.9% 正确（180 个多球场景）；模型自己的首碰正确率是 60.3% | 同上，纯几何计算 |
| SFT T4 成功 0/240。219 次有效尝试中只有 15 次触发 ≥150r 的状态/洗袋罚分；其余中位误差 50.3r | `data/research/capo-valid-qwen35-sft400-tag-v1` |
| CAPO 臂训练日志：第 2–4 步（还没学到东西）T4 误差已经是 26–29r，因为课程先给简单族；Stage 3（全族）的 T4 误差均值约 28r；大多数步的 `frac_reward_zero_std` = 0 | `artifacts/capo/capo-stage1-seed20261119-v1/capo-20261119.log` |
| 奖励按组标准化（`scale_rewards="group"`）：在线性区内，改变 20r/60r 的尺度不改变组内优势 | `capo_grpo.py:719` |
| T5 训练截断：第 103–118 步的 6 个 T5 步，`clipped_ratio` 为 25–62.5%，每步 290–601 s。原因：每轮思考上限 1024，而 6 轮合计上限只有 4096（`capo_grpo.py:561, 712, 732`） | 同上日志 |
| T5 每轮文本反馈 161 token（1 球）到 330 token（4 球），内容主要是精确坐标；只留序号、成败和失败类型时为 13 token | Qwen3.5-9B tokenizer 实测 |
| T5 反馈里有每个球的精确结果坐标、母球到目标的残差（两者相加就是目标坐标），还有 `loss_radii`（单球题打 3 杆后可用三点定位反推出目标） | `capo_env.py:255-261` |
| T4 训练奖励只有 R_goal，没有乘 G_causal；论文 `icml2027/main.tex:216` 写的是乘了 | `capo_grpo.py:505-513` |
| 摩擦扰动只在 test 生成（0.5×/2×，库边弹性 0.65/0.97）；train/valid 全是标准物理 | `benchmark/vlm_suite.py:643, 688-692` |

## 1. 技能池

所有技能都是闭式公式，不做时间积分，不调用 Pooltool；输入全部由模型自己填写。

| skill | 操作 | 可用任务 | 退火 |
|---|---|---|---|
| `billiards_geometry` | `first_contact`：射线与 2r 圆求交，返回首碰球和接触点；`ghost_aim`：假想球瞄准角和切角；`deflect`：碰后方向（目标球沿球心连线，母球沿切线，跟杆/缩杆由模型自己修正）；`mirror`：一库镜像反弹角 | forward、T4、T5 | 是 |
| `kinematics_stop_distance` | 由杆速、高低杆和摩擦（T5 中填模型自己的估计）算出滑动段距离、停球距离和停球时间 | forward、T4、T5 | 是 |
| `try_shot` | 执行一杆，返回渲染图和精简文本（见 D2） | 仅 T5，6 次预算 | 否 |
| `tactical_physics_card` | ≤ 8 张卡片，每张 ≤ 50 词。按标签规则过滤：球数（1 个 / 多个）和是否加塞（只有 forward 能用这个标签，因为动作是题目给的） | forward、T4、T5 | 是 |

- 删除 `verify_causal_rollout`。只有 T5 的 `try_shot` 会调用 Pooltool。
- `first_contact` 和 `mirror` 的输出与门控检查的内容有部分重叠。允许保留它们，理由有三：一是它们是闭式几何，可以内化；二是用真实球位时，射线的首碰正确率上限也只有 73.9%；三是它们会被退火，最终只报告无技能成绩。
- 不单独实现 `probe_table_physics`。是否学会了"试探"用行为衡量：第一杆之后，第二杆杆速的调整方向是否与真实摩擦一致（在 valid_perturbed 上测）。
- 卡片由当前模型 π_k 写，素材是同一题组内误差最低和最高的两条轨迹。每组 8 个样本中给 4 个注入卡片，只保留在匹配标签的组里 ΔÂ > 0 的卡片。不用外部模型写卡片。

## 2. 红线（代码必须守住，并配单测）

1. 技能函数只接收数字参数，不接收 scene、label 或 split。用单测检查函数签名。
2. 退火：每个训练样本以概率 p_aid 获得可退火技能（含卡片），Round 1/2/3 分别取 p_aid = 1.0 / 0.5 / 0.0。`try_shot` 不退火。不设调用惩罚 λ，只保留现有的无效调用罚分（每次 0.05）。
3. 如果启用 G_causal，就用最终提交的 (a_final, ŷ_final) 计算。
4. T5 训练使用连续物理扰动：滑动和滚动摩擦的缩放系数 ~ U[0.65, 1.75]，库边弹性 ~ U[0.72, 0.94]，避开 test 的 0.5/2.0 和 0.65/0.97。这三个参数不出现在提示里。forward 和 T4 仍用标准物理。另外用独立种子按同一分布生成并冻结 `valid_perturbed`（60 个场景，每族 15 个，只做 T5）：`layout_sha256` 不与 train/valid/test 重叠，只用于选检查点和测试探能力。
5. T5 训练长度：每轮思考上限 384 token，反馈只用精简文本（D2），总长上限保持 4096（不需要重启 vLLM）。估算：6 × (384 + 约 100) + 5 × 约 120 ≈ 3.5k token。如果保留坐标反馈，4 球题约 5.1k token，仍会被截断。验收标准：在 smoke run 中，T5 步的 `clipped_ratio` < 5%。评测协议不变。
6. 盲测协议：T4 主表只报无技能的单杆成绩；T5 只允许用 `try_shot`；有技能的成绩作为次要行。这些规则在冻结盲测协议之前写进 `configs/frozen_arms.json`。

## 3. Loop B 选题

- 用 π_k 对每个 train 场景采样 8 次。8 次中有 1–4 次误差不超过奖励尺度（默认 20r，见 D3）的场景算作边界题。不用 pass@8 成功率来选。
- 对边界题扰动杆速、旋转或隐藏摩擦，再用 Pooltool 重放。新题必须同时满足：有见证动作、不洗袋、`layout_sha256` 不重叠。另外混入 30% 原始 train 场景作为锚点。

## 4. 验收顺序

1. **离线上限检查（只用 CPU，在任何 RL 之前做）**：用 `billiards_geometry` + `kinematics_stop_distance` 串成一条规则链，分别在真实球位和模型读到的球位上算 valid 的 T1/T2。只有当真实球位下的 T2 明显高于模型现在的 20.5%（比如 ≥ 50%）时才继续；否则说明技能池覆盖不了 T2 的主要缺口，要回头重新设计。
2. `kinematics_stop_distance` 在单球、不碰库边的场景中，与 Pooltool 停球位置的误差 < 0.5r。
3. 设对照臂：从同一个 Round 1 检查点出发，用相同步数和相同奖励训练，但不给技能、不做 RSI。主要成功标准：Round 3 在 valid 上的无技能成绩（T4 误差中位数、T2、T3@3r）高于对照臂，并给出配对 bootstrap 置信区间。

## 5. 决策状态

| # | 问题 | 状态 |
|---|---|---|
| D1 | T4 是否乘 G_causal | 已定：论文按代码现状写（门控作用在 forward 和双生对比上，经共享权重和课程迁移到控制）。Stage 1 结束后检查 T4 有没有"只动母球刷分"的现象，再决定 Stage 2 是否加接触门控。注：`endpoint_loss` 对各球取 max，目标球不动时会按它应走的全部位移计入误差，所以刷分空间本来就小（`metrics.py:63-75`） |
| D2 | T5 反馈内容 | 已定（2026-11-20）：Stage 2 训练和最终 test 协议都改用新反馈，见第 6 节。Stage 1 和 GPT-6.2 Astra 继续用旧反馈跑完 valid，这些 T5 数字只供开发参考 |
| D3 | 奖励尺度 | 暂缓：保持 20r，等 Stage 1 的 valid 结果出来再定。Stage 1 结果：CAPO 的 valid T4 中位误差为 52.2r（不计无效答案），超过了约定的 40r 重议线，需要重新决定。理由是 CAPO 臂大多数步没有零方差组，而且没有达到预注册的触发条件（超过一半的组奖励全为 0） |

## 6. D2 实现

**已完成**：[`src/inverser/capo/t5_feedback.py`](../../src/inverser/capo/t5_feedback.py) 和 `tests/test_t5_feedback.py`。

- 文本只包含 `ordinal`、`success`、`failure`、`first_contact` 和 `cushions`，约 30 token。`first_contact` 和 `cushions` 是这一杆实际发生的事件，定义与 T1/T2 相同（用 `forward_truth` 计算），这样就能区分"慢台没走到"和"快台碰库弹回"。
- 图像：画出最新结果，用黄圈标出 GOAL，用蓝点标出之前每一杆的落点，母球落点旁边标上杆号。先用 INITIAL/GOAL 的渲染器按原尺寸绘制，再统一缩到长边 512 px（约 3.4 px/r，约 150 个视觉 token）。训练和评测使用的是像素完全相同的同一张图。遇到无效杆时，仍然返回包含历史的图。
- 修掉了旧协议的一个问题：旧反馈图来自 `render_vlm_goal_pair(...)[1]`，所以结果图上印着 "GOAL" 标签。新图标的是 "OUTCOME k"。
- 唯一剩下的结构差异：评测端点最多只能放 2 张图，所以评测把 GOAL 和反馈图拼成一张；训练里两者是分开的两张图。反馈图本身的像素没有差别。

**接线（2026-11-21 已完成）**：Stage 1 流水线和 GPT-6.2 Astra 都跑完后完成接线；ruff、mypy 和全部 146 个测试通过，其中 `test_vlm_suite` 端到端测试了评测端的 T5。训练端的 `try_shot` 嵌在 `run_grpo` 里，没有单测覆盖，需要在 Stage 2 的 smoke run 中确认。以下是当时的约束和步骤，留作记录：接线前要求 Stage 1 流水线的 valid 评测和选点都完成、GPT-6.2 Astra 跑完，并且没有其他评测正在运行。原因是 `select-capo-checkpoint` 要求所有候选 run 的 `producer` 哈希一致（`capo_selection.py:58-67`），而哈希覆盖 `vlm_eval.py`、`capo_env.py` 等文件；`preview.py` 和 `vlm_suite.py` 还被数据集哈希锁住，不能修改。

1. `capo_grpo.py` 的 `try_shot`：通过 `replay_action` 闭包取到 replay，改为调用 `t5_feedback(env, replay)`，并删除 224×384 缩略图那一步。在 T5 提示中加一句图例说明。
2. `vlm_eval.py` 的 T5 循环：`judge` 同样保存 replay，改为调用 `t5_feedback`；在 `Images:` 那一行补上图例；把 `"t5_feedback.py"` 加入 `producer_hashes`。
3. `blind_protocol.py` 的 `evaluation_producers` 同样加入 `"t5_feedback.py"`。
4. 保留 `GoalEpisode.feedback`，因为 T4 的 agentic 基线（DeepPhy reflection）还在用它。
5. 接线之后，旧的 valid run 和新的 valid run 哈希不同，`select-capo-checkpoint` 不能把它们放在一起比较。需要在新协议下比较的检查点，要重新跑 T5 valid。
