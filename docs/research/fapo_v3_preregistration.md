# FAPO v3 预注册（2026-11-21，开跑前冻结）

本文件在 v3 SFT 与 RL 开跑前写定。开跑后只能追加"偏离记录"，不能改动已写内容。

## 要回答的问题

三个 RL 臂只在"接触预测一致度 E 如何进入奖励"上不同，其余设置全部相同：

| 臂 | `--mode` | forward 奖励 | T4/T5 每次出杆的奖励 |
|---|---|---|---|
| Outcome | `outcome` | `C − P_len` | `R_tail + R_near` |
| FAPO-相加 | `additive` | `E + C − P_len` | `R_tail + R_near + β_E·E` |
| FAPO-门控 | `capo` | `E + (α + (1−α)E)·C − P_len` | `R_tail + R_near·(α + (1−α)E)` |

- `α = 0.25`。
- `E = (碰撞 F1 + first_contact 是否一致 + 库边集合是否一致) / 3`。逆向任务中，E 比较的是模型对自己这一杆的接触预测与模拟结果。
- `L` 为出杆后的终点误差（单位：球半径），`L0` 为不出杆时的误差。
  - `R_near = 1[Succ@1r] + max(0, 1 − L/min(20, L0))`
  - `R_tail = 0.2·max(0, 1 − L/L0)`
  - 当 `L ≥ L0`（轻推或打得更差）时，两项都为 0。
- 逆向回复缺少接触标签，或 `</think>` 后不是恰好一个 `<answer>` 块：奖励为 −1。
- 实现见 `src/inverser/capo/capo_env.py`（`event_agreement`、`inverse_reward`）、`training_env.py`（`goal_reward`、`tail_reward`）、`capo_grpo.py`（`score_goal_completion`）。

**预先写下的预测：**
- P1：门控臂的 T4 Succ@1r 与 T5@3 不低于相加臂，并高于 Outcome。
- P2：门控相对相加臂的优势，随训练中近场命中率（`shot/near_mean`）上升而扩大。理由：门控里 E 的边际价值是 `(1−α)·R_near`，会随出杆能力增长；相加臂的 `β_E` 固定在初始化时的匹配值。
- P3：在全组 `R_tail = R_near = 0` 的组里，相加臂的优势完全来自 E，与 β_E 大小无关。因此相加臂的零接触出杆率（`shot/zero_contact_rate`）高于门控臂。

## 数据与评测协议

- 划分见 `docs/research/inverser_v3.md`：SFT 用 `inverser-v3/sft`；RL 用 `inverser-v3/rl`，SFT 未见过这部分布局。
- 选 checkpoint：在 `inverser-v3/valid` 的预注册前缀上进行（`--count 2468`，即 1,234 个布局组；manifest 中的 `panel_prefix`），由 `scripts/capo_valid_queue.sh` 执行。
- Table 1：使用 `inverser-v3/test` 的预注册前缀，共 6,172 行，即 2,469 个标准组加 1,234 个 OOD episode，需要冻结的盲测协议。
- 所有模型（含零样本基线）使用同一 system prompt（`CAUSAL_PROMPT`）和同一 7 标签逆向 prompt。图片在文字之前，与 TRL 的布局一致。不设 stop 字符串。
- 评测宽松提取唯一的 `<answer>` 块。只有动作标签也能为 T4/T5 计分；接触预测单独报告覆盖率和 E。

## SFT

- 2,400 步，4×A100，等效 batch 32，学习率 2e-5，seed 20261119。
- 覆盖 76,800 条样例，约 0.31 epoch。
- LoRA：r=64，α=128。目标模块由 `capo_trainer.LORA_TARGETS` 匹配，共 58 个线性层：
  - 语言模型 full-attention 层的 q/k/v/o；
  - ViT blocks 21–26 的 `attn.qkv`、`attn.proj`、`mlp.linear_fc1/2`；
  - `visual.merger.linear_fc1/2`。

  三个 RL 臂沿用这个 adapter，LoRA 范围全程不变。
- T4 的训练目标是参考动作加上该动作产生的接触；T5 不进入 SFT。

## β_E 校准（只做一次，所有 seed 共用）

1. SFT 合并后，在 `inverser-v3/rl` 前 64 题上跑 T4 采样：`run-vlm-eval --tasks T4 --count 64 --temperature 0.8`，`--seed 0` 到 `7` 各跑一次，共 512 杆，thinking 预算 1024。
2. `calibrate-fapo-beta` 计算 `β_E = (1 − α)·mean(R_near)`，输出写入 `data/research/capo-v3-beta-e.json`。
3. 兜底规则：若 512 杆的 `R_near` 全为 0，只允许把校准集扩到前 256 题 × 8 个 seed（2,048 杆）后重算。不设下限，也不在训练中途重新校准。
4. 训练脚本拒绝 `β_E ≤ 0` 的校准结果。

## RL

- 课程 60 / 120 / 270，共 450 步。三个阶段的物理族依次为 free_roll、加入 center、全部四族；T5 只出现在第 3 阶段。
- 三个臂共用：anchor/twin 成对采样、DAPO 零方差过滤（最多重采样 8 次）、GROUP=8、温度 0.8、学习率 5e-6、β_KL=0、`loss_type=dapo`、`token_truncate` 重要性采样、thinking 预算 1024、最大生成 1,536 token，以及长度惩罚。
- twin 对比奖励（权重 0.5）只加在相加臂和门控臂。
- T5 逐轮训练：
  - 每题均匀抽取 k ∈ {1, 2, 3}，用当前策略按评测方式先走 k−1 轮；若某一轮已命中，就改为在那一轮展开。
  - 第 k 轮产生 8 个单轮样本。
  - 改进奖励为 `0.25·max(0, (R_tail+R_near)_k − (R_tail+R_near)_1)`，三臂相同，与 E 无关。
- Seeds：20261119、20261120、20261121。三个臂 × 三个 seed，共 9 个 run，无论前面的结果如何都全部跑完。启动入口为 `scripts/capo_train_queue.sh RUN_ID SFT_ADAPTER 60 120 270`。
- 每个 run 以最终 checkpoint（第 450 步）参与比较。

## 报告与统计

- 主指标：T4 Succ@1r、T5@3、S_total。
- 次要指标：
  - T4 误差小于 20r / 10r 的比例，以及"比不出杆更近"的比例；
  - T4/T5 接触预测的覆盖率与 E；
  - 漏报碰撞（False Stasis）、误报碰撞（False Motion）与平衡准确率；
  - T1、T2、T3。
- 训练过程指标：各臂 DAPO 拒绝次数、零接触出杆率、出杆速度、`near/tail` 均值、T5 的平均 k。
- 统计方法：
  - 每个 seed 内，在 test 前缀上按布局组做配对 bootstrap（10,000 次）；
  - 跨 seed 报告均值 ± 标准差，以及各 seed 的差值方向。
- 主对比为门控 vs 相加；次对比为门控 vs Outcome。

## 待定

- 附录变体：β_E = 0.5 的相加臂，单 seed。用于检验 P3 的另一端，即大权重是否诱发轻推。是否运行待定；如果运行，同样在开跑前写入本节。

## 偏离记录

（开跑后追加，注明日期和原因。）

- **2026-11-22，评测前缀改为 `data/research/inverser`。** 用户把数据集统一命名为 `data/research/inverser`，评测前缀改为：valid 前 500 组（`--count 1000`），test 前 1,000 组标准场景加 500 个 OOD episode（2,500 行）。`inverser` 的 sft、rl 与 `inverser-v3` 的 scenes/labels 哈希完全一致，valid 文件也一致，只有前缀长度不同；test 重新排过序。因此 SFT、β_E 校准和 RL 训练数据都不受影响，只有选 checkpoint 和 Table 1 的评测改用新前缀。原文中的 2,468 / 6,172 行以此条为准。
