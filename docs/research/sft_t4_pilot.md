# SFT 阶段提升 T4 的 pilot（2026-11-23 起）

## 为什么先做 SFT

2026-11-23 在 SFT-v3 的 512 条 T4 采样上测得：

- 读球位已经很准：初始布局中位误差 0.08r，目标布局 0.10r，98% 的回复里每个球都读在 1r 以内。
- 在模型自己读出的布局上，用真实 Pooltool 规划，T4 成功率 42%；在真实布局上规划为 80%。
- 模型自己出杆只有 1%。瞄准中位误差 5.0°，速度误差约 ±25%。
- pass@8 只有 4/64 个题目。

所以 T4 的缺口主要在"由读到的布局算出杆"这一步。如果 SFT 不能明显提升 T4，RL 也很难：文献里起点接近 0 时 RL 基本失效，见 SimpleVLA-RL 与 LUFFY。

目标：在 SFT 阶段让 valid 上的 T4 Succ@1r 明显上升。10–20% 是期望值，不是硬指标。训练中频繁验证，看曲线走向。

## 两个臂

两个臂只有 `<think>` 的写法不同，其余全部相同：

- 从基座 Qwen3.5-9B 开始，seed 42。
- 有效 batch 32，学习率 2e-5，线性衰减。
- 调度按 1 个 epoch 设定（7,685 步），用 `--save-every 600` 每 600 步保存一次 adapter。

| 臂 | 数据 | 机器 |
|---|---|---|
| A：推导式 think | `data/research/inverser-sft-think`（`scripts/build_sft_think.py` 生成） | msra1，4 卡 |
| B：原模板 think | `data/research/inverser/sft`，由 `sft_examples` 现场生成 | msra，GPU 2–3，grad accum 4 |

A、B 的 prompt、图片、答案和行顺序都相同。两边 world size 不同（4 卡 vs 2 卡），批次的组成会略有差异，比较时要注明。

## 验证

`scripts/rolling_t4_eval.sh` 对每个新出现的 checkpoint 依次做：合并 adapter，起 vLLM，在 `inverser/valid` 前 400 行（200 个布局组，分层前缀）上单次采样测 T4，然后删除合并出的模型。

- 解码设置与 valid 队列一致：thinking 预算 1024，输出上限 2048。
- 记录 Succ@1r、中位终点误差（单位 r）、5r 内比例、无效率，写入 `artifacts/sft-t4/rolling.tsv`。
- 基线用同一口径测：SFT-v3（2,400 步）和基座模型。
- 在 2,400 步和最终 checkpoint 上加测 forward（T1–T3），确认没有退化。

## 判定

- **2,400 步时**：若 A 的 T4 Succ@1r 比 B 高至少 2 个百分点，且按布局配对的区间下界大于 0，就认为推导式 think 有效。
- **A 的曲线在 1 epoch 内达到 10% 及以上**：用 A 作后续 RL 的起点，或继续扩 SFT 数据（速度阶梯、方位角 twin、交叉旋转）。
- **A 到 4,800 步仍低于 3% 且没有上升趋势**：停掉 A，改做以下三条路线之一：
  - 用仿真器验证过的 best-of-N 拒绝采样 SFT；
  - 数据扩充；
  - 把计算交给工具。

## 资源

- msra1 的 4 卡：A。
- msra 的 GPU 2–3：B。
- msra 的 GPU 0：滚动验证。
- msra 的 GPU 1：基线评测和 pass@k。

v3 剩下的两个 run（capo-20261120，以及从头重跑的 additive-20261120）排在 A 到达判定点之后，用空出来的 4 卡跑。

## 产物

| 内容 | 路径 |
|---|---|
| 训练输出 | `data/research/sft-think-seed42`、`data/research/sft-template-seed42` |
| 单次验证 | `data/research/t4val-<run>-<checkpoint>` |
| 汇总表 | `artifacts/sft-t4/rolling.tsv` |
