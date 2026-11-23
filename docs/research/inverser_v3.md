# InverseR v3 数据划分

`data/research/inverser/` 下有四个按初始布局组划分的分区，文件格式与旧版 InverseR-VLM 相同。`data/research/inverser-v3/` 是同一批数据的过渡副本（行内容逐字节相同，只有 manifest 不同），正被一个进行中的 RL 实验使用；该实验结束后会改成指向 `inverser/` 的链接：

- `public/scenes.jsonl`、`public/images/`：模型可见的输入；
- `private/labels.jsonl`、`private/physics.json`：真值和仿真参数；
- `groups.json`：布局组清单；
- `manifest.json`：统计信息和全部文件的哈希。

旧版 v1 分区（`inverser-vlm-v1`）已删除，不再使用；当前论文 Table 1 中的 `N=240` 结果来自它，正文暂不改动。标签中的 `source` 仍保留 v1 来源 ID 作为溯源。

| 目录 | manifest `split` | 布局组 | 场景（行） | 来源 | 用途 |
|---|---|---:|---:|---|---|
| `sft/` | train | 61,478 | 122,956 | v2 49,502 + v1-train 11,976 | SFT 监督样例 |
| `rl/` | train | 15,326 | 30,652 | v2，SFT 未见过 | 在线 RL 的任务池（初始图和目标图，不含参考动作） |
| `valid/` | valid | 9,566 | 19,132 | v2 | 选超参和选 checkpoint |
| `test/` | test | 9,566 标准 + 1,234 OOD | 20,366 | v2 + 新生成的 OOD | 冻结盲测 |

外层 train : valid : test ≈ 80 : 10 : 10（按布局组计），train 内部 SFT : RL ≈ 80 : 20。

## 构成

- **物理族**：每个分区中 free_roll 占 1/8，center、side、vertical 各占 7/24；多球族中 2、3、4 球各占 1/3。
- **约三分之二的题需要碰球**：需要碰球指目标要求某个物体球移动超过 1r 或落袋，只移动母球无法完成。多球组按 anchor/twin 的碰球类型（两个都需要 / 只有一个需要 / 都不需要）成比例抽样，因此需要碰球的场景占 sft 65.8%，rl、valid、test 66.2%，OOD 慢摩擦 64.7%、快摩擦 63.7%。构建时强制要求落在 60–70% 之间。
  - 降低 free_roll 的原因：只需移动母球的题，强模型可以从刻度尺读出坐标，再用给定的摩擦参数直接算出击球力度。GPT-6.2 Astra 在 v1 上这类题 T4 全部做对，而需要碰球的题只有 17–40%。
  - SFT 采用同样的构成，所以训练时有足够多的碰球样例。
- **T5 最多 3 次尝试**（v1 仍为 6 次）。成功定义为 3 次内任意一次在 1r 内到达目标。
- **OOD**：`test/` 中有 1,234 组隐藏摩擦的 T5 episode，慢、快摩擦各 617。它们是新生成的布局，与其他所有布局去重；多球组中 1/4 的 anchor 不发生接触。这两档摩擦值不能出现在 RL 训练中。

## 划分规则

- **按布局组划分**。同一布局的 anchor、twin 以及由它们派生的 forward / T4 / T5 任务，总是在同一个分区。
- **训练过的布局不进 RL、valid、test**。v1-train 的 11,976 组（SFT-400 与第一阶段 RL 用过）只放进 `sft/`。v1 valid/test 和旧 pilot、formal-test 的布局在所有分区都被排除，按布局哈希和 5 mm 格子比对。
- **v2 从未被训练过**（此前只做过数据加载检查），所以可以用作 valid/test。
- 为了把 free_roll 降到 1/8，约 1.6 万组 v2 free_roll 没有使用。

## 评测子集（预注册）

文件按各分层的比例交错排列，所以任何前缀的构成都接近整体；valid 和 test 的开头各有一段固定的分层子集：

- **valid**：前 500 组 = 前 1,000 行，用于日常选 checkpoint，例如 `run-vlm-eval --split data/research/inverser/valid --count 1000 ...`。
- **test**：前 1,500 组 = 前 2,500 行，即 1,000 组标准场景加 500 个 OOD episode（按物理族、球数、摩擦设定和接触类型分层），作为 Table 1 中所有方法共同评测的子集。`run-vlm-eval` 对 test 只接受整套运行，或者恰好等于 manifest `panel_prefix.scene_rows`（2,500）的 `--count`，并且都必须有冻结的盲测协议。

按布局组计、成功率约 50% 时，95% 置信区间的半宽：500 组约 ±4.4 个百分点，1,000 组约 ±3.1，9,566 组约 ±1.0；两个方法在同一批场景上配对比较时，约 650 个场景即可稳定检出 5 个百分点的差异。评测成本：本地 9B 约 17 秒/场景，GPT-6.2 Astra 约 48 秒/场景（8 路并发）。

## SFT、RL 与 RSI 的使用约定

- SFT：`train-capo-sft --split data/research/inverser/sft`。
- RL：`train-capo-grpo --split data/research/inverser/rl`。论文中分开报告任务池大小（15,326 组）、实际采样过的任务数和 rollout 数。
- RL 是否复用 SFT 布局作为消融：固定同一个 SFT checkpoint 和相同的仿真预算，只比较"全新布局"和"部分复用 SFT 布局"。
- RSI：Loop B 的练习题只能从 `sft/` 或 `rl/` 的布局扰动生成，并且必须与 valid/test 按布局哈希、5 mm 格子和初始图去重；每一轮之间的选择只用 valid；记录每一轮采样到的任务 ID。

## 复现

每条标签的 `source` 字段记录来源：v2 的 `slot/generation/candidate`（v2 种子 20261120）、v1 的原 `group_id`，或 OOD 的 `seed/slot/candidate/tier`。标签中也保存了初始布局和动作，可以直接重放。

当前数据由 `scripts/build_inverser.py` 以种子 20261122 从上一版重新划分得到，当时还用已删除的旧 v1 分区做了排除检查。当前的评测子集就是这样从过渡副本导出的（`--panels 500 1000 500`）。在现有数据上改变预注册前缀时，用 `--panels` 模式，划分成员和标准组的顺序保持不变：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src envs/main/bin/python scripts/build_inverser.py \
  --source data/research/inverser --panels <VALID> <TEST> <OOD> \
  --pilot data/research/.inverser-exclusion-pilot/privileged.jsonl \
  --formal data/research/formal-test-v1 --output <新目录> --workers 48
```

重新划分（不加 `--panels`）时默认种子为 42，`--legacy` 可选，用来指定需要额外排除的旧分区。脚本按标签重放每一组并重新渲染全部图片，标准组的 PNG 必须与来源逐字节一致；随后用 `verify_vlm_suite` 审计两遍（分别把 `sft` 和 `rl` 当作 train，T5 为 3 次尝试，每种摩擦设定中至少一半的题需要碰球），并检查碰球比例落在 60–70%，全部通过后才输出。

上一版 v3 是按相同种子链，由 v2 与 v1-train 的合并候选池（`scripts/merge_inverser_sft.py`）划分得到的；它和候选池都已被替换。
