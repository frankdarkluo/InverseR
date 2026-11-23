# InverseR SFT v2 数据构建与核验

v2 的 100,000 组已按种子重建，并划入 [InverseR v3](inverser_v3.md) 的四个分区；`data/research/inverser-sft-v2/` 已不在本地。v2 构建时主动排除了 v1 的全部分区；下文仅记录未变更的 v2 来源语料。

目标：单独发布训练语料，由 100,000 个初始布局各生成一对 anchor/twin。每个击球记录衍生一条联合 T1–T3 forward 和一条 T4 inverse，目标各 200,000 条。它不包含 T5 轨迹，不替代 v1 的验证或盲测，也不代表模型已经用这些数据训练。

数据位于 `data/research/inverser-sft-v2/`：`sft.jsonl` 是可训练的对话样例，图片路径相对于该目录；`groups.jsonl` 记录四条样例的布局组、干预维度、候选索引和五个图像 SHA-256；`manifest.json` 汇总大小、过滤计数、机制分层、来源及两个 JSONL 摘要。

2026 年 11 月 20 日的构建达到目标：**400,000 条 SFT 样例**（正向 200,000、逆向 200,000），共 100,000 个初始布局组、200,000 条击球和 500,000 张图。生成时有 2,193 个因布局或模型输入重复而被拒的候选对，均重新采样补足。27,790 对改变了目标球碰撞掩码，65,127 对改变了库边序列，两类可重叠。独立核验逐条确认输入唯一性与标签格式、校验全部图片哈希，并重新仿真抽样的 256 对击球；v1 三个 manifest 的哈希保持不变。核验回执：`artifacts/sft/inverser-sft-v2-verify.json`。

## 构建和验收规则

- 四种物理族配平；多球族按 2–4 球分层，并显式保留无接触、多目标球接触及普通接触的 anchor。每对标准条件相同；旋转族的 twin 真正改变对应旋转维度，而不是用速度变化替代。
- 排除母球落袋、静止不动即可解和两条击球终态过于接近的样例。以同色球可交换的实际终态口径，anchor/twin 的终态差至少 2 个球半径。每个击球动作确定性复演，并另外执行 **SFT 文本里四位小数的实际动作**，验证在 1 球半径内到达目标。
- 按布局坐标哈希、5 mm 同格指纹、初始图哈希排除 v1 重叠并做新集内部去重；旧 pilot 和 formal-test 精确布局也进入排除集。5 mm 同格是分箱检查，不是连续空间全局近邻证明。最终样例以完整输入文本和有序图像哈希检查唯一性，不以文件名或 completion 判重。
- 逆向只看初始及目标图；见证轨迹摘要可出现在训练 completion 中，不能出现在 prompt 中。自动核验不等同于人工审读，也不证明有下游迁移收益。

## 复现命令

```bash
# 默认 v1 pilot 路径若已清理，可用同种子恢复；构建会拒绝与 v1 来源摘要不符的文件。
envs/main/bin/python -m inverser generate-pilot \
  --output data/research/inverser-sft-legacy-pilot --count 2000 --seed 20261010

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
envs/main/bin/python -m inverser.capo.sft_corpus \
  --output data/research/inverser-sft-v2 \
  --pilot data/research/inverser-sft-legacy-pilot/privileged.jsonl \
  --pairs 100000 --workers 32

envs/main/bin/python scripts/verify_sft_corpus.py data/research/inverser-sft-v2 \
  --pilot data/research/inverser-sft-legacy-pilot/privileged.jsonl \
  --replay-pairs 256 --receipt artifacts/sft/inverser-sft-v2-verify.json
```

训练时把 `train-capo-sft --split` 指向 `data/research/inverser-sft-v2`。v2 通过 Arrow 缓存按需读训练行及图片；同 v1 一样，`--count` 若填写，指击球记录数量而非训练行数量。正式比较不要用 `--count` 取前缀代替分层抽样。
