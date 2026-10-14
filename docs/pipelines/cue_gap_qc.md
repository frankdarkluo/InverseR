# 无共享背景的母球/近邻/缺帧 QC

这是一轮真实原始帧的 CPU 数据准备实验，不是 learned world model 或 LLM 性能评测。
固定来源颜色规则后，检查可见白球是否因为前景连接形状被漏掉，以及关联器是否不必依赖
共享背景就能减少碎片。

## 1. 实测失败原因与 opt-in 修复

两个 support 录制中的失败帧仍有可见白球，但白球连接球杆、阴影或近邻球之后：

- 前景 circularity 出现 0.269–0.439，低于原 0.45 阈值；
- 白球与其他前景连成一块时，面积达到约 3 倍球盘，被判为未解决组件；
- 白球与 pink 邻近时，前景被合并或切碎，无法作为一个独立 cue 候选。

因此没有简单地全局放松“所有物体都像球”的筛选。新增 opt-in 的 native white-core
像素证据：只在来源颜色条件中，寻找尺寸受限的紧凑白色核心，抑制薄球杆、大面积白色
遮挡和重复检测；已有非 cue 候选与它冲突时保留未解决状态，不强制替换身份。

每个候选仍来自当前帧。没有填补缺失帧、补齐球库存、强迫恰好一个 cue 或重新拟合
人工参考路径。两个白色候选都满足证据时，两者都保留并触发库存不确定性。

## 2. 无共享背景消融结果

2 个 support 录制后冻结实现；表格为另外 14 个开发检查录制、6,097 原始帧，按录制
等权平均，与上一轮相同来源和空间指标。每个条件均不使用 shared background。

| 方法 | cue 未解决帧 ↓ | 新轨迹/100帧 ↓ | 点盘路径 IoU ↑ | 空间路径覆盖 ↑ |
| --- | ---: | ---: | ---: | ---: |
| 颜色修复基线 | 23.91% | 93.17 | 58.17% | 61.96% |
| 颜色＋缺帧关联 | 23.91% | 24.96 | 58.17% | 61.96% |
| 颜色＋母球像素恢复 | 17.28% | 92.05 | 58.87% | 62.87% |
| 颜色＋母球恢复＋缺帧关联 | 17.28% | 24.87 | 58.87% | 62.87% |

完整数值：[主表 CSV](../../data/quality/author-cue-gap-pilot-v2/aggregate_table.csv)、
[summary](../../data/quality/author-cue-gap-pilot-v2/summary.json)及
[配对统计](../../data/quality/author-cue-gap-pilot-v2/paired_diagnostics.json)。

相对颜色基线，母球像素恢复的 IoU 差为 **+0.70 个百分点**，录制组 bootstrap 的
95% 区间为 **[+0.36,+1.10] 个百分点**。这仅是本开发环境下的空间参考增益，不代表
多桌泛化或实际控制更好。

### 通俗解释

- 母球在原始画面可见，不代表整块前景轮廓能通过球形筛选。恢复当前帧像素候选能减少
  一部分“不知道 cue 在哪里”的帧，不需要共享背景。
- 缺帧关联主要帮助同一观测序列少启动新 ID，不能创造没有看到的球，所以它不改变
  检测位置、cue 库存或点盘路径分数。
- 17.28% 的 cue 未解决比例仍不低，颜色重复冲突也没有被这项 cue 修复解决。
  它不是独立检测 recall 或身份准确率；G1 没有因此通过。

## 3. 世界模型定位：这是观测条件，不是方法贡献终点

把这四个条件作为 empirical study 的 **perception/belief factor**，对照 oracle
状态、BEV、噪声/缺失状态与相同动作/目标，判断后续预测和规划损失来自哪里。

不能直接把“轨迹更连贯”写成“模型更懂物理”。主问题仍是指定动作之后的预测、事件
一致性、候选排序与最终目标达成，且 native 模型和 simulator-assisted 条件必须分开。

本轮 QC 限定到固定 16 个有空间参考的录制，不追求无止境 tracking 优化。与新取得的
真实 timed trajectory 一起形成下一阶段的输入条件和验证层。见
[世界模型数据衔接](../research/world_model_data_bridge.md)。

## 4. 执行与验证

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser run-author-semantic-qc \
  --mapping data/quality/author-canvas-audit-v3/audit.json \
  --episodes data/structured/author-snooker-incremental-v1/snapshots/551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e/episodes.jsonl \
  --output data/quality/author-cue-gap-next \
  --workers 2 --max-new-output-gb 2 \
  --profile-suite cue_gap_no_background
```

新产物约 100 MB，16 个录制完成、失败 0，重复执行新增处理 0。原始数据与旧 v1
保持不变。薄线、大白色遮挡、近邻球、两个 cue 候选和默认行为均有回归测试。
来源颜色与 compact-cue 规则只在明确 opt-in 的来源条件中生效。
