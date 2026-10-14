# 从数据质量到世界模型：下一阶段测什么

本文锁定 empirical study 主线，不把数据量或 tracking 修复本身作为世界模型贡献。
本轮已经下载真实辅助数据、完成有界旧数据 QC，并准备可执行的数据输入与参考基线。
研究者已接受 G1 的有限分层范围，并锁定开发诊断面板。尚未运行待测 LLM/VLM、
训练 learned dynamics 或通过正式评价 gate。

## 1. 三条证据链不能混合

| Track | 输入与证据 | 目标 | 当前状态 |
| --- | --- | --- | --- |
| Controlled action-conditioned dynamics | 合成 snooker 初始状态、指定 cue action、可见物理参数；隐藏未来与 witness | Native intervention prediction、事件与轨迹响应 | 已锁定 64 布局开发面板、基础/单变量动作、paired state/BEV；未做正式盲测冻结 |
| Prediction to planning utility | 相同候选 bank、目标、可见证据和查询预算 | 排序/regret、最终提交 goal success、成本和工具归因 | 经典/mocked harness 已验证；没有模型成绩 |
| Real factual dynamics | BSA 已发表的 past x/y/time prefix；隐藏未来标注 | Source-annotated future compatibility | 从 1,053 case pilot 锁定 64 个不同 source-round 开发 case；local-only adapter 尚待接入 |

给模型一个 Pooltool 工具不意味着它自己学到了动力学；BSA 单球轨迹也不能变成任意
cue/spin 干预的真实 GT。学习式模型、校准物理引擎、感知/latent belief 与搜索器分别命名。
二维是平面运动表示，速度、旋转、运动状态和材料参数仍可能是 latent state。

## 2. 新数据补上了哪些缺口

作者 snooker 的 manual raster 主要检验空间路径 union；新取得的 BSA 轨迹含显式
cm 和视频时间字段。现在能把 real factual forward prediction 作为单独验证层，而不
继续只看终态或路径图。

数据准备见 [BSA 标签说明](../data/billiards_sports_analytics.md)。不同球轨迹可能属于
同一击球/局，不能将轨迹文件、未来查询或多视图计成更多独立单杆任务。源单位声明不是
独立测量，完整场景、spin 和真实 cue controls 仍缺失。

## 3. 已导出的真实事实预测 pilot

来源为 2,080 个 source-declared cm/time 轨迹。严格按协议导出：

- 1,053 个去重后可用的球轨迹 forecast case，180 个 source-round 路径组；
- 前 10 个观测点作可见 prefix，时间从第一个点减去原始录像 offset；
- 请求 prefix 末端之后 0.25、0.5、1.0 秒的位置；
- 取各请求时刻之后的首个原始标注，容许最大 60 ms gap，不插值；
- 缺少 horizon 的 654 条、过短历史的 368 条和重复 annotation trace 的 5 条单独记录，
  不按预测是否成功筛选；
- Public 文件不含未来位置、原始文件/录像身份、全轨迹终点或生成 witness。

产物：

- [可见前缀](../../data/research/bsa-factual-forecast-pilot-v1/public_prefix.jsonl)。
- [隐藏未来与来源](../../data/research/bsa-factual-forecast-pilot-v1/privileged_future.jsonl)。
- [协议和统计](../../data/research/bsa-factual-forecast-pilot-v1/summary.json)。

1,053 是合格**轨迹 case**，不是 1,053 次独立真实击球。这个集合用于开发和接口验证，
不是已冻结的盲测。未来应按 video/match/round 来源保组，不把同一 source event 拆开。

## 4. 已运行的非学习参考，不是 LLM 主结果

在同样的 3 个实际 annotation query 上，先按 case 算位置误差，再对 source-round
组等权平均：

| 简单参考 | 平均 query 位置误差，source-declared cm ↓ |
| --- | ---: |
| 一直保持最后位置 | 25.99 |
| 最后 3 个 prefix 点拟合匀速并外推 | 42.10 |

这是实际来源未来位置的兼容性计算，不是物理台面独立实测误差，也不是全轨迹 ADE。
数据里有减速、接触、停止和不可见完整场景；“继续直线匀速”不一定比“保持最后位置”
更接近该实际未来。没有逐案例因果核查，不能从平均值归因给某一种物理机制。

这个参考说明**预测基线和任务已能运行**，不说明某个 learned world model 已有效。
模型之后必须在相同信息下比较，充分记录轨迹缺失、单位、完整场景不足与不确定性。
真实 prefix 输入仍只在本地使用，不发送到付费 API。

## 5. 预处理如何变成研究因素

旧数据的颜色、compact-cue 和缺帧关联条件已经冻结成可测 observation factors，
见 [cue/gap 消融](../pipelines/cue_gap_qc.md)。

首轮模型设计保留：

1. **RQ1：干预响应。** 指定相同初始状态下的动作/参数干预，对照保持物理不变的外观
   与排列变化；先测无工具预测，禁止 goal/future 泄漏。
2. **RQ2：预测—规划联系。** 在同候选与同 planner 下比较预测误差、事件、候选排序
   和真正提交动作的参考执行。成功接受任何合法可行动作，不强迫复现历史 witness。
3. **RQ3：信息与物理不确定性。** 分开 oracle 状态、图像/提取状态和 latent physics；
   Support/query 校准分离。Real factual track 是观察验证，不声称真实反事实。

若没有独立坐标 GT，不能从自动 detector 指标反推出已校准的位置噪声。当前拟议的
synthetic noise grid `0,0.25r,1r` 是声明的受控敏感性条件，不是实测相机误差分布。
未观测球也不能被 tool 偷偷补为隐藏 true state。

可核查协议保存在
[world-model-readiness-v1](../../data/research/world-model-readiness-v1/protocol.json)。

## 6. 下一步不再无界预处理

研究者已接受有限的 G1 分层范围，[开发小面板](model_diagnostic_panel.md)实际生成并
锁定：64 仿真布局＋64 不同 source-round 真实 case，共 448 请求记录；API 共同子集
仅 48 个合成请求，目前 0 次模型调用。G2/G4 正式盲测没有因此通过。

接下来：

- 明确 GPU UUID/资源或实际 API request/token quota，再做 backend smoke；
- 运行既定 RQ 的 native、强经典基线与 simulator-assisted 对照，保留负面结果；
- 有诊断证据后，再决定是否需要 learned transition baseline 或新方法训练。

现有 ModelCall 的主 API 路径仍只接受合成 forward/goal 合同，真实前缀任务不在本轮
自动送入模型。真实轨迹输入的 local-only adapter 接入属于下一阶段；没有把“导出 JSON”
当成“已经完成模型推理”。

## 7. 复现接口

```bash
.venv/bin/python -m inverser prepare-real-world-model-pilot \
  --trajectories data/BilliardsSportsAnalytics/audit-combined-v4/trajectories.jsonl \
  --output data/research/bsa-factual-forecast-pilot-next
```

使用新目录。Prefix/query 时间边界、单位、source action GT、重复轨迹和未来泄漏有
测试。没有学习式模型训练、真实物理执行或 checkpoint/API 比较结果。
