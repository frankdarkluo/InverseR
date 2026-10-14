# 小型开发诊断面板 v1

研究者已接受 G1 的分层证据范围：主任务用受控 snooker 模拟动作真值，BSA 只作真实
事实未来位置兼容性，原始弱观察用于感知因素和案例。这个确认不把真实观察升级成实测
cue/spin GT，也不授权模型/GPU/API、训练或发布。

实际面板位置：
[model-diagnostic-panel-v1](../../data/research/model-diagnostic-panel-v1/)。
这是已锁定的**开发诊断快照**，不是 G2/G4 的正式盲测 benchmark。

## 1. 已锁定的数量与请求形状

| 层 | 来源单位 | 输入/任务 | 请求记录 |
| --- | ---: | --- | ---: |
| Simulated snooker | 64 个不同 initial layout /源组 | 两个指定动作的 blind forward＋一个 goal；各有 structured/BEV 视图 | 384 |
| Real factual nine-ball | 64 个不同 source-round 组，各一条轨迹 | 10 个 prefix 点到三个 future query；仅本地 | 64 |
| 合计 | 不将 views/cases 称为更多独立真实击球 | 主/辅助指标分开 | 448 |

64 个仿真布局按 `free_roll/center/vertical/side` 各 16 个采样，固定 seed 20261014。
它们来自已经用于开发的 2,000 个 pilot，不能改称从未接触的正式测试集。
64 个真实 case 从 1,053 个可用轨迹中按 source-round 保组抽样；不根据参考预测误差选
“容易”或“困难”样本，不复制原始视频或权重。

单变量动作变化：

- free roll：cue-stick speed ×1.25；
- center：azimuth +15°；
- vertical/side：相应 tip offset 按原符号增加 0.05。

这些是 cue controls 干预；tip offset 改变可能同时影响平动与转动，不声称仅改变
angular velocity。当前所有 initial 球静止、elevation 为 0，known physics 显式给出。
Forward 采样时刻固定 0.25、0.5、1.0 秒；goal 主容差固定 1 个球半径、no scratch。

## 2. Public/private 与世界模型证据

Public 请求只引用 `public/` 下的 payload/images：

- blind forward 不给 goal、future、历史 track identity 或 witness；
- BEV 文本不夹带球坐标，使用与 structured 条件相同的物理/单位说明；
- inverse goal 允许目标图/状态，但红球按 endpoint-local exchangeable 集合评分；
- real prefix 不给未来位置、完整轨迹、文件/录像身份或真实 cue controls；
- 两个真实球轨迹来自同一 source round 时不能跨选入不同组。

Private 区保存 source group、原始 record、动作干预元数据和 reference future/events。
分离是 loader/请求合同的边界，不声称拥有一个隔离所有本地进程的安全 sandbox。
模型不能通过额外文件/代码工具读取这些隐藏内容。

`load_synthetic_panel_call` 从冻结公共文件重构 typed ModelCall，检查与已锁定 prompt
完全一致，不需要读取 private。真实 prefix 输入合同已验证，但 local-only adapter
仍属下一阶段，不将它的 `model_adapter_ready=false` 静默改成可运行。

## 3. 实际验证，不是模型成绩

- 64 个基础 witness 全部重放，终态与事件一致；
- 64 个动作干预轨迹完成参考执行/物理采样，共 128 次 CPU reference replay；
- 384 synthetic＋64 real 请求 shape、唯一 ID、不同来源组、图像角色与字段边界通过；
- 每个 public/private/image artifact 写入 SHA-256，配置、来源、producer 代码、依赖、
  resolved physics metadata 均锁定；
- 重复 prepare 只验证已有产物，不重新生成轨迹或图像；
- 原始来源与之前面板不改写，当前面板约 8.3 MB；
- 全套 187 个测试、Ruff、Mypy（35 个 source file）通过，模型调用为 0。

验证入口：
[manifest](../../data/research/model-diagnostic-panel-v1/manifest.json)、
[validation](../../data/research/model-diagnostic-panel-v1/validation.json)、
[request index](../../data/research/model-diagnostic-panel-v1/request_index.jsonl)。

## 4. API 小型共同子集与成本

预选 8 个仿真布局，各 family 2 个。每布局“两次预测＋一次 goal”×两种输入，
共 **48 个原生请求**，以 `api_common_subset=true` 标记。

这不是 API 授权。Probe、重试、格式修复和任何 simulator-feedback turn 还要另外计入
已确认总额；不能把“100 请求”理解为每格 100 episode，也不保证它够跑所有 treatment。
真实 64 请求始终 `local_only=true`，不外发。

完整本地 native 面板为 448 个计划请求，其中 real 路线需补 local adapter。需要 GPU
许可与 runtime/backend smoke 后才运行；当前不启动 server、不载入模型、不做训练。

## 5. 后续模型诊断

先跑无工具、known-physics 的 native forward/goal：

1. RQ1：同 initial/action 干预对的未来状态、event 与响应一致性。
2. RQ2：另以同 bank/同 planner/同证据预算诊断 prediction→ranking→goal utility；
   不从 native 小面板的单项误差直接推出控制能力。
3. RQ3：后续版本增加声明的 observation noise/hidden physics/support-query 条件。

面板修改必须新版本；正式评价还需未用于开发的布局/来源、模型/checkpoint/prompt、
预算和容差冻结，以及 G2/G4 决策。当前 G1 是有限证据范围接受，不是正式 test freeze。

## 6. 复现准备

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser prepare-diagnostic-panel \
  --synthetic data/pilot-v0/privileged.jsonl \
  --real-public data/research/bsa-factual-forecast-pilot-v1/public_prefix.jsonl \
  --real-private data/research/bsa-factual-forecast-pilot-v1/privileged_future.jsonl \
  --output data/research/model-diagnostic-panel-next \
  --seed 20261014
```

使用新目录；相同配置可复用既有冻结版本，来源/code/协议或 artifact 变化明确报错。
