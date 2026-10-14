# Azure GPT-5 原生开发冒烟对照

## 范围与配置

研究者明确要求参考 SoccerTactics 的 Azure 配置与认证脚本，试运行 GPT-5-mini 和
GPT-5。大批运行额度未得到进一步答复，因此没有运行每模型 48 请求的完整共同子集，
而是采用更小的工程 pilot：**2 个固定布局、每模型 12 请求，共 24 请求**。

- 输入来自已锁定的 [开发面板](model_diagnostic_panel.md)，只发送合成数据。
- 两个不同 initial-layout/source group，分别为 `free_roll`、`center`。
- 每布局 2 个指定动作的 blind forward＋1 个 goal，各有 structured/BEV 视图。
- 两模型：`gpt-5-mini-2025-08-07`、`gpt-5-2025-08-07`，实际返回版本已严格验证。
- 同一 Azure GlobalStandard endpoint，API version `2024-12-01-preview`；
  `reasoning_effort=minimal`，`max_completion_tokens=4096`，JSON-object 输出。
- 模型收到数值 output schema，但没有 API-level strict JSON-schema enforcement。
- 不发送不支持的 `temperature/top_p/seed/max_tokens`；当前不是确定性推理。
- 总请求上限 24，总 Token 上限 300,000，单请求预留 16,384；没有自动重试、格式
  修复、候选 bank、模型可用的仿真工具或新的真实轨迹输入。
- 不复制原配置/凭据；本地读取指定外部文件，DefaultAzureCredential 在运行时取 token。
  保存 endpoint hash、配置源 hash、模型 revision，不将 token 或资源 URL 写入产物。

这是只有两个独立布局的工程/开发诊断，不是正式模型排名，不支持统计显著性、spin
能力、现实控制或泛化结论。Native 中间预测不能直接当成 planning-utility 证据。

## 实际结果

位置误差按运动球的 query 位置计算，先对每个有效 forward 请求求平均，再对该格
有效请求等权平均，以球半径为单位。它是 **valid-only** 描述，失败没有伪造为零误差。
主容差为 1 个球半径；goal 由隐藏参考执行判分，允许任何可行合法动作，不要求还原
历史 witness。

| 模型 | 输入 | 合同有效输出/总请求 | 有效 forward/4 | 运动球平均误差，球半径↓ | forward 达标/4 | goal 成功/2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5-mini | structured | 3/6 | 1/4 | 7.19 | 0/4 | 0/2 |
| GPT-5-mini | BEV 图像 | 4/6 | 2/4 | 39.04 | 0/4 | 0/2 |
| GPT-5 | structured | 5/6 | 3/4 | 7.70 | 0/4 | 0/2 |
| GPT-5 | BEV 图像 | 6/6 | 4/4 | 28.76 | 0/4 | 0/2 |

**各格有效样本不同，不能用 7.19 vs 7.70 宣称 mini 的物理预测优于 GPT-5。**
同样，不能仅由这个表把图像条件的差异归因为视觉能力或物理推理能力。

6 个 native 合同失败仍计入请求和失败分母：

- mini 3 次错误 query 时间、1 次 schema/layout 组织错误、1 次不可用 contact geometry；
- GPT-5 1 次没有保留全部初始球 slot；
- 无格式修复或重试；原回复和实际 usage 保留在本地账本。

其余输出满足字段/单位/slot 约束不表示预测正确。全部有效 forward 的 contact F1 为
0；4 个完整、可评分动作干预对也均未达到 joint position/event 容差。缺失配对不
补成有效预测。8 次 goal 提交格式合法，但参考执行均未达到目标。

通俗解释：**API 能正常答题，但这次直接作答尚不能可靠预测轨迹或完成击球目标。**
样本仅两个布局、推理预算 minimal，且有合同失败，因此不能据此判定模型普遍“不懂
物理”。下一版先隔离输出可靠性，再讨论物理误差和工具收益。

## 资源与可复用性

- 实际 provider 请求 24；input 62,570、output 5,296、total **67,866 Tokens**。
- output usage 包含服务端计入的 completion/reasoning；没有成本价信息，不编造美元账单。
- 参考执行 8 次，全部是 hidden judge，不是 agent 可见工具调用。
- 顺序调用耗时合计约 136.90 秒：GPT-5-mini 55.14 秒，GPT-5 81.75 秒；这是含网络
  的小样本 elapsed，不是独立服务器吞吐 benchmark。
- 本地 run artifacts 约 161 KB。全部 24 job 有不可变 checksum envelope。
- 重复调用用禁止发送请求的 MockTransport 验证：新增模型请求和参考执行均为 0，
  使用量仍为 67,866。错误回复同样不自动重计费。
- 190 个单元/回归测试、Ruff、Mypy 36 个源文件通过。

本地机器产物：
[manifest](../../data/research/azure-gpt5-native-pilot-v2/manifest.json)、
[summary](../../data/research/azure-gpt5-native-pilot-v2/summary.json)。
原回复、账本、隐藏真值不提交到 Git；本页是去除 endpoint/凭据的可读摘要。

## API 脚本与复现

脚本实现：[azure_study.py](../../src/inverser/azure_study.py)；
transport/合同：[model_adapters.py](../../src/inverser/model_adapters.py)；
持久化请求与计费：[evaluation.py](../../src/inverser/evaluation.py)。

```bash
# 不带 --execute，只锁定计划，不申请 token 或调用模型。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser azure-panel-pilot \
  --panel data/research/model-diagnostic-panel-v1 \
  --azure-config ../SoccerTactics/codes/inference/azure_openai_config.json \
  --endpoint-index 4 \
  --output data/research/azure-gpt5-native-pilot-next \
  --layouts 2 --max-requests 24 --max-total-tokens 300000 \
  --max-output-tokens 4096 --reasoning-effort minimal
```

明确授权后增加 `--execute`；`--limit-jobs 2` 先跑两模型共同的首个请求，再原参数
恢复。endpoint index 4 是本次外部 inventory 的选择，不是通用 Azure 资源编号；
别人的配置必须选择包含这两个精确 GlobalStandard deployment 的条目。

安装 inference extra 后，用户的 Azure CLI/环境/managed identity 必须拥有该资源的
数据面访问权限。认证和服务失败明确报错，不改用个人密钥或静默切换模型/资源。
来源、endpoint、producer、参数、预算变化必须新 output 目录；不能覆盖旧结果。

## 下一阶段

在新版本中，用同题同预算比较严格结构化输出与当前数值 schema 提示，先分清“格式
失败”与“物理预测失败”；再比较相同的 reasoning effort/可见证据与经典参考。
之后才添加等候选/等查询预算的 simulator-assisted 对照，测预测、排序、执行成功、
regret 和成本。扩大到 8 布局/每模型 48 请求需要新的明确预算，不从本次试跑推定。
