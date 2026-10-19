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

## 200 个布局的目标击球扩展（2026-10-15）

研究者要求先扩大 evaluation 到 200 samples，暂不研究格式修复。本轮将 sample 定义为
**不同初始布局/源组**，而不是 API 请求、帧或视图。完整旧矩阵会需要 2,400 请求；
范围问询暂未得到答复，因此采用已说明的较小主任务范围：200 个共同布局，两个模型
各一次 structured goal 提交，共 **400 请求**。未扩展图像/forward/干预推理实验。

布局从已有 2,000 个开发记录中按 seed 20261015 固定采样，四个 source family 各 50，
initial hash 与 source group 均有 200 个不同值。来源 family 指生成 witness 的击球
条件，不证明达到该目标必须使用同样的旋转；评分仍接受任何合法、可行的单杆动作。
这不是新生成的 200 个独立真实击球或从未用于开发的正式盲测。

扩展面板采用 `development-evaluation-panel-v1`；原 64-layout snapshot 保留不改写。
所有 200 基础 witness 和 200 动作变化共 400 次 CPU preparation replay 通过。
面板也保留 64 个 local-only 真实 prefix，但这次未选入 API 请求。

推理仍使用上文的模型版本、endpoint/API version、`minimal` effort、4096 completion
cap、原数值提示/JSON-object 输出和 1 球半径容差，没有 strict schema、格式修复、
自动重试或模型可见的仿真工具。新 allocation 明确锁定为 400 请求/2,000,000 总 Tokens；
legacy paired pilot 的 8 布局/100 请求/1,000,000 Token ceiling 保持不变。

### 实际结果

| 模型 | 布局 | 合法动作输出 | 目标成功率 | 母球落袋失败 | 最大终态误差均值，球半径↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5-mini | 200 | 199/200 | 0/200（0%） | 21/200 | 61.75（177 个可算位置误差 case） |
| GPT-5 | 200 | 200/200 | 0/200（0%） | 24/200 | 61.45（174 个可算位置误差 case） |

这里的“最大终态误差”是每布局所有球的最大位置误差，红球用 exchangeable bottleneck
matching，而不是 cue-only 误差或历史动作相似度。球半径为 0.02619375 m；上表的均值
约为 1.62 m 和 1.61 m。Scratch、pocket status 不匹配和无法执行的输出没有虚构为
有限位置误差，因此均值是 conditional 描述，不能凭 61.75 vs 61.45 给两模型排名。
总体成功分母始终是各 200，包括那 1 个非法输出。

| Source family | 每模型布局 | GPT-5-mini 成功 | GPT-5 成功 |
| --- | ---: | ---: | ---: |
| free_roll | 50 | 0/50 | 0/50 |
| center | 50 | 0/50 | 0/50 |
| vertical | 50 | 0/50 | 0/50 |
| side | 50 | 0/50 | 0/50 |

其他失败：mini 有 177 个位置不达标、1 个 pocket inventory/status 不符、1 个非法动作；
GPT-5 有 174 个位置不达标、2 个 pocket inventory/status 不符。
这轮 399/400 动作满足输出合同，故观察到的零成功不能主要归因为格式问题。

两模型各自的 exact-binomial 95% interval 为 [0, 1.83%]，只描述这组受控开发布局的
Bernoulli 采样假设，不是现实桌面或所有 snooker 问题的能力上界。按 source group 配对，
200 对均双方失败；bootstrap 的成功率差区间退化为 [0,0]，不能证明能力等价。

### 判分正控与完整性

为排除 public/private 状态转换或 judge 错误，另外将隐藏 known witness 在**所有
200 个公共 goal 输入**上重放：200/200 在 1e-8 m 容差下通过，最大终态偏差约
6.75e-16 m。见 [positive control](../../data/research/azure-gpt5-goal-200-v1/positive_control_validation.json)。
这些 200 次 CPU audit 不暴露给模型，也不算模型工具预算。

又从已保存的 hidden-judge cache 重算全部 399 个合法提交：399/399 分数一致，
没有新增仿真或 API 请求。见 [judge cache audit](../../data/research/azure-gpt5-goal-200-v1/judge_cache_validation.json)。
398 次实际 hidden-judge rollout 对应 399 个合法提交，其中一个相同动作复用缓存。

- 实际 400 provider 请求，input 828,868＋output 19,341＝**848,209 Tokens**。
- 两个返回版本各 200 次都匹配 `2025-08-07`；没有 reserved/unknown-response 遗留。
- 模型耗时合计约 729.98 秒（含网络与判分）：mini 342.83 秒，GPT-5 387.15 秒。
- 新面板约 25.10 MB，API 产物约 4.25 MB；无真实资料外发、GPU、训练或发布。
- 禁止 HTTP 请求和 simulator construction 的重复执行通过：400 个 job 全部复用，
  新请求/仿真均为 0。193 测试、Ruff、Mypy 36 个源文件通过。

冻结输入：[200-layout panel](../../data/research/model-evaluation-panel-200-v1/)；
结果：[summary](../../data/research/azure-gpt5-goal-200-v1/summary.json)、
[manifest](../../data/research/azure-gpt5-goal-200-v1/manifest.json)。
原回复和账本仍只保存在 ignored 本地 data。

### 复现扩展

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser prepare-diagnostic-panel \
  --synthetic data/pilot-v0/privileged.jsonl \
  --real-public data/research/bsa-factual-forecast-pilot-v1/public_prefix.jsonl \
  --real-private data/research/bsa-factual-forecast-pilot-v1/privileged_future.jsonl \
  --output data/research/model-evaluation-panel-200-next \
  --seed 20261015 --synthetic-layouts 200

# 先锁定零调用计划；有明确授权再增加 --execute。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser azure-panel-pilot \
  --panel data/research/model-evaluation-panel-200-next \
  --azure-config ../SoccerTactics/codes/inference/azure_openai_config.json \
  --output data/research/azure-gpt5-goal-200-next --endpoint-index 4 \
  --design goal_structured --layouts 200 --max-requests 400 \
  --max-total-tokens 2000000 --max-output-tokens 4096 --reasoning-effort minimal
```

新版本目录是必要的：不能覆盖原两布局/64-layout 记录，不能将已有开发 pilot 改名为
正式 test。任务子集、模型、提示、预算、producer 和数据 hash 全部进入协议。

## 后续研究方向（待授权）

按最新决定，暂不优先格式修复。当前已建立 200 布局的无工具单次 goal baseline；
下一步可在相同布局、物理条件和明确的新预算下，比较经典搜索与 simulator-assisted
规划，再分开测预测、候选排序、实际提交成功与查询成本。该结果尚不回答预测精度
是否改善规划，也不检验本地模型、real dynamics、更多 reasoning effort 或反事实能力。
不要从本轮授权推定已批准完整 2,400 请求矩阵、仿真反馈轮次、GPU、训练或正式评价。
