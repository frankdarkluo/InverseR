# 规划 pilot：经典搜索预算与已有模型提案的价值

实际运行日期：2026-10-17。开发研究，不是正式盲测或新方法复现。

## 1. 本轮回答什么

在 [200 布局原生实验](azure_native_pilot.md)之后，本轮先测两个问题：

1. 不使用 LLM，只用公开几何、物理参数和有界数值搜索，能做到多少？
2. 保留相同搜索器和预算，把已有 GPT 动作作为首个数值起点，是否比几何起点更好？

这是经典物理模型与候选提案的研究，不是学习式 world model、盲测后果预测或
SimLM/CueTip 全流程复现。研究者提出先做 pilot；新增 API 额度的问询暂未得到回复，
因此只执行已有 CPU 范围内的实验，**新增 API 请求和 Tokens 均为 0**。
无工具/真实反馈/无新证据反思的模型对照仍待后续额度。

## 2. 固定数据、信息和判分

- 从既有 200 布局开发面板中选 32 个不同 initial-layout/source group，每个 source
  family 8 个。按 seed 20261017 与 source-group hash 排序选择，不依据模型成绩。
- 四组为 `free_roll`、`center`、`vertical`、`side`。这是生成 witness 的条件，
  不意味着目标必须使用同样的 spin 或路线。
- 所有方法读取相同 public structured initial/goal、known physics、cue/台面规格。
  搜索器不读取生成动作、未来轨迹或 event truth。
- 复用原 200 布局实验中的 64 条模型回复，逐条验证公共 panel、模型/请求/案例身份
  和 checksum。历史使用量为 mini 67,838、GPT-5 67,871 Tokens；本轮没有重新调用它们。
- 目标接受任何合法可行的单杆动作；红球使用 exchangeable bottleneck matching。
  所有球最大终态位置误差不超过 1 球半径（0.02619375 m），且母球不落袋。
- 现有数据是 1–4 球的模拟开发布局，不代表完整专业 snooker 或真实物理执行。

原 panel 和两模型请求没有改写。子集中的公共目标已被原先全部 200 个
[known-witness 正控](../../data/research/azure-gpt5-goal-200-v1/positive_control_validation.json)
覆盖，且全部通过。正控答案不进入候选生成。

## 3. 方法与查询计数

| 方法 | 可见信息与策略 | 模型成本 | 搜索反馈 |
| --- | --- | --- | --- |
| Geometry direct | 按目标几何与公开摩擦/impact 规格，直接计算第一个候选 | 0 | 无 |
| GPT direct | 复用原 `minimal`、单次数值 goal 提交 | 历史调用，不重计 | 无 |
| Random / Sobol | 相同动作范围内的独立/低差异采样，选已反馈的最低 loss | 0 | 已知物理引擎 |
| Geometry＋Powell | 几何候选优先，再补随机化 Sobol 起点，最多 4 starts | 0 | 已知物理引擎 |
| GPT＋Powell | 原模型动作排在第一，其余沿用相同几何/Sobol 序列，最多 4 starts | 复用历史提案 | 已知物理引擎 |

搜索沿用 cue-stick speed `[0.05,3]` m/s、azimuth、半径不超过 0.5 的归一化 tip
contact disk、elevation=0。所有合法历史提案的 speed 位于原搜索范围内，没有裁剪。
一个 mini 输出原本非法，直接和 warm-start 方法均保留为失败，不替换成几何答案。

每个搜索 trajectory 的最大 horizon 为 **128 次 objective attempts**，在其 8、32、
128 次前缀选已知最优候选，分别交给隐藏 judge。成功可提前停止。**这些是固定
128-horizon 算法的 anytime 前缀，不是为每个预算另调参数或重分配 starts 的独立实验。**

缓存命中和失败尝试仍占 objective budget，实际新 engine rollout 另计。搜索反馈与
hidden judge 用不同证据 role/namespace；judge 的终态不反馈给搜索。直接方法的
search budget=0，但事后 judge 仍有 CPU 成本。

## 4. Pilot A：经典搜索预算曲线

所有结果的分母都是同一批 32 布局。

| 方法 | 8 查询前缀 | 32 查询前缀 | 128 查询前缀 |
| --- | ---: | ---: | ---: |
| Random | 0/32 | 0/32 | 0/32 |
| Sobol | 0/32 | 0/32 | 0/32 |
| Geometry＋Powell | 10/32（31.25%） | 12/32（37.50%） | **14/32（43.75%）** |

零搜索的 **Geometry direct 为 10/32（31.25%）**。它不是读取 witness，而是直接用
目标几何和已声明物理计算动作。它在 `free_roll` 8/8、`vertical` 1/8、`side` 1/8
成功，`center` 0/8。不同路线或不同 spin 达成相同目标按主任务规则正常计成功。

由此，这批布局中，纯随机/低差异覆盖即使改善误差，也没有在 128 次内碰到 1r
可行区域；结构化几何起点比盲目覆盖更有用。该结果不证明所有采样算法无效，也不是
对所有优化器或更大预算的排名。

## 5. Pilot B：模型提案作为首个数值起点

| 方法 | 无搜索直接提交 | 8 查询前缀 | 32 查询前缀 | 128 查询前缀 |
| --- | ---: | ---: | ---: | ---: |
| Geometry / Geometry＋Powell | 10/32 | 10/32 | 12/32 | **14/32** |
| GPT-5-mini / GPT-5-mini＋Powell | 0/32 | 0/32 | 2/32 | 12/32（37.50%） |
| GPT-5 / GPT-5＋Powell | 0/32 | 0/32 | 1/32 | 12/32（37.50%） |

这里的 “GPT＋Powell” 不表示模型根据反馈作了新推理：**模型只提供旧动作，后续全部
由确定性数值搜索控制**，并且仍可使用几何候选。因此 0→12 的改善不能归给模型学习了
物理，也不是已经测得的语言反思能力。

128-query 分层：

| Source family | 每方法布局 | Geometry＋Powell | Mini＋Powell | GPT-5＋Powell |
| --- | ---: | ---: | ---: | ---: |
| free_roll | 8 | 8/8 | 8/8 | 8/8 |
| center | 8 | 1/8 | 0/8 | 0/8 |
| vertical | 8 | 3/8 | 2/8 | 2/8 |
| side | 8 | 2/8 | 2/8 | 2/8 |

两种 warm-start 的 12 个成功布局均包含在几何 baseline 的 14 个成功中，没有新增成功
布局。各自相对几何 baseline 的配对成功率差为 −6.25 个百分点，按 source group
bootstrap 的 95% interval 为 [−15.625,0] 个百分点。样本小、只有一个冻结搜索 seed
分配，且涉及多项探索，不能据此建立普遍的模型强弱结论。

### 起点顺序是一个重要解释，不应隐藏

本协议把模型动作放在第一，最多 4 starts、每个初始分支可先占约 32 次查询。trace
审计发现：

- 几何 baseline 在全部 32 个案例的第 1 个 objective 就访问首个解析几何候选。
- 在确实访问了该候选的 warm-start 案例中，mini 的 29 个、GPT-5 的 31 个案例都到
  **第 33 个 objective** 才访问它。其余案例提前成功或原提案非法。

因此，8/32 前缀的下降部分反映“错误起点优先消耗预算、延后几何候选”的调度效应。
**本轮检验的是这种具体注入策略，而不是证明所有语言提案、事件路线或 seed 排序
都没有价值。** 后续可以固定 seed set、交换顺序或限制首分支占额来检验该解释；
那是新的消融，不按本轮结果回改旧协议。

## 6. 不确定性、误差与实际资源

128-query 的 exact-binomial 95% interval：

- Geometry＋Powell 14/32：[26.36%,62.34%]。
- 两种 GPT warm-start 12/32：[21.10%,56.31%]。
- 无工具模型和 Random/Sobol 的 0/32：[0,10.89%]。

这只是受控开发样本的统计描述，不是现实执行概率。原来 200-layout 原生结果的 CI
仍按其原分母报告，不用 32 个子样本替代。

128-query 最低规划 loss 的中位数分别为：Random 9.34r、Sobol 10.73r、Geometry＋Powell
1.81r、Mini＋Powell 2.42r、GPT-5＋Powell 3.01r。该 loss 包括 status/scratch penalty，
不是无条件平均位置误差；mini 有一个非法种子没有有限 loss，成功分母仍保留 32。

实际执行：

- 32 不同 source/layout groups，四组各 8；8 个方法，共 **256 个 checksum job**。
- 总 objective attempts **16,575**，含缓存命中；实际 `agent_search` rollout **16,551**。
- 实际独立 `hidden_judge` rollout **374**，合计 **16,925**；未解决 reservation 为 0。
- 新模型请求/Tokens **0/0**；没有 GPU、模型训练、真实媒体外发或任意代码工具。
- 最多 4 CPU workers，BLAS/OMP 各 1 thread；当前产物约 142.33 MB，低于已批准存储额度。
- 重复运行用禁止 simulator construction 的验证通过：新增 job/physics/model request
  均为 0，所有源回复、配置、代码、依赖和 job hash 锁定。
- 另从 hidden-judge cache 重算 572 个有动作的 checkpoint，分数/loss 全部一致；
  核查每个预算的候选只来自已执行前缀，成功曲线单调。64 条源回复 hash 未变。
- 全部 **197 测试**、Ruff、Mypy（37 个源文件）通过。

产物：[manifest](../../data/research/planning-budget-warmstart-pilot-v1/manifest.json)、
[summary](../../data/research/planning-budget-warmstart-pilot-v1/summary.json)、
[validation](../../data/research/planning-budget-warmstart-pilot-v1/validation.json)、
[分案例输出](../../data/research/planning-budget-warmstart-pilot-v1/cases/)。
这些本地数据和 ledger 不提交到 Git。

## 7. 对下一轮的作用

**已经支持的观察：** 不能只与 0/200 的直接 LLM 比较；已知物理的几何/数值 baseline
必须保留。把原始模型动作简单插到搜索首位，本轮没有比几何起点更好；收益主要来自
几何与数值反馈，且紧预算调度会改变结果。

**仍未回答：** 盲测后果预测、指定动作干预、同 bank 的预测到选择关联、VLM 的真实
反馈修正、事件级路线、学习式转移和真实世界控制。本轮不是这些任务的负面结果。

后续优先：

1. 固定 seed set 和实际 query budget，分离起点质量与调度。
2. 做小型 blind forward＋共同候选选择，测接触事件/位置预测是否影响 ranking。
3. 在独立预算下比较真实仿真反馈、无新证据反思与纯数值搜索；这才接近
   [方法调研](world_model_methods_2023_2026.md)中的 agent/world-model 归因。

不优先格式修复，不因本轮成功自动扩训练或变更正式评价 gate。

## 8. 复现

实现：[planning_pilot.py](../../src/inverser/planning_pilot.py)；
复用 [baselines.py](../../src/inverser/baselines.py) 的 bounded search/warm-start；
[测试](../../tests/test_planning_pilot.py)覆盖小样本工程正控、invalid seed、预算、
source grouping、job corruption 和零新增工作恢复。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser run-planning-pilot \
  --panel data/research/model-evaluation-panel-200-v1 \
  --native-run data/research/azure-gpt5-goal-200-v1 \
  --output data/research/planning-budget-warmstart-pilot-next \
  --layouts 32 --workers 4 --seed 20261017
```

可先加 `--limit-cases 1`，随后同参数去掉限制恢复；未完成的固定子集不称总体结果。
输入、代码或配置变化必须使用新目录，不覆盖旧 API/开发面板或本次产物。
