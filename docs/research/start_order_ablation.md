# 实验 A：搜索起点顺序消融

写定日期：2026-10-21，在运行之前。开发研究：复用[规划 pilot](planning_pilots.md)的 32 个开发布局
和 64 条已有模型回复，新增模型请求 0；只用 CPU（4 workers，BLAS/OMP 各 1 线程）。

## 1. 问题

v1 把模型动作放在第一个起点时，8/32 查询的成功率低于几何起点；trace 审计显示几何候选被推迟到
第 33 次查询。本实验区分两种解释：

1. 调度：模型起点只是先占用了预算；
2. 起点质量：模型动作作为起点本身就不如几何候选。

## 2. 设计

同一 32 布局、同一 Powell 搜索器和 seed、128 次查询 horizon，在 8/32/128 前缀交隐藏 judge；判分
不变（1 球半径，母球不落袋）。

| 臂 | 起点（按顺序） | starts |
| --- | --- | ---: |
| `geometry_powell` | 几何候选，再补 Sobol（同 v1） | 4 |
| `{model}_first` | 模型动作＋前 3 个几何/Sobol 起点（同 v1 warm-start） | 4 |
| `{model}_last` | 同一组 4 个起点，模型动作放最后 | 4 |
| `{model}_only` | 只用模型动作 | 1 |
| `geometry_g1_only` | 只用第一个几何候选 | 1 |
| `sobol_s1_only`…`sobol_s4_only` | 只用第 k 个 Sobol 点（随机单起点参照） | 1 |

模型是 GPT-5-mini 和 GPT-5（v1 的 minimal 提案）。非法提案（mini 有 1 个）在三个模型臂都记失败，
不替换。共 12 臂 × 32 布局 = 384 个 job。

## 3. 预注册的预测与比较

- 复现检查：`geometry_powell` 和 `{model}_first` 的查询序列应与 v1 trace 完全一致。
- 预测：两个模型的 `{model}_last` 在 8/32/128 前缀分别为 10/12/12（共 32）。即 8 和 32 与几何持平
  （v1 的下降来自调度），128 仍是 12（起点集合与 v1 warm-start 相同）。
- 配对比较（按 source group bootstrap，95% 区间）：last − first、last − geometry_powell、
  only − geometry_g1_only，各在 8/32/128。
- 判读：若 last 达到 10/12 且 last − first 在 8/32 为正，v1 的下降是调度效应；only − g1_only 回答
  起点本身的质量。32 个布局的统计力有限，结果只用于开发判断。

## 4. 运行偏差（在看到结果之前记录）

- 第一次运行（`start-order-ablation-v1`）完成 384 个 job 中的 367 个后中断：两个 Sobol 单起点臂
  （`sobol_s2_only`、`sobol_s3_only`）里，有界 Powell 因浮点舍入给出约 −1e−17 的坐标，被搜索立方体
  检查拒绝。这是数值边界问题，不是物理失败。
- 修复：`baselines.unit_action` 只把超出立方体不到 1e−9 的点夹回边界，真正越界仍报错。立方体内的
  点走原来的路径，所以已完成运行（v1 规划 pilot、测试集经典基线）的轨迹不受影响；修改前已确认测试集
  经典基线在原代码下重跑新增工作为 0。修改前的 `baselines.py` 见存档提交 `a9d9e84`（到 `d001933` 未变）。
- producer 变了，所以全部 384 个 job 在新目录 `start-order-ablation-v2` 重跑。v1 目录保留，不删除，
  也不使用它的部分结果。

## 5. 结果（v2）

384 个 job 全部完成，4 workers 用时 5 分钟，新增模型请求 0。95 条 v1 trace（geometry_powell 32、
GPT-5 first 32、mini first 31）逐次相同；v1 中断前完成的 367 个 job 在 v2 中逐条相同。物理：搜索
29,929 次、隐藏 judge 474 次，未解决预留 0；禁止模拟的重跑新增工作为 0。

成功数（共 32 个布局）：

| 臂 | 8 次 | 32 次 | 128 次 |
| --- | ---: | ---: | ---: |
| geometry_powell（4 起点） | 10 | 12 | 14 |
| geometry_g1_only | 10 | 12 | 14 |
| GPT-5 first / **last** / only | 0 / **10** / 0 | 1 / **12** / 1 | 12 / **12** / 7 |
| GPT-5-mini first / **last** / only | 0 / **10** / 0 | 2 / **12** / 2 | 12 / **12** / 5 |
| Sobol 单起点 s1 / s2 / s3 / s4 | 0 / 1 / 1 / 0 | 5 / 4 / 2 / 4 | 11 / 10 / 8 / 10 |

预注册预测（last 为 10/12/12）在两个模型上都完全命中。

配对差（成功率百分点，按 source group bootstrap 的 95% 区间）：

| 比较 | 8 次 | 32 次 | 128 次 |
| --- | --- | --- | --- |
| GPT-5 last − first | +31.3 [15.6, 46.9] | +34.4 [18.8, 53.1] | 0 |
| mini last − first | +31.3 [15.6, 46.9] | +31.3 [15.6, 50.0] | 0 |
| last − geometry_powell（两模型相同） | 0 | 0 | −6.3 [−15.6, 0] |
| GPT-5 only − geometry_g1_only | −31.3 [−46.9, −15.6] | −34.4 [−53.1, −18.8] | −21.9 [−37.5, −9.4] |
| mini only − geometry_g1_only | −31.3 [−46.9, −15.6] | −31.3 [−50.0, −15.6] | −28.1 [−43.8, −12.5] |

判读：

1. v1 在 8/32 次的下降完全来自调度：模型起点放到最后，8/32 次与几何完全相同。
2. 作为起点，模型动作明显差于第一个几何候选，128 次时仍少 22–28 个百分点；也不比随机 Sobol 单
   起点好（模型 5–7/32，Sobol 平均 9.75/32；这一条是描述性的，不在预注册比较里）。
3. 128 次时，模型动作占掉第 4 个起点，比几何/Sobol 的第 4 个起点少解出 2 个布局。
4. 在这批开发布局上，已有的单个模型动作作为数值搜索起点没有带来收益。模型利用真实反馈的能力
   （见 [C](c_feedback_attribution.md)）留给正式测试的 P5 再检验。

## 6. 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser start-order-ablation \
  --panel data/research/model-evaluation-panel-200-v1 \
  --native-run data/research/azure-gpt5-goal-200-v1 \
  --reference data/research/planning-budget-warmstart-pilot-v1 \
  --output data/research/start-order-ablation-v2 --workers 4
```

实现：[start_order.py](../../src/inverser/start_order.py)；
[测试](../../tests/test_start_order.py)覆盖臂构造、非法提案、v1 trace 复现和零新增工作恢复。
