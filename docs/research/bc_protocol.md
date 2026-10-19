# B0 / B / C 实验协议与代码准备

日期：2026-10-19。本轮只完成代码和 mock 测试：**没有调用 API、GPU 或训练，也没有冻结
新的正式数据产物**。研究者已确认路线 B0 → B（B1＋B2）→ C 和推荐档；B0 的实际额度另行确认。

## 1. 为什么是这个顺序

- 已有证据只覆盖"直接出动作"：GPT-5/mini 在 200 布局上 0/200（见
  [200 布局结果](azure_native_pilot.md)），而后果预测只测过 2 个布局、16 个 forward 请求。
- 经典已知物理搜索在 32 布局上 128 次查询达到 14/32；缓存模型动作做起点没有增益，
  差距来自起点调度与被挤掉的随机起点（见[规划 pilot](planning_pilots.md)）。
- 因此先用 B0 定推理强度和真实 token 用量，再用 B 测"后果预测是否可用、能否变成选择"，
  最后用 C 测"模型是否真正利用反馈内容"。

## 2. 实验设计

| 实验 | 面板与案例 | 臂 | 请求数（两模型） | 主要读数 |
| --- | --- | --- | ---: | --- |
| B0 推理强度探针 | `model-diagnostic-panel-v1` 的 8 布局 API 共同子集 | structured；每布局 2 forward＋1 goal；minimal/low/medium/high | 192 | 输出 token（含 reasoning）、`finish_reason`、forward/goal 指标 |
| B1 后果与干预 | 同一面板 64 布局 | 基础＋单变量干预 forward × structured/BEV；另加 structured 的击后母球 v/ω（oracle） | 768 | 运动/静止球误差、静止终态、接触、干预对联合正确性 |
| B2 共同 bank | 新冻结 bank 面板：64 布局、K=8、其中 16 布局加 BEV | 每个候选盲测 forecast；看 goal 直接选；CPU 侧确定性 selector、free-roll 参考、oracle、随机期望 | 1,440 | 选中成功、regret、Spearman、配对差 |
| C 反馈归因 | [规划 pilot](planning_pilots.md) 的同 32 案例 | 每局最多 8 轮：genuine / shuffled / reflection | ≤1,408 | 最终成功、按轮上限曲线、任一轮成功、配对差 |

B1/B2/C 只用一档推理强度，在 B0 之后按实测结果确定。

## 3. 信息边界与对照

- **Oracle 运动学**：给出模拟器测得的击后母球线速度与角速度，用来区分"cue→运动"映射与
  后续动力学。它是额外信息的上界，不并入原生成绩。基础动作的值必须与冻结记录逐位一致。
- **Bank**：候选从与 goal 无关的 Sobol prior 采样、顺序随机；goal 由 bank 内一个可执行、
  不落袋的候选生成，所以有限 bank 的可达上界可解释。生成 goal 的候选只写在 private 区。
  Forecaster 看不到 goal；直接选择看得到 goal 和全部候选。
- **C 的三种条件**：
  - genuine：展示所提交动作的仿真结果（终态、事件、相对 goal 的误差）；
  - shuffled：展示同布局上一个独立的种子随机动作的仿真结果，但当作该次尝试的结果展示；
  - reflection：不执行，只看到自己之前的动作。
- 三种条件**共用同一个第 1 轮请求**，之后只在展示的信息上分叉，配对比较只差反馈内容。
- genuine 的反馈和独立 hidden judge 使用不同 role/namespace；两者不一致即报错。
- 非法输出直接结束该局，不做格式修复或重试；服务/传输错误停止整轮，不自动重发。
- 1 球半径主阈值不变；预注册 0.25/0.5/1/2/5 球半径的敏感性计数，避免结果全部贴地。
- 经典对照：C 的同 32 案例已有几何＋Powell 在 8/32/128 次查询的结果（10/12/14），
  不重跑。注意信息不对称：模型看到终态和事件，Powell 只看到标量 loss。

## 4. 实际核查（plan-only，0 API）

在真实冻结面板上用假的 Azure inventory 和临时目录跑 plan-only，之后已删除临时目录：

- B0 计划 192 请求，B1 768，C 192 局、最多 1,408 请求，B2 1,440。
- Bank 面板 64 布局全部首次抽样即得到合法 goal 候选；每个布局恰有 1 个可达候选，
  所以随机选择的期望成功率是 1/8＝12.5%；512 次 CPU 回放、0 次仿真失败、约 11 MB。
- 正式 bank 面板**尚未冻结**；需要时用第 6 节命令生成（确定性种子，结果可复现）。

## 5. 预算

单请求 token 用量以已有运行实测为准：minimal 下 forward 约 2.3k 输入/0.3k 输出，goal 约
2.1k 输入/50 输出。推荐档合计约 3,800 请求（B0 192、B1 768、B2 1,440、C ≤1,408）：
minimal 约 13M token；若中等推理每请求多 4k 输出，约 28M token。这是估计，B0 之后用实测替换。

默认上限（CLI 参数，可在批准时改）：

| 命令 | 请求 | 总 token | 单请求输出上限 | 单请求预留 |
| --- | ---: | ---: | ---: | ---: |
| `native-probe`（B0） | 192 | 2,000,000 | 32,768 | 40,960 |
| `bank-study`（B2） | 1,440 | 20,000,000 | 4,096 | 16,384 |
| `feedback-study`（C） | 1,408 | 20,000,000 | 4,096 | 16,384 |

B1 用 `native-probe --design forward_intervention` 并显式给出 768 请求和对应 token 上限。
若 B0 选 medium/high，B1/B2/C 的输出上限和预留要同步调高，否则长推理会被截断成非法输出。
CPU 侧合计不到 3,000 次回放，在已批准的 4 worker / 20 GB 额度内。

## 6. 命令

不加 `--execute` 时只锁定计划、写 manifest，不申请 token、不发请求。加 `--execute` 前必须
已有该实验的明确额度。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser native-probe \
  --panel data/research/model-diagnostic-panel-v1 \
  --azure-config ../SoccerTactics/codes/inference/azure_openai_config.json \
  --output data/research/native-effort-probe-v1
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser prepare-bank-panel \
  --source-panel data/research/model-diagnostic-panel-v1 \
  --output data/research/bank-panel-v1
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser feedback-study \
  --panel data/research/model-evaluation-panel-200-v1 \
  --native-run data/research/azure-gpt5-goal-200-v1 \
  --azure-config ../SoccerTactics/codes/inference/azure_openai_config.json \
  --reasoning-effort minimal --output data/research/feedback-study-v1
```

`bank-study` 同理（`--bank-panel`、`--reasoning-effort`）。`--limit-jobs`/`--limit-episodes`
可先跑少量再用相同参数恢复；输入、代码或配置一变就必须换新目录。

## 7. 代码与验证

- 新模块：[study_calls.py](../../src/inverser/study_calls.py)（新请求合同、可放宽到 32k 输出的
  backend、运行锁与账本辅助）、[native_probe.py](../../src/inverser/native_probe.py)（B0/B1）、
  [bank_study.py](../../src/inverser/bank_study.py)（B2）、
  [feedback_study.py](../../src/inverser/feedback_study.py)（C）；CLI 新增四条命令。
- 写这些代码时，规划 pilot、200 布局 goal 运行和 200 布局面板的 producer 代码还没有提交，
  所以本轮**没有改动这 14 个 producer 文件**（逐个 hash 核对一致）。`StudyAdapter` 因此复制了
  冻结的 `ModelAdapter._complete` 的 HTTP 交换，并用测试保证两者请求形状一致。
- 随后这些代码已连同本页做了本地存档提交（未推送）。之后可以合并两条 HTTP 路径；合并会改变
  上述运行记录的 producer hash，届时需检出该存档提交来重新核验这些运行。
- 211 个测试、Ruff、Mypy（41 个源文件）通过。新测试全部用 mock transport。

## 8. 未完成与限制

- 只验证了 mock：真实服务的延迟、限流、reasoning token 实际用量和拒答都未测。
- 开源模型/本地 GPU 路线未实现；需要 GPU 分配和推理环境许可。
- C 的 shuffled 是本协议的定义（独立随机动作的结果），不是跨案例置换。
- 这些都是开发诊断，不是 G2/G4 正式评价；现有 2,000 条开发记录不能改称未见 test。
