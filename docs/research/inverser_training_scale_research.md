# InverseR 训练集扩容调研：从 12,000 到 100,000 个初始布局

调研日期：2026 年 11 月 20 日。结论针对当前仓库和稿件；表中的拟建数据量不是已完成的实验结果。

## 摘要

InverseR v1 有 12,000 个按坐标指纹区分的训练布局，每个布局一对 anchor/twin，合计 24,000 条击球记录，可构造 48,000 条 SFT 样例。当前 400 步 SFT 只处理了约 12,800 次样例呈现。大规模数学与代码 SFT 的确常报告几十万到数百万条训练记录，但这些记录可能来自少得多的原题；可验证奖励的 RL 也有用 17K、24K 题的成功先例。最接近的规模先例 CoPhy 按原文比例约有 20 万个训练实验，而 Sim2Reason 的 20 万条 SFT 并未改善其 OOD 评测。建议将 100,000 个布局组、200,000 条验证过的击球记录定为训练集扩容目标，在不改动现有验证和测试分区的前提下，先做布局规模曲线和同规模机制覆盖对照。应先修正数据规则与论文表述的不一致，再扩量。

## 1. 问题与比较口径

1. RQ1：代表性数学和代码工作中的“几十万”，数的是独立问题、SFT 题解、RL prompts，还是在线 rollout？是否足以作为 InverseR 的最低规模？
2. RQ2：最接近 InverseR 的反事实物理、桌球和视觉规划工作达到何种规模？哪些结论可以迁移？
3. RQ3：本仓库目前有多少数据、实际训练过多少？如何扩容并区分数据量、物理分布和计算量的贡献？

本调研检索 2019 年至 2026 年 11 月 20 日的原始论文、会议版本和官方数据卡，按数学 SFT/蒸馏、代码 SFT/蒸馏、实际用于 RL 的 prompts、模拟物理与视觉控制四个角度交叉核对。比较表选入 15 项可核验且与决策相关的工作；这不是对所有训练报告的代表性抽样，不估计行业中位数，也不以闭源、单位不明或仅见二手转述的数字填补空白。本地证据是源码、三个 manifest 与组指纹索引、训练标签，以及 SFT 运行回执；没有使用盲测结果或运行额外训练。

## 2. 文献证据：同为十万级，增加的东西不同

**表 A：大规模 SFT 大多按训练行数报告；题库规模和优化阶段必须另列。**

| 工作与阶段 | 原文规模 | 对 InverseR 的含义 |
|---|---:|---|
| 数学 SFT：MetaMath，ICLR 2024 [1] | 395K 训练行 | 从 GSM8K/MATH 训练集做题面改写、答案增强与其他转换；不是 395K 道原始题。 |
| 数学 SFT：Orca-Math [2] | 200K 合成应用题 | 题目层合成的直接例子；后续偏好训练另计。 |
| 数学 SFT：OpenMathInstruct-2，ICLR 2025 [3] | 14M 题解对，约 600K 去重题目 | 同时扩问题与解法；去重题目不代表同等数量的数学机制。 |
| 代码 SFT：OpenCodeReasoning，COLM 2025 [4] | 736,712 条 Python 解答与推理，28,904 道去重题 | 数十万条训练行来自不到三万道题；采用论文 Table 1 口径，不与发布卡的版本数混用。 |
| 代码 SFT：rStar-Coder，NeurIPS 2025 [5] | 数据集 Table 1：418,314 道验证题，其中 380,560 道合成；另报约 580K 长推理解答 | 确实在题目层面扩到几十万，而非只重采样解答；题目数采用数据表口径，不取附录组成表中的不同数字。 |
| 数学 RL：DAPO [6] | 17K prompts，每次每 prompt 采样 16 个回答 | 在线生成量大不意味着题库就必须大。 |
| 代码 RL：DeepCoder [7] | 约 24K 题目及对应测试 | 与 InverseR 相近的量级，但代码测试奖励与桌球奖励不可直接类推。 |
| 数学 RL：DeepMath-103K，ICLR 2026 [8] | 103K 题，95K 新筛出 + 8K SimpleRL | 也有十万级 RL；其三条解法/题是另一计数。 |
| 同模型家族：Olmo 3 Think [9] | 32B SFT 混合语料约 2.25M 行；Think RL 全领域 104,869 个 prompts，表中数学合计 30,186、代码 23,110 | 同一报告内，训练行与 RL prompts 的数量差距很大；全领域合计不能当成单任务题库。 |

与只报 395K 扩增行的 MetaMath [1] 相比，OpenMathInstruct-2 [3] 的消融将训练行控制在约 256K，在 1K 与 6.5K 不同题目之间看到 MATH 验证表现相差约 10 个百分点；代码方面，rStar-Coder [5] 扩合成题后 LiveCodeBench 提升，但其 Table 7 的 MBPP+ 从 `Seed+50K` 的 76.5 降至 `Seed+all` 的 74.0。两者共同支持测试“增加新布局”而非只重复同源记录，也提醒我们不要只报单项进步。相较于 DAPO 与 DeepCoder 的 17K/24K [6,7]，DeepMath-103K [8] 的实际 RL 证明十万级可以做，但不是开展 RL 的门槛。

## 3. 更接近的物理数据先例

**表 B：反事实物理有二十万级训练先例，直接桌球工作却主要报告评测量。**

| 工作 | 规模和单位 | 和 InverseR 的区别 |
|---|---:|---|
| CoPhy，ICLR 2020 [10] | 三类合计 286K 实验 / 143K 底层几何；按 70% 划分约 200K 训练实验（推算） | 每个实验已含原始和反事实两条序列；干预初始位置或物体，而非反求连续击球动作。 |
| CLEVRER，ICLR 2020 [11] | 10K 训练视频；每个视频衍生多种问答 | 物理与反事实推理先例，但不是交互控制，也不能把问答行数当独立视频。 |
| DINO-WM，ICML 2025 [12] | PushT 的 200／1K／5K／10K／18.5K 训练轨迹，对应成功率 8／48／72／88／92% | 与稿件中世界模型基线直接相关；其规模曲线值得效仿，成功率和饱和点不能外推。 |
| BilliardPhys-Bench，2026 预印本 [13] | 5 个时间窗，每窗 200 场景，共 1,000 场景–时间窗评测单元 | 只报告前向评测，未报告可对照的训练规模。 |
| DeepPhy，AAAI 2026 [14] | Pooltool 子项 100 次评测实验／模型，每次最多 15 次尝试 | 9 球评测，预设力度与旋转，和连续击球控制的任务不同；尝试数不是训练题数。 |
| Sim2Reason，ICML 2026 [15] | 200K 条用于 SFT 的模拟题答；另跑在线 RL | 32B 实验中 IPhO 基线 19.8%、SFT 15.9%、RL 25.2%。训练方法与数据形态都不同，不是仅改变规模的实验。 |

CoPhy [10] 支持几十万量级的合成物理数据可行，而且在构建时要求同一设置下有物性变化引发明显不同的轨迹；与之相比，BilliardPhys-Bench 和 DeepPhy [13,14] 不是可直接对比的训练集。DINO-WM 的量级远小于 CoPhy 但有清晰的规模曲线 [10,12]；Sim2Reason 则说明即便能生成几十万条，训练阶段和分布不对也未必带来 OOD 收益 [15]。未找到与 InverseR 同时匹配 BEV 视觉、连续动作、成对干预和在线隐藏物性适应的规模实验，因此不能借外部工作推断本项目的最优数据量。

## 4. InverseR v1 本地审计

| 分区 | 初始布局组 | 标准击球记录 | OOD T5 |
|---|---:|---:|---:|
| train | 12,000 | 24,000（12,000 anchor + 12,000 twin） | 0 |
| valid | 120 | 240 | 0 |
| test | 420（300 标准 + 120 OOD） | 600 | 120 |

- **Table 1 的 `N=240` 是验证集评测量，不是训练量。** 稿件 [Table 1 caption](../../icml2027/main.tex#L234) 和 [Datasets 段](../../icml2027/main.tex#L283) 对应不同数字。按 `groups.json` 核对，三个分区的布局指纹两两交集均为零。按 manifest 中的 `-init.png` 哈希，三分区也没有跨分区像素级完全相同的初始图；但 train 的 12,000 个布局指纹只对应 11,998 个不同初始图哈希。扩大后不能只依赖精确坐标哈希。
- **现有 SFT 主要是短程热身。** [SFT 构造器](../../src/inverser/capo/capo_trainer.py#L132) 每个标准击球记录形成一条联合 T1–T3 forward 和一条 T4 inverse，T5 不在这 48,000 行里。400 步、有效 batch 32、约 0.267 epoch，相当于 12,800 次呈现，不等于 12,800 个不同布局。在线 RL 应另报实际见过的布局、有效 rollout groups 与模拟器调用数。
- **族数均衡不代表机制均衡。** train 四类各 3,000 个布局，但多球 anchor 中 1,229/9,000（13.7%）的 `first_contact` 为 `null`，分别是 center 247、vertical 382、side 600。这与[附录](../../icml2027/main.tex#L364)称所有多球 anchor 都通过有效母球–目标球接触筛选不一致。[构建器](../../src/inverser/benchmark/vlm_suite.py#L716)实际检查仿真成功、无 scratch 和 twin 差异至少 2r，没有显式检查 anchor 首碰。训练标签中只有 376/24,000（1.57%）条记录涉及至少两个标记为碰撞的目标球。不要事后清理冻结分区；应如实写 v1 规则，并在新增训练数据中设显式的有接触和无接触层。
- **正文另有一个可核对的引文问题。** BilliardPhys-Bench §2.3 的位置判定阈值为 **1r** [13]，[当前稿](../../icml2027/main.tex#L286)却称 3r 与之对齐。可将 3r 写成本工作自己的宽松诊断；即便同用 1r，两工作的时间窗和聚合方式也不等价。

## 5. 建议：先扩独立布局，再检验训练收益

### 5.1 构建目标与可行性

以 **100,000 个布局组 / 200,000 条击球记录**为目标，保留每组一个 anchor 和一个只改一个动作维度的 twin。沿用目前转换口径，最多得到 400,000 条 SFT 行和约 500,000 张 PNG；这既不是 400,000 个独立布局，也不是 T5 的已执行轨迹。300,000 条击球记录相当于 150,000 个布局组，可留作第二阶段，不必先承诺。

目前 `prepare-vlm-suite` 不能直接完成：CLI 没有训练规模或 train-only 开关；构建器硬限每族 3,000 对，单循环最多 300,000 个候选索引，并拒绝写入已有版本。同格式现有训练文件逻辑字节数约 1.17 GB；线性估算 200K/300K 条记录约 9.8/14.7 GB，不含文件系统块、临时构建、缓存及 checkpoint。200K 条记录按现有核验至少需要 400K 次模拟执行来生成并复演所接受的见证，候选拒绝与额外 twin 尝试另计。现阶段没有生产吞吐数据，不承诺工期；先用 2K 个新布局测各层接受率、CPU 时间、PNG I/O 与哈希时间。现工作区也没有默认旧 pilot 路径下的 `privileged.jsonl`，正式构建前应恢复可复验的旧排除集或其等价受控指纹来源，并核对来源摘要。

建议新建训练专用版本，按 seed 和索引区间确定性分片，保留 v1 train 的布局和既有验证／测试的 manifest 不变；新增布局显式排除 v1 train、valid、test 与 legacy 布局，合并时再做跨分片精确哈希、渲染哈希、近邻检查。对于由平移、镜像、微扰或 frontier 衍生的布局，保留父布局或生成模板来源，不能把同源变体当独立物理机制。近重复的等价规则需考虑球身份、桌边和摩擦条件，不能无条件把镜像等价化。

扩量优先补充多球连锁、切球角度与击球距离、薄碰/无碰对照、多库反弹、跟杆/拉杆/侧旋和真实改变碰撞或库边序列的 twin；保持一部分 endpoint-only twin 作为受控对照。若做 T5 摩擦扩展，单开训练专用物性范围并和冻结 OOD 条件隔离；[当前 RL episode](../../src/inverser/capo/capo_grpo.py#L403)复用名义 `PhysicsConfig()`，只有数据文件而不增加每回合私有物性配置与可见性检查，不能训练真实的隐藏物性适应。现有 twin 的可达验证可复用；没有理由放宽 2r 干预或无 scratch 门槛来凑数。

### 5.2 规模曲线与对照

| 独立 train 布局组 | 击球记录 | 可构造的 SFT 行 |
|---:|---:|---:|
| 12K：v1 | 24K | 48K |
| 25K：第一梯 | 50K | 100K |
| 50K：第二梯 | 100K | 200K |
| 100K：目标 | 200K | 400K |

1. **先做等预算曲线。** 使用分层、嵌套、可复现的布局组子集，不让 anchor/twin 被拆散，不用 loader 的顺序截断代替分层抽样。固定 base、SFT/LoRA 配置、输入处理和 token/optimizer 预算，固定 RL 的组大小、尝试预算与实际 simulator calls；记录见过的唯一布局、每类覆盖、reward 方差和有效组率。追加与 v1 同分布的布局先估计量的作用；再在相同大小下比较覆盖强化版，避免把分布变化全归于样本量。
2. **再做同等数据行数下的新布局对照。** 在相同曝光及物理族配比下，比较“新增初始布局”与“固定 v1 布局，多采经过验证的击球或重复原样例”。两者分别回答布局多样性和同布局内动作覆盖。若等预算只能见到小比例数据，再为最有希望的臂增加预算，按实际 tokens、GPU 小时、unique groups 和 sim calls 记账。不能把固定 400 步的结果写成完整使用了 400K 行：按 batch 32，一整轮要约 12,500 步；按现运行吞吐只是约 48 小时的 SFT 粗算，不是 RL 成本。
3. **选规模只看开发数据。** 预先规定 T4 `Succ@1r` 与 T5 标准/开发物性适配为控制指标，配合 forward `Joint`、反事实分辨能力、按球数和机制分层、保留能力评测；区间以布局组做 paired bootstrap。有需要可另建独立的训练外难例开发集，不能用冻结 test 调配额和选择 checkpoint。Table 1 目前 SFT 的 valid T4 是 0.0%、同稿 oracle 在其有限搜索预算下是 14.6%，不能单把低分归为数据不足；先确认奖励是否给出可学习的差异、当前搜索预算是否形成瓶颈。
4. **主表不混淆算法效应和数据效应。** Table 1 保留统一评测和匹配预算；若 CAPO 用了扩展数据或更长 SFT，至少为支持算法主张的 Outcome／Uncoupled／CAPO 控制臂匹配初始化、训练池和预算。否则把 data scaling 单列为消融，明确指明各结果使用的 manifest 和实际训练规模。未训练的扩展语料只能称为候选／已发布语料（视完成状态），不能写成主表模型已使用。

## 6. 结论与未决问题

- **RQ1：** 猜测对大型合成 SFT 语料成立；对独立问题数和 RL 题数不普遍成立。开放的数学／代码资料适合提供计数规范，而非给桌球规定最低门槛。
- **RQ2：** CoPhy 是最贴近的二十万级可行性先例；Sim2Reason 和 DINO-WM 分别说明质量／训练范式以及规模曲线值得测，不能据此宣布 InverseR 会受益。
- **RQ3：** v1 为 12K 布局组、24K 击球、48K 可用 SFT 行，至少有匹配现有明细的可扩容方案。先审计并处理方法差异，跑 2K 新布局试生成，再按 12K→25K→50K→100K 布局推进。真正未决的是稀有物理机制的接受率、rollout 预算下的实际覆盖、以及新增布局能否改善控制和 OOD。扩容只在这些指标优于较小、同预算方案时成为主结论。

## 参考文献

[1] Yu, L., Jiang, W., et al., “MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models,” ICLR, 2024.

[2] Mitra, A., Khanpour, H., et al., “Orca-Math: Unlocking the Potential of SLMs in Grade School Math,” arXiv:2402.14830, 2024.

[3] Toshniwal, S., Du, W., et al., “OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data,” ICLR, 2025.

[4] Ahmad, W., Narenthiran, S., et al., “OpenCodeReasoning: Advancing Data Distillation for Competitive Coding,” COLM, 2025.

[5] Liu, Y., Zhang, L. L., et al., “rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset,” NeurIPS, 2025.

[6] Yu, Q., Zhang, Z., et al., “DAPO: An Open-Source LLM Reinforcement Learning System at Scale,” arXiv:2503.14476, 2025.

[7] Agentica & Together AI, “DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level,” official dataset card `agentica-org/DeepCoder-Preview-Dataset`, 2025.

[8] He, Z., Liang, T., et al., “DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning,” ICLR, 2026.

[9] Olmo Team, “Olmo 3,” arXiv:2512.13961, 2025.

[10] Baradel, F., Neverova, N., et al., “CoPhy: Counterfactual Learning of Physical Dynamics,” ICLR, 2020.

[11] Yi, K., Gan, C., et al., “CLEVRER: Collision Events for Video Representation and Reasoning,” ICLR, 2020.

[12] Zhou, G., Pan, H., et al., “DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning,” ICML, 2025.

[13] Wang, B., Li, X., et al., “BilliardPhys-Bench: Benchmarking Physical Reasoning and Visual Dynamics of Multimodal LLMs,” arXiv:2605.30900, 2026.

[14] Xu, X., Bu, P., et al., “DeepPhy: Benchmarking Agentic VLMs on Physical Reasoning,” AAAI, 2026.

[15] Prabhudesai, M., Satpathy, A., et al., “Sim2Reason: Solving Physics Olympiad via Reinforcement Learning on Physics Simulators,” ICML, 2026.
