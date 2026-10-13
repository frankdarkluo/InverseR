# 二维台球中的物理推理：从观测可信度到预测与规划的一致性

研究工作报告，2026-10-13。本文区分已经完成的数据工程、已核查的文献结论和待批准的实验方案；不包含尚未运行的模型结果。

## 摘要

给定台球的初始状态与目标终态，输出可执行的击球策略，需要视觉状态估计、动力学预测和连续动作规划共同成立。本报告通过三个独立检索视角，核查学习式世界模型、物理推理评测及逆向规划与系统辨识的代表工作，并结合 InverseR 的实际数据审计，研究这些能力应如何分别测量。文献支持将可查询的动作条件动力学、外部物理引擎和语言模型的工具控制区分开来；端点吻合、视频合理或工具辅助成功，不能单独判定模型具有原生物理预测能力。现有真实视频可以支持观测动力学学习与空间路径验证，但没有实测击球控制，不能直接提供精确动作或旋转监督。建议用真实数据约束观测分布，用明确标注的仿真提供动作干预与可执行评测，再通过预算匹配的实验研究预测误差、接触事件和规划效用的关系。百万任务是后续构建目标，不是目前已经完成的数据规模。

## 1. 问题：能描述运动，是否就能选对动作？

本项目的主任务保持不变：给定初始台面状态和期望终态，寻找任意满足目标与约束的单杆动作。它不要求复现视频中球员的历史动作。历史轨迹重建是另一项诊断任务，不能把“与历史路线不同”直接当成目标规划失败。

台球适合受控研究，因为空间几何容易表达，动作可以连续参数化，碰撞、库边反弹和摩擦可以执行检验。但这个便利不等于任务已经解决：目标可能对应多个动作，相机可能漏球，旋转和材料参数可能不可见，微小误差可能改变接触事件。

本次调查回答三个问题：

- **Survey-RQ1：** 如何区分学习式动力学、外部模拟器与 LLM 工具控制，测量物理预测和干预一致性，而不只是输出合理性？
- **Survey-RQ2：** 已完成的作者数据与比赛视频实际支持哪些标签、任务和干预？如何扩大数据而不虚增独立样本或制造伪 GT？
- **Survey-RQ3：** 哪些预算匹配的实验能够区分感知、物理辨识、正向建模和逆向规划的瓶颈，并提供可推广的 insight？

“二维”在这里指球心的平面运动，不是完整状态只有二维位置。球在运动时，速度、滚动或滑动状态、三维角速度和接触参数仍会影响未来。合成初始状态可以声明所有球完全静止；真实图像中球心没有明显位移，并不能单独验证所有潜在旋转都已消失。

## 2. 证据范围与调查方法

### 2.1 检索与纳入标准

检索截止日为 2026-10-13。三个独立视角分别调查：

1. 学习式、动作条件世界模型：`action-conditioned world model`、`latent dynamics planning`、`video model action conditioning`。
2. 物理理解的测量与反证：`physical reasoning benchmark`、`billiards collision prediction`、`video physics generalization`、`executable reconstruction`。
3. 最近邻方法与可辨识性：`billiards inverse planning spin`、`LLM simulator feedback`、`physical parameter identification`、`Bayesian simulator inference`。

纳入标准是可核查的主要论文、正式出版记录或作者公开实现，且能回答上述 RQ。最终核心方法与评测文献为 15 篇，另纳入作者数据集论文，共 16 篇。每条研究线路至少覆盖三篇代表工作。检索不是覆盖所有世界模型或机器人控制文献的系统性穷举；没有运行或独立复现这些论文。

核查关注输入、动作来源、动力学由谁提供、评价条件和结论边界。代码检查用于确认接口及计算预算，不把实现中的默认值当成论文实验配置。正式出版与预印本题名不同的工作按版本区分。PhysBench 的 ICLR 2025 记录已通过官方会议页面及 OpenReview 核实。

### 2.2 本地证据与文献证据分开

作者论文报告了低分辨率、低帧率的固定相机采集与轨迹重建方法 [1]。本地的文件数、哈希、转换帧数和目录引用来自实际审计，不是从论文推算。证据入口是：

- [完成上传的源文件审计](../../data/quality/author-completed-upload-snapshot-v1/source_independence_audit.json)。
- [100 个 snooker 录制的转换验证](../../data/structured/author-snooker-incremental-v1/snapshots/551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e/final_upload_validation.json)。
- [统一目录的统计](../../data/research/physics-world-model-catalog-v1/summary.json)与[引用验证](../../data/research/physics-world-model-catalog-v1/validation.json)。
- [现有数据卡](../data/data_card.md)与[统一目录使用说明](../data/research_corpus_catalog.md)。

本报告中的模型机制来自已核查文献；下文的假设、预算档位与扩建规模是实验建议。现阶段尚未运行待测模型的推理、训练或 API 实验；研究辅助与抽帧复核不作为模型性能实验。

## 3. 能力分解：世界表示、动力学与动作控制不是同一件事

为了研究控制用途的世界模型，采用动作可查询的预测接口：

$$
p_\theta(z_{\mathrm{future}}\mid z_{\mathrm{history}}, A_{\mathrm{hypothetical}}).
$$

这是一项操作性定义，不是否认广义的、无动作条件的视频预测也可以称为世界模型。它要求本研究的模型能够对指定的假设动作给出可检验的未来预测；不要求生成像素视频，状态或 latent 预测也可以。

| 模块 | 问题 | 典型输出 | 不能由它单独证明的能力 |
| --- | --- | --- | --- |
| 观测与 belief 估计 | 当前有哪些球？哪些状态或参数不确定？ | 坐标、观测掩码、潜在状态假设 | 指定动作之后会发生什么 |
| 学习式动力学 | 数据学到的转移如何响应动作？ | 状态或 latent 转移、事件、分布 | 选择的动作一定能达到目标 |
| 参数化可执行物理模型 | 给定或拟合参数后如何演化？ | 引擎 rollout | LLM 自身学到了同样的动力学 |
| 规划与工具控制 | 怎样选动作与分配查询？ | 候选动作、数值优化、验证决策 | 不使用工具时的原生预测能力 |
| 独立评价 | 动作实际在参考环境中产生什么？ | 目标成功、约束违例、误差与成本 | 未执行的真实世界目标成功 |

分类按模块而不是按论文互斥分组。一个系统可以同时有训练过的视觉编码器、拟合的解析动力学和 LLM 控制器。参数从数据拟合的物理引擎可以是数据驱动的参数化世界模型，但与学习状态转移的神经模型不同；两者都应与固定参考引擎区分。

还要区分价值代理与转移模型。只预测一个 shot 的质量，能够帮助搜索，但并不单独展示可滚动的状态或 latent 动力学。反过来，预测图像或坐标准确，也不自动保证对候选动作的排序正确。

## 4. 学习式世界模型：动作信息与规划接口决定了可迁移范围

Ha 和 Schmidhuber 的世界模型与 DreamerV3 都通过动作相关经验学习动力学，但前者研究在学习环境中演化策略，后者使用想象轨迹训练 actor-critic [2, 3]。它们支持“预测模型可以服务控制”，同时也提示必须检查策略是否利用了模型误差，而不只是检查平均预测质量。

DreamerV3 与 TD-MPC2 的区别不在于谁更像视频生成器，而在于如何使用 latent 动力学：前者训练想象中的策略，后者进行在线模型预测控制 [3, 4]。V-JEPA 2 则把大规模无动作视频预训练与机器人动作条件扩展分开；其控制实验使用机器人状态与动作数据，不能等同于从没有控制标签的台球视频直接得到击球接口 [4, 5]。

**比较表：控制所需的动作条件不能由视频预训练名称替代。**

| 工作 | 动力学表示 | 控制或规划方式 | 动作与训练数据的条件 | 对本项目的意义 |
| --- | --- | --- | --- | --- |
| Recurrent world models [2] | 学习式视觉 latent 与 recurrent 转移 | 在想象环境中进行策略演化 | 带动作的环境 rollout | 检验模型利用与外部参考评价 |
| DreamerV3 [3] | 学习式 latent world model | 想象轨迹中的 actor-critic | 环境状态观察、动作、奖励等经验 | 不是即插即用的 cue planner |
| TD-MPC2 [4] | 无像素 decoder 的 latent 动力学 | 在线 MPC | 连续控制的转移与任务数据 | 小型状态模型可比大视频生成器更适合本任务 |
| V-JEPA 2 [5] | 视频预测表示；另有动作条件扩展 | 动作条件模型辅助规划 | 基础视频无动作；机器人扩展有实测状态与动作 | 需要建立台球控制与其动力学表示的连接 |

对于单杆任务，击球后通常不能继续改变运动中的球。TD-MPC2 与 V-JEPA 2 的连续交互设置提供模型规划的参考，却不能直接支持“球滚动时持续闭环纠正”的台球叙事 [4, 5]。本项目中的多轮反馈主要是执行前的计算式搜索；真实物理执行仍是一杆开放环动作。

## 5. 物理测量：答案、轨迹与可执行假设各自解决不同问题

PhysBench 通过物理问答覆盖视觉物理知识，而 PHYRE 让求解器在二维环境中尝试动作并按预算计分 [6, 7]。前者的答案正确性不等于连续控制能力；后者的动作成功也不必然揭示模型内部的物理表示。两者需要互补的归因实验，而不能用一个综合 accuracy 替代。

BilliardPhys-Bench 测试球的位置与碰撞预测，VisPhyWorld 测试从观测帧构建可执行的重建假设 [8, 9]。前者的固定时间预测不是“最终静止状态”预测，后者看到后续帧之后的重建不是盲测未来。两项工作都提供可用测量思路，但不能混成一个正向推理分数。

VisPhyWorld 的可执行结果与物理规律视角的生成模型研究，进一步说明评价条件会改变结论：有目标观测的重建与没有未来信息的 OOD 预测是不同任务 [9, 10]。后者报告的分布内与组合泛化能力也不应忽略。因此不能从特定 OOD 失败推导“生成模型完全没有物理能力”。

**比较表：不同指标承诺不同证据；规模与成功率不能直接横向排行。**

| 工作 | 输入与任务 | 参考证据 | 能支持的结论 | 本项目需要补测的维度 |
| --- | --- | --- | --- | --- |
| PhysBench [6] | 图像或视频的物理问答；10,002 条条目 | 问答标签 | 被测条件下的答案能力 | 动作、执行误差、工具归因 |
| PHYRE [7] | 在二维场景放置球等操作达到接触目标 | 可执行模拟器与预算评测 | 整体求解器的预算内成功 | cue 速度、角度、旋转与非唯一解 |
| BilliardPhys-Bench [8] | 简化合成台球的位置与碰撞预测 | 明确的合成物理条件 | 规定时刻的状态和事件预测 | 静止终态、spin、逆向动作与未知参数 |
| VisPhyWorld [9] | 首帧及后续帧到可执行重建；209 视频、108 模板 | 合成观测与执行比较 | 端点或观测条件下的重建兼容性 | 无未来泄漏的预测与动作干预 |
| 物理规律视角的生成研究 [10] | 受控物理视频的 ID、OOD 与组合测试 | Box2D 等受控参考 | 被测生成器的物理泛化条件 | 相同误差是否导致规划失败 |

因此，除了端点误差，评价还应包含接触事件、干预响应和动作决策的外部参考效用。这里的“干预”应来自我们可控制动作或参数的仿真，不是给真实视频改写一句提示后就声称获得了真实反事实 GT。

## 6. 逆向规划与物理辨识：最近邻已覆盖击球和 spin

CueTip 与 SimLM 已经让语言模型参与有物理引擎的台球动作搜索。CueTip 提出事件候选再做连续优化，SimLM 利用模拟器反馈修订动作；两者都包含五个击球控制，不能把 spin 或 LLM 加模拟器本身当成新贡献 [11, 12]。

与 CueTip 的模拟评价不同，Mathavan 等人的机器人台球工作还进行了物理击球与球位控制实验，但披露了手动定位或定向辅助 [11, 13]。它证明真实逆向控制不是未被探索的方向，也说明真实执行与视频上的拟合需要明确区分。

LLMPhy 与 PhysMind 都从观测拟合可执行物理假设，而不是从头训练神经状态转移 [14, 15]。BayesSim 进一步研究参数后验，但它依赖状态与动作经验，其论文中部分称为真实观察的测试由指定参数模拟器生成；因此不能用它替代真实 cue 控制标签 [14, 16]。

**比较表：外部动力学、参数拟合与真实执行应独立标注。**

| 工作 | 物理由谁提供 | LLM 或学习模块的角色 | 验证层级 | 必须对照或避免的重复 |
| --- | --- | --- | --- | --- |
| CueTip [11] | Pooltool 与连续动作优化 | 事件提案、解释及 shot-value 代理 | 模拟比赛；解释用户研究 | 事件搜索、五控制及价值代理不是新设定 |
| SimLM [12] | 固定模拟器 | 基于反馈进行参数化动作修订 | 合成环境与模拟台球 | LLM simulator-in-the-loop 是必要 baseline |
| Mathavan 等 [13] | 力学模型与拟合模块 | 遗传搜索与机器人控制 | 实际击球，含人工辅助 | 不能声称首次真实逆向球位控制 |
| LLMPhy [14] | MuJoCo/PyBullet 等引擎 | 从给定干预与观察拟合参数、布局 | 合成任务及真实 tray-pushing 观察 | 系统辨识必须测试未用于拟合的干预 |
| PhysMind [15] | 拟合的解析运动与碰撞更新 | 感知、重建和可执行推理 | 视频物理问题 | 可执行重建不等于新学到的神经动力学 |
| BayesSim [16] | 参数化模拟器 | 根据经验推断参数分布 | 论文中的模拟控制与参数测试 | 候选集合不自动是校准后验 |

论文中的优化迭代数不等于完整的函数评价次数。例如群体优化的一步可能调用很多次引擎，LLM 的一次纠错也可能包含多轮模型请求。不能用相同“迭代数”建立公平性，更不能直接比较这些不同任务的 potting 或 QA 百分比。

## 7. 跨线路结论：文章要研究能力链，而不是包装一个工具

三条文献线路收敛到一个测量问题：正向预测、可执行重建和动作搜索都有有效结果，但其成功可能来自不同模块。原生预测和外部模拟器控制可以各自有价值；只有把输入、工具与预算固定，才能判断提升来自哪里。

本项目可以聚焦三个具体缺口，而不是“现有方法都不懂物理”：

1. **预测到决策的缺口。** 小的平均位置误差是否仍会改变接触拓扑、候选排序或终态成功？预测质量与规划效用需要在同一物理案例上连接。
2. **工具归因的缺口。** 在相同候选和相同 rollout 证据下，LLM 是否超过几何或简单选择器？在自由提案下，事件组织是否超过几何初始化的数值搜索？
3. **可见信息与泛化的缺口。** 未知材料参数、不可见 spin 和视觉误差各造成多少损失？真实视频可以支持哪部分辨识，又有哪些不确定性不能靠语言消除？

这些是待检验的研究问题，不是已经得到的新发现。World Models 等工作已有模型利用问题，CueTip 已有事件优化，本项目不能把这两个概念重新命名就当成方法创新。

若实验证明 LLM 在 oracle state 与匹配预算下并无增益，应保留结果；若增益只在视觉端出现，应归因给感知；若小模型加数值优化更稳定，也应作为实际结论。方法设计应由这种诊断决定。

## 8. 数据构建：按标签能力分层，而不是按文件量合并

### 8.1 现在实际拥有什么

| 来源 | 当前实际数量 | 可用证据 | 现阶段不能称为 |
| --- | ---: | --- | --- |
| 作者 snooker | 100 个内容哈希去重后的原始录制；37,876 个 raw frame；16 个人工空间路径参考 | 固定相机原始观察、自动轨迹、来源人工路径图 | 116 个独立击球；精确动作或逐帧物理 GT |
| 比赛视频 | 32 视频、228.38 小时；7,114 个 raw 候选 | 多场景观察、弱端点与轨迹、质量状态 | 7,114 个已验收的真实单杆任务 |
| 作者辅助领域 | blackball 100、carom 100 个原始录制；人工参考分别为 22、16 | 跨球类观察与空间路径参考 | 主 snooker 数量；200 个已全部转换的训练样本 |
| 仿真开发 pilot | 2,000 个 shot | 已知 cue 控制、参数、事件与轨迹；witness 重放通过 | 真实动作标签；正式冻结测试集 |

作者目录共有 354 个原始视频文件副本，对应 300 个内容去重的录制。54 个 Manual 副本与原始录制关联，不增加物理事件数。哈希证明文件内容不同或相同，不证明所有录制在统计上独立；同一桌、相机或采集 session 仍应分组。

统一目录已有 9,414 个引用：7,214 个主域真实观察、200 个辅助来源、2,000 个仿真 shot。所有引用和来源哈希检查通过。**已独立验收的真实物理 benchmark 任务仍为零；实测真实 cue 动作仍为零。**

当前自动筛选保留 16 个优先复核候选、6,004 个待重处理、887 个不适合当前保守协议、207 个疑似重复。另一次抽帧复核因漏黑球单独暂缓一个优先项，因此留下 15 个未验收的优先项和 370 条定向修复队列。疑似重复不是已经证明的重复，也不能把剩余数量直接称为独立击球。

### 8.2 四层数据产品

| 层 | 构建方式 | 主要用途 | 标签边界 |
| --- | --- | --- | --- |
| Real qualified observations | 校准与完整性检查后的作者录制及比赛子集 | 盲测空间路径、状态估计、观测迁移 | 保留原始人工空间标签；新自动标签为弱标签 |
| Real weak observations | 稳定视角、可观测运动区间的重处理 | 感知预训练、动作后观测预测、候选发现 | 缺失与遮挡不是落袋；不假定完整全局状态 |
| Controlled simulation | 前向生成状态、动作、参数和事件，再隐藏 witness | 动作条件学习、反事实、目标动作评测 | 是声明模型下的 simulator truth |
| Real-layout-conditioned simulation | 从可信真实布局或布局分布出发进行仿真 | 真实几何上的受控挑战 | 布局来源真实，动作和未来仍是模拟的 |

不能进入完整单杆逆向任务，不等于完全没有用途。一段缺少可信终态的视频，仍可能包含可用的短时运动预测区间。但必须针对新用途重新筛选，不能把全部 6,004 个待修复候选自动升级成合格训练集。

### 8.3 真实视频可学动力学，但不能补造 cue 标签

击球后的运动通常是无新增控制的演化。因此，真实视频可以用于学习

$$
p_\theta(z_{\mathrm{future}}\mid I_{\mathrm{post\text{-}impact\ prefix}}),
$$

例如减速、反弹、接触前后的观测变化与遮挡恢复。它不是完全无用的“没有动作数据”。

完整的击球模型可以按模块连接：由 cue 控制映射到击球后的速度与角速度，再由动力学预测后续运动。仿真可提供前一模块的明确动作监督，真实运动前缀可约束后一模块的观测预测。真实的隐含 spin 和未测 cue 速度必须保持为 latent 或缺失标签，不进行硬填零，不把拟合出的一个动作冒充球员实际动作。

原始 X3D 与源算法轨迹也应保留独立角色：它们的动画时钟、scene units 与原始相机帧时钟尚未逐一对齐。已经复现的 16 个 snooker 人工 raster 上的 160 次源算法比较，是作者算法的评分核对，不是我们的检测器或世界模型成绩。

### 8.4 优先修质量，再增加 raw 覆盖

建议先修 100 个固定相机 snooker 录制和 370 个强运动候选，而不是重新遍历全部比赛：

1. 对固定相机复用背景与相机几何；在原始图像中检测球，再投影球心。针对黑球、密集红球、杆影、运动模糊与遮挡分别记录错误。
2. 恢复并独立核查原始视频到人工 raster 的坐标映射。相同 canvas 或 homography 的正反变换一致，不等于物理映射正确。
3. 用 native fps 重处理强运动区间。现有 5 fps 适合粗扫描，却不足以可靠解析快速球和短接触链；重处理保留原始时间戳，不对比赛视频强制套用作者的 20 fps。
4. 再用两阶段发现扩大覆盖：低频定位稳定台面与运动区间，原始帧率精处理。连续观察片段、重播和相邻窗口归并到同一来源事件。

作者报告 640×480、20 fps、焦距 525 px、相机高度 1.853 m、实验台 2.115×1.205×0.78 m，以及 baize-plus-cushion 区域 1.854×0.935 m；snooker 球直径为 51 mm [1]。这些是来源元数据，不是我们的独立实测标定。不能用标准比赛桌尺寸覆盖作者小桌，也不能把包含库边的区域直接作为精确 playing bounds。

球心高度与镜头参数仍可能产生系统误差。实际终态容差必须大于经独立检查的观测误差；不能因为模拟重放误差为零，就宣布真实毫米级控制成立。

### 8.5 如何达到大规模

百万量级的可执行动作监督主要来自明确标注的仿真，而不是把现有真实录制扩写成百万“真实击球”。建议按开发阶段评估 10k、100k，再决定是否生成 1M；这些数字是候选档位，尚未执行。

生成时先采样合法布局、动作与声明的物理 profile，再执行前向模拟。保留失败、擦边、落袋与不同事件类型，用任务约束决定哪些案例能进入 goal-synthesis 子集。不能把正向学习集只筛成漂亮的成功动作。对逆向任务可以使用已知合法 witness 构建可达目标，但要公开其生成与筛选分布。

一个 shot 可以导出正向预测、逆向规划、反事实比较与事件解释视图；这些视图不增加独立物理 shot 数。干预对、相同布局的多个目标和视觉增强也需要作为同一个统计 family 分组。

规模统计至少同时报告独特原始录制、布局 family、物理 profile、shot 数及任务视图数。若承诺百万独立主任务，就不能用 100 个真实布局的近邻扰动、帧数或工具调用数凑足。

为了控制 1 TB 预算，默认保存结构化状态、稀疏物理采样、事件和渲染 seed；按需渲染与缓存，避免为百万条任务存储完整视频。时间采样应由物理演化产生，不能用跨碰撞的简单线性插值伪造中间状态。

## 9. 实证设计：三个可证伪的论文 RQ

以下是拟议实验，沿用原计划的证据与归因、结构化搜索、真实迁移三条主线。它们不是完成后的 Findings。

### 9.1 Paper-RQ1：原生预测能否响应动作干预？

**假设：** 在不查询参考引擎时，模型能响应真正改变物理的动作或参数，同时对保持物理不变的视觉变化保持一致。

使用相同布局的动作干预对，分别改变角度、cue 速度或 tip offset；参数干预改变声明的摩擦或恢复系数。改变一个 cue 控制可能同时影响平动与转动，因此不能把 tip-offset 干预直接宣称为“只改变角速度”。加入纹理、相机及正确变换动作与 spin 的几何对称控制；对称关系先由参考物理检查。

比较 oracle state、带噪坐标、BEV 与图像输入；每一条件记录实际给了哪些速度、spin 或参数。无工具条件先提交未来状态与事件预测，之后才隐藏执行评价。加入动作抹除或打乱对照，检查模型是否实际上忽略动作。

指标包括时刻对齐的位置误差、接触与落袋事件 F1、事件顺序及干预响应误差。对相似端点但事件不同的动作单独统计。运动球与静止球分别计分，避免静止多数掩盖动力学错误。

**反证条件：** 正确率主要来自固定布局、外观或目标泄漏；动作被打乱后基本不变；在相同端点条件下不能分辨实际不同的事件。这反驳本任务下的操作性能力假设，不是“模型完全不懂物理”的证据。

### 9.2 Paper-RQ2：预测准确是否转化为规划效用？LLM 的增益在哪里？

先用固定候选 bank 分离“提案”和“选择”。所有方法看相同初始状态、目标及候选动作；正预算条件获得相同预先指定的 rollout 证据。候选顺序随机化。再做自由提案搜索，检验模型是否更会探索，而不是只看谁得到更多反馈。

必要 baseline 如下：

| 方法族 | 作用 | 公平比较要求 |
| --- | --- | --- |
| Random 与几何初始化 + 数值优化 | 不依赖大模型的强动作搜索 | 相同 bounds、损失与真实 forward-evaluation 预算 |
| 直接 VLM | 无外部物理反馈的原生输出 | 不暗中运行几何修复或引擎校正 |
| SimLM-style | 数值动作与 simulator feedback | 计入第一轮、重试和全部模型请求 |
| CueTip-style | 事件候选与连续优化 | 不能用弱版随机搜索代替最近邻方法 |
| 小型 set/GNN 动力学 + 相同 planner | 测试学习式转移的价值 | 共享候选或 planner，并计入训练成本 |
| 小型提案/价值代理 | 分离预测动力学与直接决策学习 | 不能把价值模型的收益都归给世界模型 |

对固定合法候选集 \(\mathcal A\)，定义：

$$
\operatorname{Regret}
=L_{\mathrm{ref}}(\hat A)-\min_{A\in\mathcal A}L_{\mathrm{ref}}(A).
$$

它是候选 bank 内的 regret，不是对全动作空间最优解的证明。自由搜索报告相对于声明的预算内参考搜索的 gap，不能称其为全局最优 regret。

检验位置误差、事件误差、候选排序与 goal success 的关系，并按布局、事件复杂度和参数条件分层。相关性不是因果机制；应配合动作条件抹除、感知替换和统一 planner 消融。

再增加搜索预算，比较选中动作与同一支持范围内随机动作的模型误差：

$$
\operatorname{OptimismGap}
=L_{\mathrm{ref}}(\hat A)-L_M(\hat A).
$$

若优化越充分，预测损失越低而参考损失不降甚至上升，说明在被测条件下可能存在优化诱导的模型利用。该解释需要排除候选支持范围变化等混杂。

主 goal-synthesis 指标接受任何满足目标的动作，不惩罚与 witness 的参数或路径不同。历史事件相似度仅用于单独的重建任务。候选 bank 中“无可行候选”可以由穷举这个有限 bank 确认；这仍不意味着全动作空间不可达。

**反证条件：** LLM 收益在相同候选与反馈下消失，或几何数值 baseline 在同成本下已达到相同效果。届时文章应报告归因结果，不强行包装新 agent 方法。

### 9.3 Paper-RQ3：不确定物理如何适应，真实观察到底改善哪一环？

从少量参数开始，例如滚动摩擦与库边恢复系数；不要一开始同时辨识全部 spin、相机和材料参数。对合成的未知环境提供相同支持集，支持动作有记录，支持布局与查询布局分离。检查支持轨迹是否真正激发待识别参数。

比较固定 nominal 参数、点估计、声明 prior/likelihood 的分布式辨识，以及直接学习式预测。将真实观察加入感知或动作后动力学学习的实验，与仅用仿真的对应方法配对。测量 held-out 动作、目标和物理组合的效用，而不是只测对支持轨迹的拟合。

未知物理条件必须明确：

- 参数已知、允许查询真实引擎：已知世界的规划。
- 参数未知、只有固定支持经验、查询阶段不开放真实引擎：辨识或适应。
- 参数未知、但目标阶段允许访问实际隐藏参数引擎：更强的交互或 oracle 条件，单独报告。

对真实支持视频，没有测得 cue 动作时，应把动作或击球后状态作为 nuisance；不能套用“已知动作参数辨识”的标签。能做的是基于运动前缀的观测预测、合理模型集合及其 held-out 兼容性。参数误差只在可辨识且有 GT 的合成条件计分。

**反证条件：** 改善只出现在拟合的同一支持轨迹；换目标或布局后消失；扩大假设集合即可取得路径覆盖但预测不够 sharp；“真实数据收益”由重复来源或未来轨迹进入输入解释。

### 9.4 真实验证的两个协议

1. **Prefix-only factual prediction：** 只给击球后的严格前缀，隐藏目标帧、未来检测、人工图和 full-video smoothing。预测空间路径，对照来源人工 raster。由于人工图是时间无关的 union，报告全路径的空间一致性、prefix-only baseline 和静态 baseline，不声称获得逐时刻未来 ADE、接触顺序或 spin 准确率。
2. **Endpoint-conditioned reconstruction：** 给初始与终态，保持中间观察和人工图隐藏，拟合候选物理解释并测试空间兼容性。它是重建验证，不是盲测未来，也不是证明找到历史真实动作。

两种协议都先验证视频到人工 canvas 的映射；固定前景与画线协议，不能按方法调线宽获取更高 IoU。除 overlap 外，报告对称空间距离；对多假设报告数量或覆盖面积，避免把整桌涂满就获得高 coverage。

16 个 snooker 人工案例是选择过的小样本。它们支持 pilot 与案例分析，不自动代表完整比赛分布。Blackball 与 carom 的 38 个参考可作为单独的迁移诊断，不能并入主 snooker 成功率。

### 9.5 划分与统计

所有原始/Manual 副本、重播、相邻窗口、相同布局衍生及校准支持都按来源 family 处理。合成训练、开发和正式测试使用独立布局 family 与 seed；当前 2,000 条属于已使用的开发数据，不重新命名成盲测测试集。

分别进行 layout、物理参数组合、相机与领域留出，不随机把所有帧打散。报告多个训练或搜索 seed、配对差值、效应量和 layout/source-clustered 置信区间。合成干预 twins 不能当作独立布局重复计入显著性。

主终态分数使用球型和角色匹配。红球作为可交换集合，使用最小瓶颈匹配后的最大误差，不能使用静止球稀释后的平均误差。建议在合成条件预注册 \(0.25r,0.5r,1r\) 三档目标容差并做敏感性分析；这些不是已经验证的真实控制精度。

## 10. Agent harness：先约束接口，再比较能力

下面是待实现或扩展的设计，不是现有 CLI 已完成全部模型评测的声明。

输入应声明初始状态、目标、约束、可见物理信息、动作 bounds 和预算。输出为可解析的数值动作与被要求的预测，而不是“中等力度、稍微加塞”。每轮保留提案、工具输入输出、错误及最终提交。

本项目的“力度”首先用 cue-stick impact speed 表示，不是从视频直接测到的牛顿力，也不是 cue-ball speed。方向与抬杆角使用声明单位；side/vertical tip offset 是按球半径归一化的击点，不是实际角速度。所有控制显式设置，不能依赖引擎默认击点。

建议局部开发比较 \(B=0,8,32,128\) 的工具预算，先选必要的实验格子，不做全部条件的笛卡尔积：

- 每次 forward evaluation、支持校准、失败提案、重试、模型请求和 perception pass 都记录。
- 优化器 iteration 与 engine call 分开；缓存命中、请求尝试和实际计算也分开。
- 真值引擎 rollout 与学习模型 rollout 分开计数；离线生成、训练和在线成本同时报告。
- 给所有方法统一隐藏的事后 judge。其计算不提供反馈，也不混入模型可使用的查询预算。
- 输出格式错误、非法动作、超时和引擎失败计入总体结果；记录原因，不能丢弃后只报成功样本。
- 到达模拟事件上限、被强制停止的轨迹不当作自然静止终态。
- 不允许模型通过未计费代码执行或读取 privileged witness 获得额外引擎与答案。

同时给出固定查询和固定 wall-time 曲线。多目标共用一次编码、一次参数拟合或一个模型时，明确摊销单位。一个 cheap learned rollout 不等于一个 truth rollout，但它们都不是免费计算。

当前预算仍是最多两块明确分配的空闲 A100 80GB、16 个 CPU worker、1 TB 项目数据。此前 GPU/NVML 有运行时不匹配记录，正式模型执行前需要再次确认设备与环境，不假定 GPU 已可用。

本地模型先用已指定的 Qwen3.6-27B；闭源部署按实际 API 可用性确认，不静默替换。初始每个闭源模型最多 100 次请求，含 probe、修复和循环。可以先选择小型配对样本，不能把 100 请求误解成每个实验格子 100 个 episode。无充分预算的闭源结果仅作 pilot，不声称充分 leaderboard。

## 11. 论文定位、推进顺序与投稿选择

建议暂时按 empirical study 加可执行 benchmark 组织，不先承诺新方法胜出。可用工作题名是：

> InverseR: Measuring the prediction-planning gap in multimodal physical reasoning

它只是工作题名；“gap”仍需由实证建立。

### 11.1 五个支柱的当前状态

| 支柱 | 当前覆盖 | 缺口 |
| --- | --- | --- |
| 研究缺口 | 已有可测量问题与最近邻比较 | 需要用同任务实验建立实际差异，而非概念重新命名 |
| 构建管线 | 已有完整采集、增量转换、版本化目录及来源审计 | 真实子集的标定、完整性与人工 raster 映射尚未验收 |
| 评价框架 | 已有任务契约、simulator witness 审计及上述实验设计 | 模型 harness、全部指标与盲测划分未完成 |
| 实证发现 | 未完成 | 没有模型性能、消融或外部执行 Findings |
| 配套方法 | 未决定，非必须 | 根据瓶颈证据决定；必须超过强数值与 CueTip-style baseline |

若事件分歧确实是瓶颈，可以研究学习式多拓扑提案与有限预算验证；若隐藏参数是瓶颈，可以研究支持经验下的 belief 与鲁棒规划。两者都需要新的实证收益和适当最近邻比较，不能把“事件搜索”或“加 simulator”本身称为创新。

### 11.2 分阶段交付与研究者确认

1. **数据质量阶段：** 修复固定俯视录制和定向队列，核查人工 canvas 映射，给出通过率、失败类型及真实误差边界。G1 尚未通过。
2. **开发诊断阶段：** 构建小型独立布局的动作干预与 goal bank，完成几何数值 baseline、原生预测和匹配反馈归因。选择扩建档位前测量实际生成、存储和推理成本。
3. **Benchmark 冻结：** 审核标签能力、任务分布、划分、容差和泄漏防护后通过 G2。百万扩建不是当前默认许可。
4. **方法选择：** 依据诊断选择 benchmark/empirical 主贡献或机制型方法，通过 G3 再进行对应训练。
5. **正式评价：** 确认模型、设备、成本与测试方案，通过 G4。数据发布、API 扩额和投稿另行确认。

### 11.3 时间与单一投稿策略

截至本报告日期，官方页面列出：

- [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)：注册 2026-11-10 AoE，正文 2026-11-16 AoE。
- [IJCAI 2027](https://2027.ijcai.org/)：摘要 2027-01-04 AoE，正文 2027-01-11 AoE。

CVPR 正文截止距当前约一个月。应先判断能否完成可信数据、强 baseline、可证伪分析和盲测结果，而不是为赶时间把全部弱标签包装成高质量基准。建议在 10 月下旬进行是否冲刺 CVPR 的 go/no-go 判断。

CVPR 审稿期覆盖 IJCAI 截止，不能理解为同时投两个会议。选择 CVPR 或继续完善至 IJCAI，应在提交前确定；若已提交，后续处理必须遵守双方当年政策。不能保证中稿。

## 12. 可核查来源与当前工程结果

本地代码阶段已有 103 个测试通过，Ruff 与 Mypy 通过；100 个主 snooker 录制转换完成，9,414 个目录引用与来源哈希验证通过。它们验证软件与引用契约，不验证 learned world model、LLM 性能或真实击球精度。

下表保存主要检索入口。正式参考文献在文末；这里的入口用于复查版本、方法与证据条件。

| 编号 | 主要来源 | 核查重点 |
| --- | --- | --- |
| 1 | [作者论文](https://doi.org/10.1007/s10489-023-04542-3) | 来源设备、尺度、算法与路径评价；本地 PDF |
| 2 | [正式会议记录](https://papers.nips.cc/paper_files/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)及[项目](https://worldmodels.github.io/) | 正式题名、动作经验与模型利用 |
| 3 | [Nature](https://www.nature.com/articles/s41586-025-08744-2) | 2025 正式版本与 imagined policy learning |
| 4 | [OpenReview](https://openreview.net/forum?id=Oxh5CstDJU) | ICLR 2024、latent dynamics 与 MPC |
| 5 | [主要论文](https://arxiv.org/html/2506.09985v1) | 无动作预训练与机器人 AC 阶段分离 |
| 6 | [OpenReview](https://openreview.net/forum?id=Q6a9W6kzv5)及[官方会议记录](https://iclr.cc/virtual/2025/poster/32088) | PhysBench 题名、ICLR 2025 与问答协议 |
| 7 | [正式会议记录](https://papers.nips.cc/paper_files/paper/2019/hash/4191ef5f6c1576762869ac49281130c9-Abstract.html) | 二维动作任务、预算、模板划分 |
| 8 | [主要论文](https://arxiv.org/abs/2605.30900) | 初始箭头与参数、七球、无 spin、固定时刻预测 |
| 9 | [主要论文](https://arxiv.org/abs/2602.13294) | 观测条件重建与执行评价，不是盲测未来 |
| 10 | [正式会议记录](https://proceedings.mlr.press/v267/kang25g.html) | ID/OOD/组合测试与物理误差条件 |
| 11 | [正式出版入口](https://doi.org/10.1145/3721238.3730742)及[扩展论文](https://arxiv.org/html/2501.18291v2) | 事件提案、五个 cue 控制与数值优化 |
| 12 | [主要论文](https://arxiv.org/html/2312.14215v2) | 第 7 节台球动作与反馈条件 |
| 13 | [正式出版入口](https://doi.org/10.1109/TMECH.2015.2461547) | 实际击球、位置误差与人工辅助 |
| 14 | [机构记录](https://www.merl.com/publications/TR2026-052)及[扩展论文](https://arxiv.org/html/2411.08027v3) | AISTATS 2026、给定 forcing、参数拟合与真实观察 |
| 15 | [主要论文](https://arxiv.org/html/2608.04575v1) | 解析重建、拟合与视频推理，而非实际动作执行 |
| 16 | [主要记录](https://arxiv.org/abs/1906.01728) | 参数后验、状态动作经验与模拟测试边界 |

World Models 的 2018 预印本题名为 “World Models”，正式会议版本题名为 “Recurrent world models facilitate policy evolution”。DreamerV3 的 2023 预印本题名与 2025 Nature 题名也不同。本文优先使用正式版本；仍未核实正式 venue 的近期工作保留 arXiv 身份，不推测发表去向。

## 13. 未解决问题与结论适用范围

真实数据不具备实测 cue 控制或新采集的反事实执行。当前能支持模拟中的动作干预，以及真实观察上的兼容性；不能声称已经证明机器人或人类按模型方案击球可成功。要提升到真实 goal-control 结论，需要未来的受控动作记录或执行装置。

拟合相同起终态可能让错误材料参数与隐藏动作互相补偿。有限搜索找到多个解释是歧义证据，但没有找到第二个解不证明唯一，也不证明不可达。优化候选集合没有明确 prior、likelihood 与 coverage 检查时，不称为概率后验。

真实路径参考规模小且有选择偏差，固定相机作者样本也不等于多桌泛化。低帧率、不可见旋转、球心投影高度、误检与 ID 错换构成信息边界，而不只是模型弱点。必须保留这些不确定性，避免把无法观察的答案作为确定 GT。

Pooltool 的固定参数、接触模型和近似不等于真实桌面规律。参数偏移实验称为 synthetic domain-shift stress test，不能代替 sim-to-real 验证。对于摩擦、库边或旋转的 OOD 结果，不能直接外推到所有物理场景。

比赛视频的训练、再分发与版权来源仍未厘清。公开可观看和用户批准本地处理不是发布许可。作者论文开放许可也不自动解决每个媒体资产的许可。当前不自动发布视频、发送私有素材到 API 或对外上传数据。

## 14. 对三个调查问题的回答

**Survey-RQ1：** 将 belief、学习式转移、参数化物理模型与规划控制按模块区分，并分别测量无工具干预预测、工具辅助选择与独立参考执行。世界模型可以不生成像素，但控制用途必须验证动作条件响应；成功工具调用不能替代该证据。

**Survey-RQ2：** 现有来源支持大量弱观察、有限的人工空间路径参考和小型仿真真值开发集。扩建应以分层标签、来源去重与独立布局为单位；真实视频用于感知及动作后观测动力学，明确仿真用于可执行 cue 监督与干预。不能把帧、Manual 副本或仿真未来包装为更多真实击球。

**Survey-RQ3：** 使用配对动作与参数干预、相同候选及反馈证据、oracle/视觉输入替换、支持/查询分离和 source-clustered 统计，可以定位能力链中的损失。最值得先验证的是预测质量是否能转化为候选排序和终态成功，以及工具与真实观察分别改善了哪一环。

下一阶段应先获得可信的数据质量与小规模诊断，而不是立即训练大模型或生成百万视频。只有完成这些证据，才能决定文章是方法主导还是 benchmark 与 empirical study 主导。

## 参考文献

[1] Francisco J. Rodriguez-Lozano, Juan C. Gámez-Granados, Héctor Martínez, et al., “3D reconstruction system and multiobject local tracking algorithm designed for billiards,” Applied Intelligence, 53:21543-21575, 2023.

[2] David Ha, Jürgen Schmidhuber, “Recurrent world models facilitate policy evolution,” Advances in Neural Information Processing Systems, 2018.

[3] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, et al., “Mastering diverse control tasks through world models,” Nature, 640:647-653, 2025.

[4] Nicklas Hansen, Hao Su, Xiaolong Wang, “TD-MPC2: Scalable, robust world models for continuous control,” International Conference on Learning Representations, 2024.

[5] Mido Assran, Adrien Bardes, David Fan, et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv:2506.09985, 2025.

[6] Wei Chow, Jiageng Mao, Boyi Li, et al., “PhysBench: Benchmarking and enhancing vision-language models for physical world understanding,” International Conference on Learning Representations, 2025.

[7] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, et al., “PHYRE: A new benchmark for physical reasoning,” Advances in Neural Information Processing Systems, 2019.

[8] Ben Wang, Xiaogang Li, Ruochen Gao, et al., “BilliardPhys-Bench: Benchmarking physical reasoning and visual dynamics of multimodal LLMs,” arXiv:2605.30900, 2026.

[9] Jiarong Liang, Max Ku, Ka-Hei Hui, et al., “VisPhyWorld: Probing physical reasoning via code-driven video reconstruction,” arXiv:2602.13294, 2026.

[10] Bingyi Kang, Yang Yue, Rui Lu, et al., “How far is video generation from world model: A physical law perspective,” International Conference on Machine Learning, 2025.

[11] Sean Memery, Kevin Denamganai, Jiaxin Zhang, et al., “CueTip: An interactive and explainable physics-aware pool assistant,” SIGGRAPH Conference Papers, 2025.

[12] Sean Memery, Mirella Lapata, Kartic Subr, “SimLM: Can language models infer parameters of physical systems?” arXiv:2312.14215, 2023.

[13] Senthan Mathavan, Michael R. Jackson, Robert M. Parkin, “Ball positioning in robotic billiards: A nonprehensile manipulation-based solution,” IEEE/ASME Transactions on Mechatronics, 21(1):184-195, 2016.

[14] Anoop Cherian, Radu Corcodel, Siddarth Jain, et al., “LLMPhy: Parameter-identifiable physical reasoning combining large language models and physics engines,” International Conference on Artificial Intelligence and Statistics, 2026.

[15] Chen Yang, Shenxiang Zeng, Haoyang Zhao, et al., “PhysMind: From video to executable worlds for training-free physical reasoning,” arXiv:2608.04575, 2026.

[16] Fabio Ramos, Rafael Carvalhaes Possas, Dieter Fox, “BayesSim: Adaptive domain randomization via probabilistic inference for robotics simulators,” Robotics: Science and Systems, 2019.
