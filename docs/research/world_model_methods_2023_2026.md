# 世界模型怎样帮助理解物理后果：2023–2026 方法及 InverseR 迁移路线

研究工作报告。检索截止：2026-10-15。本文区分论文已经验证的结果、我们的实际实验、
可以借用的模块，以及尚待检验的方法假设。本轮没有新增模型实验、训练或权重加载。

## 摘要

给定初始状态和目标终态，直接生成击球参数，与预测指定动作的物理后果是不同问题。
本报告通过四个独立检索视角及主要论文、出版记录和公开实现核查，整理 2023–2026 年
的 32 篇相关工作。比较集中于动作条件预测、对象状态、可执行物理、视觉语言规划和
功能评价，而非视频生成质量本身。证据支持两条可分开的路线：训练可查询的视觉或
状态转移模型，或者让多模态模型构造、调用和校正显式物理世界。两者都需要与规划器、
动作接口及独立执行评价分开归因。现有论文已覆盖台球仿真反馈、事件到连续控制、
视频到可执行世界及预测—规划关联，因此这些成分不能单独作为新颖性声明。对 InverseR，
较务实的起点是复用强数值搜索、建立盲测后果预测和同候选排序对照，再决定是否需要
训练世界模型或提出接触敏感的查询分配机制；真实轨迹只能支撑其标签范围内的观察验证。

## 1. 从“给出动作”到“理解动作后果”

我们的 [200 布局实验](azure_native_pilot.md) 已提供一个起点：GPT-5-mini 和 GPT-5
在数值输入、`minimal` 推理预算、无工具的单次目标击球条件下，均为 0/200 成功。
399/400 个输出满足动作接口，且 200/200 个隐藏 witness 在公共输入上通过正控。
这说明该条件下的直接策略不可靠，但**不说明它们已经在盲测动力学或世界模型任务上
失败**：这轮测的是策略 \(\pi(S_0,G)\)，并没有大规模测转移 \(F(S_0,a)\)。

本次调查回答：

- **RQ1：** 2023–2026 的方法怎样表示并预测指定动作的后果？哪些是真正的动作条件
  转移模型，哪些是视觉计划、价值代理或外部引擎的控制器？
- **RQ2：** 哪些证据能区分视觉合理、事实预测、物理干预响应和最终规划效用？
- **RQ3：** 哪些模块可在 InverseR 的现有条件下借用，哪些需要新的动作数据、训练或
  资源？最近邻工作已经排除了哪些宽泛的新颖性主张？

下文先定义接口和分类，再比较方法，最后给出迁移优先级与可证伪研究方案。它不是
正式论文稿，也不是已完成这些方法复现的声明。

## 2. 检索与证据边界

### 2.1 范围和核查

四个视角分别检索：学习式动作条件动力学与规划；对象表示和可执行物理；多模态
内部推理与仿真工具控制；物理评价、失败证据与功能效用。各视角先广搜，再按具体
方法、引用和最近邻窄查。核心关键词包括 `action-conditioned world model`、
`visual latent planning`、`physical consequences`、`object-centric dynamics`、
`executable physics`、`billiards simulator feedback` 和 `prediction planning utility`。

保留 32 篇身份和相关论点可核实的工作，其中 2 篇为综述。正式发表年份或首稿年份在
2023–2026 范围内：例如 SlotFormer 首稿为 2022 年，正式 ICLR 2023 版本在本次范围内。
已发表论文优先引用正式题名和年份；未核实去向的工作明确保留 arXiv 身份。
产品演示、搜索摘要中的录用断言和无法核实的新投稿不作为方法证据。

Li 等 [1] 用功能、时间建模和空间表示三个轴组织世界模型，而 Long 等 [2] 强调外部
模拟器与内部模型的互补。这两种综述框架都有帮助，但本项目还需要进一步核对动作的
物理单位、允许修改的世界变量，以及最终动作是否真正执行成功。因此本文按模块
接口重新比较，不把所有冠以 world model 的系统视作同一种能力。

来源：[Li 等的主要论文](https://arxiv.org/abs/2510.16732v3)、
[Long 等的主要论文](https://arxiv.org/abs/2507.00917v3)。两篇综述的正式 venue 未核实，
不采信搜索摘要将它们统一称为已同行评审论文的表述。

### 2.2 核查强度与限制

方法描述依据主要论文的 method、experiment、appendix 或已读取的官方源码；
出版身份用会议、出版机构、登记记录或明确的论文版本核查。本文没有独立复现实验，
不将作者结果改写成我们已经测得的结果。

部分 OpenReview 入口遇到浏览器验证，DBLP/API 也有访问限制。DINO-WM 的正式 venue
未据此确认，仍引用 2024 预印本；这不等于认定它未发表。JEPA-WMs 的 TMLR 2026
身份由正式版首页、arXiv 修订说明及官方仓库标注支持，但未读取受限的审稿决定。
一个带占位链接的 `Phys-WorldBench` 检索线索无法核实，没有纳入参考文献。

公开仓库不等于可以任意复制；代码、checkpoint、训练数据和媒体许可分开看。
本文只在实际读取许可文本后标注许可，未核实处不猜测。

## 3. 操作性分类：表示、转移来源和控制用途

控制用途的世界模型需要能够回答：

$$
\hat S_{1:H}(a),\ \hat E(a),\ U(a)
=F_\theta(S_{\leq 0},a,\phi),
$$

其中 \(a\) 是指定动作，\(\phi\) 是已知或需要推断的物理条件，\(\hat E\) 是预测事件，
\(U\) 是尚需校准的不确定性指标。预测可以发生在像素、latent 或对象状态中，不必
生成视频。这里采用的是控制研究的操作性要求，不否认无动作条件视频模型的广义命名。

分类用三个正交轴，不要求一篇论文只属于一个模块：

| 轴 | 类别 | InverseR 必须核对什么 |
| --- | --- | --- |
| 世界表示 | 像素、视觉 latent、对象状态、语言/事件记录 | 是否保留位置、身份、速度、旋转和物理条件？ |
| 后果提供者 | 学习式动作转移、无动作预测、显式物理引擎、只有价值/策略 | 后果是谁算的？输出能否响应同状态下的动作变化？ |
| 控制用途 | 想象训练 actor、MPC/候选优化、直接策略、工具验证/修正 | 动作怎样产生？用了哪些信息和实际查询？ |

例如 DINO-WM [3] 的 latent predictor 是学习式转移；CueTip [4] 的 PoolTool 提供转移，而
神经网络提供价值；VAGEN [5] 学习显式状态推理，但原环境提供真实反馈。这些跨类别系统
需要拆开评测，不能仅凭系统总成功率把能力全部归给多模态模型。

对台球，“2D”只描述球心所在的平面。动态状态通常还涉及 \(v_x,v_y\)、三轴角速度、
滑动/滚动阶段和接触参数。单帧位置、漂亮 mask 或稳定对象 slot 不是完整 Markov 状态。

## 4. 学习式后果模型：训练接口比模型名字更重要

### 4.1 想象策略与优化动作不是同一种使用方式

DreamerV3 [6] 在 RSSM 想象轨迹里训练 actor–critic，交互时执行 actor；TD-MPC2 [7]
则把 latent 转移、奖励和 Q 用于在线 MPPI。它们都说明世界模型可以支持控制，但
前者的收益不能解释为“LLM 在线试打”，后者的任务相关价值也不能自动换成任意终态
的评价函数。两者都需要动作轨迹和训练，不能直接通过现有 GPT API 得到其方法能力。

### 4.2 冻结视觉模型不等于冻结整个世界模型

DINO-WM [3] 和 V-JEPA 2 [8] 都允许复用视觉表征，但动作条件预测器仍需要学习：
前者在 DINOv2 patch features 上训练下一状态预测，后者把无动作视频预训练与
机器人动作条件阶段分开。PLDM [9] 则从目标域数据学习 JEPA 表征和转移，并加入
防坍塌与规划相关机制。三者可借的共同接口是“编码状态—预测候选动作—优化目标”，
不是现成的台球控制 checkpoint。

2026 年的潜动作工作 [10] 将动作发现推进到无标注视频，但规划阶段仍训练真实控制到
latent action 的映射。与 V-JEPA 2 [8] 的已记录机器人动作相比，它改变了预训练阶段
对动作标签的依赖，没有把潜动作自动变成有单位的 cue-stick speed 和 tip offset。
这正是我们不能仅凭大量比赛视频就宣称已经获得真实击球监督的原因。

**比较表：可借的是动作条件预测和规划接口，不是“zero-shot”标题所暗示的免训练控制。**

| 方法与来源 | 转移和规划机制 | 可以借用的模块 | 不可直接迁移之处 |
| --- | --- | --- | --- |
| [DreamerV3](https://doi.org/10.1038/s41586-025-08744-2) [6]，2023 首稿/Nature 2025 | RSSM；想象训练 actor–critic | 长期后果和终态价值的分工 | 需要训练；不是任意 goal 的冻结策略 |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) [7]，ICLR 2024 | 控制相关 latent/Q；MPPI＋policy prior | 候选采样、短 rollout＋terminal value | 需动作/奖励训练；原目标函数不等于我们的全终态目标 |
| [DINO-WM](https://arxiv.org/abs/2411.04983) [3]，2024 预印本 | 冻结 patch encoder；动作 ViT；latent-goal CEM | 视觉 encoder、动作预测器、独立 planner | 需目标域动力学训练；latent 距离需与球位置/事件核对 |
| [V-JEPA 2/2-AC](https://arxiv.org/abs/2506.09985) [8]，2025 预印本 | 视频预训练；动作条件 predictor；CEM | 可复用 encoder；动作条件扩展 | 原动作是机器人位姿/夹爪增量，不是 cue 参数；相机敏感 |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) [11]，NeurIPS 2024 | 分时间噪声；联合 action/reward/observation 去噪与 guidance | 一致轨迹假设和目标引导 | 主要控制证据是低维 Maze2D；不是已验证的视觉台球控制 |
| [PLDM](https://arxiv.org/abs/2502.14819) [9]，NeurIPS 2025 | JEPA＋inverse dynamics；ensemble MPPI | 不确定性惩罚、reward-free goal planning | reward-free 仍需 action；ensemble 分歧不等于校准风险 |
| [Latent actions in the wild](https://arxiv.org/abs/2601.05230) [10]，2026 预印本 | IDM 潜动作＋forward model；再学真实动作映射 | 无动作视频表征/转移学习的分工 | 潜动作不等于真实力、旋转或击球控制；官方代码未核实 |

Diffusion Forcing [11] 展示了联合序列生成的用途，而 DINO-WM [3] 把规划留在不需解码
视频的特征空间。对当前 2D、精确终态任务，没有证据要求先训一个大视频生成器；
紧凑状态/latent 转移更适合作为首轮学习式对照。但这只是资源与接口判断，不是认定
视频生成方法无效。

### 4.3 一个比“预测—规划 gap”更近的 empirical study 先例

JEPA-WMs [12]，即
[What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/pdf/2512.24497v4)，
已经系统比较视觉编码器、动作条件方式、rollout/context、模型/数据规模、planner
和规划代价，并直接研究 prediction metrics 能否代理成功率。它比单纯视频评分论文
更接近我们的经验研究；不能把这些问题重新称为首次提出。

该文 [12] 与 WorldArena [13] 的结果也不支持“预测质量从来没用”这种反向口号。
JEPA-WMs 报告代理关系随任务变化，并出现验证 loss 相近而规划效果不同的配置；
WorldArena 则同时报告正相关和较弱的规划关联。更合理的问题是：**哪种误差、表征和
动作接口，在什么决策条件下影响成功？**

JEPA-WMs 的有用控制包括对齐 encoder 规模/token 数、固定环境内 planner 配置、
匹配动作条件比例，以及检查训练与规划 context 是否一致。但其部分消融同时改变
batch、encoder/predictor 宽度或信息输入；DROID 的离线 Action Score 也不是实际
机器人任务成功。我们应借控制逻辑，不照搬综合结论或声称成本已完全匹配。

## 5. 对象和可执行世界：状态清晰不等于参数真实

### 5.1 对象历史、接触与物理状态

SlotFormer [14] 用对象 slot 的历史模拟视觉动力学，OCVP [15] 将对象内时间注意力与
对象间关系注意力分开；两者都比一个全局视觉向量更方便讨论多球交互，但原方法并没有
识别有单位的三轴 spin。相较之下，Dynamics/Δynamics [16] 把速度、角速度、材料和
几何显式写入 YAML 再交给 MuJoCo，代价是大量合成配置监督，而非仅换一个 prompt。

我们可借对象级接口及“自由运动与相互作用分开”的结构，不能把高 slot/mask 分数当作
物理状态已完整恢复。对初始完全静止的合成场景可以声明 \(v=\omega=0\)；对真实运动
前缀，需要把未观测 spin、漏球和参数不确定性保留为未知或假设集合。

### 5.2 外部物理与外观生成的边界

PAC-NeRF [17] 通过 posed video、NeRF 和可微 MPM 拟合物理，而 PhysGaussian [18]
主要将已有几何与给定材料放入前向模拟再渲染。PhysGen [19] 更接近平面刚体的
“感知—物理—外观”流程，但单图中由预训练模型估计的摩擦或质量仍是先验猜测，
不是动态测量。三者都提醒我们：渲染一致、参数拟合和真正的动作后果是不同证据。

**比较表：借状态与仿真接口，不把 3D 渲染器或 continuum solver 直接替换成台球真值。**

| 方法与来源 | 核心机制 | 可借模块 | 台球适配限制 |
| --- | --- | --- | --- |
| [SlotFormer](https://arxiv.org/abs/2210.05861) [14]，ICLR 2023 | 对象 slot 自回归预测；另有 PHYRE success head | 对象历史、身份和交互表示 | 需训练；不是已标定 cue-action/spin 模型 |
| [OCVP](https://arxiv.org/abs/2302.11850) [15]，ICIP 2023 | temporal/relational attention 分解 | 自由运动与碰撞交互的结构化预测 | 原 formulation 未提供显式动作条件 |
| [PAC-NeRF](https://arxiv.org/abs/2303.05512) [17]，ICLR 2023 | 多视图几何＋可微物理拟合 | observe/render–simulate–fit 分工 | 需标定、逐段优化；continuum 材料不是 sphere contact |
| [PhysGaussian](https://arxiv.org/abs/2311.12198) [18]，CVPR 2024 | Gaussian 几何与 MPM 前向模拟/渲染耦合 | 物理 rollout 与外观分离 | 不提供真实 cue、摩擦或旋转辨识 |
| [PhysGen](https://arxiv.org/abs/2409.18964) [19]，ECCV 2024 | 预训练感知/参数先验＋刚体引擎＋视频 refinement | training-free hybrid 的模块边界 | 平面圆盘旋转不是球体完整 spin/滑动/滚动 |
| [Dynamics](https://arxiv.org/abs/2605.20576) [16]，CVPR 2026 | optical flow/VLM→物理 YAML→MuJoCo | 明确的世界状态、参数及重模拟接口 | 使用 400K 合成场景训练；重建吻合不证明参数唯一 |

### 5.3 最值得借的真实观察路线：先拟合可执行假设

PhysMind [20] 从视频建立可复用的、问题无关的可执行世界：感知/位姿轨迹之后，拟合
分段解析运动和瞬时接触。它与 PAC-NeRF [17] 都采用观察约束物理，但避免固定步长
unrolling 的方式不同；其“training-free”也不等于没有昂贵的预训练感知模块。
PhysMind 识别的是接触连通组内的质量比，部分尺度需要先验，不能把标题理解为恢复
了所有真实物理参数。

对我们最有用的迁移是“前缀轨迹—运动阶段—接触时刻—参数假设—严格未来预测”。
但 [PhysMind 的原文](https://arxiv.org/pdf/2608.04575v1)及已读实现主要采用平移/法向
冲量模型，物体移除是已展示的 active edit；这不是完整 sphere-spin 台球后端，也不
等于可以恢复真实 cue 动作。相关代码的许可尚未建立，只借设计，不直接复制。

LLMPhy [21] 则把参数辨识和新布局推理分为两阶段：LLM 提程序，物理执行后依据
重建误差修改。与 PhysMind [20] 的视觉前处理相比，LLMPhy 的合成实验直接提供辅助
3D simulator trajectories，且 query pusher velocity 已给定。借这条路线时，不能
把这些额外观测悄悄加进我们的真实视频条件，也不能把“parameter-identifiable”
当作任意动作/参数都唯一可识别的定理。

来源：[PhysMind](https://arxiv.org/abs/2608.04575)、
[LLMPhy 正式记录](https://proceedings.mlr.press/v300/cherian26a.html)。

## 6. 多模态模型的角色：构建世界、提出路线，而非替代连续物理

### 6.1 最接近我们的先例已经很具体

VisPhyWorld [22] 从两个观测帧推断场景、运动参数和可执行程序；SimLM [23] 在台球
中提连续参数并读仿真反馈；CueTip [4] 更进一步用语言事件序列驱动连续优化。
因此，“初始/后续图像到物理世界”“LLM＋simulator”“语言路线到力度/角度/旋转”
都不是足够的新贡献。

CueTip 的神经 surrogate 预测 win-rate 分布，不是状态转移；PhysAgent [24] 则用
Genesis 验证并修复物理场景/控制程序，再引导视频生成。两者与 SimLM [23] 的共同
价值在于让语言模型负责提案和诊断，把数值后果交给外部引擎。不能将系统总收益全部
归因为 VLM 内部学到了这些动力学。

**比较表：先把这些当强 baseline 和设计来源，再讨论新的方法模块。**

| 方法与来源 | 实际流程 | 现在最可借的部分 | 必须保留的区别 |
| --- | --- | --- | --- |
| [LLMPhy](https://proceedings.mlr.press/v300/cherian26a.html) [21]，2024 首稿/AISTATS 2026 | 参数假设/程序→重建误差→修订→新布局 | 辨识、感知和 query 分离 | 辅助轨迹/已知 excitation 简化了原问题；代码 AGPL |
| [VisPhyWorld](https://arxiv.org/html/2602.13294v3) [22]，2026 预印本 | 帧 1/10→scene/code→执行/重建 | 可检查的场景假设、执行/参数/运动分项验证 | 不是受约束 cue goal control；已有 forecast-only 消融及参数 MAE |
| [SimLM](https://arxiv.org/abs/2312.14215) [23]，2023 预印本 | 参数提案→仿真→反馈/自评→修改；例子检索 | 有限反馈、失败记忆、reset 后重新试打 | 原台球目标是 pot；台球用 3 retries/3 examples，不照搬主 2D 实验的 5 attempts |
| [CueTip](https://arxiv.org/abs/2501.18291) [4]，SIGGRAPH 2025 | 事件路线→simulated annealing→surrogate/tuner→选择解释 | 事件提案＋连续优化；改用终态目标 loss | 原 value surrogate 需训练；300 optimization steps 不是完整 simulator-query 总数 |
| [PhysAgent](https://arxiv.org/abs/2607.16355) [24]，2026 预印本 | 物理程序→验证→诊断→局部修正→视频 | 因素级错误反馈和修正记忆 | 原任务是视频事件合成；force/torque/pose API 不是 cue 执行接口 |

VisPhyWorld [22] 还提示一个重要限制：成功运行代码只保证能生成输出，未必意味着
程序实际采用物理求解器；部分 prompt 只建议使用 Cannon.js。其 v3 已报告物理参数
恢复 MAE，并采用时间 offset/DTW 对齐运动。我们应借分项验证，但对 forward prediction
固定时间轴，并把图像相似、可运行、参数误差及真实执行达标分别报告。

对 goal planning，更安全也更可测的迁移是**固定初始状态、物理和台面，只开放 typed
cue-action API**。不能让模型通过改重力、球的位置、材料或任意施加后续力来凑目标。
这是一项拟议适配，不是我们已实现 VisPhyWorld/PhysAgent 的结果。

### 6.2 内部推理和视觉未来计划可以借，但要看训练与动作层

VAGEN [5] 显式记录当前 belief、动作推理和预期 next belief；ThinkAct [25] 把
视觉潜计划连接到 diffusion controller；CoT-VLA [26] 先生成视觉 subgoal，再出动作
chunk。三者都可启发“先说清后果，再给控制”的接口，但其已发表增益包含 RL、
视觉/动作对齐或示范训练，不由复制推理模板自动继承。

ThinkAct [25] 的 slow-plan/fast-control 与 CoT-VLA [26] 的闭环动作 chunk 也不能原样
搬到单杆台球：杆头离开母球后不能继续改角度、力度或旋转。我们的多轮反馈发生在
执行前的计算搜索中，或另一次重置仿真试验中；不能包装成滚动过程的实体闭环干预。

来源：[VAGEN](https://arxiv.org/abs/2510.16907)、
[ThinkAct](https://arxiv.org/abs/2507.16815)、
[CoT-VLA 正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html)。

## 7. 物理理解的证据：问答、画面和执行必须分开

### 7.1 有用的困难来自受控变化，而不是一个总分

Kang 等 [27] 用受控二维物理视频区分 ID、参数 OOD 和组合泛化；PhysBench [28]
覆盖更广的物理 QA，却没有执行连续动作。前者同时有 ID/组合改善和参数外推局限，
后者还受模型子集、帧输入差异影响，不能简单把某个统一总分叫作“理解物理的程度”。

BilliardPhys-Bench [29] 是必须认真对照的 2026 最近邻。它给定已经建立的母球线速度，
预测碰球、碰壁与 1–5 秒位置，不要求模型选择击球。相比 PhysBench [28] 的答案准确率，
它提供更直接的事件/坐标后果评价；但它采用简化合成碰撞与摩擦，明确不覆盖真实 spin
等效应。因此我们可以借它的分层指标，而不能称自己为首次 MLLM 台球物理 benchmark。

来源：[物理规律测试](https://proceedings.mlr.press/v267/kang25g.html)、
[PhysBench](https://arxiv.org/abs/2501.16411)、
[BilliardPhys-Bench](https://arxiv.org/html/2605.30900v1)。

其输入图中的速度箭头和数值是任务明确提供的信息，对 forward 任务不是泄漏；
但如果改成 inverse goal 输入，必须去掉它们，否则暴露生成该终态的运动信息。
球的初速度也不是 cue-stick speed。它的固定时间位置评分不能直接改称自然静止终态成功。

### 7.2 视频物理评价能补充解释，但不能代替 ground-truth 后果

VideoPhy [30] 将语义遵循和 physical commonsense 分开，VideoPhy-2 [31] 进一步保留
规则违反、遵循和无法判断，并报告联合指标。与 WorldScore [32] 的相机/场景控制和
几何/纹理一致性相比，这些人类评价更接近直观物理，但都不提供我们所需的精确 cue
干预与目标执行。WorldScore 的 motion accuracy 也主要度量运动发生在哪里，不是
球碰撞冲量或各球的正确轨迹。

**比较表：指标各有用途，不能用视觉/代码有效性代替约束动作的执行结果。**

| 方法与来源 | 真正测量的东西 | 可借到 InverseR 的设计 | 不支持的外推 |
| --- | --- | --- | --- |
| [Kang 等](https://arxiv.org/abs/2411.02385) [27]，ICML 2025 | 视频解析的速度/规律误差；ID/OOD/组合 | 独立参数留出、组合留出、parser noise floor | 无显式动作输入，不证明反事实控制 |
| [PhysBench](https://arxiv.org/abs/2501.16411) [28]，ICLR 2025 | 图像/视频/文本物理 QA | 同案例/帧预算、属性与 dynamics 错误分层 | QA 正确不等于坐标预测或可执行控制 |
| [BilliardPhys-Bench](https://arxiv.org/abs/2605.30900) [29]，2026 预印本 | 已给速度下的 contact/wall/coordinate | 分事件、运动球/静止球、horizon 与 1r 容差 | 没有选择 cue/spin；不一定到静止 |
| [VideoPhy](https://arxiv.org/abs/2406.03520) [30]，ICLR 2025 | 人类语义/物理合理性及 joint | 分项与联合评分、盲化、运动难度 | 并非 simulator-grounded factual truth |
| [VideoPhy-2](https://arxiv.org/abs/2503.06800) [31]，ICLR 2026 | 细粒度规则与联合成功 | 保留无法判断；区分 full/hard subset | 模型失败筛出的 hard set 不是总体失败率 |
| [WorldScore](https://arxiv.org/abs/2504.00983) [32]，ICCV 2025 | 相机、外观、几何和运动一致性 | 相机/物体运动分离；不只报 aggregate | 没有 object-force/shot action 效果 |
| [WorldArena](https://arxiv.org/abs/2602.08971) [13]，2026 预印本 | 视频质量＋训练/排序/规划的功能试验 | 独立执行效用与模块替换 | robot simulation 不等于真实台球执行 |

### 7.3 预测—规划关系：已有证据，也有条件依赖

WorldArena [13] 的 EWMScore 是 16 个归一化视频指标的平均，不包含最终 task success；
它报告与人类评价 0.825、与 action planning 0.360 的 Pearson 相关。JEPA-WMs [12]
则在不同物理规划任务中发现代理关系强弱不同，还展示 loss/可视化相近而规划较差的
例子。两者的共同启示是测量接口和决策过程，不能只挑一个数字宣布“world model
没有用”或“更真实的视频必然更会规划”。

这些研究也有正面结果：Kang 等 [27] 的组合泛化会随数据改善，WorldArena [13] 的
一些生成数据条件能帮助 policy，JEPA-WMs [12] 的某些任务中预测代理与成功率关系
较强。我们的调查因此支持有条件的 utility 假设，而非预设所有模型必然失败。

## 8. 对 InverseR 当前结果的解释与缺失证据

### 8.1 0/200 不能直接归因为“不懂动力学”

BilliardPhys-Bench [29] 给定 post-impact 线速度；V-JEPA 2 [8] 使用已记录机器人动作；
SimLM/CueTip [4,23] 则经具体 cue 接口与引擎交互。这些差异提示我们分开：

1. **动作到初始运动：** cue speed、击点怎样映射成球的 \(v,\omega\)；
2. **初始运动到后果：** 摩擦、接触、spin 和停止如何演化；
3. **后果到决策：** 从候选中选什么，或者怎样反求动作。

目前的 0/200 同时涉及这些环节与 minimal 推理预算。即使正控验证了任务/判分，
也不保证模型掌握了引擎特定 impact convention。应追加独立 blind forward，以及
post-impact \(v,\omega\) oracle 对照；后者是额外信息的诊断上界，不能混入原生成绩。

### 8.2 “更懂后果”必须在固定决策过程里检验

DINO-WM/PLDM [3,9] 将预测交给优化器，而 CueTip [4] 将事件提案交给数值求解。
因此直接策略弱并不否定这些组合路线；反过来，组合成功也不能证明 LM 内部动力学准。
需要让同一初始布局、候选、信息和 selector/planner 下的预测变化产生可量化决策变化。

固定 bank 尤其需要保证可解释的上界。可以先从与 goal 无关的 prior 构造 bank，再从
其中一个合法动作生成可达目标；顺序随机化、forecaster 不见 goal。若沿用当前 200 个
goal 再抽随机 bank，必须报告 bank 内是否有可行候选，不能把“这个有限 pool 没解”
写成全动作空间不可达，也不能拿全为零的 bank success 研究有意义的排序改善。

### 8.3 真实数据能补哪一层

PhysMind/LLMPhy [20,21] 的观察—物理拟合路线与我们的真实前缀最接近，但它们的感知、
激发条件和引擎假设不能忽略。现有 BSA 是 single-track prefix、来源标注的 cm/time，
没有完整场景、实测 cue 或 spin；作者/manual raster 也不是逐时刻动力学真值。

因此真实数据先用于感知、严格 prefix-only factual compatibility 和参数假设的 held-out
检验。不能把拟合出的历史动作当 GT，不能把看到目标帧的重建称为 blind prediction，
也不能把无动作视频中的 latent action 写成可执行杆速/击点。实际数据边界见
[世界模型数据衔接](world_model_data_bridge.md)。

## 9. 可借用路线、代码许可和实现优先级

### 9.1 推荐顺序

| 优先级 | 借用哪些方法 | 要落地的模块 | 与现有工程的衔接 | 运行前提 |
| --- | --- | --- | --- | --- |
| **P0** | JEPA-WMs＋BilliardPhys＋WorldArena [12,13,29] | blind consequences、事件、ranking、goal、成本分开 | 已有 forward/event/paired/bank metrics 和 hidden judge | 先冻结小型共同 bank；新 API 额度另确认 |
| **P1** | CueTip＋SimLM＋PhysAgent [4,23,24] | 事件路线→数值优化→物理反馈/针对性修改 | 已有 random/Sobol/geometry-Powell 与持久预算 | baseline 适配，不默认新方法；所有实际调用计费 |
| **P2** | PhysMind＋LLMPhy [20,21] | 观察前缀→状态/参数假设→可执行后果 | 现有 tracking、真实 prefix/private future；用原 Pooltool 后端 | 本地处理；支持/query 分离；不伪造历史动作 |
| **P3** | DINO-WM＋PLDM＋SlotFormer [3,9,14] | compact object/visual latent dynamics＋goal planner | 用仿真动作真值；对比 oracle-state 与 BEV | 需要独立训练/验证数据及资源许可；没有现成台球权重 |
| **P4** | V-JEPA 2＋潜动作工作＋Dynamics [8,10,16] | 大型视觉表征、动作对齐、完整状态 schema | 可借 backbone/schema；重新适配 cue 控制 | 训练/算力成本更高，不是当前最小起步路线 |

P0 的作用是定位瓶颈，P1 是强工具 baseline，P2 补真实观察，P3 才明确加入学习式
transition。该顺序没有要求长期停留在预处理，也没有把 P1 的组合包装成新 world model。
复用接口和 baseline 可以立即设计；**实际新推理或训练仍需新的资源授权**。

已存在的代码入口：[规划与候选](../../src/inverser/baselines.py)、
[状态/动作合同](../../src/inverser/study.py)、
[后果与决策指标](../../src/inverser/diagnostics.py)、
[请求/物理预算](../../src/inverser/evaluation.py)。CEM/MPPI、真实 local adapter 和上述
学习式方法尚未因本次调查自动实现或运行。

### 9.2 哪些可以借代码，哪些只能借设计

以下是已读许可的范围，不是所有素材和权重的再授权，也不是完整法律/安全审计：

| 资源 | 核实的代码许可/状态 | 建议 |
| --- | --- | --- |
| [CueTip](https://github.com/SeanMemery/cuetip)、[VAGEN](https://github.com/mll-lab-nu/VAGEN)、[VisPhyWorld](https://github.com/TIGER-AI-Lab/VisPhyWorld) | MIT | 可研究模块适配并保留许可/引用；不原样搬 robot/control 或任意 code-exec 接口 |
| [DINO-WM](https://github.com/gaoyuezhou/dino_wm)、[PLDM](https://github.com/vladisai/PLDM)、[TD-MPC2](https://github.com/nicklashansen/tdmpc2) | MIT | 可借 predictor/planner 代码；发布 checkpoint 不是台球域可用性证明 |
| [V-JEPA 2](https://github.com/facebookresearch/vjepa2) | 主体 MIT，部分文件 Apache-2.0 | 分文件和模型资产核对，不将代码许可泛化给全部 weights/data |
| [SlotFormer](https://github.com/pairlab/SlotFormer)、[PAC-NeRF](https://github.com/xuan-li/PAC-NeRF) | MIT | 借表征/拟合结构；目标域训练或逐场景优化仍有成本 |
| [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing) | MIT 文本另带 README 署名要求 | 按其具体条款处理，不简化成无条件复制 |
| [LLMPhy](https://github.com/merlresearch/llmphy) | AGPL-3.0-or-later | 不当作 MIT 代码直接混入；借方法或独立、许可兼容地复现 |
| [PhysAgent](https://github.com/IApple233/PhysAgent) | CC BY-NC-SA 4.0 标注 | 注意非商业和 share-alike 条件，不能称 unrestricted |
| [PhysMind](https://github.com/ccyydd/PhysMind)、[SimLM](https://github.com/SeanMemery/SimLM) | 代码可读，许可未建立 | 当前借方法描述，不复制未确认许可的代码 |
| [WorldArena](https://github.com/tsinghua-fib-lab/WorldArena) | 只核实 `embodied_task` 子树 MIT | 不把子树许可扩成整个仓库；改成我们的 cue/interface/judge |

## 10. 可能形成文章的方法假设，而不是已经确认的创新

### 10.1 不建议再主张什么

SimLM/CueTip [4,23] 已覆盖台球连续控制、仿真反馈和事件到数值优化；
PhysMind/LLMPhy/VisPhyWorld [20,21,22] 已覆盖视觉物理辨识、可执行世界与后果查询；
JEPA-WMs/WorldArena [12,13] 已研究预测与规划关联。
因此“不需要训练”“加一个 simulator”“给模型画未来”“发现 prediction–planning gap”
都不能独立支撑新的方法贡献。加入 spin 也不是首次，CueTip 的 PoolTool 已支持相关控制。

较明确的项目条件是：**固定世界、真实 cue 接口、任意可行的多解终态目标、接触/旋转
敏感性、含未知状态的观察条件，以及同预算下的预测到选择到执行归因。**
这些是有价值的检验条件，但只换环境不保证方法新颖性，当前 200 个开发结果也不等于
可投稿的正式 benchmark。

### 10.2 可优先检验的三个假设

**H1：决策关键的接触错误，比全局平均预测误差更能解释策略失败。**  
依据是对象/交互分工 [14,15] 与 JEPA 代理的任务依赖 [12]，而不是我们已经看到这种
因果关系。保持 candidate bank、planner 和信息固定，比较位置误差、first contact、
contact order、事件时刻与 selected regret/goal success；再用 oracle event 或 oracle
continuous rollout 替换定位损失。若位置指标已充分解释且事件项无额外作用，就否定 H1。

**H2：用任务相关的不确定性分配有限验证预算，比平均给每条路线更多尝试更有效。**  
可借 PLDM 的 ensemble/规划分工 [9] 和 CueTip 的事件候选 [4]：廉价 predictor 提供
各动作的后果与分歧，规划器优先验证可能改变候选排序、scratch 风险或 contact topology
的候选。ensemble 分歧须先在 held-out 数据校准，不能直接叫真实失败概率。比较均匀、
随机、纯预测代价、普通 BO/CEM 和任务相关分配，固定实际 truth calls；若强数值搜索
同预算已达到同效果，或者只省调用不省总体时间，就不宣称新机制更有效。

这是候选模块，不是已证实首创。Pooltool 本身很便宜，API/视觉模型可能更慢，因此
必须同时报告 wall time、Tokens、learned rollouts、true rollouts 和 offline 成本；
不能人为给 cheap engine 设置预算后只报“减少仿真次数”。

**H3：保留多种可兼容的隐藏状态，能够改善未见目标的后果/鲁棒规划，而非只改善轨迹拟合。**  
PhysMind 的尺度/接触辨识限制 [20] 与 LLMPhy 的辅助经验条件 [21] 支持研究这个问题。
在合成中有真实 spin/physics、在真实前缀中保留未知，比较单一最优拟合与明确的假设
集合；用未参与拟合的动作、布局和严格未来检验。没有 prior/likelihood/calibration 的
候选集合不称 Bayesian posterior。若收益只发生在同一支持轨迹，H3 不成立。

这些假设不是必须同时做。先完成 P0/P1 与强 baseline，再按实际失败选择一个，
比先堆大型视频模型、RL 和复杂 agent 更能定位文章贡献。

### 10.3 下一轮的最小诊断框架

1. 保留已有 200 布局的 native goal baseline；新诊断子集 outcome-independent 选取。
2. **后果预测：** 给 initial＋指定 action、不见 goal；同时测坐标、事件与单变量
   action twins。用 post-impact \(v,\omega\) 对照定位 actuator mapping。
3. **有限候选选择：** 用共同 bank 和同一确定性 selector，比较 native forecast、
   学习式/显式模型 forecast 与 GT-bank upper bound；区分 bank 外可行目标。
4. **受控搜索：** 比较 geometry/Sobol/Powell、CueTip/SimLM-style 与候选新模块；
   拆开实际反馈、无新证据反思和 shuffled feedback，报告允许预算及实际消耗。
5. **真实层：** 只在本地做 strict prefix factual validation，既不反推实测 cue/spin GT，
   也不把 synthetic physics shift 当作已证明 sim-to-real。

这不是当前新增实验授权。预算、API轮次/模型、学习式训练、GPU 和正式测试划分需另定。
正式评价使用新的布局/seed/source groups，不能把现有 2,000 个开发记录改称未见 test。

## 11. 对三个研究问题的回答

**RQ1：** 可借的后果模型有动作条件视觉/latent 预测、对象结构化动力学，以及观察约束
的可执行物理。多模态模型可以负责状态/事件/参数假设和规划控制，但价值 surrogate、
视觉 subgoal 和 simulator wrapper 不能单独证明其原生连续动力学。

**RQ2：** 需要分开视觉合理、时间对齐后的重建、严格未来状态、action intervention、
候选排序和最终执行。先控制信息、候选、planner、动作接口和真实查询，再检验预测
与效用的关系；已有正/负面证据都不支持一个跨任务通用结论。

**RQ3：** 当前优先借 CueTip/SimLM 的受控提案—优化—验证，PhysMind/LLMPhy 的观察—
物理假设分工，以及 JEPA-WMs/WorldArena/BilliardPhys 的经验评价逻辑。之后再选择
DINO-WM/PLDM 等学习式 predictor。可争取的贡献是更具体的错误归因或预算机制，
不是重复“台球＋LLM＋世界模型”口号。

本调查新增的作用是把这些文献接到同一个可执行 cue/control 问题上，指出哪些部件
能借、哪些标签/权限不足，以及现有 0/200 还没有回答什么。它不预先保证任何方法收益
或会议录用。

## 参考文献

文内编号按首次出现排序；来源链接见对应表格和段落。正式 venue 未核实的项保留
arXiv，不能把它们解读为匿名会议录用。

[1] Xinqing Li, Xin He, Le Zhang, et al., “A comprehensive survey on world models for embodied AI,” arXiv:2510.16732, 2025.

[2] Xiaoxiao Long, Qingrui Zhao, Kaiwen Zhang, et al., “A survey: Learning embodied intelligence from physical simulators and world models,” arXiv:2507.00917, 2025.

[3] Gaoyue Zhou, Hengkai Pan, Yann LeCun, et al., “DINO-WM: World models on pre-trained visual features enable zero-shot planning,” arXiv:2411.04983, 2024.

[4] Sean Memery, Kevin Denamganai, Jiaxin Zhang, et al., “CueTip: An interactive and explainable physics-aware pool assistant,” SIGGRAPH Conference Papers, 2025.

[5] Kangrui Wang, Pingyue Zhang, Zihan Wang, et al., “VAGEN: Reinforcing world model reasoning for multi-turn VLM agents,” NeurIPS, 2025.

[6] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, et al., “Mastering diverse control tasks through world models,” Nature, 2025.

[7] Nicklas Hansen, Hao Su, Xiaolong Wang, “TD-MPC2: Scalable, robust world models for continuous control,” ICLR, 2024.

[8] Mido Assran, Adrien Bardes, David Fan, et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv:2506.09985, 2025.

[9] Vlad Sobal, Wancong Zhang, Kyunghyun Cho, et al., “Learning from reward-free offline data: A case for planning with latent dynamics models,” NeurIPS, 2025.

[10] Quentin Garrido, Tushar Nagarajan, Basile Terver, et al., “Learning latent action world models in the wild,” arXiv:2601.05230, 2026.

[11] Boyuan Chen, Diego Marti Monso, Yilun Du, et al., “Diffusion Forcing: Next-token prediction meets full-sequence diffusion,” NeurIPS, 2024.

[12] Basile Terver, Tsung-Yen Yang, Jean Ponce, et al., “What drives success in physical planning with joint-embedding predictive world models?” TMLR, 2026.

[13] Yu Shang, Zhuohang Li, Yiding Ma, et al., “WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models,” arXiv:2602.08971, 2026.

[14] Ziyi Wu, Nikita Dvornik, Klaus Greff, et al., “SlotFormer: Unsupervised visual dynamics simulation with object-centric models,” ICLR, 2023.

[15] Angel Villar-Corrales, Ismail Wahdan, Sven Behnke, “Object-centric video prediction via decoupling of object dynamics and interactions,” ICIP, 2023.

[16] Chia-Hsiang Kao, Cong Phuoc Huynh, Chien-Yi Wang, et al., “Dynamics: Language-based representation for inferring rigid-body dynamics from videos,” CVPR, 2026.

[17] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, et al., “PAC-NeRF: Physics augmented continuum neural radiance fields for geometry-agnostic system identification,” ICLR, 2023.

[18] Tianyi Xie, Zeshun Zong, Yuxing Qiu, et al., “PhysGaussian: Physics-integrated 3D Gaussians for generative dynamics,” CVPR, 2024.

[19] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, et al., “PhysGen: Rigid-body physics-grounded image-to-video generation,” ECCV, 2024.

[20] Chen Yang, Shenxiang Zeng, Haoyang Zhao, et al., “PhysMind: From video to executable worlds for training-free physical reasoning,” arXiv:2608.04575, 2026.

[21] Anoop Cherian, Radu Corcodel, Siddarth Jain, et al., “LLMPhy: Parameter-identifiable physical reasoning combining large language models and physics engines,” AISTATS, 2026.

[22] Jiarong Liang, Max Ku, Ka-Hei Hui, et al., “VisPhyWorld: Probing physical reasoning via code-driven video reconstruction,” arXiv:2602.13294, 2026.

[23] Sean Memery, Mirella Lapata, Kartic Subr, “SimLM: Can language models infer parameters of physical systems?” arXiv:2312.14215, 2023.

[24] Qirui Li, Jinkun Hao, Yibo Li, et al., “PhysAgent: Reflective agentic physics control for physically plausible video generation,” arXiv:2607.16355, 2026.

[25] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, et al., “ThinkAct: Vision-language-action reasoning via reinforced visual latent planning,” NeurIPS, 2025.

[26] Qingqing Zhao, Yao Lu, Moo Jin Kim, et al., “CoT-VLA: Visual chain-of-thought reasoning for vision-language-action models,” CVPR, 2025.

[27] Bingyi Kang, Yang Yue, Rui Lu, et al., “How far is video generation from world model: A physical law perspective,” ICML, 2025.

[28] Wei Chow, Jiageng Mao, Boyi Li, et al., “PhysBench: Benchmarking and enhancing vision-language models for physical world understanding,” ICLR, 2025.

[29] Ben Wang, Xiaogang Li, Ruochen Gao, et al., “BilliardPhys-Bench: Benchmarking physical reasoning and visual dynamics of multimodal LLMs,” arXiv:2605.30900, 2026.

[30] Hritik Bansal, Zongyu Lin, Tianyi Xie, et al., “VideoPhy: Evaluating physical commonsense for video generation,” ICLR, 2025.

[31] Hritik Bansal, Clark Peng, Yonatan Bitton, et al., “VideoPhy-2: A challenging action-centric physical commonsense evaluation in video generation,” ICLR, 2026.

[32] Haoyi Duan, Hong-Xing Yu, Sirui Chen, et al., “WorldScore: A unified evaluation benchmark for world generation,” ICCV, 2025.
