# 多模态物理世界模型中的递归自我进化（RSI）与技能调用后训练方案（CAPO-RSI）

> **定位**：面向 ICML 2027（摘要截稿 2027-01-14，全文截稿 2027-01-22），在已完成无泄漏 SFT 预热与 `CAPO` 因果门控强化学习框架的基础上，将 **递归自我进化（Recursive Self-Improvement, RSI）** 与 **多技能智能体后训练（Skill-Invoking Agentic Post-Training）** 有机融入 `InverseR-VLM` 的 2D-to-3D 连续物理仿真闭环，形成既有前沿高度、又可在 4×A100 上低风险落地的完整故事线与实验方案。

---

## 一、先读懂当前 RSI 与 Skill-RL 文献大趋势（2024–2026 已核查文献综述）

近年来（尤其是 NeurIPS 2025、ICLR 2026 主会及 **ICLR 2026 Workshop on AI with Recursive Self-Improvement**，以及 2026 年涌现的一批 Agentic Skill-RL 工作），“模型自我进化（RSI）”已经从早期的概念炒作分化为 **四条清晰的技术路线**。通读这些文献可以提炼出一个最核心的行业共识：**没有可靠外部验证器（Verifier）的自我进化必然坍缩；而“可执行环境验证 + 技能库与策略协同进化”正在成为新一代主流范式。**

### 1. 路线一：自指代码与脚手架演化（Self-Referential Code & Scaffold Evolution）
- **代表工作**：
  - **STOP**（Zelikman et al., COLM 2024, `arXiv:2310.02304`）：用语言模型迭代改写自身的优化脚手架代码，但底层模型权重固定不变。
  - **ADAS**（Hu et al., ICLR 2025, `arXiv:2408.08435`）与 **AlphaEvolve**（Novikov et al., 2025, `arXiv:2506.13131`）：用评估器反馈在代码空间搜索更优智能体结构或算法。
  - **Gödel Agent**（Yin et al., ACL 2025, `arXiv:2410.04444`）与 **Darwin Gödel Machine**（Zhang et al., ICLR 2026, `arXiv:2505.22954`）：让编程智能体直接修改自身运行逻辑并保留多样性演化档案。
- **趋势判断与局限**：这类方法改的是**外部代码壳子（Scaffold）**，不更新 VLM 的视觉-物理模型参数；此外 ICLR 2026 RSI Workshop 的最新审视指出，若代理评测设计不当，代码自演化极易触发指标投机（**Reward Hacking in Self-Improving Code Agents**, Zhao et al., ICLR 2026 RSI Workshop），有时甚至拉不开与简单采样基线的差距（Gideoni et al., ICLR 2026 RSI Workshop, `arXiv:2602.16805`）。

### 2. 路线二：无外部验证器的“自打分”迭代 vs 可验证器锚定的自博弈课程（Self-Play & Curriculum RSI）
- **反面教训（纯自打分必遇瓶颈甚至坍缩）**：
  - **Self-Rewarding LMs**（Yuan et al., ICML 2024, `arXiv:2401.10020`）与 **Meta-Rewarding LMs**（Wu et al., EMNLP 2025）让模型自己给自己当裁判；**Shumailov et al.**（*Nature* 2024, `arXiv:2305.17493`）证明纯靠模型递归消费自身生成分布会导致尾部丢失与 **Model Collapse**；**Stechly et al.**（ICLR 2025, `arXiv:2402.08115`）也证明无外部健全验证器时 LLM 自检常把对的改错。
  - 极具警示意义的是 **R-Zero**（Huang et al., ICLR 2026, `arXiv:2508.05004`）：它让 Challenger 出题、Solver 用**多数投票（Majority Vote）** 当伪标签做自进化，作者在论文中诚实报告：随着迭代轮数增加，伪标签准确率下降，后期出现性能退化！
- **正面突破（外部可执行验证器破除坍缩）**：
  - **Yi et al.**（ICLR 2026 RSI Workshop, `arXiv:2510.16657`）从理论与实验证明：只要在递归训练回路中引入**可靠的外部验证器（External Verifier）** 过滤合成轨迹，就能避免模型坍缩并向验证器的真实知识中心收敛。
  - **Absolute Zero / AZR**（Zhao et al., NeurIPS 2025, `arXiv:2505.03335`）：单一模型同时扮演 **Proposer（出题者）** 与 **Solver（解题者）**，以 Python 解释器作为零幻觉验证器，专挑“不太难也不太简单”的能力边界题实现自我强化。
  - 多模态方向的跟进包括 **Vision-Zero**（Wang et al., ICLR 2026, `arXiv:2509.25541`）、**Agent0-VL**（Liu et al., ICLR 2026 RSI Workshop, `arXiv:2511.19900`）与 **MM-Zero**（Li et al., 2026, `arXiv:2603.09206`），但它们在困难视觉题上仍部分依赖模型自投票或自评判，缺少连续物理定律级别的绝对真值裁判。

### 3. 路线三：Agentic Post-Training 中的技能调用、演化与内化（导师点出的最前沿风口！）
早期技能库工作如 **Voyager**（Wang et al., TMLR 2024, `arXiv:2305.16291`）、**TroVE**（Wang et al., ICML 2024, `arXiv:2401.12869`）、**LATM**（Cai et al., ICLR 2024, `arXiv:2305.17126`）和 **SkillWeaver**（Zheng et al., 2025, `arXiv:2504.07079`）只把技能攒在外部提示词或脚本库里，**完全不训练底层模型参数**。而 2025–2026 年爆发的最新趋势，正是导师所说的 **“在 RL Post-Training 过程中让 Agent 学会调用、演化并内化不同 Skill 来实现 RSI”**：
- **多轮工具调用 RL 基础**：**ReTool**（Feng et al., ICLR 2026, `arXiv:2504.11536`）、**ToolRL**（Qian et al., NeurIPS 2025, `arXiv:2504.13958`）、**GiGPO**（Feng et al., NeurIPS 2025, `arXiv:2505.10978`）与 **ARPO**（Dong et al., ICLR 2026, `arXiv:2507.19849`）解决了多轮工具交互中的奖励设计与步骤级信用分配（Credit Assignment）。
- **技能库 × 策略协同进化（2026 核心浪潮）**：
  - **SkillRL**（Xia et al., 2026, `arXiv:2602.08234`）：把成功与失败轨迹蒸馏成层级技能库，再用 GRPO 训练基于技能的条件策略，并根据验证集失败案例递归精炼技能库。
  - **SKILL0**（Lu et al., 2026, `arXiv:2604.02268`）：提出 **技能内化（Skill Internalization）**——训练初期给模型提供技能上下文辅助拿高分，随后按掌握程度**逐步撤掉外部技能**，把外部技能硬生生“蒸馏进模型自身权重”！
  - **Skill1**（Shi et al., 2026, `arXiv:2605.06130`）、**ReSkill**（He et al., 2026, `arXiv:2606.01619`）、**RSIPO**（Zhang et al., 2026, `arXiv:2610.15772`）与 **TrackRL**（Zhu et al., 2026, `arXiv:2610.25126`）：让同一策略在 GRPO 组内对比“调某技能 vs 不调该技能”的真实收益，淘汰无效技能、保留真正涨点的技能，实现 **Skill–Policy Co-Evolution**。
- **必须警惕的坑——工具滥用幻觉**：**Zeng et al.**（2026, `arXiv:2604.19749`, *The Tool-Overuse Illusion*）发现，若只给结果奖励而不计调用代价，RL 会把模型训成“连最简单算术也无脑调工具的懒汉”。因此技能调用必须绑定**效率成本或渐进内化机制**。

### 4. 路线四：物理仿真器驱动的具身/多模态自我进化（离我们最近的赛道空白）
- **Sim2Reason**（Prabhudesai et al., ICML 2026, `arXiv:2604.11805`）：用物理引擎随机生成题目训练 LLM 解物理奥赛题，但**纯文本无视觉、无交互控制**。
- **PhysEvolve**（Hu et al., 2026-10, `arXiv:2610.34921`）：让 VLM 提议可执行物理场景做自演化 RL，但**只做被动看图问答（VQA）**，没有连续动作控制、没有未知摩擦主动试探，也没有技能调用机制。
- **PEAR**（Tang et al., 2026-11, `arXiv:2611.18562`）：在仿真中训练推理模型操控真实机械臂，但采用单一二值成败奖励，**没有显式因果世界模型门控，也没有递归技能演化与自主难例生成**。

---

## 二、核心洞察：为什么我们的 2D-to-3D 台球物理引擎是做 RSI 的“天选实验场”？

把上面的文献拼图合在一起，你会发现前人做 RSI 最头疼的两个死穴，在我们的 `InverseR-VLM`（基于 `Pooltool` 刚体接触与旋转摩擦物理引擎）里**天然迎刃而解**：

1. **牛顿力学是绝对无法“钻空子（Reward Hacking）”的完美验证器**  
   代码生成 RSI 里模型会篡改单元测试，数学自博弈（如 `R-Zero`）里多数投票会集体算错；而在 `InverseR-VLM` 中，弹性碰撞、库边反弹、滑动转纯滚动摩擦与高低杆/左右塞三维旋转偏转全由确定性物理方程裁决，再加上 `CAPO` 的**乘法因果门控（出杆结果 × 过程事件预测准确度）**，连“瞎蒙进袋”都被彻底堵死——因此多轮递归自训练**不会发生 Model Collapse**。

2. **物理推理与未知环境控制天然具有“模块化技能结构（Compositional Physical Skills）”**  
   人类台球高手面对复杂球局或陌生球桌时，绝不是凭感觉瞎抡一杆，而是按需组合调用一套**物理微技能（Skills）**：
   - 看球位切角难算时 $\rightarrow$ 调 **假想球几何瞄准技能（Ghost-Ball Geometry Skill）**；
   - 到了新球桌不知道桌布滑不滑（`T5`）时 $\rightarrow$ 调 **低能量诊断试杆技能（Friction Probe Skill）** 反推摩擦系数；
   - 担心母球碰库洗袋或路线被挡时 $\rightarrow$ 调 **心理前向推演自检技能（Mental Rollout Check Skill）** 验证碰撞因果链；
   - 练球遇到反复打丢的薄球/塞球时 $\rightarrow$ 针对该失误专门摆球加练（**反事实难例自生成技能，Counterfactual Drill Proposer**）。

> **一句话讲清我们的创新定位（对比 `SkillRL`、`PhysEvolve`、`PEAR`）**：  
> 现有 `SkillRL / RSIPO` 只在离散文本/网页里演化文字提示技能，不懂连续 3D 物理；现有 `PhysEvolve` 只用物理引擎出被动问答题，不动手控制也不调技能。我们提出 **`CAPO-RSI`**：**首个以确定性刚体物理引擎为防坍缩验证器，让多模态模型在强化学习后训练中自主学会“按需路由调用物理技能 $\rightarrow$ 从失败轨迹中演化新技能与边界难例 $\rightarrow$ 将外部技能逐步内化进自身权重”的视觉物理递归自我进化框架。**

---

## 三、`CAPO-RSI` 具体怎么做？（通俗易懂的三层架构设计）

为了既体现导师强调的 **“在 Post-Training 中调用不同 Skill 强化自己实现 RSI”**，又不把工程复杂度搞失控（确保稳赶 1 月底 ICML 2027），我们将系统设计为建立在现有 `capo_env.py` + `capo_grpo.py` 之上的 **“四大可调用物理技能 + 双闭环递归迭代”** 架构：

### 第 1 层：智能体在 Post-Training 中可自主选择调用的【物理技能池（Skill Library）】

在多轮强化学习轨迹中，模型在 `<think>...</think>` 推理后，不再只有单一的盲打动作，而是可以根据当前题目难度与不确定性，从以下 **4 类技能** 中按需调用（每次调用计入统一的交互预算 `max_calls`，避免刷量作弊）：

| 技能名称 (`skill_name`) | 技能类型 | 输入与执行逻辑 | 解决什么痛点？（模型何时该调它） |
| :--- | :--- | :--- | :--- |
| **`probe_table_physics`**（桌面摩擦探针技能） | 环境交互技能（消耗 1 次试杆预算） | 对当前球局打出受控试探杆，返回试打后双视角新图像 + 球的实际滑行距离摘要 | 专治 **`T5` 未知摩擦/库边弹性桌面**：用 1 杆探针算出当前桌布是 `slow`（涩）还是 `fast`（滑），避免盲打 |
| **`solve_ghost_geometry`**（假想球几何解算技能） | 视觉-几何辅助技能（训练期可退火） | 根据模型从 2D 图中目测出的母球与目标球粗略坐标，解算球心相距 $2R$ 时的接触点切角与入射角参考范围 | 专治 **多球大角度薄切与库边折线球**：帮 9B 模型跨越早期 2D→3D 三角函数算不准的冷启动门槛 |
| **`verify_causal_rollout`**（心理世界模型预演技能） | 自检技能（不碰真实未知球桌） | 在标准物理假设下快速核验候选出杆的首碰球与洗袋风险，输出因果冲突告警 | 专治 **多球遮挡与母球洗袋风险**：出正式杆前先排除致命失误（对应人类出杆前的“预演瞄准”） |
| **`tactical_physics_card`**（可演化的物理战术经验卡） | 递归蒸馏文本技能（随 RSI 轮次生长） | 按当前视觉球局特征（如“近库强低杆”“高速右塞投掷偏转”）检索由上一轮失败轨迹提炼出的补偿准则 | 专治 **旋转偏转（Throw / Spin Deflection）与二次碰库系统性偏差**：承接上一轮自我复盘总结出的物理修正规律 |

### 第 2 层：防偷懒、防幻觉的【因果门控 + 技能效率奖励函数】

在 `CAPO` 原有乘法因果门控的基础上，加入对技能调用策略的精准激励（呼应 `ToolRL` 与 `The Tool-Overuse Illusion` 的发现）：

$$R_{\text{CAPO-RSI}}(\tau) = \underbrace{R_{\text{goal}}(\text{final\_shot})}_{\text{终局到位/进袋得分}} \times \underbrace{G_{\text{causal}}(\hat{y}_{T1,T2,T3}, y^*)}_{\text{物理过程预测因果门控}} - \underbrace{\lambda_{\text{skill}} \sum_{t} c(\text{skill}_t)}_{\text{非必要技能调用微惩罚}} + \underbrace{\beta_{\text{info}} \cdot \mathbb{I}[\text{有效探针改进}]}_{\text{未知环境有效辨识奖励}}$$

- **为什么这样设计极妙？**
  1. **简单题自动戒掉拐杖**：在单球直线自由滚动（`free_roll`）上，模型自己心算就能打进，乱调技能反而扣 $\lambda_{\text{skill}}$，于是 RL 自动学会 **“简单题零调用直接一杆进”**；
  2. **困难题/未知摩擦题精准调技能**：在 `T5` 未知摩擦或复杂高低杆双胞胎球局上，不调 `probe_table_physics` 根本拿不到 $R_{\text{goal}}$，调用技能带来的巨额收益远超微小的 $\lambda_{\text{skill}}$，于是 RL 自动学会 **“看不清摩擦时先调探针技能、角度复杂时调几何/预演技能”**；
  3. **因果门控防止瞎蒙**：哪怕调了技能，最终出杆时依然必须准确预测 `T1` 首碰球、`T2` 库边与 `T3` 停球位置，确保模型真正读懂了技能返回的信息。

### 第 3 层：三轮闭环【递归自我进化（RSI Outer Loop）】

整个训练不再是“跑完一遍静态数据集就结束”，而是以当前刚训完的 **400 步无泄漏 SFT 模型作为 `Round 0` 种子**，运行 $k = 1 \to 2 \to 3$ 的三步闭环自我进化：

```text
          ┌─────────────────────────────────────────────────────────────────┐
          │                    Pooltool 确定性刚体物理引擎                   │
          │          （零幻觉、零作弊的牛顿力学验证器 External Verifier）      │
          └───────────────▲─────────────────────────────────┬───────────────┘
                          │ 验证题目可解性 & 物理因果奖励      │ 返回真实碰撞/停球图像
                          │                                 ▼
  ┌───────────────────────┴────────┐               ┌────────────────────────┐
  │ Loop B: 边界难例自生成          │               │ 策略模型 π_k (Qwen-9B) │
  │ (Frontier Scene Challenger)    │◄── 失败诊断 ───│ + 物理技能路由调用     │
  │ 专挑 0 < pass@8 ≤ 0.5 的薄弱球局 │               │ (CAPO GRPO Post-Train) │
  │ 做反事实旋转/速度/摩擦变异出新题 │               └───────────┬────────────┘
  └────────────────────────────────┘                           │
                                                               ▼
                                          ┌─────────────────────────────────┐
                                          │ Loop A: 技能蒸馏与渐进内化        │
                                          │ 1. 归纳高方差轨迹 → 更新战术技能卡│
                                          │ 2. TrackRL 组内对照剔除无效技能   │
                                          │ 3. SKILL0 渐进遮蔽外部辅助技能，  │
                                          │    把物理直觉内化进 π_{k+1} 权重！│
                                          └─────────────────────────────────┘
```

每一轮递归（`Round k -> Round k+1`）同时发生两件真正的“自我进化”：
1. **向上生长技能 + 向内吸收权重（`SkillRL` + `TrackRL` + `SKILL0` 融合）**：
   - 收集上一轮 GRPO 中“调了某技能就成功、不调就失败”（组内方差最大）的轨迹，提炼/更新最多 8 条精炼的物理补偿战术卡（用组内随机注入对照 $\Delta \hat{A} > 0$ 做硬筛选，不涨点的技能直接淘汰）；
   - 同时按轮次提高“无外部几何辅助技能”的裸考采样比例（例如 `Round 1: 30%` $\to$ `Round 2: 65%` $\to$ `Round 3: 100%` 独立推理，仅保留 `T5` 物理上必需的环境试探杆 `probe_table_physics`），逼迫 VLM 把借由技能学会的三维碰撞与旋转规律**彻底内化进自己的视觉-语言参数里**！
2. **向能力边界自出题（`AZR` + `PhysEvolve` 物理版）**：
   - 统计 `Round k` 在四类物理族（`roll / center / vert / side`）及不同摩擦系数上的 `pass@8`；
   - 对全对（`pass@8 = 1`，已学会）的场景降权，专门抽取 `0 < pass@8 <= 0.375`（模型似懂非懂、刚好卡住）的失误球局，自动对出杆旋转、球距、或隐藏摩擦系数做扰动，经 `Pooltool` 验算筛出无洗袋、反事实位移 $\ge 2R$ 的合法新题，并混入 30% 原始训练集锚点（遵循 Gerstgrasser et al., `arXiv:2404.01413` 的真实+合成累积原则，100% 杜绝遗忘），喂给 `Round k+1`！

---

## 四、面向 ICML 2027 的分阶段落地计划（算力可控、不拖工期！）

为了保证 **1 月中旬前稳稳拿到全部主表和消融曲线**，我们绝不推倒重写现有代码，而是把当前已跑通的进度直接作为第 0–1 阶段，增量叠加 `CAPO-RSI`（仅需在 `src/inverser/capo/` 新增约 150–200 行轻量代码）：

| 阶段 | 对应模型与对比组 | 核心验证目标（对应论文哪张表/图） | 4×A100 预计耗时 | 当前状态 |
| :--- | :--- | :--- | :---: | :---: |
| **Stage 0：地基校准** | `Qwen3.5-9B` Base vs `Round 0 (400-step SFT)` | 验证无泄漏 `<think>+<answer>` SFT 打通格式与基础 2D→3D 空间感 | 已完成（91 min） | **SFT 已完成并合并，Valid 评测运行中** |
| **Stage 1：核心因果消融** | `Outcome-GRPO` vs `Uncoupled-GRPO` vs `CAPO (Round 1)` | 证明**乘法因果门控**在相同时长下消除“蒙答案”、同步提升 `T1–T3` 与 `T4/T5`（论文 **Table 1 主消融**） | ~12–16 小时 | 代码与冒烟已就绪 |
| **Stage 2：多技能后训练与递归自进化（RSI 核心新增）** | `CAPO (Round 1)` $\to$ `CAPO-RSI (Round 2)` $\to$ `CAPO-RSI (Round 3)` | 展示导师强调的核心亮点：<br>1. **Figure 3**：随 RSI 轮次推进，`T4/T5` 成功率与 `S_total` 单调爬升；<br>2. **Figure 4**：简单题技能调用率自动下降、未知摩擦题探针调用精准度上升，且**撤掉外部辅助技能后的裸考分数持续上涨（证明真正内化进权重！）**；<br>3. **消融对照**：去掉外部物理验证器改用自投票伪标签（复现 `R-Zero` 式退化），反证物理引擎验证器对防坍缩的决定性作用 | ~14–18 小时（两轮各 80–100 步 GRPO） | 待 Stage 1 后无缝衔接 |
| **Stage 3：盲测与跨基准泛化** | 冻结最优检查点，跑 `inverser-vlm-v1/test` + `PhysBench` + `DeepPHY (I-PHYRE)` + `CueTip / DeepPHY-WM / Self-Verify` 基线 + 可选 `Qwen3-VL-32B` 跨尺寸对比 | 证明 **9B 经 `CAPO-RSI` 自进化后超越未做因果自进化的 32B 大模型与推理期脚手架基线**，并在外部物理基准零样本涨点（论文 **Table 2 & Figure 5**） | ~4–6 小时（4 卡并发 vLLM） | 评测并发器已就绪 |

---

## 五、论文叙事升级对比（你可以直接拿去跟 Mentor 对齐！）

- **升级前叙事**：我们在台球仿真器上提出了 `InverseR-VLM` 基准，并设计了带因果门控奖励的 `CAPO` 强化学习算法来提升物理预测与击球控制。*(扎实，但容易被审稿人觉得只是特定环境上的 Reward Shaping。)*
- **引入 RSI + Skill Post-Training 后的新叙事**：
  > **“多模态大模型能否在没有人类无限标注的情况下，通过与物理世界交互、调用并内化物理技能来实现持续自我进化（Physical RSI）？”**  
  > 我们揭示了纯自评判 RSI 在物理推理中必然因因果幻觉而坍缩（呼应 `R-Zero` 与 ICLR 2026 RSI Workshop 发现），进而提出 **`CAPO-RSI`**：以确定性 2D-to-3D 物理引擎为不可作弊的因果裁判，让模型在 Agentic Post-Training 中自主学习**按需调用诊断探针、几何解算与心理预演技能**，通过**乘法因果门控**对齐“物理预测”与“连续控制”，并在多轮递归中**自动生成自身能力边界上的反事实物理难例、蒸馏失误技能并逐步将外部技能内化进模型参数**。最终仅用 `9B` 模型通过三轮自我进化，在未知摩擦在线适应与因果世界模型预测上全面击败推理期复杂智能体脚手架（`CueTip`、`DeepPHY-WM`）与更大规模静态视觉模型。

---

## 六、已核查核心参考文献清单（全部经真实检索验证，可直接用于论文 Related Work）

1. **Zelikman et al.** (COLM 2024). *Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation*. `arXiv:2310.02304`.
2. **Shumailov et al.** (*Nature* 2024). *AI models collapse when trained on recursively generated data*. `arXiv:2305.17493`.
3. **Gerstgrasser et al.** (2024). *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. `arXiv:2404.01413`.
4. **Yuan et al.** (ICML 2024). *Self-Rewarding Language Models*. `arXiv:2401.10020`.
5. **Wang et al.** (TMLR 2024). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. `arXiv:2305.16291`.
6. **Wang et al.** (ICML 2024). *TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks*. `arXiv:2401.12869`.
7. **Stechly et al.** (ICLR 2025). *On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks*. `arXiv:2402.08115`.
8. **Yin et al.** (ACL 2025). *Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement*. `arXiv:2410.04444`.
9. **Zhang et al.** (ICLR 2026). *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*. `arXiv:2505.22954`.
10. **Zhao et al.** (NeurIPS 2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. `arXiv:2505.03335`.
11. **Zhou et al.** (NeurIPS 2025). *Self-Challenging Language Model Agents*. `arXiv:2506.01716`.
12. **Huang et al.** (ICLR 2026). *R-Zero: Self-Evolving Reasoning LLM from Zero Data*. `arXiv:2508.05004`.
13. **Yi et al.** (ICLR 2026 RSI Workshop). *Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence*. `arXiv:2510.16657`.
14. **Liu et al.** (ICLR 2026 RSI Workshop). *Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning*. `arXiv:2511.19900`.
15. **Wang et al.** (ICLR 2026). *Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play*. `arXiv:2509.25541`.
16. **Feng et al.** (ICLR 2026). *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*. `arXiv:2504.11536`.
17. **Qian et al.** (NeurIPS 2025). *ToolRL: Reward is All Tool Learning Needs*. `arXiv:2504.13958`.
18. **Feng et al.** (NeurIPS 2025). *Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)*. `arXiv:2505.10978`.
19. **Dong et al.** (ICLR 2026). *Agentic Reinforced Policy Optimization (ARPO)*. `arXiv:2507.19849`.
20. **Zeng et al.** (2026). *The Tool-Overuse Illusion: Why Does LLM Prefer External Tools over Internal Knowledge?*. `arXiv:2604.19749`.
21. **Xia et al.** (2026). *SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning*. `arXiv:2602.08234`.
22. **Lu et al.** (2026). *SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization*. `arXiv:2604.02268`.
23. **Shi et al.** (2026). *Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning*. `arXiv:2605.06130`.
24. **He et al.** (2026). *ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL*. `arXiv:2606.01619`.
25. **Zhang et al.** (2026). *RSIPO: Recursive Self-Improving Policy Optimization for Agentic Reinforcement Learning via Skill-Policy Co-Evolution*. `arXiv:2610.15772`.
26. **Zhu et al.** (2026). *Trackable Skill Library Lifecycle Management for Agentic Reinforcement Learning (TrackRL)*. `arXiv:2610.25126`.
27. **Prabhudesai et al.** (ICML 2026). *Sim2Reason: Solving Physics Olympiad via Reinforcement Learning on Physics Simulators*. `arXiv:2604.11805`.
28. **Hu et al.** (2026). *PhysEvolve: Verifier-grounded Self-evolution for Physical Reasoning with Executable Worlds*. `arXiv:2610.34921`.
29. **Tang et al.** (2026). *Training Reasoning Models in Simulation to Natively Use Real Robots (PEAR)*. `arXiv:2611.18562`.
30. **Wang et al.** (NeurIPS 2025). *VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents*. `arXiv:2510.16907`.
