# InverseR 的 empirical study 参考：预测质量与规划效用怎样分开测量

这是一份计划阶段的定向文献补充，不是实验结果。此前的完整三视角调查保留在 [研究报告](world_model_research.md)。

## 1. ICLR 2027 的访问与核查边界

对用户提供的 [ICLR 2027 active submissions](https://openreview.net/group?id=ICLR.cc/2027/Conference#tab-active-submissions) 进行了公开检索。研究助手检查了物理推理、动作条件世界模型、prediction/planning utility、干预一致性及 simulator-feedback attribution 等方向。

本次普通访问无法同时核实某篇新投稿的身份、状态和方法内容：提交 API 返回 403，发现的 forum 与 PDF 入口遇到 browser verification。没有认证、绕过限制或尝试去匿名化。不能据此说 ICLR 2027 没有相关论文，也不能把索引中的标题或候选链接当成已核查参考。

因此，本轮采用可读取主要论文的 arXiv 与已发表工作。后续如得到可普通访问的主要论文，可增补比较，但不在实验中途按新论文的结果调整冻结测试。

## 2. 主参考：WorldArena 的“质量不等于用途”

WorldArena [1] 将 world model 的感知/视频质量与实际功能效用分开，而 Kang 等人的工作 [3] 用受控二维物理环境区分 ID、OOD 和组合泛化。这两种设计互补：前者问预测是否能服务任务，后者问成功来自规律还是分布相关性。

**建议以 WorldArena 的测量逻辑为主参考，不照搬其机器人算法或综合分数。**

- 主要来源：[arXiv abstract](https://arxiv.org/abs/2602.08971)、[version-pinned PDF](https://arxiv.org/pdf/2602.08971v2)。
- 已核查作者顺序：Yu Shang、Zhuohang Li、Yiding Ma，等。
- 身份：2026 年 arXiv preprint；本次未核实后续正式 venue，不称为 ICLR 2027 投稿或录用。
- 论文定位：§3.2 至 §4.1、PDF pp.5-6；规划/功能比较：§4.2.2、Figure 5、Table 5、PDF p.8。

作者报告 EWMScore 与人类评价、与 action-planner performance 的相关性分别为 0.825 和 0.360。这是被测模型条件下的跨模型关联，不是每个案例上的因果证据； inspected Table 5 只展示两个任务，不能推导普遍规律或沿用一个未经验证的相关性阈值。

对 InverseR 的迁移：

1. 把 native prediction error、contact-event correctness、candidate ranking 和参考执行的 goal success 分开。
2. 使用同一布局、相同候选、相同 planner 的配对实验，补足单看跨模型平均相关性的归因不足。
3. 记录模型预测更准但动作没有更好、以及优化放大模型误差的反例。

不能直接迁移：

- RoboTwin 中的闭环机器人干预；
- 视频经 inverse dynamics 转成动作的标签；
- 已进行 post-training 的模型与我们的 inference-only 模型比较；
- 用视频综合分数证明实际 billiards cue/spin 能力。

## 3. 辅参考：CLEVRER 的事实、预测与反事实分层

CLEVRER [2] 将描述、解释、未来预测和反事实问题分开；WorldArena [1] 则更关注预测输出的下游任务用途。前者适合定位物理事件错误，后者适合组织预测到动作的证据链。

- 主要来源：[arXiv abstract](https://arxiv.org/abs/1910.01442)、[version-pinned PDF](https://arxiv.org/pdf/1910.01442v2)。
- 已核查作者顺序：Kexin Yi、Chuang Gan、Yunzhu Li，等。
- 正式身份：ICLR 2020；主要 PDF 首页明确标为已发表会议论文。初始预印本为 2019 年，引用正式会议年份。
- 官方 OpenReview 的普通访问仍受 browser verification 限制；不猜测 forum ID。
- 任务与反事实：§3.1-3.2、PDF pp.3-5；指标与实验：§4.2、Table 3、PDF pp.6,9。

可借鉴：

- 同时保留 factual、predictive 和 counterfactual 任务，不用一项总分代替。
- 使用语言/视觉信息删除对照，检验模型是否实际依赖当前场景。
- 除边际正确率外，报告同一干预对的联合正确性。
- 将事件监督与额外 privileged 信息单独消融。

不能直接迁移：物体移除 QA 不是连续击球；event graph 标签不是已测真实 cue 控制；程序、mask 等额外监督不能在模态比较中不加说明地提供给某个方法。

## 4. 泛化参考：物理规律视角的受控 ID/OOD 测试

Kang 等人的主要论文 [3] 可作为实验构建的第三参考，尤其适合与 WorldArena [1] 的效用评价结合。

- 主要来源：[正式 PMLR 记录](https://proceedings.mlr.press/v267/kang25g.html)。
- 作者：Bingyi Kang、Yang Yue、Rui Lu，等。
- 身份：ICML 2025，PMLR 267:28991-29017。
- 受控二维物理视频测试 ID、OOD 和组合泛化；论文报告正面的 ID 与组合结果，也报告 OOD 局限，不能只引用负面部分。

对 InverseR，分别留出布局、物理组合和视觉扰动。图像外观变换必须保持语义与物理不变；真实 cue action 未测时，不把提示词修改称为有真实 GT 的干预。

## 5. 设计对照表

| 维度 | WorldArena [1] | CLEVRER [2] | Kang 等 [3] | InverseR 拟议设计 |
| --- | --- | --- | --- | --- |
| 核心问题 | 预测质量与功能用途是否一致 | 事实、未来与反事实能否区分 | 是否学到可泛化物理规律 | 原生预测是否帮助连续、非唯一解的目标规划 |
| 操控对象 | 模型、条件及数据/策略条件 | 合成事件与物体移除 | 数据分布与物理/外观组合 | 指定 cue 控制、参数及合法 nuisance |
| 下游检验 | 模拟机器人任务成功 | 事件答案 | 生成运动的物理误差 | 隐藏参考引擎对最终提交动作评分 |
| 同案例归因 | 不等于充分因果归因 | 受控合成任务 | 受控分布变化 | 同布局、同候选、同 optimizer、同反馈预算 |
| 真实证据边界 | 参考执行为模拟器 | 合成 GT | 合成 GT | 真实视频只做事实空间路径/前缀验证 |
| 需要补强 | 完整成本、重复 seed 与置信区间 | 实际动作效用、完整成本 | 预测到规划的联系 | 联合报告 calls/tokens/rollouts/成本及 grouped uncertainty |

### InverseR 自己提出的控制，不冒充参考论文已验证的结论

- Genuine feedback、self-reflection-without-new-evidence 与 shuffled feedback。
- 同一候选 bank 的 blind forward prediction、预测驱动选择与直接 goal selection。
- 已知物理、nominal physics shift、固定支持集辨识和 oracle-feedback 上界分离。
- 微小平均位置误差是否改变接触、排序和实际目标达成。
- no-spin ablation 使用对各 action space 都有合法 witness 的共同目标，避免把可达性变化误解释成推理变化。

这些控制仍需实现、测量和检验。不能在正文中预写它们产生了某项 Finding。

## 6. 最近邻与原创性边界

本补充不替代对 SimLM、CueTip 与机器人台球的最近邻比较。它们已经覆盖 simulator feedback、事件候选和 spin 控制；新的文章不能只重复这些成分。

可争取的区别是：可交换红球与多解目标评分、同实例的预测/接触/排序/规划连接、无工具与工具辅助的成本匹配归因，以及标签边界明确的真实观测验证。是否足够构成论文贡献，应由实际结果决定。

## 参考文献

[1] Yu Shang, Zhuohang Li, Yiding Ma, et al., “WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models,” arXiv:2602.08971, 2026.

[2] Kexin Yi, Chuang Gan, Yunzhu Li, et al., “CLEVRER: Collision events for video representation and reasoning,” International Conference on Learning Representations, 2020.

[3] Bingyi Kang, Yang Yue, Rui Lu, et al., “How far is video generation from world model: A physical law perspective,” International Conference on Machine Learning, 2025.
