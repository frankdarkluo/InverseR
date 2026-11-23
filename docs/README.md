# InverseR 文档导航

InverseR 已接受 G1 的有限分层证据范围，并锁定小型开发诊断面板。项目概览、数据规模、
当前进展和分阶段 TODO 见[根 README](../README.md)。新 goal 的 GPU/API 范围以
[CAPO 当前执行计划](research/capo_execution.md)为准；真实动作 GT、新 benchmark 冻结、
贡献判断和盲测 gate 不会因该授权自动通过。

## Research：研究路线与文献

- [多模态物理世界模型中的 RSI 与技能调用后训练方案（CAPO-RSI）](research/rsi_skill_post_training_plan.md)：
  系统梳理 2024–2026 年 RSI（自指代码演化、无验证器坍缩 vs 外部验证器自博弈）与 Agentic Skill-RL
  （SkillRL、SKILL0、ReSkill、RSIPO、TrackRL、PhysEvolve 等 30 篇已核查文献），提出基于 2D-to-3D
  物理引擎防坍缩验证、四大可调用物理技能、乘法因果门控与技能渐进内化的 `CAPO-RSI` 三轮闭环方案。
  [Stage 2 技能规格（v0.2 草案）](research/capo_rsi_skill_spec.md)按 2026-11-20 的数据与源码核查修订技能池、红线与待拍板项。

- [当前 InverseR-VLM / CAPO 执行计划](research/capo_execution.md)：八阶段验收、
  GPU/API 路线、模型实际可用性、待验证假设和分区/盲测边界；
  [研究者原始计划](research/capo_requested_plan.md)完整保留。
  [本地运行与验收](research/capo_runtime.md)记录实际 GPU smoke、代理失败与外部资产边界。

- [Table 1 Baseline 填表计划](research/table1_baselines_plan.md)：论文 Table 1 各行的输入/输出、
  valid 初步结果、GPU 排队与指标口径。

- [训练路线协议](research/training_protocol.md)：后训练阶梯 × 能力链的预注册——训练集与 hold-out、
  单轮环境与奖励、T0–T5 六臂、H1–H4 假设、本地 vLLM 评测路线和 11/15、12/1、12/15 决策点。

- [论文主线与引言设计](research/paper_framing.md)：主线"知道会发生什么不等于知道该怎么做"，
  同实例能力链 L1–L5，逆向推理为主打任务；六段式引言、两种发现写法、Table 1 草稿与待补清单。

- [正式测试：五个模型的能力链结果](research/formal_model_sweep.md)：GPT-5.5/5.4、o4-mini（low）与
  Kimi-K2.6、DeepSeek-V4-Pro（none）；直接出杆 ≤1/200（几何 77/200），排序不优于直线近似，只有 GPT
  旗舰能利用真实反馈（36/64、29/64）；统一重算、事故与事后诊断。

- [正式测试：GPT-5.5 结果](research/formal_gpt55_results.md)：后果预测 63% 优于直线近似但 2r 内
  仅 2%，排序高于随机但不优于直线近似，直接出杆 2/200；反馈利用 genuine 36/64，比同预算几何＋Powell 多
  15.6 个百分点。

- [正式测试集协议与预注册分析](research/formal_test_protocol.md)：新种子生成、与开发数据零重叠的
  200 个布局；forward v2、强干预、候选完整排序；5 项预注册主要分析与请求量。

- [InverseR-VLM v1 数据与评分协议](research/capo_vlm_protocol.md)：T1–T5 视觉多模态合同、
  train/holdout/test 三分区生成与验证协议。

- [2023–2026 世界模型方法与迁移](research/world_model_methods_2023_2026.md)：32 篇核查
  文献；动作条件预测、对象状态、可执行物理和多模态规划；借用模块、许可与新颖性边界。

- [研究计划与实验方案](research/plan.md)：当前执行方案见 Section 13；保留此前的决策、
  工程记录与研究边界。
- [Empirical study 参考](research/empirical_study_references.md)：WorldArena、
  CLEVRER、受控物理泛化实验的借鉴点及检索边界。
- [世界模型研究报告](research/world_model_research.md)：能力分解、已核查文献、
  数据分层、研究问题、baseline 与证据边界。

## Benchmark：任务与评价

- [Inference-study 准备与 mock dry-run](benchmark/study_preparation.md)：安全视图、
  等比例渲染、物理时刻标签、经典搜索、诊断指标、可恢复账本及真实映射验证边界。

- [开发 pilot 协议](benchmark/benchmark_protocol.md)：任务契约、模拟器、
  public/privileged 视图、重放与命令；当前不是冻结 benchmark。

## 本地材料与生成结果

文档中的 `data/...` 和 `artifacts/...` 路径指向可复现的本地运行结果；这些目录除说明文件外
均被 Git 忽略，因此 GitHub 页面不会包含原始媒体、转换数据或运行产物。第三方论文 PDF
和原始讨论材料同样保留在本地，不由本仓库重新分发。文档中的 shell 命令默认从项目根目录执行。
