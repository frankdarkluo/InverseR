# InverseR 文档导航

InverseR 已接受 G1 的有限分层证据范围，并锁定小型开发诊断面板。项目概览、数据规模、
当前进展和分阶段 TODO 见[根 README](../README.md)。这不代表 GPU/API 额度已确认、
真实动作 GT 已合格，或已通过正式 benchmark 冻结、贡献选择和评价 gate。

## Research：研究路线与文献

- [Azure GPT-5 原生开发对照](research/azure_native_pilot.md)：实际 24 请求，保留合同
  失败、预测/执行指标和 Token 账本；两个布局的工程结果，不是正式排名。

- [小型开发诊断面板](research/model_diagnostic_panel.md)：已锁定 64 仿真布局＋64 保来源组
  真实 case、配对动作/输入、隐藏标签、48 请求 API 共同子集；尚未调用模型。

- [世界模型数据衔接](research/world_model_data_bridge.md)：区分受控动作预测、预测到规划
  和真实事实前缀预测；明确下一阶段模型实验，而非无限优化 tracking。

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

## Data：数据来源、标签与目录

- [Billiards Sports Analytics 已获取数据](data/billiards_sports_analytics.md)：布局与真实
  timed trajectory 的实际规模、单位/时钟审计和辅助九球边界。

- [数据卡](data/data_card.md)：实际规模、版本、许可及标签局限。
- [真实观察 schema](data/real_data_schema.md)：BEV、观测、缺失、时间与来源字段。
- [跨来源研究目录](data/research_corpus_catalog.md)：真实观察、辅助来源与仿真真值
  的引用/能力分层，避免将目录行数误作物理 ground truth。
- [Hugging Face 来源](data/huggingface_source.md)：本地快照、组织方式及媒体权利边界。
- [数据访问与许可](data/data_access_requests.md)：访问记录、许可问题和未发送的请求草稿。

## Pipelines：预处理与质量筛选

- [无共享背景 cue/gap 实测](pipelines/cue_gap_qc.md)：母球可见像素恢复、近邻前景合并和
  缺帧关联，作为世界模型实验的观测条件。

- [语义 QC 实测消融](pipelines/semantic_qc_pilot.md)：16 个原始录制的颜色、背景与
  短缺帧关联对照表，包含负面结果、空间表示诊断及备选来源核查。

- [Native canvas 审计](pipelines/native_canvas_audit.md)：逐录制像素对齐、冻结映射后的
  点兼容性与颜色/库存冲突；不把空间覆盖当作物理 GT。

- [视频处理](pipelines/video_processing.md)：采集、标定、原图检测及候选导出。
- [质量筛选](pipelines/quality_screening.md)：自动 triage、疑似重复、复核与修复队列。
- [作者直接上传](pipelines/author_direct_uploads.md)：不依赖不完整 ZIP 的来源读取与转换。
- [增量 snooker 管线](pipelines/incremental_snooker_pipeline.md)：内容去重、缓存、
  版本化转换、人工 raster 审计与可复现实验输入。

## 本地材料与生成结果

文档中的 `data/...` 和 `artifacts/...` 路径指向可复现的本地运行结果；这些目录除说明文件外
均被 Git 忽略，因此 GitHub 页面不会包含原始媒体、转换数据或运行产物。第三方论文 PDF
和原始讨论材料同样保留在本地，不由本仓库重新分发。文档中的 shell 命令默认从项目根目录执行。
