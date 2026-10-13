# InverseR 文档导航

InverseR 当前处于 pre-G1 数据质量与实验基础设施阶段。项目概览、数据规模、当前进展和
分阶段 TODO 见[根 README](../README.md)。研究计划获批不代表 GPU/API 额度已确认，
也不代表已通过数据质量、benchmark 冻结、贡献选择或正式评价 gate。

## Research：研究路线与文献

- [研究计划与实验方案](research/plan.md)：当前执行方案见 Section 13；保留此前的决策、
  工程记录与研究边界。
- [Empirical study 参考](research/empirical_study_references.md)：WorldArena、
  CLEVRER、受控物理泛化实验的借鉴点及检索边界。
- [世界模型研究报告](research/world_model_research.md)：能力分解、已核查文献、
  数据分层、研究问题、baseline 与证据边界。

## Benchmark：任务与评价

- [开发 pilot 协议](benchmark/benchmark_protocol.md)：任务契约、模拟器、
  public/privileged 视图、重放与命令；当前不是冻结 benchmark。

## Data：数据来源、标签与目录

- [数据卡](data/data_card.md)：实际规模、版本、许可及标签局限。
- [真实观察 schema](data/real_data_schema.md)：BEV、观测、缺失、时间与来源字段。
- [跨来源研究目录](data/research_corpus_catalog.md)：真实观察、辅助来源与仿真真值
  的引用/能力分层，避免将目录行数误作物理 ground truth。
- [Hugging Face 来源](data/huggingface_source.md)：本地快照、组织方式及媒体权利边界。
- [数据访问与许可](data/data_access_requests.md)：访问记录、许可问题和未发送的请求草稿。

## Pipelines：预处理与质量筛选

- [视频处理](pipelines/video_processing.md)：采集、标定、原图检测及候选导出。
- [质量筛选](pipelines/quality_screening.md)：自动 triage、疑似重复、复核与修复队列。
- [作者直接上传](pipelines/author_direct_uploads.md)：不依赖不完整 ZIP 的来源读取与转换。
- [增量 snooker 管线](pipelines/incremental_snooker_pipeline.md)：内容去重、缓存、
  版本化转换、人工 raster 审计与可复现实验输入。

## 本地材料与生成结果

文档中的 `data/...` 和 `artifacts/...` 路径指向可复现的本地运行结果；这些目录除说明文件外
均被 Git 忽略，因此 GitHub 页面不会包含原始媒体、转换数据或运行产物。第三方论文 PDF
和原始讨论材料同样保留在本地，不由本仓库重新分发。文档中的 shell 命令默认从项目根目录执行。
