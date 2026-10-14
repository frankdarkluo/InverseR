# 作者 snooker 语义 QC pilot：颜色、背景与短缺帧关联的消融

本轮已经实际处理视频原始帧，不是 mock 或模型 API 结果。它可以作为数据构建与预处理
消融的初步实验表，但不能写成 LLM 物理推理、真实控制或已验收 benchmark 的主结果。

## 1. 实验范围与冻结协议

- 16 个作者原始 snooker 录制，按内容来源组去重。
- 2 个 support 录制用于来源格式与颜色证据检查；其余 14 个录制用于开发检查。
- 总计 6,609 个原始帧；表格使用后 14 个录制的 6,097 帧，按声明的原始 20 fps 处理。
- 每个录制使用已冻结的 native crop 映射；人工路径没有用于重新拟合变换。
- 5 个固定消融条件共用相同输入、几何、球形筛选和帧时钟。
- 最多 4 个 CPU worker、20 GB 新产物、项目总量上限 1 TB；实际产物约 125 MB。
- 模型/API/GPU、人工新增标注、训练次数均为 0。G1-G4 未自动通过。

`held_out` 在产物中表示来源颜色规则的开发检查组，不是从未接触的正式 benchmark test。
这些录制来自同一作者实验环境，不能据此宣称多桌或比赛域泛化。

原始 BGR 解码与源 AVI 像素关系已检查。来源内一个绿色球的代表色相接近 90–95，
旧通用阈值会归到 blue；部分高饱和红色也落在旧 red 范围之外。修复采用显式 opt-in
的来源颜色规则，不改变比赛视频的默认规则，也不强行限制分类结果以凑齐库存。

## 2. 主表：固定点盘渲染下的消融

以下均为**每录制先计算、再对 14 个录制等权平均**，不是把所有帧视为独立样本的统计。
原始逐录制数值见 [CSV](../../data/quality/author-semantic-pilot-v1/per_recording.csv)，
汇总数值见 [主表 CSV](../../data/quality/author-semantic-pilot-v1/aggregate_table.csv)。

| 管线 | 平均检测球数 | 路径 IoU ↑ | unknown 检测占比 ↓ | 颜色重复冲突帧 ↓ | cue 未解决帧 ↓ | 新轨迹/100帧 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧版 | 16.64 | 58.17% | 27.80% | 67.99% | 23.91% | 234.2 |
| 只修来源颜色 | 16.64 | 58.17% | 2.41% | 19.69% | 23.91% | 93.2 |
| 只用背景差分 | 16.14 | 58.13% | 29.67% | 53.83% | 25.03% | 244.9 |
| 颜色＋背景 | 16.14 | 58.13% | 2.30% | 17.79% | 25.03% | 92.1 |
| 颜色＋背景＋短缺帧关联 | 16.14 | 58.13% | 2.30% | 17.79% | 25.03% | 26.3 |

### 指标究竟测什么

- **路径 IoU：** 将每个实际检测中心画成半径 7 px 的圆盘，与 native 人工 path union
  做空间 overlap。主表不连接观测间的线、不插值缺失帧。它不是逐时刻坐标准确率。
- **unknown 检测占比：** 当前已经检测到的点中，颜色未能确定的比例。没有检测到的球
  不在这个分母中；不能改称颜色准确率。
- **颜色重复冲突帧：** yellow/green/brown/blue/pink/black 等唯一色球出现多个假设
  的帧比例。没有重复不证明每个分类都正确。
- **cue 未解决帧：** cue 假设数量不等于 1。它混合了漏检、多重 cue 和可能不可见等情况，
  不能在没有标注时称为 cue 漏检率。
- **新轨迹/100帧：** 算法启动新 track ID 的频率，是碎片化代理。数值可以超过 100，
  因为一帧有多颗球；更低也可能来自错误合并，不是 IDF1 或身份准确率。
- **平均检测球数：** 是防止靠丢掉检测获得更漂亮冲突率的辅助统计，不是完整库存 recall。

### 通俗解释

**颜色规则有可保留的收益。** 同一批点的位置和数量完全不变，只修颜色就使 unknown
和唯一色球重复冲突显著减少。不是通过少找球来降低 unknown，但没有人工颜色 GT，
仍不能断言每个重新分类都正确。

**背景差分并没有证明更好。** 它看上去让误画的路径少一些，点盘 precision 从
90.34% 到 92.87%，但人工路径的空间覆盖也从 61.96% 到 60.73%；平均球数减少，
cue 未解决比例和 unresolved-component 帧比例也上升。不能只挑一个变好的指标。

**短缺帧关联主要减少碎片。** 在相同“颜色＋背景”检测上，新轨迹频率从 92.1 降到
26.3。实现只报告实际检测点，缺失帧仍为空，未知颜色保持 unknown；对匹配歧义新建
假设而不是硬连。但没有独立 ID 标签，无法证明减少的碎片都连对了。

## 3. 负面结果与统计边界

背景差分相对旧版的点盘路径 IoU 平均差为 **−0.04 个百分点**；
按录制组配对 bootstrap 的 95% 区间为 **[−1.81, +1.50] 个百分点**。
这不支持“共享背景稳定改善空间轨迹”的结论，也不是证明两种方法严格等价。

只修颜色不改变任何点盘 mask，IoU 差值严格为 0；这不是没有实验效果，而是该指标
本来就不看颜色。

主协议半径固定为 7 px，并额外报告 6/8 px 的敏感性。旧版点盘 IoU 对应
49.33% / 64.61%，表明画线宽度会明显影响空间 overlap。不能看结果后选择更有利宽度
并重新包装成主分数。

完整配对统计保存在
[paired_diagnostics.json](../../data/quality/author-semantic-pilot-v1/paired_diagnostics.json)。
录制组是统计单位，但相同相机与实验桌的共享偏差不由这个置信区间消除。

## 4. 补充诊断：点图与连线图会得到不同分数

检查一个低 IoU 案例后发现，来源人工 raster 包含连续路径，而主协议只画采样圆盘。
因此追加了明确标为 exploratory 的渲染诊断，**没有替换主表**。

补充协议：仍画半径 7 px 的观测圆盘，仅在同一个 track ID 的两个**连续原始帧**
都有实际观测时连接固定 15 px 宽的线段。不连接缺失帧、不生成新坐标标签，不认为帧间
实际运动一定沿直线，也不证明匹配 ID 正确。

| 管线 | 主协议：观测圆盘 | 补充：相邻观测线段 |
| --- | ---: | ---: |
| 旧版 | 58.17% | 71.12% |
| 只修来源颜色 | 58.17% | 73.96% |
| 只用背景差分 | 58.13% | 70.35% |
| 颜色＋背景 | 58.13% | 73.75% |
| 颜色＋背景＋短缺帧关联 | 58.13% | 73.88% |

**同一批观测，仅画法不同，空间分数就变化了。** 因此不能把主表的 58% 简单理解为
“42% 的球看错了”，也不能把补充的 74% 解释为运动模型准确率。颜色修复在连线协议下
的变化与关联持续性相关，但仍需要独立的身份和检测证据验证。

数字与逐录制表保存在
[补充 JSON](../../data/quality/author-semantic-pilot-v1/rasterization_diagnostic.json)及
[补充 CSV](../../data/quality/author-semantic-pilot-v1/rasterization_diagnostic.csv)。
所有 observed-only JSON 坐标不变。

## 5. 是否继续使用当前数据

建议先保留作者集。它有可用的真实原始帧、native 空间参考和一致来源，当前的问题包含
可定位的规则不匹配与评价表示差异，不能笼统归因为“数据没有用”。

它目前仍不适合作为实测 cue/spin 的监督集，也没有完整逐时刻状态 GT。颜色修复后
自动 ontology 冲突帧比例仍为 61.75%，cue 未解决比例仍为 23.91%；不能因此通过 G1。

建议下一轮：

1. 以颜色修复版为保守起点，单独验证“不使用共享背景”的短缺帧关联条件。
2. 专门诊断 cue 在模糊、紧邻球和遮挡下的漏观测或多重假设。
3. 用相同空间表示、观测缺失率、库存不确定性与失败清单评估，保留旧版和负面结果。
4. 达到阶段证据条件后，再决定扩大到 100 个作者录制及 370 条候选。GPU/API 仍不启动。

这些是后续建议，不是当前已经执行或已经获得 G1 批准的扩建。

## 6. 备选真实资源：已回查主要来源

已检索替代资源，但没有发现并核验一个可以直接替代“完整单杆状态＋实测 cue/spin”
的来源。搜索摘要可能混淆静态标签、轨迹和许可证，下面只写主要页面支持的事实。

| 资源 | 已核实的信息 | 用途与阻塞 |
| --- | --- | --- |
| [Billiards Sports Analytics: Datasets and Tasks](https://arxiv.org/html/2407.19686v1) | 9-ball 数据；论文 Table 1 列出 3,019 个局级 frame 条目、6,637 个 turn 条目、2,082 个 strike 条目；公开目录含 code、data_trajectories、data_layouts | 值得检查实际 schema，可作辅助布局/轨迹资源；frame 在该论文指比赛局，不是视频帧。不是 3,019 条 snooker 单杆或已测旋转；未下载包，不确认时间精度及实测 cue 控制 |
| [billiardsadd](https://platform.ultralytics.com/li-zhe/datasets/billiardsadd) | 页面显示 93 图像、973 个检测标注；俯视台球/斯诺克图像 | 静态检测补充候选，不是动作轨迹；主要页面 license 字段为 null，使用权尚未确认 |
| [hole01](https://platform.ultralytics.com/zhe-li/datasets/hole01) | 页面显示 500 图像、2,996 个检测标注；俯视场景 | 静态检测补充候选；具体球型 taxonomy、原始媒体来源与许可待查，license 字段为 null |

Billiards Sports 的作者在论文中声称数据公开且无版权限制，并提供
[Google Drive](https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG?usp=sharing)
和 [Dropbox](https://www.dropbox.com/sh/644aceaolfv4rky/AAAKhpY1yzrbq9-uo4Df3N0Oa?dl=0)
入口。已确认前者公开目录存在，但发布包内的具体许可证和广播媒体的权利边界尚未检查。
论文的 arXiv 许可不是数据/媒体许可，不能将作者声明扩展为任意原始视频可再分发。

Roboflow 的 Snooker Vision 页面普通访问返回 403，本轮没有绕过，也不把搜索摘要中的
“MIT”“281 图像”等说法作为已核查事实。没有下载或合并任何新的媒体数据。

## 7. 复现与产物

从项目根目录执行，使用新输出目录。代码/数据/协议变化会拒绝复用旧 run。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser run-author-semantic-qc \
  --mapping data/quality/author-canvas-audit-v3/audit.json \
  --episodes data/structured/author-snooker-incremental-v1/snapshots/551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e/episodes.jsonl \
  --output data/quality/author-semantic-pilot-next \
  --workers 4 --max-new-output-gb 20 --limit-recordings 2
```

去掉 `--limit-recordings 2` 并使用同一配置可继续余下录制。已经完成的 report、观测和
图像均经 hash 检查复用；改变配置不能静默重算，未完成目录需要显式恢复版本。

主要产物：

- [manifest](../../data/quality/author-semantic-pilot-v1/manifest.json)：冻结参数、代码、
  来源、支持/开发组、依赖与 GPU/API/人工标签边界。
- [summary](../../data/quality/author-semantic-pilot-v1/summary.json)与
  [validation](../../data/quality/author-semantic-pilot-v1/validation.json)。
- 各录制的 checksummed report、observed-only gzip、native 端点预览和路径图。
- 主表、逐录制表、配对统计和补充渲染诊断。

实际运行完成 16 个录制、失败 0 个；重复运行新增处理 0 个。全套 172 个测试、Ruff、
Mypy（32 个 source file）通过。默认比赛视频检测行为保持不变，来源颜色和背景条件
只在显式 opt-in 的 QC 消融中使用。原始来源与旧结构化快照不改写。
