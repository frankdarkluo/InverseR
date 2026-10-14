# Native canvas 审计：源像素对齐不等于结构化状态验收

这是 autoresearch 的一轮真实源数据质量诊断，不是模型推理结果或正式 benchmark。
它使用既有录制的少量抽帧，不重新处理完整视频，也不改变原始数据、动作标签或 G1 状态。

## 1. 检验的问题

- **H-QC1：** 作者的人工参考图是否能共用一个 source-to-native crop？
- **H-QC2：** 已检测点全部落在人工空间路径内，是否足以验收结构化状态？

先检查视频与源展示图的像素关系，再冻结映射检查人工空间参考。人工路径不能用于调 crop、
resize、alignment 或选择有利的阈值。

## 2. 配准方法与证据范围

作者数据的 `MOLT-VACHASPATI-LEGG.avi` 是派生的算法比较展示，不是新的原始击球或实测
动作 GT。本审计只使用其中的 native 图像区域来检查像素坐标关系，不将其算法轨迹当作 GT。

展示图是两列、两行，native body 上方有 20 px header。每个 body 的尺寸与该录制的
manual raster 一致。固定取右下 body，在台布色掩码与 erosion 下做 translation-only
像素匹配；不读取人工路径来估计 offset。这个掩码主要抑制非台布和覆盖物，不是独立验收
过的完整球分割标签。

策略在两个 support 录制上确定，其余 14 个录制用于本轮 policy check。每个录制仍使用
自己的首帧作像素校准，再以中间帧检查该 crop 是否保持一致；这不是无需校准的跨桌泛化。
这些数据已用于开发，不能改称从未接触的正式测试集。

预设门槛：

| 条件 | 门槛 |
| --- | ---: |
| 台布掩码占比 | 至少 0.5 |
| normalized pixel residual | 不超过 0.02 |
| 与相隔超过 1 px 的次优 offset 的 margin | 至少 0.001 |
| 首帧与中间帧 offset 差异 | 不超过 1 px |

输入索引、inventory、实现文件与抽样像素的 SHA-256 均写入结果。原始 `.dat` 的 snapshot
signature 在配准前后检查，但本命令没有重新计算所有完整原始文件的 SHA-256；结果明确区分
既有完整 hash 与新抽样 hash。

## 3. 实际结果

本轮最终证据保存在：

- [配准审计](../../data/quality/author-canvas-audit-v3/audit.json)。
- [冻结映射后的点兼容性](../../data/quality/author-canvas-audit-v3/point_compatibility.json)。

### H-QC1：共享同一个 crop 的假设不成立

| 录制数 | Native canvas | Source crop 原点 |
| ---: | --- | --- |
| 1 | 571×300 | (34, 70) |
| 1 | 574×305 | (34, 65) |
| 14 | 571×303 | (34, 67) |

16 个录制的 32 对抽样图像均通过 policy。最大 normalized residual 为
0.004147；最小 separated-peak margin 为 0.004346。各录制首帧与中间帧的 offset
差异都是 0 px。

这支持逐录制的 source-render crop 映射，不证明毫米级物理误差为零，也不证明相机、
球心高度或 playing bounds 已标定。源像素到 native image 的变换与 source-to-metric
homography 是不同对象。相同背景文件也不能证明作者的每个导出 crop 都相同。

### H-QC2：空间覆盖不能替代语义和库存检查

在冻结映射后，只检查 14 个 policy-check 录制的 28 个抽样帧：

| 项目 | 实际结果 |
| --- | ---: |
| 已检测点 | 468 |
| Native 人工 path union 内或距离不超过 3 px 的点 | 468 / 468 |
| 共享 support crop 对照的相同指标 | 468 / 468 |
| 有 unknown-colour 假设的帧 | 28 / 28 |
| 有多个 blue 假设的帧 | 19 / 28 |
| Cue 库存未解决的帧 | 9 / 28 |
| 有任意上述语义/库存冲突的帧 | 28 / 28 |

这里的 100% 是**条件于已检测点的空间兼容性**，不是 detector accuracy、完整库存 recall、
身份、颜色准确率或未来轨迹准确率。没有检测到的球不在这个分母中，path union 也不含时间
或逐球身份。3 px 偏移的共享 crop 得到相同分数，说明该粗指标在本样本上无法区分这项误差。

因此不能因空间分数漂亮而把当前自动观察升级成高质量物理任务。下一项瓶颈是颜色、cue
库存和身份连续性；不是先扩大 raw 数量或直接训练一个模型。

## 4. 复现命令

命令从项目根目录执行。使用新的输出目录，不覆盖旧证据。当前命令限制为 2-16 个录制、
每个两个 native/render 抽样帧；它们不是完整视频重处理的入口。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser audit-author-canvases \
  --episodes data/structured/author-snooker-incremental-v1/snapshots/551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e/episodes.jsonl \
  --inventory data/quality/author-completed-upload-snapshot-v1/inventory.json \
  --output data/quality/author-canvas-audit-next \
  --limit-recordings 16

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m inverser audit-native-point-compatibility \
  --mapping data/quality/author-canvas-audit-next/audit.json \
  --episodes data/structured/author-snooker-incremental-v1/snapshots/551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e/episodes.jsonl \
  --output data/quality/author-canvas-audit-next/point_compatibility.json
```

第二条命令检查 frozen index、manual hash 和抽样 source pixel hash，只评价映射已通过的
policy-check 组。映射失败会明确记录为排除；缺乏检测时 coverage 为 unknown，而不是填零
或返回成功。它不会根据 query 人工路径重新配准。

## 5. 后续与停止边界

下一轮假设是：source-specific 颜色证据与时间连续性能够减少库存冲突，同时不制造缺失
球、身份或落袋标签。这仍待检验。不能通过强行凑齐一套球、把 unknown 改成某个颜色、
或照抄源算法的轨迹来获得虚假的通过率。

完整的 100 个作者录制、370 条比赛候选重处理仍需要明确 CPU/storage 额度。GPU/API、
训练、benchmark 扩建与发布没有启动；G1-G4 没有通过。原始 manual raster 仍只是空间
参考，真实 cue/spin 未测。

本轮代码有 source-registration、模糊配准拒绝、跨帧 crop 改变、native header/尺寸、
query-refit 禁止和 source/index 改变拒绝的测试。全套 165 个测试通过，Ruff 与 Mypy
通过（31 个 source file）。AI 辅助范围是源证据组织、代码和文档；这份结果不代表研究者
已确认数据验收、文章贡献或投稿披露政策。
