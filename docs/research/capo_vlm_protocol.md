# InverseR-VLM v1 数据与评分协议

本协议仅用于 `InverseR/`。旧版 `pilot-v0` 与 `formal-test-v1` 不修改；只读取其初始布局，生成排除指纹。不调用反代，也不进行 CPU-only 基线或搜索；Pooltool 只构造与复核真值。

## 分区

`prepare-vlm-suite` 在 `data/research/inverser-vlm-v1/{train,valid,test}` 写统一分区目录：

| 分区 | 标准 anchor | 标准 twin | 标准场景总数 | 额外 OOD |
| --- | ---: | ---: | ---: | ---: |
| train | 12,000 | 12,000 | 24,000 | 0 |
| valid | 120 | 120 | 240 | 0 |
| test | 300 | 300 | 600 | 120 |

训练按原计划的 **12,000 个 anchor 各配一个 twin**。valid 和 test 的 240/600 指标准场景总数（包括 twins）。四个 family 各占四分之一；`free_roll` 只有母球，其余 family 在 2、3、4 球间平衡。每个 twin 只改变一个动作维度，静止时至少一个身份保持的球移动 ≥2 个球半径，或发生落袋状态变化（用桌面对角线作状态差惩罚，不假造落袋坐标）。母球落袋的见证不入库。

`layout_sha256` 仅依赖初始布局（存放在 `private/labels.jsonl` 与 `groups.json`），排除 solver 参数、goal、动作和分区名；公开与私有的 `group_id`、`scene_id` 及 `public/images/{roll,center,vert,side,ood}/` 下的 PNG 文件采用可读短编码（标准题为 `{roll|center|vert|side}-{1..4}ball-{0001}-{init|anchor-fwd|anchor-goal|twin-fwd|twin-goal}.png`，测试集 OOD 为 `ood/{fast|slow}-{family}-{1..4}ball-{0001}-{init|goal}.png`）。标准 anchor/twin 共用组；分区之间以及与两个历史排除集零重叠。分组分析与配对 bootstrap 使用独立布局数，不将 twins 当作独立样本。

## 输入与输出

- T1–T3 联合请求一次，便于测事件与状态是否一致。初始位置只从 BEV 读取；文本提供精确击球控制、公开物理参数、时间点和输出 schema。等比例 BEV 最大边长 1024 px，球号 `0` 是母球；按初始编号固定 forward slot。青色方向箭头的标尺是杆速 × 0.16 s，不是假装观测到撞击后的球速。
- T1 比较物体球的碰撞向量及母球首次接触的物体编号，两者均正确才记 1。T2 同时比较边界集合和有序边界事件（保留重复）。Pooltool 线性/圆形库边面按其几何参考点到四边的最近距离映射，平局用固定 `LEFT, RIGHT, BOTTOM, TOP` 顺序；映射记录在 manifest。
- T3 在 1、2、3、4、5 秒和真正静止终态逐球比较。坐标欧氏距离 ≤1r 或正确落袋记命中；场景 `T3` 是所有时刻/球编号上的命中率，另外记录 `T3_all_slots`。`S_total = 0.3*T1 + 0.3*T2 + 0.4*T3`；跨场景宏平均，不用球多的布局提高权重。
- T4 看到初始和目标两张图，输出方向、速度、杆头横向/纵向偏移。目标同色物体球可交换；以隐藏执行的 1r endpoint 成功为准。目标图不携带出杆动作，编号不是跟踪真值。
- T5 最多六次，均从同一初始状态执行，命中可提前结束。标准条件与 OOD 分开；OOD 各一半使用滑动/滚动摩擦 ×0.5 或 ×2 并使用相应库边恢复系数 0.65 或 0.97。公开 prompt 只包含被隐藏参数的名字，不包含数值、条件名或私有文件；原始 outcome 与相对目标残差是交互观测，非隐藏摩擦参数。
- `public/scenes.jsonl` 是索引而非可整体传给模型的 prompt。正向与逆向必须分别组装；尤其不得把同一行 forward 的动作、forward PNG 或私有行传给 T4/T5。`private/` 只供训练标签构造、事后盲测评分及仿真 judge 使用。模型选择只用 valid 结果。

## 验证

```bash
.venv/bin/python -m inverser prepare-vlm-suite
.venv/bin/python -m inverser prepare-vlm-suite --verify-only
```

生成阶段将每个接受的动作重复执行一次，要求事件、时间、状态均一致且见证满足 endpoint；存的是经过该检查的记录。`--verify-only` 不运行模型或重新仿真：重新计算 SHA-256、重算布局指纹和 twin 偏移、确认所有公开图像结构可读、回查交集、任务 schema、T1–T3 评分自洽及 OOD 隐藏参数。评分规则和数据计数是固定协议，不因模型效果事后改变。仅凭数据验证不能得出 CAPO 提升、消除 Stasis Bias 或达到 70% 的结论。

2026-11-19 已生成正式三分区。生成日志 `artifacts/capo-vlm-freeze-v1.log`；
独立验证回执 `artifacts/capo-vlm-verification-v1.json`。
模型请求入口 `run-vlm-eval` 当前只允许开发/训练/valid，拒绝无冻结协议的 test；
T5 的图像反馈将目标与最新 outcome 并排打包，始终遵守两个图像输入的限制。
`train-capo-sft` 在 `.venv-train` 运行且仅接受 train manifest。
