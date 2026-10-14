# Billiards Sports Analytics：已下载的数据与可用标签

本地目录为 `data/BilliardsSportsAnalytics/`。主要来源是用户指定的
[论文](https://arxiv.org/html/2407.19686v1)链接的公开 Dropbox 与 Google Drive。
这是 nine-ball 辅助域，不增加主 snooker benchmark 的真实击球数量。

## 1. 实际获取结果

| 来源 | 已完成获取 | 验证 |
| --- | ---: | --- |
| Dropbox 原始发布 ZIP | 293,325,698 bytes | 原始归档保留，ZIP CRC、SHA-256 通过 |
| 内嵌数据 ZIP | 290,531,556 bytes | 保留并安全解压，3,850 个源文件 |
| Google 轨迹与属性 | 3,100 / 3,100 个公开计划文件，56,592,899 bytes | 2,082 XML、1,018 XLSX；逐文件大小/hash receipt 验证 |

公开目录的普通 crawler 曾超时，个别普通客户端请求失败。随后使用有超时、检查点、
大小限制的公开目录读取与标准下载确认链接完成获取；未登录、读取个人 cookie、
绕过访问限制或下载 YouTube 原始媒体。

输入计划和 acquisition receipts 保存在
[采集目录](../../data/BilliardsSportsAnalytics/acquisition/)；
总体验证是 [validation.json](../../data/BilliardsSportsAnalytics/acquisition/validation.json)。
本地 BSA 原始获取产物约 981 MB，低于本轮 20 GB 新产物与 1 TB 项目总量额度。

数据包附带的训练脚本、notebook 和模型权重没有执行或加载。原始 ZIP 的零字节 `/`
目录标记不被当作可提取路径；内嵌数据使用现有安全解压、CRC 与路径检查机制处理。

## 2. 静态布局不是连续视频帧

| 项目 | 本地审计结果 |
| --- | ---: |
| 坐标 XML 文件/解析布局条目 | 2,964 |
| 精确文件 hash 去重后 | 2,913 |
| 源 match 路径组 | 298 |
| Tournament 目录 | 94 |
| 有效比赛变量 XLSX | 298 |
| Office 临时锁文件 | 1，明确排除为非数据 |

论文中的 3,019 个 `frame` 是比赛局/rack 条目，不是视频帧。本地发布包的 2,964 个
坐标文件也不能强行报告为论文承诺的 3,019 条。文件 hash 去重不是统计独立性的证明。

布局保留 `Marker 1...` 等源标记、原生 x/y、时间表达和原始 video title：

- 2,728 个布局的 `Time=0.00` 等表达没有被默认为实际录像秒数。
- 2,497 个布局有负坐标；没有将它们默认为 pocketed。
- 10 个 marker 的文件有 2,963 个，另一个有 11 个；没有强行裁成完整球库存。
- 目录年份无法在源标题中核对的标记不等于确认配错视频，尤其标题可能是
  `videoplayback` 或其他通用名称。
- 没有自动证明 Marker 顺序等于 cue/1–9 的球身份或坐标单位是米。

完整字段与 flags 见
[layouts.jsonl](../../data/BilliardsSportsAnalytics/audit-combined-v4/layouts.jsonl)及
[审计统计](../../data/BilliardsSportsAnalytics/audit-combined-v4/summary.json)。

## 3. 轨迹有时间与单位，但没有完整动作状态

实际轨迹 XML 直接声明 `Coords (x,y:cm; t:time)`，列为 x、y、t，
并带 `cue ball`、`object 2` 等源标签。这补上了作者 snooker 的 manual path union
所缺少的逐时刻位置证据。

| 项目 | 本地审计结果 |
| --- | ---: |
| 获取的轨迹 XML | 2,082 |
| 可解析为 cm/time 的轨迹 | 2,080 |
| 保留但不纳入 cm 协议的 px 轨迹 | 2 |
| cm/time 标注点 | 143,792 |
| 源 annotation 文件夹组 | 1,005 |

**2,082 个轨迹文件不是 2,082 次独立击球。** 同一文件夹可以含 cue 与目标球轨迹，
同一局有多条记录，源标签也有拼写不一致；原表达保留，不静默修正球身份。

时间格式兼容有源证据的 Kinovea 秒数表达、`MM:SS`/`HH:MM:SS` 及 Excel
`1899-12-31T...` 时间表示。转换保留原始字符串，不补造采样点。两个原生 px 曲线原样
保存，但不能用不明 scale 转成 cm；它们在
[显式排除记录](../../data/BilliardsSportsAnalytics/audit-combined-v4/parse_errors.json)中列出。

声明的 cm 与视频时间是来源软件标注，不是我们独立实测的标定。完整桌面 inventory、
隐藏 spin、cue-stick impact speed、材料参数和实测 cue controls 仍不可用。球的位置
差分可以估计球速，但不能把它写成实测 Pooltool `V0` 或杆头击点。

## 4. 如何用于世界模型 empirical study

| 数据层 | 可做 | 不应声称 |
| --- | --- | --- |
| 静态局级布局 | 辅助场景/布局分析；校准后可作为明确标为模拟的布局种子 | 连续动力学或真实动作 GT |
| 单球 timed trajectory | Past prefix 到 factual future 的观测预测检验 | 指定任意 cue/spin 动作的真实反事实 |
| 比赛属性表 | 源 layout/outcome/context 研究 | 新学到的转移动力学或实际机器人执行 |

主 snooker 的 action-conditioned prediction、干预一致性和 goal planning 仍使用
动作与物理参数明确的受控模拟。BSA 加入的是**真实事实预测/观察兼容性**，不是把辅助
九球数据混入主域成功率。见[世界模型数据衔接](../research/world_model_data_bridge.md)。

## 5. 许可与发布边界

作者论文声明数据公开且无版权限制，并提供数据链接；本轮按用户明确要求做匿名本地获取。
没有发现并独立核验发布包中覆盖所有媒体/标注/脚本的正式许可证。论文 arXiv 许可不能
代替数据/广播媒体许可。没有自动发布、对外上传或用它训练模型；真实标注也不发送到
付费模型 API。

## 6. 审计复现

使用新输出目录，保留旧版本：

```bash
.venv/bin/python -m inverser audit-billiards-analytics \
  --dataset data/BilliardsSportsAnalytics/release/files/dataset \
  --trajectories data/BilliardsSportsAnalytics/google_trajectories \
  --output data/BilliardsSportsAnalytics/audit-next
```

源文件不改写。cm/time 类型、未验证布局单位、office lock 与时钟/像素单位排除分开记录。
