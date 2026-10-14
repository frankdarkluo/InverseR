# InverseR

InverseR 是一个面向二维台球物理推理的研究原型。项目的核心任务是：给定斯诺克台面的
初始状态和目标终态，生成任意一个物理可行的单杆动作，包括击球方向、速度以及归一化的
杆头横向/纵向偏移。复现历史球员的原始击球仅作为独立诊断任务，不是主任务的唯一正确答案。

> **项目状态（2026-10-14）：** G1 已接受限定的分层证据范围，小型开发诊断面板已锁定。
> 已完成仿真开发集、真实视频原始候选收集、自动质量筛选和 mock 实验链路；尚未冻结正式
> benchmark；未进行模型训练或正式性能评测，已完成小型 Azure 原生 API 开发诊断。

已准备的 [64＋64 开发诊断面板](docs/research/model_diagnostic_panel.md)含 384 个仿真
配对请求和 64 个 local-only 真实前缀请求。本地 GPU 与扩大 API 运行额度仍待确认，
开发快照不等于正式盲测或实测真实动作 GT。随后已授权并完成
[GPT-5-mini/GPT-5 原生小对照](docs/research/azure_native_pilot.md)：2 个布局、24 次
合成请求、67,866 Tokens，8 次隐藏参考执行；没有真实资料外发或模型可用工具。

## 研究问题

InverseR 重点研究“预测准确”能否转化为“规划有效”：

1. 模型能否对动作和物理参数干预作出正确的未来状态与事件预测？
2. 更准确的预测能否改善候选动作排序和目标终态成功率？
3. 在相同信息、候选集和物理查询预算下，LLM/VLM 相比几何与数值优化基线的增益来自哪里？
4. 当感知、材料参数和 spin 不确定时，真实观察能改善能力链中的哪一部分？

完整的可证伪假设、对照实验和证据边界见
[研究与实验计划](docs/research/plan.md)及
[世界模型研究报告](docs/research/world_model_research.md)。

已实现的安全视图、等比例渲染、预算账本与 mock 命令见
[实验准备说明](docs/benchmark/study_preparation.md)。这是工程验证，不加载模型或产生模型成绩。

## 已实现能力

- [BSA 获取与标签审计](docs/data/billiards_sports_analytics.md)：完整公开数据计划已下载，
  timed trajectory 已转换为 local-only 的真实事实预测输入；不充当真实 cue/spin GT。
- [世界模型数据衔接](docs/research/world_model_data_bridge.md)：准备 action-conditioned
  prediction→planning 主线和单独的真实 prefix 验证；已跑有界 API 诊断，未做正式评测。

- [CPU 语义质量 pilot](docs/pipelines/semantic_qc_pilot.md)：已实际运行 16 个作者录制、
  五个预处理消融条件；报告空间参考和自动一致性，不宣称模型性能或检测准确率。

- [真实源数据 canvas 审计](docs/pipelines/native_canvas_audit.md)：已核查 16 个录制的
  抽样像素映射；空间路径兼容性与颜色/库存质量分别计分，不把弱观察升级为 Gold。

- 基于 Pydantic 的台面状态、击球动作、物理参数、事件和实验 case 合同。
- 基于 Pooltool 0.6.0 的可重放斯诺克仿真、witness 审计和终态评分。
- 数据来源登记、许可状态记录、断点下载、上传快照和安全归档导入。
- 比赛视频的台面检测、BEV 观测、候选切分、失败恢复和自动质量筛选。
- 作者数据的内容哈希去重、增量转换、人工路径 raster 核查和跨来源研究目录。
- state/BEV 两种实验视图、typed 模型适配器、请求/物理预算账本和 mock dry-run。
- random、Sobol、几何初始化＋Powell 的预算受控经典搜索，以及分组的预测、事件、
  干预响应和候选 regret 诊断；`baseline-dry-run` 支持检查点恢复与产物校验。

当前命令列表可通过以下命令查看：

```bash
python -m inverser --help
```

## 实验数据集

原始媒体、转换数据和运行产物不会提交到本仓库。下表描述的是当前本地研究快照，而不是
公开发布的数据集。

| 数据层 | 当前规模 | 标签与用途 | 当前限制 |
| --- | ---: | --- | --- |
| Pooltool 仿真开发 pilot | 2,000 个可重放 shot；2,039 次生成尝试，39 次 scratch 被拒绝 | 已知动作、参数、事件和轨迹，用于开发与合同验证 | 仅 development partition；不代表真实物理 |
| 作者 Billiard-dataset 主域 | 100 个内容去重的 snooker 录制，37,876 个 raw frame，16 个人工路径 raster | 固定相机真实观察、自动轨迹和来源人工空间路径 | 自动标签尚未人工验收；没有实测 cue/spin 标签 |
| 作者辅助域 | blackball 100 个、carom 100 个内容去重录制 | 感知或跨球类迁移诊断 | 不计入主 snooker benchmark 数量 |
| 比赛视频快照 | 32 个视频，228.38 小时，7,114 个 raw 候选 episode | 多场景弱观察、质量筛选和后续重处理 | 媒体权利未核实；0 个已验收真实 benchmark 任务 |
| Wikimedia Commons pilot | 3 个短片，共约 15.86 秒 | 许可与解码流程 smoke test | 画面裁切/机位移动，不满足主任务要求 |
| 统一研究目录 | 9,414 条引用：7,214 条真实主域、200 条辅助域、2,000 条仿真 | 统一索引并保留来源、质量和标签边界 | 目录行数不是独立物理任务数 |

关键事实：

- 已独立验收的真实 benchmark 任务：**0**。
- 实测真实击球动作记录：**0**。
- 当前 7,114 个比赛候选中有 15 个未验收复核优先项和 370 条定向修复队列。
- Hugging Face 比赛视频来源的媒体许可仍未确认，不能据此推断训练或再分发权限。
- 作者数据、比赛视频、辅助球类和仿真真值始终分层统计。

详细来源、许可、规模和标签定义见
[数据卡](docs/data/data_card.md)、
[研究语料目录](docs/data/research_corpus_catalog.md)和
[真实观察 schema](docs/data/real_data_schema.md)。

## 当前实验进展

| 工作流 | 状态 | 已取得结果 / 停止条件 |
| --- | --- | --- |
| 工程与任务合同 | 已完成开发版 | Python 包、CLI、schema、仿真、指标已建立；190 个测试、Ruff 和 Mypy 通过 |
| 仿真 pilot | 已完成 | 2,000/2,000 导出记录通过 schema 与 witness 重放 |
| 比赛视频原始收集 | 已完成 | 1,387/1,387 窗口完成，得到 7,114 个 raw episode |
| 自动质量筛选 | 已完成 | 16 个自动优先项、6,004 个待重处理、887 个不适用项、207 个疑似重复 |
| 作者数据同步 | 已完成原始转换 | 100 个 snooker 内容组完成转换；55 个复用、45 个新增 |
| 统一研究目录 | 已完成 | 9,414 条引用通过来源与标签边界检查 |
| E0 实验基础设施 | 已实现 mock 链路 | state/BEV renderer、typed client、预算账本和 dry-run；不代表真实模型结果 |
| G1 分层证据范围 | 已接受有限开发范围 | 受控 snooker 动作真值主任务、BSA 真实 factual 辅助、原始弱观察用于感知/案例；真实 GT/许可问题未消失 |
| 小型开发诊断面板 | 已锁定并验证 | 64 仿真布局＋64 来源 round；448 请求、128 reference replay；48 请求 API 子集尚未执行 |
| 模型开发诊断 | 已完成首个有界 API pilot | GPT-5-mini/GPT-5 共 24 调用；本地 GPU 未分配，完整 48/model 扩展未批准；无训练 |
| 正式 benchmark / evaluation | 未开始 | 开发快照不是 G2/G4；正式布局/来源、split、指标、预算和协议须另行冻结 |

## 后续实验 TODO

| 阶段 | 计划 | 验收条件 |
| --- | --- | --- |
| P1：数据质量 | 已接受首轮有限证据范围；大规模真实主任务扩建仍需核查固定相机录制、修复队列、完整性和许可 | 当前弱观察不得冒充完整真实 GT；不因继续预处理延迟已接受的小面板诊断 |
| P2：benchmark 冻结 | 固定数据 family split、任务输入输出、renderer、终态/事件指标、候选 bank、物理预算和统计协议 | 通过 G2；当前 2,000 条开发数据不得改名为盲测测试集 |
| P3：backend smoke | 在明确资源授权后验证本地 vLLM/API 的模型身份、图像支持、单位、JSON 输出、usage 和错误路径 | 仅确认工程可用性，不报告性能 finding |
| P4：开发诊断 | 先运行 E1 原生预测/直接规划与 E2 动作/物理干预，再按额度扩展 E3 候选排序和 E4 feedback 归因 | 所有方法共享案例、可见信息、候选和真实物理查询预算 |
| P5：迁移与不确定性 | 条件满足时运行 E5 物理/观测 shift 和 E6 真实 factual compatibility | 不把未知真实动作拟合值当作 ground truth |
| P6：贡献选择 | 根据正结果、负结果和失败归因，在 empirical benchmark 与新机制之间选择论文主线 | 通过 G3；不为追求正结果强行加入新方法 |
| P7：正式评价 | 冻结 checkpoint、prompt、seed、测试集和统计方案后执行正式比较 | 通过 G4；产出可复现表格、失败记录和证据包 |

近期可执行清单：

- [ ] 重新确认 GPU、API、CPU worker 和 storage 的实际额度。
- [ ] 完成 100 个作者 snooker 录制和 370 条比赛候选的定向 QC。
- [ ] 独立验证 native frame、BEV 与 manual raster 的坐标映射误差。
- [ ] 明确比赛视频的处理、衍生数据发布和原媒体再分发权限。
- [ ] 固定 common development panel、candidate bank、指标阈值和预算账本。
- [ ] 在 mock 测试通过后执行最小 backend smoke，再决定 E1-E4 的共同样本量。
- [ ] 根据诊断结果决定是否需要小型 dynamics/proposal 模型；默认不提前训练新模型。

## 安装

### 前置条件

- Linux 环境（部分数据管线使用 `fcntl`）。
- Python `>=3.12,<3.14`。
- FFmpeg/FFprobe（仅视频采集与验证流程需要）。

建议使用独立虚拟环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://archive.panda3d.org/simple/ \
  panda3d==1.11.0.dev3702
.venv/bin/python -m pip install -e '.[dev,data,processing,inference]'
```

Pooltool 0.6.0 在当前 Linux 配置下依赖 Panda3D 官方归档中的开发 wheel，因此先单独安装
固定版本；不要绕过依赖检查或静默替换物理引擎版本。

## 快速验证

以下命令只创建小型本地开发产物，不调用真实模型：

```bash
.venv/bin/python -m inverser source-catalog \
  --registry configs/sources.json

.venv/bin/python -m inverser generate-pilot \
  --output data/demo-pilot \
  --count 20 \
  --seed 20261010

.venv/bin/python -m inverser audit-pilot \
  --output data/demo-pilot \
  --replay-count 5

.venv/bin/python -m inverser study-dry-run \
  --input data/demo-pilot/privileged.jsonl \
  --output artifacts/demo-study-dry-run
```

完整数据处理命令和限制见
[benchmark pilot 协议](docs/benchmark/benchmark_protocol.md)与
[视频处理协议](docs/pipelines/video_processing.md)。

## 开发与验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy
```

运行产物应写入新的版本化目录；使用相同配置可恢复同一运行，改变源码、输入或配置时不得
静默追加到旧结果。

## 项目结构

```text
InverseR/
├── configs/                 # 数据来源、许可和访问状态
├── data/                    # 本地原始/转换数据（Git 忽略）
├── artifacts/               # 本地报告、预览和运行回执（Git 忽略）
├── docs/
│   ├── benchmark/           # 任务合同与评测协议
│   ├── data/                # 数据卡、来源和 schema
│   ├── pipelines/           # 视频处理、同步和质量筛选
│   └── research/            # 研究问题、文献边界和实验计划
├── src/inverser/            # Python 包与 CLI 实现
├── tests/                   # 单元与回归测试
├── pyproject.toml           # 包、依赖与工具配置
└── requirements.lock.txt    # 当前已验证环境的解析版本快照
```

完整文档导航见 [docs/README.md](docs/README.md)。

## 数据、复现与安全边界

- `data/`、`artifacts/`、模型权重、缓存、凭据和本地 PDF 不进入 Git。
- 原始媒体保持只读；转换结果必须记录来源、配置、哈希、失败和缺失值。
- API 凭据只在运行时传入，不写入配置、日志、请求 URL 或仓库。
- 自动提取的真实观察属于 Silver/weak labels，不是人工验证 ground truth。
- 逆向拟合的速度、触点和 spin 是候选解释，不是球员实测控制。
- simulator success 不等价于真实球桌执行成功。
- 可公开访问不等价于允许训练、生成衍生数据或重新分发原媒体。

## License

本仓库中的原创代码和文档使用 [MIT License](LICENSE)。第三方数据、视频、论文、
模型权重和衍生材料不因本仓库许可证而改变其原始权利；使用前请分别核对来源许可证、
署名要求和允许用途。
