# InverseR

InverseR 研究大模型能否在二维斯诺克物理中**逆向规划**：给定台面初始状态和目标终态，
输出一个物理可行的单杆动作（方向、速度、杆头横向/纵向偏移）。核心问题是
“知道会发生什么”能否转化为“知道该怎么做”。

## 能力链

同一组布局上依次测五个环节：

| 环节 | 内容 |
| --- | --- |
| L1 后果预测 | 给定动作，预测静止终态 |
| L2 干预响应 | 改变动作或物理参数后，预测终态如何变化 |
| L3 候选排序 | 在共同候选 bank 中挑出最接近目标的动作 |
| L4 直接出杆 | 直接输出达到目标的动作 |
| L5 反馈利用 | 根据真实物理执行反馈多轮修正动作 |

## 主要结果（正式测试集，200 个布局）

- **直接出杆几乎全部失败**：五个模型 ≤2/200，几何直接方案 77/200。
- **排序与直线近似相当**：最好的 MRR 为 0.548，直线近似基线 0.512，随机 0.340。
- **只有 GPT 旗舰能用真实反馈**：GPT-5.5 36/64、GPT-5.4 29/64；打乱反馈后均为 0，
  同预算几何＋Powell 为 26/64。

完整结果见[五模型能力链结果](docs/research/formal_model_sweep.md)，
协议见[正式测试协议](docs/research/formal_test_protocol.md)，
论文主线见[论文框架](docs/research/paper_framing.md)。

## 安装

需要 Linux、Python `>=3.12,<3.14`；视频管线另需 FFmpeg。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://archive.panda3d.org/simple/ panda3d==1.11.0.dev3702
.venv/bin/python -m pip install -e '.[dev,data,processing,inference]'
```

Pooltool 0.6.0 依赖 Panda3D 归档中的开发 wheel，需先单独安装。

## 快速开始

以下命令只生成本地小型仿真数据，不调用模型：

```bash
.venv/bin/python -m inverser --help
.venv/bin/python -m inverser generate-pilot --output data/demo-pilot --count 20 --seed 20261010
.venv/bin/python -m inverser audit-pilot --output data/demo-pilot --replay-count 5
.venv/bin/python -m inverser study-dry-run --input data/demo-pilot/privileged.jsonl --output artifacts/demo-study-dry-run
```

测试与检查：

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy
```

## 项目结构

```text
configs/        数据来源与许可登记
docs/           研究计划、实验结果、数据卡和管线说明（导航见 docs/README.md）
src/inverser/   Python 包与 CLI
tests/          单元与回归测试
data/, artifacts/  本地数据与运行产物（不入 Git）
```

## 数据与边界

- 原始媒体、转换数据、运行产物、模型权重和凭据都不进入仓库。
- 仿真成功不等于真实球桌成功；自动提取的真实观察是弱标签，不是人工验证的真值。
- 比赛视频的媒体许可尚未核实，不能据此推断训练或再分发权限。

详见[数据卡](docs/data/data_card.md)。

## License

原创代码和文档使用 [MIT License](LICENSE)；第三方数据、视频和模型不受本许可证覆盖。
