# CAPO：本地多模态调用与外部资产检查

所有命令从 `InverseR/` 执行。下面的 smoke 只使用已有 `training-set-v1`，
不访问 formal/frozen test，也不评估物理准确率。实际新 benchmark 的 T1–T5 尚未生成。

## 1. 已验证的 A100 启动配置

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"

CUDA_VISIBLE_DEVICES=0 .venv-train/bin/vllm serve \
  /data/v-guoqingluo/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 18000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
  --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching \
  --reasoning-parser qwen3
```

首轮权重加载成功但 sampler 在初始化时要求 `nvcc`，系统不存在该工具；
使用已安装的 PyTorch sampler 后服务成功启动，没有安装/修改系统 CUDA。
失败日志 `artifacts/capo-vllm-smoke.log` 与成功启动日志
`artifacts/capo-vllm-smoke-v2.log` 均保留。

本地与代理新路由使用明确的“输出符合 output_schema 的答案，不复制 schema/task”指令。
Azure 历史路线的 system message 不变。本地训练环境复用相同 system message，
避免训练/推理提示不一致。JSON mode 是语法约束，不保证模型遵循答案 schema；
schema 错误仍按 invalid 记账，绝不从错误文本里修造正确答案。

## 2. 有界 smoke 命令

```bash
.venv/bin/python -m inverser vlm-backend-smoke \
  --training-set data/research/training-set-v1 \
  --output data/research/capo-local-qwen35-thinking-off-smoke-v2 \
  --backend local --model Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-vllm-url http://127.0.0.1:18000/v1 \
  --reasoning-effort none --count 8 --max-output-tokens 2048 --execute
```

- 去掉 `--execute` 只生成协议，不调用模型。每次最多 8 个训练场景。
- thinking-on 使用 `--thinking --reasoning-effort high` 和新的 output 目录。
  本地 `high` 是实验记录标签，不意味着 vLLM 会按云端 effort 标准分配推理长度；
  实际 template 开关、输出上限和 usage 才是生效条件。
- 已验证的有界 thinking-on 参数为
  `--thinking --thinking-token-budget 1024 --max-output-tokens 2048`。
  这是 vLLM 0.33.0 本地参数，不发送到反向代理。
  JSON mode 同时约束最终答案，reasoning parser 分离推理段。
- 输出目录锁定 prompt/producer/数据/模型 revision 与 token 预算；
  `JobEnvelope` 和 `TrialLedger` 支持相同协议恢复，已有失败不会自动再次请求。
- 首个 HTTP、transport、身份或 schema 失败停止 smoke。所有失败保留，不能只报成功重跑。
- provider reasoning tokens 原样记录；缺失保持 `null`，不拿字符数猜 token。
  返回的可见 reasoning 文本与最终 JSON 分开存储，不声称这些文本证明内部因果机制。

实际结果：

| 运行目录（`data/research/` 下） | 结果 | token 记录 |
| --- | --- | --- |
| `capo-local-qwen35-thinking-off-smoke-v1` | 首请求复制 schema、length 截断，停止 | 6,133 total |
| `capo-local-qwen35-thinking-off-smoke-v2` | 8/8 schema 通过 | 33,611 total，0 reasoning |
| `capo-local-qwen35-thinking-on-smoke-v1` | 首请求推理耗尽输出，停止 | 7,192 total，3,072 reasoning |
| `capo-local-qwen35-thinking-on-budget1024-smoke-v1` | 预算有效，但答案带 Markdown 围栏，停止 | 5,223 total，1,023 reasoning |
| `capo-local-qwen35-thinking-on-budget1024-smoke-v2` | 有界推理＋最终 JSON mode，8/8 通过 | 41,787 total，8,184 reasoning |

不通过剥掉代码围栏、猜测缺字段或忽略截断，把早期失败重算成正确。
这些工程阶段协议发生了变更；旧目录只读保留。代码 hash 不匹配时应创建新目录，
而不是强行用当前代码恢复旧协议。

## 3. 代理：最新限制仅试 GPT-5.6

`configs/proxy_model_aliases.json` 只是请求别名映射，不是模型可用性声明。
`INVERSE_PROXY_BASE_URL` 和 `INVERSE_PROXY_API_KEY` 从运行环境注入；
不要把真实密钥写入文档、命令记录或配置文件。

远程代理必须 HTTPS，回环代理允许 HTTP；实际调用需资源授权及非 unverified revision。
默认严格核对返回模型 ID。只有明确使用 `--allow-routed-model-id` 才允许不同返回 ID，
且同时保存 requested_model_id / 实际 model_id，不把路由名伪装成供应商版本。

2026-11-19：仅对 GPT-5.6 Sol 进行了 **1 次**合成双 BEV 图像请求，
输出上限 768，单次总 token 保留预算 16,384。返回 HTTP 400，无 usage；
不能称成本为零、模型支持图像或 smoke 成功。没有重试，没有测试其他系列。
记录：`data/research/capo-proxy-gpt56-smoke-v1/`。
GPT-5.6 Terra 未在当次目录中发现，不进行替代请求。

## 4. 评测阶段扩展到四卡（命令草案，未完成四卡验收）

先结束本项目已有的 smoke 服务，检查各卡占用；不要终止未知进程。
沿用第 1 节的 cache/sampler 环境变量。

Qwen3.5-9B 四副本评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/vllm serve \
  /data/v-guoqingluo/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 18000 \
  --data-parallel-size 4 --dtype bfloat16 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching --reasoning-parser qwen3
```

Qwen3-VL-32B-Instruct 两个 TP=2 评测服务（分别执行）：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-train/bin/vllm serve \
  /data/v-guoqingluo/models/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct --host 127.0.0.1 --port 18010 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching

CUDA_VISIBLE_DEVICES=2,3 .venv-train/bin/vllm serve \
  /data/v-guoqingluo/models/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct --host 127.0.0.1 --port 18011 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching
```

以上是评测分配，不与 GPUs 0/1 rollout、2/3 trainer 的训练分配同时运行。
四卡实际吞吐、显存和两种并行配置均仍需实测。

## 5. 外部资源

```bash
.venv/bin/python -m inverser verify-external-suites \
  --output data/external/verification-v2.json
```

- 固定的五个仓库 commit、HF revision 和逐文件 SHA-256 位于
  `data/external/manifest.json`；不在执行时拉 latest。
- PhysBench 媒体保留原始 ZIP，通过安全成员检查及内存 PNG decode 检查样本，
  不需要先重复解压 7GB。问题与标签按官方 idx 关联。
- PhysBench 当前公开 snapshot 有 200 val + 9,802 test；dynamics 共 3,003，
  其中 collision 663。原始标签全库中有 17 个空答案，另有多答案/大小写/空格形式。
  原样报告，不在 verifier 中默默 normalize、丢题或改分母。
- `asset_status=passed` 只说明下载文件与当前 manifest 匹配且样本可读。
  DeepPHY 三个环境的 runtime smoke 仍未通过，因此总 status 为 incomplete，
  命令返回非零，不能把仓库 clone 等同于基准可运行。
- DINOv2-small 新权重保存在本项目 `models/dinov2-small/`；
  不向共享 `/data/v-guoqingluo/models/` 写入额外文件。
- 不安装 DeepPHY 的完整 requirements：包含旧 torch/pooltool 和 macOS 专用包，
  会破坏现有训练环境。I-PHYRE/PHYRE 仍需隔离 runner 与明确版本审计。

## 6. 软件验收与源码快照

- 全量 Pytest：263 passed。日志 `artifacts/capo-regression-tests-v4.log`。
  唯一 warning 来自原有极远坐标测试中的 NumPy overflow；该测试通过，未作无关重构。
- Ruff lint 与全量 Mypy（47 个源码文件）通过。
- `ModelAdapter` / `StudyAdapter` 均覆盖真实 HTTP shape 的 mock 测试：
  双 BEV、thinking on/off、1024-token 预算、严格默认模型身份/显式 routed-ID、
  缓存恢复、reasoning usage 不重复相加、malformed/截断及运行时 credentials。
- A100 成功 on 模式的 producer hash 与
  `artifacts/capo-verified-source-20261119.tar.gz` 中的源码匹配。
  archive SHA-256：
  `38768b95f7b375632bac2a72241633962648596c3790fb3e641ee8a8f8ca09d6`。
- 随后兼容性复核将 local factory 的 thinking-off 明确编码为 `False`，
  确保基础 `ModelAdapter` 也关闭 thinking；`StudyAdapter` 的已测 HTTP payload 不变。
  两种 client 均已纳入回归测试。当前源码另存 final checkpoint，不冒充旧协议的原始源码。
- 没有运行 CAPO 梯度训练、四卡训练、32B 评测、正式 public/test sweep 或 blind evaluation；
  没有 Git commit，也没有对外发布数据或权重。
