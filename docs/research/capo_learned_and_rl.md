# 学习型基线与 GRPO 联调

## 已实测的学习型基线

训练只用冻结的 24,000 个 train 场景（12,000 对），包含全部四个
family。单种子 `20261119`，评测同一套 240 个 holdout；未打开正式 test。

| 适配基线 | 说明 | 训练 | T4 holdout |
|---|---|---:|---:|
| Object-State Transformer + CEM | 结构化初始球位与四维控制；六个时刻、落袋、碰撞及有序库边分类 | 30 epochs，5,640 步 | 7/240 |
| DINO-WM + CEM | 冻结 DINOv2-small；六层动作条件 ViT 预测下一决策点（静止终态）的 patch features | 20 epochs，10,000 步 | 2/240 |

DINO 路线只用初始与目标的 BEV patch 特征规划，**不读取结构化目标或见证动作**；冻结
encoder 权重与预处理文件均记录哈希。它是面向单杆终态的 DINO-WM 适配，不应称为
对上游所有实验的复现。Object-State 是带结构化输入的对照，不能视为相同视觉输入。
未用 holdout 选择 epoch，表中是事先指定步数的最终 checkpoint。

公平的预算是每个场景 **5×128** 个 learned-model 候选（每轮 16 elite），而不是
640 次仿真。CEM 规划仿真调用均为 **0**，选出动作后才用 Pooltool 隐藏裁判一次；
每模型共有 240 次裁判。DINO 不报告 T1–T3，状态基线的 T1/T2/T3/S_total
分别是 0.8792/0.7500/0.0843/0.5225。DINO 减 State 的 T4 差异为 −2.08
个百分点；10,000 次按布局配对 bootstrap 的 95% 区间是 [−4.17, 0.00] 个百分点。

回执分别位于 `data/research/capo-{state,dino}-wm-holdout-seed20261119-v1/`，
配对分析在 `data/research/capo-learned-wm-paired-holdout-v1/`。
这些是开发 holdout 结果，不是最终排行榜。

## 事件引导 / 反思提示适配

`run-agentic-baseline` 提供两条本地 VLM 对照：`cuetip` 先提议动作和语义接触序列，再
用最多六次真实结果做 dual annealing；`wm_reflection` 逐次提出接触预测与动作，根据
真实结果和图像修正，最多六次。两者各保留最多一次独立裁判，以明确区分搜索与评分。
它们采用公开方法的算法或提示结构，但连续动作、交换性 endpoint 和预算是本研究适配。
两条八场景 smoke 分别执行 8/48 次模型调用和各 56 次模拟器评估，均只作为验收。

## GRPO 联调证据

- T4 两步、G=8 已实测，16 个有效动作、16 次模拟奖励；梯度 norm
  0.43956 与 0.60713，LoRA checkpoint 已保存。vLLM NCCL 权重同步之后实际生成；
  首次缺少 reasoning parser 的 400 记录保留。
- Twin forward + DAPO 两步已实测；每次共同采样 anchor/twin 两组各 8，
  共 32 个 completion、28 个通过 forward schema，梯度 norm 0.03483 与
  0.03103。两个接受组都非零方差；这次 GPU smoke 没有实际触发拒绝分支，
  不把它当成实际拒绝/重采验证。单 GPU trainer 上才允许这版配对实现。
- 方差过滤发生在优化前；两个组中任何一个全同分，就丢弃整个配对并从同阶段重新采样，
  最多尝试八次，不把零优势等价为 DAPO 动态过滤。
- T5 使用 TRL environment/tool loop，最多六次试杆，每次返回带刻度的低分辨率
  终态图和残差，工具返回 token 不参与策略梯度。实测两步共 55 次工具提交、54 次
  真仿真；梯度 norm 0.29503 与约 6e−15，第二步近零不作学习证据。
- 同一优化器的 2/2/3 步联合轮转（正向/T4、正向/T4、正向/T4/T5）单 rank 已完成：
  7 次有限且非零梯度、3 个零方差批被换样、186 次仿真，阶段 1/2 和最终权重均已保存。
  此候选已合并为 bf16，尚不是正式选优 checkpoint。

## 两 rank 及后训练对照

`--fsdp` 配合 `torchrun --nproc-per-node=2` 启用 FSDP2。初始 GPU 0/1
为回环 TP=2 vLLM，GPU 2/3 是各持有一组 G=8 的训练 rank。forward 的 twin
奖励在 rank 汇合后统一计算；反向/T5 的方差过滤也使用两组完整回执，保证没有
rank 独自丢弃采样。这个拓扑和 NCCL 权重同步已经跑过前六次真实优化。

首次四卡 T5 暴露上游 TRL 在各 rank 有不同工具续写数量时的等长切片错误。
恢复 run 从第 4 步 collective 权重、optimizer、scheduler 和 RNG 回执重启阶段 3；
保留原始失败日志，不把技术恢复计作独立 seed。每 rank 的后续工具调用改为
对同一 vLLM 权重快照的本地请求，不让较快结束的 rank 进入额外 collective。
恢复为阶段续跑，不宣称私有 pair sampler 会重现失败运行的逐条抽样。
恢复 run `capo-grpo-joint-fsdp-smoke-v2` 已完成第 5–7 步，七步梯度记录全部非零，
最后一步 norm 0.003862；恢复段有 61 次 T5 工具提交、59 次真实仿真。最终
`checkpoint/full_adapter` 68/68 个 LoRA 张量均通过验证。59 是恢复段的仿真
总数（含该段的 T4），不是只计 T5 或整条七步的计数。

首次 FSDP stage 目录的普通 PEFT safetensors 只含 4/68 个张量，不能加载评估；
`pytorch_model_fsdp.bin` 的 collective 文件含齐 68 个 LoRA 张量。已从两个阶段
文件独立导出并核对 68/68 键、形状、有限值和 SHA-256。后续训练使用
`checkpoint/full_adapter`，遇到部分保存将拒绝导出。完整 LoRA 需要合并视觉投影，
不以 text-only 运行时 adapter 替代。

`outcome`/`uncoupled` 两个对照使用纯坐标预测与 endpoint 奖励，不要求事件字段，
不使用 twin、DAPO 动态过滤或 T5。两者与 CAPO 保持相同总优化步和 forward 步：
interactive 预算在非交互对照中分给 T4；`uncoupled` 先放全部 forward，再放 T4；
`outcome` 交替。vLLM 请求、实际采样及仿真成本需要分别报告，不声称三个臂总成本相同。
`scripts/capo_train_queue.sh` 支持三 seed 顺序执行且等待四卡验收。14 步开发
pilot（4/4/6）排在两个对照 smoke 之后，固定 3 seeds × 3 RL 臂；pilot 不等同于
正式训练。阶段候选、最终适配器与选优结果分别存档。

32B TP=2 完整 holdout T1–T4 已结束：480 个请求均完成，T4 0/240、T1 40.83%、
T2 12.08%、T3 0.40%、S_total 16.03%。134 个真实碰撞场景中 27 个输出格式无效，
其余 107 个格式有效的全部预测为无物体球碰撞。两类失败分别报告。
见 `capo-holdout-qwen32-and-wm-analysis-v1/analysis.json`；首次超时运行是单独账本，
未并入结果。
同一回执可无模型/仿真地生成 `causal_alignment.svg` 与 `regime_transfer.svg`；
两次运行逐字节一致。reasoning token Pareto 只采用报告了该字段、同场景集合的
forward 结果；未知 usage 不作零处理。未知摩擦曲线只在实际 OOD 评估后生成。

Checkpoint 选优用 `select-capo-checkpoint`，只接受完整 240×3 的 holdout
任务和无未决请求的评估。固定指标为 `mean(S_total, T4, T5)`，平分按 checkpoint
manifest hash 判定。冻结 test 仍未启用；没有完整 holdout 就不签发选优回执。

## 外部 DeepPHY 本地基线

I-PHYRE 使用固定渲染补丁与真实评分，9B thinking-off 覆盖全部 40 题，解出 6；
347 次模型调用、336 次仿真，账本没有未决请求。PHYRE 官方 0.2.2 wheel
在 Python 3.9 隔离运行，使用 DeepPHY 8×8 格与 8 档半径；首八个 fold-0 测试题
为 0/8（80 次请求与仿真），这不是 500 题全量分数。两条协议只提供初始图和实际
文本反馈，不复现上游多帧 history。

九球使用 Pooltool 0.6，保留上游离散控制、最低球自动瞄准、继续局面与 scratch。
首个 smoke 八局共用同一开局，结果完全相同，只能用于接口验收；
后续版本按 game ID 固定不同 rack seed，跨进程复核同 ID 双跑一致、不同 ID 不同，
并重新运行八局：八张不同开局 PNG、8 条不同对局轨迹，4/8 解出，107 次模拟
（16 次 scratch，不曾选择最高档）。两版不可混作同一组样本。反代无新增请求。

自验证的 9B base 两场景 smoke 共 32 次模型调用、两次隐藏裁判，规划阶段零仿真；
两个场景都未命中。该入口不是经过 CAPO checkpoint 的最终自验证结果。
MMMU validation 和 RealWorldQA test 的本地入口各完成 8 个任务的检查（3/8、5/8）。
计分是显式记录的规范化 exact match，不把子串当作正确或当作官方 MMMU 开放题评分。
RealWorldQA 固定数据卡标注 `CC-BY-ND-4.0`；不向外部服务发送图像、不发布衍生图片。

服务用隔离回环地址、`--reasoning-parser qwen3`、prefix caching 和禁用 FlashInfer
sampler 的本地配置。已安装的 TRL 1.17.1 会对 vLLM 0.33.0 发出版本范围警告，
Transformers 训练暂用 SDPA 和正确但较慢的 Qwen reference kernels。混跑需要监控显存，
不能将单 rank 训练等同于正式 2+2 A100 方案。
