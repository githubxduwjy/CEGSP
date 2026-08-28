# TDBT Clean-Room Protocol

日期：2026-08-26

状态：active protocol for future TDBT idea refinement and experiment planning.

## 1. 目的

本协议把 TDBT idea 与 R014-R058 以及早期 G4090 诊断报告隔离。旧实验不删除、不否认，但从现在起只作为背景参考、工程约束和风险提示；它们不能直接决定新的方法叙事、超参数、gate 或实验结论。

核心原则：

> New TDBT ideas must be derived from the ternary PTQ problem itself, then tested by fresh pre-registered experiments. Legacy results may motivate caution, but may not steer the answer.

## 2. 新问题锚点

后续 TDBT 只围绕这个锚点展开：

- Bottom-line problem：固定三值码本下，PTQ 一次性选择的离散 codeword 与预训练模型函数之间存在 compatibility gap；QAT 能通过量化前向下的状态调整缓解该 gap，但成本高。
- Must-solve bottleneck：找到一个不训练 latent FP weights、不依赖目标模型 QAT teacher 的 PTQ 级机制，利用三值 `{-1,0,+1}` 的支撑/极性结构改善量化后函数保持。
- Non-goals：不是继续修补 PT² 的 hard-T gate；不是证明旧 zero-mediated sign flip 必然成立；不是把 QAT checkpoint/logits/latent weights 蒸馏进主方法；不是通过事后调参救单次结果。
- Constraints：单张 RTX 4090 24GB 可验证；每个核心实验必须有 fit/validation/untouched 分割；strict PTQ 方法 wall-clock 目标不超过 direct PTQ 的 3 倍。
- Success condition：新方法在预注册对照下优于 direct/one-shot/endpoint-only PTQ，并能说明收益来自三值支撑/极性机制，而不是更多计算或 teacher 信息。

## 3. Legacy Evidence Quarantine

| Legacy source | 允许用途 | 禁止用途 |
|---|---|---|
| R014-R058 hard-T / trust-gate 系列 | 提醒低比特离散搜索容易过拟合、nonfinite、成本膨胀；作为负面历史背景 | 不能作为 TDBT 的失败证据；不能沿用其中的 epsilon、mask、projection、layer selection 或 gate |
| `PTQ158_DIRECTION_REVIEW_AFTER_R058.md` | 参考“不要钻牛角尖”和“需要转向”的判断 | 不能把里面的方案自动视为 TDBT 前提 |
| `EXPERIMENT_LOG_G4090_20260826.md` | 参考 4090 可运行配置、显存、脚本 bug、correct next-token CE 修正 | 不能用其中的小 holdout 数字决定新实验阈值或宣称最终 gap |
| `TDBT_EVIDENCE_NEXT_STEPS_20260826.md` | 作为旧阶段证据盘点和风险清单 | 不能作为 active experiment plan；其中的 next steps 必须重新预注册后才可执行 |
| `G4090-TDBT-00/01/02*` | harness、局部 barrier、QAT/PTQ gap 的诊断证据 | 不能证明 TDBT 主方法有效；不能让 zero-mediated sign flip 成为默认核心 claim |
| Li et al. 2026 paper digest | 理论动机：QAT 的 quantized-weight gradients 可能把量化点推回 low-loss basin | 不能推出 TDBT 一定有效；不能要求真实三值轨迹必须显式经过 zero state |

## 4. 可继承与不可继承

可以继承：

- 修正后的 explicit next-token CE 评估方式；
- 4090 24GB 下可行的模型规模、batch/sequence 显存经验；
- “QAT reference 不等于 teacher”的术语边界；
- 对过拟合、calibration split leakage、nonfinite 和成本膨胀的审计规则。

不可继承：

- R014-R058 的 hard-T gate、projection mask、epsilon、trust threshold；
- 旧实验里因结果好坏形成的 layer 偏好；
- 旧计划中的任何未重新预注册的 success gate；
- `QAT-assisted` 分支作为 strict PTQ 主方法；
- “zero-mediated sign flip 是核心机制”的默认假设。

## 5. Future Idea Rules

1. 每个新 TDBT idea 必须先写“从三值量化本身推出的机制”，再列“旧证据只是如何提示风险”。
2. 每个实验计划必须有独立编号，使用 `TDBT2-*` run namespace 和独立 tracker。
3. 每个 plan 必须包含 `Legacy evidence used` 小节，说明用了哪些旧结果、只用于什么作用。
4. 每个 plan 必须包含 `Fresh falsification gate`，新 idea 只由新 gate 判断。
5. 若旧结果与新实验冲突，以新实验的预注册设计为准；旧结果只能触发复核，不能直接推翻新 idea。
6. strict TDBT 不允许读取目标模型 QAT checkpoint、QAT logits、QAT latent weights 或 QAT state prior。
7. 若运行短 QAT 作为诊断，它必须标记为 `diagnostic QAT reference`；若它参与生成 prior，则整条分支必须标记为 `QAT-assisted`，不得冒充 strict PTQ。

## 6. 新实验入口

后续不再从 `EXPERIMENT_PLAN_TDBT_4090.md` 直接启动实验。新的顺序是：

```text
TDBT_CLEANROOM_PROTOCOL.md
  -> independent method note
  -> pre-registered TDBT2 experiment plan
  -> EXPERIMENT_TRACKER_TDBT_CLEANROOM.md
  -> run only the next minimal discriminative experiment
```

新方法 note 至少回答：

- 这个机制为什么是三值特有，而不是普通 2-bit/low-bit 都能套用？
- 它弥合的是 QAT-PTQ gap 的哪一部分？
- 它是否需要 teacher 信息？如果需要，为什么不再叫 strict PTQ？
- 它的最小反证实验是什么？

## 7. 当前默认立场

当前不把 TDBT 写成已经成立的方法。更合适的状态是：

> TDBT is a clean-room hypothesis family inspired by QAT/PTQ gap diagnostics. Legacy experiments motivate the problem and reveal hazards, but the next method and its claims must stand on newly pre-registered evidence.

