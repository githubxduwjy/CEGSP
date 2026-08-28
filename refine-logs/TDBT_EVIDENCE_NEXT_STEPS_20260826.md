# TDBT 当前证据盘点与下一步验证计划

> Clean-room notice：本文档现在仅作为 historical evidence memo 保留。它可以提供背景、bug 记录和风险提示，但不再作为 TDBT 后续 idea 修改或实验启动的 active plan。新的 idea 和实验必须从 `TDBT_CLEANROOM_PROTOCOL.md` 与 `EXPERIMENT_TRACKER_TDBT_CLEANROOM.md` 重新预注册。

日期：2026-08-26

## 0. 研究目标定位

最终目标不是做一串实验记录，而是形成一篇可辩护的论文。当前阶段只负责回答三个问题：

1. 三值 PTQ 与 QAT 之间是否存在不能被普通 FP fine-tuning 解释的 gap？
2. 这个 gap 是否和固定三值网格上的 quantization compatibility / basin correction 有关？
3. 三值状态空间是否提供了一个低成本 PTQ 式弥合路径，而不是只能回到完整 QAT？

因此，目前不能急着启动更大系统。先确认主 claim 的证据边界，再决定方法是否收缩、转向或继续。

## 1. 可以作为可靠证据的结果

只采用修正后的 explicit next-token cross entropy 结果。早期 HuggingFace labels 版本存在二次 shift 问题，保留为 legacy harness 记录，不进入论文主证据。

### 1.1 OPT-125M 固定三值网格 gap

配置：`facebook/opt-125m`，Wikitext-2 train calibration / validation holdout，32/16 batches，batch size 4，seq length 128，group size 128，threshold factor 0.7，W-only fixed-grid ternary QAT 256 steps，lr 5e-5。

| Method | Holdout NLL | 与 direct PTQ 差值 | 备注 |
|---|---:|---:|---|
| FP | 4.0347 | -5.6628 | 全精度参考 |
| direct ternary PTQ | 9.6975 | 0.0000 | 一次性三值化 |
| W-only ternary QAT | 5.4503 | -4.2472 | gap closure 0.7500 |
| FP-FT then ternary PTQ | 9.6225 | -0.0750 | 同预算普通 FP 微调 control |

解释：QAT 的收益远大于同预算 FP-FT 后再 PTQ。当前最强结论是：固定三值网格上的 QAT 改善不是普通 FP 微调能解释的，它更像 Li et al. 所说的 quantization compatibility correction。

### 1.2 OPT-125M QAT 轨迹

| Step | f(Q(w_k)) holdout | f(w_k) holdout | 主要状态变化 |
|---:|---:|---:|---|
| 0 | 9.6389 | 4.0347 | 初始 direct PTQ 附近 |
| 1 | 8.9420 | 4.0219 | 0->nonzero 35875；nonzero->0 35943；sign flip 0 |
| 8 | 7.9887 | 4.0130 | 支撑变化继续主导；sign flip 0 |
| 32 | 6.6309 | 4.1328 | 支撑变化继续主导；sign flip 0 |
| 64 | 6.1087 | 4.1307 | 支撑变化继续主导；sign flip 0 |
| 128 | 5.7854 | 4.2537 | sign flip 21 |
| 256 | 5.4503 | 4.3639 | sign flip 402 |

解释：部署量化 loss 大幅下降，而 latent FP loss 没有同步下降，甚至后期略升。这与 Li et al. 的机制预测高度一致：QAT 不是简单让 FP 模型更好，而是在调整 latent 权重，使其量化后的 codeword 回到低损失区域。

更重要的是，早期改善主要来自 `0 <-> nonzero` 的支撑变化，而不是显式 `+1 -> 0 -> -1` sign flip。这个结果要求我们修改原始 TDBT 叙事：zero state 不能被写成“必要的符号翻转通道”，更合理的三值特性是“零态提供可变支撑预算，使 QAT/PTQ 可以重新分配哪些参数应该参与函数表达”。

### 1.3 zero-mediated path 局部 barrier 诊断

| Run | Model / Layer | FP32 packets | bridge 降低 barrier | mean direct | mean bridge |
|---|---|---:|---:|---:|---:|
| G4090-TDBT-01-FP32 | OPT-125M layer-0 q_proj | 8 | 3/8 | 0.001276 | 0.000669 |
| G4090-TDBT-01-OPT350M-FP32 | OPT-350M layer-0 q_proj | 8 | 7/8 | 0.0000205 | 0.0000104 |

解释：同一终点下，经过 zero intermediate 能降低单步最大 barrier，这支持“三值状态图存在低障碍路径”的局部几何可能性。但它不是端到端性能证据，也不能证明 zero-mediated sign flip 是 QAT 成功的主要机制。

## 2. 当前不能声称的东西

1. 不能声称已经提出并验证了完整 TDBT 算法。我们只验证了 gap、QAT 轨迹和局部 path barrier，还没有跑 endpoint/one-shot/path 的全模型或跨层对比。
2. 不能声称 zero-mediated sign flip 是核心原因。轨迹显示早期几乎没有 sign flip，支撑变化更重要。
3. 不能声称 350M 上已经有可靠 gap。旧 350M 结果属于 legacy measurement，且小 holdout 上 direct PTQ 曾出现不劣于 FP 的异常现象，必须用 corrected metric 重跑。
4. 不能声称方法已经是 PTQ 级低成本。现在的 QAT 诊断用了 256 step，只是 teacher/mechanism probe，不是最终方法成本。
5. 不能声称跨层函数保持已经有效。Q/K、V/O 组合算子 distortion 还没有测。

## 3. 需要明确修改的研究思路

### 3.1 从 zero-channel claim 改成 support-transport claim

旧说法：三值零态是符号翻转的低障碍中间通道。

新说法：三值零态的核心价值首先是支撑可变性。三值权重 `{-alpha, 0, +alpha}` 不只是 2bit 的粗网格，而是把“幅值是否参与表达”和“参与后的极性”拆成两个离散自由度。QAT 轨迹显示，早期部署 loss 改善主要通过支撑集合重分配发生。

论文主线应从 `+alpha -> 0 -> -alpha` 的路径故事，收缩到：

> fixed ternary grid 下，PTQ 的一次性 codeword 选择破坏 quantization compatibility；QAT 通过大量小的 ternary support transitions 改变可量化函数。我们要把这种兼容性修正蒸馏成 PTQ 级的离散支撑运输。

### 3.2 把 QAT 当作机制参照，而不是最终算法

QAT 256 step 的作用是证明 gap 和产生轨迹证据。这里的 QAT 是 diagnostic reference，不是 strict PTQ 的 teacher。最终方法不应依赖完整 QAT 训练循环、QAT logits、QAT checkpoint 或 QAT latent weights，否则必须降级标记为 `QAT-assisted`，并会被质疑接近 ATQ/QAT 成本。

下一步方法应只使用 QAT 中可压缩的信号，例如：

- 初始少量 quantized-gradient；
- 支撑转移方向的 ranking；
- 单层或窗口级 trust region；
- 少量候选验证，而不是持续训练所有 latent weights。

### 3.3 跨层约束应服务于三值支撑，不应成为通用低比特重构

如果只写跨层输出重构，这个方法可以套到 2bit/4bit，创新会变弱。跨层部分必须问一个三值特有问题：

> 哪些支撑位置应该被保留、释放、迁移，才能保持注意力组合算子或 MLP 函数？

因此跨层实验要围绕 support movement、zero budget、Q/K 和 V/O 组合失真，而不是泛泛地加一个 window reconstruction loss。

## 4. 下一步必须验证什么

### E1. corrected 350M gap and trajectory replication

目的：确认 OPT-125M 不是偶然，且旧 350M legacy 结果被正确替换。

最小实验：

- OPT-350M；
- explicit next-token CE；
- Wikitext-2 train/validation，尽量增加 holdout token；
- direct PTQ、W-only fixed-grid QAT、FP-FT then PTQ；
- 记录 `f(Q(w_k))` 与 `f(w_k)` trajectory。

判据：

- direct PTQ 明显劣于 FP；
- QAT 的 deployed loss 下降显著超过 FP-FT then PTQ；
- early transitions 仍以 support movement 为主。

如果失败：不否定三值 PTQ 方向，但要先关闭“350M W2 gap replication”作为证据，换 C4/更大 holdout/另一个模型族确认。

### E2. support transition necessity

目的：证明三值特性不是包装，而是方法有效性的来源。

最小实验：

- 固定相同候选预算；
- 比较 direct PTQ、random matched support changes、gradient-ranked support-only transport、sign-only transport、support+sign transport；
- 在 OPT-125M 上先做 layer/block 级诊断，再上全模型小预算。

关键指标：

- held-out NLL；
- layer output distortion；
- transition count；
- 支撑稀疏率变化或固定预算下的支撑交换收益。

判据：

- support-aware 方法必须稳定优于 random matched 和 sign-only；
- 如果 support-only 已经接近 support+sign，主 claim 应集中到 support transport，而不是 sign path。

### E3. cross-layer composed-operator distortion

目的：回答“为什么不是普通单层 PTQ/GPTQ 类方法”的创新性问题。

最小实验：

- layers 0/7/15/23；
- 测 Q/K composed operator distortion，例如 attention score 或 `QK^T` 误差；
- 测 V/O composed operator distortion，例如 attention value path 或 output projection 后误差；
- 对比 direct endpoint、one-shot quantized-gradient、support transport、support transport + single-layer trust region。

判据：

- 支撑运输若只改善单层 weight/output loss，不改善 composed operator，则跨层 claim 不成立；
- 如果 composed operator 明显改善且 held-out NLL 不退化，才进入跨层联合算法。

### E4. one-shot vs path planning

目的：证明“路径”不是多余的。

最小实验：

- same candidate budget；
- one-shot gradient update；
- K-step support transport with barrier constraint；
- K-step transport without barrier；
- random K-step matched control。

判据：

- path + barrier 必须优于 one-shot 或 no-barrier，否则 paper 主线应改成 quantized-gradient support correction，而不是 basin transport。

### E5. cost gate

目的：防止方法滑向 QAT。

必须记录：

- wall-clock；
- peak VRAM；
- number of teacher/FP forward passes；
- number of quantized-gradient backward passes；
- 相对 direct PTQ 和 256-step QAT 的倍数。

判据：

- 最终 PTQ-style 方法最好控制在 direct PTQ 的 2-3 倍；
- 若超过 5 倍且收益只接近 QAT，不适合作为 PTQ 论文主方法。

## 5. 仍不明确的问题

1. QAT 早期支撑变化到底是泛化机制，还是 OPT-125M/Wikitext/threshold=0.7 的局部现象。
2. 支撑变化是否真的能被少量 gradient/proxy 捕捉，还是必须依赖完整 QAT 训练过程。
3. 单层 trust region 是否会过强，导致跨层有益支撑交换无法发生。
4. zero-mediated barrier 的局部优势能否转化成 endpoint NLL 或 composed operator 改善。
5. 校准集上选出的支撑迁移是否会过拟合，validation/untouched split 是否能复现。
6. 这个方向最强论文 claim 是“解释和诊断 QAT-PTQ gap”，还是“提出一个 PTQ-style 弥合算法”。目前数据更支持前者，后者还需要 E2-E5。

## 6. 下一阶段建议

首选路线：先把主线收缩为 Ternary Support Compatibility Transport。

一句话版本：

> QAT 在固定三值网格上的主要早期收益来自支撑集合的兼容性修正；我们用少量校准梯度和 trust-region 筛选，把这种支撑运输蒸馏成 PTQ-style 的离散更新。

下一批实验不要大铺开。推荐顺序：

1. 先做 corrected OPT-350M trajectory replication；
2. 再做 OPT-125M support-only/sign-only/random matched 消融；
3. 然后做 layers 0/7/15/23 的 Q/K、V/O composed-operator distortion；
4. 最后才实现最小 K-step support transport。

止损条件：

- 如果 E1 不能复现 PTQ-QAT gap，转向“何时存在 gap”的诊断论文；
- 如果 E2 显示 support transport 不优于 random/sign-only，关闭三值支撑 claim；
- 如果 E4 显示 path planning 不优于 one-shot，关闭 basin transport claim，保留 quantized-gradient support correction；
- 如果 E5 成本接近 QAT，方法必须重新设计为少步 proxy，而不能继续堆搜索。
