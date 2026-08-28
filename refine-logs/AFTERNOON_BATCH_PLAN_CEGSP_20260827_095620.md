# CEGSP 下午论文级验证批次

日期：2026-08-27

## 1. 批次目标

当前已有证据表明：CEGSP 在 direct ternary PTQ 之上能够跨 offset、Wikitext/C4、OPT 多个规模以及 Pythia-1B 改善 NLL；support relocation 也已经在 matched control 中显示出三值结构信号。

但这些结果还不能回答论文最重要的竞争性问题：

> CEGSP 是只修复了一个较弱的 direct 初始化，还是对已有强三值 PTQ 也有独立价值？

因此下午不再做新的局部模块扫描，而执行一次“强基线审计 + 2×2 组合实验”。

## 2. 固定主张与反主张

**主张 C1：** 在部署后的三值点使用 CE gradient 进行少量离散编辑，可以在不使用 QAT teacher、latent FP weight 或 optimizer update 的条件下改善函数保持。

**主张 C2：** CEGSP 的收益不是 direct baseline 特有的；它至少能够作为现有强三值 PTQ 的低成本修复层，或在同口径下取得有竞争力的结果。

**必须排除的反主张：** CEGSP 的提升完全来自弱 baseline、更多计算、不同有效 bit-width 或 calibration/test 泄漏。

## 3. 下午实验矩阵

### M0：强基线审计（必须先完成）

Run ID：`CEGSP-11A-AUDIT-OPT350M`

模型：OPT-350M；RTX 4090 24GB；单 seed。

系统：

1. FP16 reference；
2. direct ternary（当前 canonical CEGSP 起点）；
3. PT² ATQ full（ITF+AGA，优先使用官方实现；SSR 作为预注册的单独配置）；
4. 若 PT² 接口无法迁移，再记录 `baseline-reproduction-failed`，不把 CEGSP 对 direct 的提升写成 SOTA。

固定设置：W1.58A16 weight-only、group/block size 128、相同 tokenizer、相同 calibration token budget、相同 Wikitext/C4 split、相同 FP16 推理 dtype。PT² 的 `mu/alpha`、outlier、residual、mixed precision 和 metadata 必须单独记录，并计算有效 bpw。

M0 只验证：指标、split、finite、模型输出、耗时、显存和码本是否可公平比较；不在 untouched holdout 上调参。

### M1：决定性 2×2 组合实验（M0 通过后立即运行）

Run ID：`CEGSP-11B-2X2-OPT350M`

| 起点 | 不加 CEGSP | 加 CEGSP |
|---|---|---|
| direct ternary | direct | direct + CEGSP joint-k25 |
| PT² ATQ | PT² | PT² + CEGSP joint-k25 |

`PT² + CEGSP` 使用 PT² 实际导出的三值状态。若其形式为

\[
W_q=\mu+\alpha S,\qquad S\in\{-1,0,+1\},
\]

则第一版只编辑 (S)，冻结 (mu,alpha)，不重新拟合量化器。这样能隔离 CEGSP 的离散函数修复能力，并保持严格 PTQ 定义。

### M2：规模确认（M1 接口通过后）

Run ID：`CEGSP-11C-2X2-SCALE`

模型优先级：OPT-1.3B → OPT-2.7B；Pythia-1B 继续作为已有跨架构 direct/CEGSP 证据，不因 PT² 没有 adapter 而冒充强基线比较。

系统：direct、direct+CEGSP、PT²、PT²+CEGSP。主配置固定 `k25`；`k50` 只作为预注册敏感性点，不由 holdout 选择。

数据：Wikitext-2 32 个 untouched batches + C4 32 个 untouched batches；若 OPT-2.7B 资源不足，启动前固定为 24+24，不能跑完后修改。

## 4. 统一评价与 gate

首要指标：Wikitext/C4 untouched per-batch NLL、相对 direct 和相对 PT² 的 paired delta、95% bootstrap CI。

次要指标：PPL、编辑数量、非零率、有效 bpw、量化/编辑 wall-clock、峰值显存。

通过条件：

- M0：PT² 至少在 OPT-350M 上可复现，或明确标记不可复现；所有结果 finite，配置和 split 完整；
- CEGSP standalone：相对 PT² 不劣，且 Wikitext 与 C4 的 aggregate delta 不恶化；
- CEGSP complementary：即使 CEGSP standalone 弱于 PT²，只要 `PT²+CEGSP` 相对 PT² 在两个 holdout 上稳定改善，仍保留论文主线，但改称 quantized-point function-repair layer；
- 若 standalone 和组合均失败，只能保留“优于 direct 的诊断性结果”，不得继续添加模块来掩盖该失败。

## 5. 结果解释规则

| 结果 | 论文定位 |
|---|---|
| CEGSP > PT² | 独立三值 PTQ 方法，具备竞争性主张 |
| CEGSP < PT²，但 PT²+CEGSP > PT² | 现有三值 PTQ 的低成本正交修复层 |
| CEGSP < PT²，且 PT²+CEGSP ≈ PT² | 当前方法主要依赖 direct 初始化，主张收窄或停止算法主线 |
| PT² 无法复现 | 只说明 baseline harness 问题，不作方法结论 |

单次负结果不改变方法方向；只有统一 M1/M2 后的组合失败才触发止损判断。

## 6. 暂不运行的内容

- 不做新的 projection mask、epsilon、threshold 或逐层枚举；
- 不加入 QAT teacher、QAT logits、latent weight 或 optimizer update；
- 不在下午同时堆叠 PTQTP、ScaleQ-1.58、TWLA 等不同 bit/calibration/activation 口径的方法；它们先做有效 bpw 与 calibration 假设审计；
- 不立即训练 QAT。QAT 只在强基线 2×2 完成后，用于单独测量 PTQ–QAT gap 和成本。

PT² 是当前最优先的共同强基线；PT-BitNet 只有在下午 M0 完成且接口、bit accounting 可比时，才作为后续 secondary baseline。PT² 的 ATQ/ITF/AGA/SSR 与其他三值 PTQ 的表示差异必须按原论文和实现记录，不能只比较最终 PPL。[PT²-LLM](https://arxiv.org/abs/2510.03267) [PT-BitNet](https://www.sciencedirect.com/science/article/pii/S089360802500735X)

## 7. 预期成本与交付物

- M0：约 0.5–1.5 GPU 小时，主要风险是官方 PT² 接口和当前 harness 不一致；
- M1：约 1–2 GPU 小时；
- M2：约 3–6 GPU 小时，视 OPT-2.7B 的加载和 PT² 运行速度决定；
- 首批交付：`result.json`、每 batch NLL、配置快照、有效 bpw、finite 检查、耗时/显存和 2×2 汇总表。

下午的真正决策点是 M1：先证明 CEGSP 对强三值起点仍有价值，再决定是否投入 QAT gap、下游任务和更多架构的论文级资源。
