# Research Direction Lock: PTQ 1.58-bit / Ternary PTQ

日期：2026-08-26

## 1. Frozen Research Problem

后续实验不得改变主问题：

> 三值/1.58-bit PTQ 为什么显著弱于 QAT，以及能否在不进行 QAT、不使用 QAT teacher 的条件下，用 PTQ 级别的量化点信息弥合这部分 gap。

这不是一个“追单次最好结果”的工程调参任务，而是一个机制验证任务。

## 2. Frozen Method Family

当前方法族固定为：

> Quantized-Point Gradient Guided Ternary Editing

它包含若干候选模块：

- CE gradient at deployed ternary weights；
- zero-support relocation；
- nonzero sign / polarity correction；
- layer budget selection；
- calibration split robustness；
- possible future regularization or block constraints。

这些是同一方法族内的模块，不是每次实验后的重新换题。

## 3. What Experiments Are Allowed to Change

实验可以更新：

- 哪个模块可信度更高；
- 哪个模块需要作为 control；
- 哪个超参数范围值得保留；
- 哪个 proxy/objective 不可靠；
- 哪些层/预算存在跨层干扰。

实验不能轻易改变：

- 研究问题；
- PTQ-only 定位；
- 不使用 QAT teacher 的边界；
- 三值量化作为核心对象；
- 最终论文要解释 PTQ-QAT gap 的主线。

## 4. Evidence Levels

单次实验只允许产生以下结论：

- `diagnostic`: 发现 harness 或 metric 问题；
- `module-positive`: 某模块在当前设置有信号；
- `module-negative`: 某模块在当前设置弱；
- `robust-positive`: 多 split/seed/model 后仍稳定；
- `paper-claim-ready`: 通过跨 split、跨 seed、至少两个模型或数据集。

禁止把单次 `module-positive` 直接写成最终论文主张。

## 5. Direction Change Rule

只有满足以下条件之一，才允许正式转向：

1. 同一模块在至少 3 个独立 split/seed 上失败，并且有更简单模块稳定成功；
2. 当前模块违反 PTQ-only 成本边界；
3. 当前模块与三值特性无关，且 control 证明收益来自普通 low-bit trick；
4. 当前模块无法迁移到 untouched/C4 或第二模型。

否则只能说“调整模块权重”，不能说“换方向”。

## 6. Current State After CEGSP-01B

可靠但仍未成论文 claim 的观察：

- CE-gradient editing 在 OPT-350M / Wikitext-2 上有强信号；
- top-k 层预算控制必要；
- joint support/sign editing 当前优于 support-only；
- all-layer editing 会造成跨层干扰；
- 还没有完成 split/seed 稳定性验证。

因此下一步必须是稳健性验证，而不是扩大叙事或扩大模型。

## 7. Next Locked Experiment

`CEGSP-02A`:

- 目的：验证 CEGSP top-k / joint editing 是否对 calibration split 稳定；
- 设置：OPT-350M、Wikitext-2、Q/K、strict PTQ；
- offsets：至少 3 个 calibration/validation offset；
- k：`4,6,8,12`；
- 比较：support-only、signflip-only、joint-best；
- 成功标准：joint 或某一简单 family 在多数 offset 上稳定改善 untouched NLL。

