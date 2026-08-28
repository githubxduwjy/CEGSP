# CEGSP-V2-P4 Gap/Cost 闭环实验分析

日期：2026-08-28  
Run ID：`CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2`  
远端结果：`/root/tqgsp-runs/CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2/result.json`  
远端日志：`/root/tqgsp-runs/CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2/console.log`

## 1. 实验定位

本实验不是新模块消融，而是论文闭环验证：

> 在同一 direct ternary 初始化、同一 calibration/validation/untouched split 下，比较 Direct PTQ、CEGSP、One-Step QAT 与 small-step QAT，测量 CEGSP 能恢复多少 PTQ–QAT gap，以及它相对 QAT 的成本优势。

本实验对应 P4-2。  
P3/P2 已经完成 fixed-rule robustness 和跨架构验证，因此本轮不重复跑 P3，也不继续扩展 action/mixed 模块。

## 2. 去重检查

| 候选 | 是否已由既有实验覆盖 | 本轮处理 |
|---|---|---|
| fixed-rule 新 offset 验证 | P3A/P3B 已覆盖 | 不重复 |
| OPT-350M W2/C4 CEGSP 改善 | P2A/P2B/P3A 已覆盖 | 不重复作为主问题 |
| Pythia 跨架构 | P2C2/P3B 已覆盖 | 不重复 |
| QAT gap/cost on OPT-350M | P0 只在 OPT-125M 做过 | 本轮运行 |
| score-validity on OPT-350M | P0 只在小模型做过，本轮作为附带机制指标 | 本轮记录 |
| strong PTQ + CEGSP | 尚未完成可比 baseline 审计 | 暂缓，不盲跑 |

## 3. 配置与完整性

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| layers | 0–23 |
| 数据 | Wikitext-2 + C4 validation |
| fit / val / untouched W2 / untouched C4 | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| CEGSP layer top-k | 6 |
| score layers | 13,17,14,19 |
| score candidates | 128 total |
| QAT etas | 0, 0.003, 0.01, 0.03, 0.1 |
| QAT steps | 1, 10, 50 |
| dtype | bf16 |
| GPU | RTX 4090 24GB |
| peak memory | 1.20 GB |
| total elapsed | 138.34 s |

环境说明：

- 第一次启动失败，原因是远端 pandas 安装损坏，导致 datasets 无法导入；
- 已只修复 pandas，不重装 PyTorch；
- CUDA witness 通过：PyTorch 2.5.1+cu124，GPU RTX 4090；
- 第二次启动成功并写出 result.json。

该环境故障不计为方法失败。

## 4. 主结果

### 4.1 NLL / PPL

| Method | val NLL | val PPL | W2 untouched NLL | W2 PPL | C4 untouched NLL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|
| FP | 3.7920 | 44.35 | 3.6120 | 37.04 | 3.4365 | 31.08 |
| Direct ternary | 8.4695 | 4766.93 | 8.4652 | 4746.86 | 8.1304 | 3395.99 |
| CEGSP top6 | 8.2894 | 3981.38 | 8.2971 | 4012.02 | 7.8402 | 2540.80 |
| 1-step QAT, val-best | 8.4695 | 4766.93 | 8.4652 | 4746.86 | 8.1304 | 3395.99 |
| 10-step QAT, val-best | 8.1095 | 3325.96 | 8.0249 | 3056.12 | 7.8827 | 2651.08 |
| 50-step QAT, val-best | 8.3580 | 4264.14 | 8.3060 | 4047.89 | 7.9871 | 2942.70 |

### 4.2 Delta vs Direct

| Method | Δval NLL | ΔW2 untouched NLL | ΔC4 untouched NLL |
|---|---:|---:|---:|
| CEGSP top6 | -0.1801 | -0.1682 | -0.2901 |
| 1-step QAT, val-best | 0.0000 | 0.0000 | 0.0000 |
| 10-step QAT, val-best | -0.3599 | -0.4403 | -0.2476 |
| 50-step QAT, val-best | -0.1115 | -0.1593 | -0.1433 |

## 5. Gap closure

使用 validation-selected best multi-step QAT 作为 QAT reference。  
本轮 best multi-step 是 10-step QAT, eta=0.003。

定义：

```text
R_gap = (L_PTQ - L_CEGSP) / (L_PTQ - L_QAT)
```

结果：

| Split | Direct NLL | CEGSP NLL | QAT-ref NLL | Gap closure |
|---|---:|---:|---:|---:|
| W2 untouched | 8.4652 | 8.2971 | 8.0249 | 38.20% |
| C4 untouched | 8.1304 | 7.8402 | 7.8827 | 117.16% |

解释：

- W2 上，CEGSP 用一次量化点 CE gradient + 离散 support relocation，恢复了约 38% 的 10-step QAT gap；
- C4 上，CEGSP 比 validation-selected 10-step QAT 更好，因此 ratio > 1；
- 不能将 C4 ratio > 1 解释为“CEGSP 全面优于 QAT”，因为 QAT eta/steps 是按 W2 validation 选出的，C4 是 report-only transfer。

## 6. 成本对比

| Component | 时间 |
|---|---:|
| tokenizer/data loading | 38.52 s |
| model loading | 3.52 s |
| FP eval/snapshot | 1.28 s |
| direct PTQ + eval | 1.02 s |
| CE gradient collection | 0.20 s |
| CEGSP edit/select/eval | 4.02 s |
| score-validity | 8.64 s |
| QAT controls | 81.14 s |
| total | 138.34 s |

关键成本结论：

- CEGSP 核心额外成本约为一次 CE gradient + 离散编辑/选择评估，约 4.22 s，不含共同的数据/模型加载；
- QAT controls 约 81.14 s，主要来自多 eta × 多 step 的 latent update/quantize/eval；
- 在当前 4090 设置下，CEGSP 显著低于 QAT sweep 成本。

## 7. Score-validity

| 指标 | 数值 |
|---|---:|
| candidates evaluated | 128 |
| Spearman(score, actual improvement) | 0.4347 |
| top10% true improvement rate | 100% |
| all candidate true improvement rate | 98.44% |
| mean top10% Δval NLL | -0.00377 |
| mean all Δval NLL | -0.00221 |

解释：

- 一阶 score 与实际 improvement 有正相关；
- top-score candidate 的平均收益强于全集平均；
- 但 all-candidate improvement rate 已经很高，说明在这些高敏感层内，合法 support relocation 的整体方向大多是有益的；后续机制图应加入随机 layer 和 score-bin 曲线，避免只在高收益层里看相关性。

## 8. 关键观察

### 8.1 One-Step QAT 在本设置下没有形成有效 baseline

1-step QAT 的 validation-best eta 是 0，也就是保持 direct 不动。  
这说明：

- 不能在这组结果里声称 CEGSP 优于有效 one-step QAT；
- 只能说：在当前 eta 网格和归一化 latent step 下，一步 QAT 没有找到 validation 改善；
- 这可能是 QAT step 归一化、eta 网格或只更新 Q/K 的限制导致，需要单独审计。

### 8.2 10-step QAT 是更合理的 QAT reference

10-step QAT 明显改善 W2/C4，且 validation 最优。  
因此本轮 gap closure 使用 10-step QAT 更合理。

### 8.3 50-step QAT 反而弱于 10-step

这提示小校准集下多步 latent update 可能过拟合或偏离；不能简单把更多 step 当作更强 teacher。  
这对 CEGSP 的低成本定位反而是有利的，但仍需谨慎：QAT baseline 还不是充分调优的 SOTA QAT。

## 9. Gate 判定

| Gate | 结果 |
|---|---|
| CEGSP improves W2 untouched | PASS |
| CEGSP improves C4 untouched | PASS |
| QAT reference improves direct | PASS，10-step QAT 有效 |
| CEGSP gap closure positive | PASS，W2 38.20%，C4 117.16% |
| CEGSP cheaper than QAT controls | PASS |
| one-step QAT valid comparison | WEAK / eta=0 selected |

总判定：

```text
PASS_GAP_COST_WITH_WEAK_ONE_STEP_QAT
```

## 10. 对论文 claim 的影响

当前可以支持的说法：

> CEGSP converts a single quantized-point CE gradient into discrete ternary support relocation and recovers a measurable fraction of the PTQ–QAT gap at substantially lower cost than iterative latent-weight QAT controls.

当前不能说：

- CEGSP 优于 QAT；
- CEGSP 优于最新 strong ternary PTQ；
- one-step QAT 被严格击败；
- CEGSP 已经是完整 SOTA 方法。

## 11. 下一步

下一步优先级：

1. **Strong PTQ baseline audit / combination**
   - 需要确认 PT² 或其他 strong ternary PTQ 的 official/reproducible 口径；
   - 核心比较是 `Strong PTQ + CEGSP` vs `Strong PTQ`。

2. **One-Step QAT audit**
   - 重新检查 latent step normalization 与 eta 网格；
   - 不应把本轮 eta=0 当作 one-step QAT 的最终负结论。

3. **Score-validity 机制图**
   - 加入随机 layer；
   - 做 score-bin curve 与 scatter；
   - 用于论文机制图，而不是继续调 CEGSP 方法。

4. **主表前冻结 canonical**
   - support relocation 是主方法；
   - signflip/joint 作为 ablation；
   - PPL 与 NLL 同报；
   - 不用 untouched 选择 k 或 eta。
