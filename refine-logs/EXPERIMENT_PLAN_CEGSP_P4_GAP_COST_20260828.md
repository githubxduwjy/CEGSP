# CEGSP P4 论文闭环实验方案与去重检查

日期：2026-08-28  
目标：从“CEGSP 有效”推进到“CEGSP 在 PTQ–QAT gap 与成本维度上的论文闭环证据”。

## 1. 已有证据与去重结论

### 已完成，不重复

| 已完成实验 | 已回答问题 | 是否重复 |
|---|---|---|
| P2A/P2B OPT-350M | 固定 top-6 在 W2/C4、不同 offset 上是否改善 | 不再重复 |
| P2C2 Pythia-1B | 非 OPT 架构 adapter 是否有效 | 不再重复 |
| P3A OPT-350M | canonical fixed-rule 新 offset，W2+C4 是否改善 | 不再重复 |
| P3B Pythia-1B | canonical fixed-rule 新 offset，W2 是否改善 | 不再重复 |
| P0 OPT-125M | 小模型上初步 QAT gap 与 score validity | 不作为论文主设置，不能替代 P4 |

### 暂缓，不在本次跑

| 候选实验 | 暂缓原因 |
|---|---|
| 继续换模型 | P2C2 已经回答初步跨架构；当前瓶颈不是模型数量 |
| mixed/joint 新消融 | 容易回到模块搜索；support relocation 已成为主方法 |
| P4-3 score-validity 大规模候选 | P0 已有小模型证据；应在 gap/cost 后再做主设置机制图 |
| P4-4 多 seed/offset replication | P3 已经完成一次新 offset；等主表规则完全冻结后再做 |
| P4-1 strong PTQ + CEGSP | 重要，但需要先审计强 ternary PTQ/official PT² 的可比口径，避免又跑出不可比 baseline |

## 2. 本次实际运行：P4-2 QAT gap/cost

本次只跑：

```text
CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2
```

核心问题：

> 在同一模型、同一 direct ternary 初始化、同一 calibration/validation/untouched split 下，CEGSP 能恢复多少 PTQ–QAT gap？它的成本是否仍然是 PTQ 级别？

## 3. 方法对照

| 方法 | latent FP update | optimizer step | backward | 作用 |
|---|---:|---:|---:|---|
| Direct ternary PTQ | no | 0 | 0 | 部署基线 |
| CEGSP | no | 0 | 1 | 一次量化点 CE 梯度 + 三值 support relocation |
| One-Step QAT | yes | 1 | 1 | 回答“为什么不直接做一次 QAT” |
| 10/50-Step QAT | yes | 10/50 | 10/50 | 给出小步 QAT upper reference |

CEGSP 不使用 QAT teacher、不使用 QAT checkpoint、不使用 QAT logits、不更新 latent FP 权重。

## 4. 固定配置

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| layers | 0–23 |
| 数据 | Wikitext-2 + C4 validation report-only |
| seq len | 128 |
| batch size | 1 |
| fit / val / untouched W2 / untouched C4 | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| group size | 128 |
| threshold factor | 0.7 |
| CEGSP layer top-k | 6 |
| max edits | 64 |
| grad batches | 1 |
| QAT etas | 0.0, 0.003, 0.01, 0.03, 0.1 |
| QAT steps | 1, 10, 50 |
| score layers | 13,17,14,19 |
| score candidates | 32 |
| dtype | bf16 |
| GPU target | RTX 4090 24GB |

说明：

- CEGSP top-k 沿用 P3A canonical small-budget，不用 untouched 重新选择；
- QAT eta 只按 validation 选择，untouched/C4 只报告；
- PPL 由 NLL 直接换算，同时保留 NLL 为主分析指标。

## 5. 主要指标

必须保存：

| 类别 | 指标 |
|---|---|
| Accuracy | val/W2/C4 NLL |
| 展示 | val/W2/C4 PPL |
| Gap | W2 gap closure ratio |
| Gap | C4 gap closure ratio |
| Cost | wall-clock |
| Cost | peak GPU memory |
| Cost | backward 次数 |
| Cost | optimizer steps |
| Mechanism | score-validity Spearman、top10% success |

Gap closure：

```text
R_gap = (L_PTQ - L_CEGSP) / (L_PTQ - L_QAT)
```

其中 QAT 使用 validation-selected best multi-step QAT，untouched 只用于最终报告。

## 6. 判据

### Strong PASS

- CEGSP 改善 direct 的 W2 untouched；
- CEGSP 改善 direct 的 C4 untouched；
- QAT reference 优于 direct，使 gap denominator 有意义；
- CEGSP 的 gap closure 为正；
- CEGSP wall-clock 明显低于 50-step QAT。

### PASS

- CEGSP W2 改善，C4 持平或轻微退化；
- QAT 明显更强，但 CEGSP 以明显更低成本恢复部分 gap。

### FAIL，但不推翻方向

- QAT reference 不稳定或不优于 direct：说明 QAT 控制设置需要重新审计；
- CEGSP 只在 direct 上改善但 gap closure 小：claim 收缩为 direct ternary repair，不声称逼近 QAT。

## 7. 云端执行文件

本地脚本：

`/home/x1shan/文档/ChatGPT/PTQ_paper/remote-tools/cegsp_v2_p4_gap_cost_4090.py`

远端计划目录：

`/root/tqgsp-runs/CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2/`

本次不执行 strong PTQ 组合实验，原因是强基线协议尚未完成可比审计。
