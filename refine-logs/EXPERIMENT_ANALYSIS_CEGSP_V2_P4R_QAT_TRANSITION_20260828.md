# CEGSP-V2-P4R QAT Transition Audit 实验报告

日期：2026-08-28  
Run ID：`CEGSP-V2-P4R-OPT350M-QAT-TRANSITION-OFFSET2`  
远端结果：`/root/tqgsp-runs/CEGSP-V2-P4R-OPT350M-QAT-TRANSITION-OFFSET2/result.json`  
远端日志：`/root/tqgsp-runs/CEGSP-V2-P4R-OPT350M-QAT-TRANSITION-OFFSET2/console.log`

## 1. 实验定位

P4 已经证明 CEGSP 在 OPT-350M 上能恢复 PTQ–QAT gap 的一部分，但 P4 有一个弱点：

> One-Step QAT 的 validation-best eta 为 0，导致 one-step baseline 不够强。

P4-R 的目的不是修改 CEGSP，而是加强 QAT 对照：

1. 密扫 one-step QAT 的 eta；
2. 记录 requantization 后实际发生的 ternary transition；
3. 构造 edit-matched one-step QAT baseline；
4. 补充 5/10/20/50-step QAT 曲线。

CEGSP 仍然保持 canonical：

```text
Direct ternary → one quantized-point CE gradient → Q/K → support relocation → fixed top-6
```

## 2. 配置

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| 数据 | Wikitext-2 + C4 validation |
| fit / val / W2 untouched / C4 untouched | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| CEGSP layer top-k | 6 |
| CEGSP selected layers | [13,17,14,19,23,16] |
| CEGSP changed coordinates | 768 |
| one-step eta sweep | 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 |
| multi-step eta | 0.001, 0.003, 0.01 |
| multi-step steps | 5, 10, 20, 50 |
| GPU | RTX 4090 24GB |
| peak memory | 1.20 GB |
| elapsed | 163.33 s |

## 3. 主表结果

| Method | val NLL | W2 untouched NLL | C4 untouched NLL | changed coords | backward/steps |
|---|---:|---:|---:|---:|---:|
| Direct ternary | 8.4695 | 8.4652 | 8.1304 | 0 | 0 |
| CEGSP | **8.2894** | **8.2971** | 7.8402 | 768 | 1 backward, 0 optimizer |
| One-Step QAT val-best | 8.3327 | 8.3245 | **7.8298** | 16,799 | 1 |
| One-Step QAT edit-matched | 8.4594 | 8.4545 | 8.1123 | 528 | 1 |
| 5-step QAT val-best | 8.2372 | 8.2095 | 7.8450 | 144,089 | 5 |
| 10-step QAT val-best | **8.1095** | **8.0249** | 7.8827 | 243,108 | 10 |
| 20-step QAT val-best | 8.2235 | 8.1551 | 7.9301 | 398,550 | 20 |
| 50-step QAT val-best | 8.3178 | 8.3042 | 7.9184 | 332,820 | 50 |

注：

- one-step validation-best 为 eta=0.001；
- edit-matched one-step 为 eta=3e-5，changed coords=528，最接近 CEGSP 的 768；
- multi-step validation-best overall 是 10-step eta=0.003。

## 4. One-Step QAT eta transition audit

| eta | changed coords | val NLL | W2 untouched | C4 untouched |
|---:|---:|---:|---:|---:|
| 1e-6 | 22 | 8.4695 | 8.4643 | 8.1296 |
| 3e-6 | 52 | 8.4686 | 8.4644 | 8.1294 |
| 1e-5 | 169 | 8.4663 | 8.4605 | 8.1242 |
| 3e-5 | 528 | 8.4594 | 8.4545 | 8.1123 |
| 1e-4 | 1,732 | 8.4327 | 8.4293 | 8.0656 |
| 3e-4 | 5,099 | 8.3859 | 8.3827 | 7.9894 |
| 1e-3 | 16,799 | **8.3327** | **8.3245** | 7.8298 |
| 3e-3 | 50,571 | 8.5105 | 8.5340 | **7.7984** |
| 1e-2 | 169,563 | 8.7375 | 8.8360 | 7.9870 |
| 3e-2 | 506,151 | 9.0324 | 9.1245 | 8.2455 |
| 1e-1 | 1,659,876 | 9.3668 | 9.4485 | 8.6549 |

### Transition 类型观察

one-step QAT 的 transition 基本由四类构成：

- 0→+；
- 0→−；
- +→0；
- −→0。

直接符号翻转非常少，直到 eta=1e-2 以上才出现少量 +↔−：

- eta=1e-2：+→− 4，−→+ 6；
- eta=3e-2：+→− 132，−→+ 152；
- eta=1e-1：+→− 1796，−→+ 1817。

这支持一个很重要的三值机制判断：

> 在实际 requantized one-step QAT 中，主要变化不是直接符号翻转，而是经过零态/非零态边界的 support transition。CEGSP 显式选择 support relocation，确实抓住了三值 PTQ 中最主要、最自然的离散变化通道。

## 5. CEGSP vs One-Step QAT

### 5.1 对 validation-best one-step QAT

| Method | val NLL | W2 untouched NLL | C4 untouched NLL | changed coords |
|---|---:|---:|---:|---:|
| CEGSP | **8.2894** | **8.2971** | 7.8402 | 768 |
| One-Step QAT val-best | 8.3327 | 8.3245 | **7.8298** | 16,799 |

结论：

- CEGSP 在 W2 val 和 W2 untouched 上优于 properly swept one-step QAT；
- one-step QAT 在 C4 上略优；
- 但 one-step QAT 需要 16,799 个 ternary coordinate changes，而 CEGSP 只改变 768 个。

### 5.2 对 edit-matched one-step QAT

| Method | val NLL | W2 untouched NLL | C4 untouched NLL | changed coords |
|---|---:|---:|---:|---:|
| CEGSP | **8.2894** | **8.2971** | **7.8402** | 768 |
| One-Step QAT edit-matched | 8.4594 | 8.4545 | 8.1123 | 528 |

结论：

> 在近似相同 discrete modification budget 下，CEGSP 明显优于 one-step latent QAT。

这是目前最强的机制正证据之一：同样使用一次梯度，CEGSP 的价值不是“拿了 gradient”这么简单，而是把梯度转化为结构化、预算受控的三值支撑重分配。

## 6. Multi-Step QAT curve

| steps | eta | val NLL | W2 untouched | C4 untouched | changed coords |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.003 | 8.2372 | 8.2095 | **7.8450** | 144,089 |
| 10 | 0.003 | **8.1095** | **8.0249** | 7.8827 | 243,108 |
| 20 | 0.003 | 8.2235 | 8.1551 | 7.9301 | 398,550 |
| 50 | 0.001 | 8.3178 | 8.3042 | 7.9184 | 332,820 |

观察：

- 10-step QAT 是 W2 validation / W2 untouched 最强；
- 5-step QAT 在 C4 上最接近 CEGSP，甚至略差于 CEGSP；
- 20/50-step 并没有继续变好，说明小校准集下 QAT steps 增加可能带来过拟合或不稳定；
- QAT 的 changed coords 是 CEGSP 的数百倍。

## 7. Gap closure

使用 validation-best multi-step QAT 作为 reference，即 10-step QAT eta=0.003。

| Split | Direct | CEGSP | QAT-ref | Gap closure |
|---|---:|---:|---:|---:|
| W2 untouched | 8.4652 | 8.2971 | 8.0249 | 38.20% |
| C4 untouched | 8.1304 | 7.8402 | 7.8827 | 117.16% |

解释：

- W2 上，CEGSP 恢复了约 38.2% 的 multi-step QAT gap；
- C4 上，CEGSP 优于 validation-selected QAT reference，因此 ratio > 1；
- 这不能写成“CEGSP 优于 QAT”，但可以写成“CEGSP 在 cross-domain transfer 上表现出更强稳健性，需要后续验证”。

## 8. 成本

| Component | time |
|---|---:|
| load tokenizer/data | 38.31 s |
| load model | 2.96 s |
| FP eval/snapshot | 1.06 s |
| direct PTQ/eval | 1.13 s |
| CE gradient collection | 0.18 s |
| CEGSP edit/select/eval | 4.05 s |
| one-step QAT transition grid | 14.17 s |
| multi-step QAT curve | 101.46 s |
| total | 163.33 s |

成本结论：

- CEGSP 核心部分仍是秒级；
- one-step dense eta sweep 已经比 CEGSP 核心更贵；
- multi-step QAT curve 是主要成本来源；
- P4-R 支持 CEGSP 的定位：optimizer-free、预算受控、低成本离散修复。

## 9. Gate 判定

| Gate | 结果 |
|---|---|
| one-step QAT eta transition audit | PASS |
| one-step QAT 不再 eta=0 | PASS，val-best eta=0.001 |
| CEGSP vs one-step val-best | PASS on W2，C4 one-step 略优 |
| CEGSP vs edit-matched one-step | STRONG PASS |
| CEGSP positive gap closure | PASS |
| multi-step QAT reference 有效 | PASS，10-step 最强 |
| CEGSP 成本低于 QAT curve | PASS |

总判定：

```text
STRONG_PASS_QAT_TRANSITION_AND_EDIT_MATCHED_BASELINE
```

## 10. 对论文主张的影响

P4-R 允许我们把 CEGSP 的机制 novelty 说得更强：

> CEGSP is not merely using the same quantized-point gradient as QAT. It converts that gradient into a small, structured ternary support relocation. Under a matched discrete-change budget, this direct discrete repair is substantially more effective than one-step latent QAT.

中文解释：

> CEGSP 的核心价值不是“用了 CE 梯度”，而是“在三值离散空间中直接执行少量、结构化、预算受控的 support transition”。这正是 one-step QAT 通过连续 latent update 很难精确控制的部分。

## 11. 仍需谨慎的地方

1. Strong PTQ baseline 尚未完成，因此不能声称优于最新三值 PTQ。
2. one-step QAT 在 C4 上略优于 CEGSP，说明 cross-domain 结论还要更谨慎。
3. 目前 QAT 只更新 Q/K latent weights，不是 full-model QAT。
4. multi-step QAT 不是 SOTA QAT，只是 matched lightweight QAT reference。

## 12. 下一步建议

下一步应进入 P5：

> Strong PTQ protocol audit。

这一步先不一定需要 GPU。必须确认 strong ternary PTQ 的 codebook、scale、group size、zero-state、calibration/eval 协议能否与 CEGSP 合法对接。  
只有协议审计通过后，才跑：

```text
Strong PTQ
Strong PTQ + CEGSP
```

P4-R 已经足够支撑 QAT baseline strengthening，不建议继续扩大 one-step eta 或 QAT step 搜索。
