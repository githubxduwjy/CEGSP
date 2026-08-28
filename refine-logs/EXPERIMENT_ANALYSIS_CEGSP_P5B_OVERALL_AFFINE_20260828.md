# CEGSP-P5-B 结果分析：整体 affine CEGSP compatibility

日期：2026-08-28  
结果文件：[p5b_overall_affine_result.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5b_overall_affine_opt350m_20260828/p5b_overall_affine_result.json)  
原始日志：[cegsp_p5b_overall_affine_opt350m_20260828.log](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5b_overall_affine_opt350m_20260828/cegsp_p5b_overall_affine_opt350m_20260828.log)

## 结论

P5-B 通过预注册的 `PASS_OVERALL_AFFINE_COMPATIBILITY` gate。

在 OPT-350M 的全部 24 个 decoder layers 上，将 affine ternary codebook 固定为 `Q = mu + alpha*T`，使用 fit split 的量化点 CE gradient 进行 layer ranking 和 relocation selection，主预算 top-6 layers × 64 relocation pairs 在 validation、Wikitext-2 untouched 和 C4 untouched 上都改善 affine baseline；同时 Wikitext-2 untouched 上明显优于严格匹配的 random affine relocation。

这证明了：P5-A 的 affine index-space adapter 不只是第 13 层的局部 feasibility，而能在全候选 Q/K 空间上以固定的整体 layer-selection/budget rule 工作。

本轮没有运行 PT² checkpoint，因此不能据此声称优于 PT² 或其他 strong ternary PTQ；它完成的是进入真实 PT² pipeline 对接前的整体兼容性 gate。

## 协议核验

| 项目 | 实际配置 | 预注册要求 | 结果 |
|---|---|---|---|
| 模型 | `facebook/opt-350m` | OPT-350M | 一致 |
| 候选层 | 0--23，共 24 层 | 全部允许 decoder layers | 一致 |
| 模块 | Q/K | Q/K only | 一致 |
| codebook | per-row-group `mu + alpha*T` | `T in {-1,0,+1}` | 一致 |
| group size | 128 | 128 | 一致 |
| threshold factor | 0.75 | 0.75 | 一致 |
| fit / val / W2 / C4 | 8 / 8 / 8 / 8 batches | 8 / 8 / 8 / 8 | 一致 |
| sequence / batch | 128 / 2 | 128 / 2 | 一致 |
| 梯度 | fit split，1 batch | fit-only CE gradient | 一致 |
| 主预算 | top-6 layers × 64 pairs = 384 pairs | 固定 primary | 一致 |
| 次预算 | top-4 layers × 64 pairs = 256 pairs | 固定 secondary | 一致 |
| `mu/alpha` | relocation 后冻结 | 冻结 | 一致 |
| QAT teacher / optimizer | 均未使用 | 不允许 | 一致 |
| untouched 参与选择 | 否 | 不允许 | 一致 |

## 原始 NLL/PPL

NLL 越低越好；PPL 为 `exp(NLL)`，没有做模型自归一化。

| Variant | layers | relocation pairs | changed coords | Val NLL / PPL | W2 untouched NLL / PPL | C4 untouched NLL / PPL |
|---|---:|---:|---:|---:|---:|---:|
| FP reference | -- | -- | -- | 3.803895 / 44.876 | 3.987563 / 53.923 | 3.438045 / 31.126 |
| Affine baseline | all 24 | 0 | 0 | 4.253630 / 70.360 | 4.381343 / 79.945 | 3.767555 / 43.274 |
| Random matched top-4 | [3,6,4,8] | 256 | 512 | 4.252506 / 70.281 | 4.380354 / 79.866 | 3.768578 / 43.318 |
| Affine CEGSP top-4 | [3,6,4,8] | 256 | 512 | 4.185267 / 65.711 | 4.321534 / 75.304 | 3.683665 / 39.792 |
| Random matched top-6 | [3,6,4,8,5,7] | 384 | 768 | 4.252978 / 70.317 | 4.381070 / 79.923 | 3.768179 / 43.301 |
| **Affine CEGSP top-6** | **[3,6,4,8,5,7]** | **384** | **768** | **4.178845 / 65.290** | **4.309960 / 74.438** | **3.673551 / 39.392** |

上表 PPL 保留到小数后三位；原始 JSON 中保存了 exact PPL，以下 gate 和 delta 全部基于 exact NLL。正式论文表格应直接从结果 JSON 生成 PPL，避免人工转写误差。

## Delta vs affine baseline

| Variant | Val ΔNLL | W2 untouched ΔNLL | C4 untouched ΔNLL | W2 vs matched random |
|---|---:|---:|---:|---:|
| Random matched top-4 | -0.001123 | -0.000990 | +0.001023 | -- |
| **Affine CEGSP top-4** | **-0.068362** | **-0.059809** | **-0.083890** | **-0.058819** |
| Random matched top-6 | -0.000652 | -0.000274 | +0.000625 | -- |
| **Affine CEGSP top-6** | **-0.074785** | **-0.071383** | **-0.094004** | **-0.071109** |

相对于 affine baseline 到 FP reference 的 gap，top-6 CEGSP 回收约 16.6% 的 validation gap、18.1% 的 Wikitext-2 gap 和 28.5% 的 C4 gap。这里的 gap closure 只是对 affine initialization 的诊断，不是与 QAT 的 gap closure。

## Gate 结果

| Gate | 机器判定 |
|---|---|
| codebook legality | PASS；所有 variant `num_illegal_states=0`，最大 codebook residual 为 0 |
| support cardinality | PASS；所有 variant cardinality violations 为 0 |
| changed-coordinate budget | PASS；top-4 为 512，top-6 为 768 |
| finite metrics | PASS；FP、baseline、CEGSP、random 所有 val/W2/C4 NLL finite |
| primary validation improvement | PASS；top-6 ΔNLL = -0.074785 |
| primary W2 improvement | PASS；top-6 ΔNLL = -0.071383 |
| primary beats matched random | PASS；top-6 比 random top-6 低 0.071109 NLL |
| secondary top-4 | PASS；同时改善 val、W2，并优于 matched random |
| **overall affine compatibility** | **PASS** |

## 观察、解释与含义

### 1. 整体 affine extension 通过，而非 layer-13 偶然

预先指定的全层候选空间自动选出的 top-6 是 `[3, 6, 4, 8, 5, 7]`，没有查看 untouched 结果后手工挑层。top-4 与 top-6 都通过，说明 affine index-space relocation 在整体流程上具有可运行的正向信号。

### 2. CEGSP 的信号不是“改几个位置就有效”

在完全相同 selected layer set、相同每层 relocation 数和相同 affine feasible space 下，random matched top-4/top-6 的 W2 改善接近 0，而 CE-selected relocation 有明显改善。该对照支持：收益主要来自 fit-split quantized-point CE gradient 对 support/sign 的排序，而不是随机扰动本身。

### 3. top-6 比 top-4 更好，但本轮不能据此继续扩预算

top-6 在三个 split 上都优于 top-4；这只说明本轮预注册的两个小预算没有出现 over-edit。不能由此推导更大预算必然更好，也不能把 top-6 之外的预算当作后验调参方向。

### 4. 这轮没有回答 strong PTQ compatibility

当前 affine baseline 是按 P5-A 固定规则从 FP 权重构造的整体 affine ternary initialization，不是已经完成协议审计并可复现的 PT² checkpoint。因而结论应限定为：

> affine ternary index-space CEGSP can operate as a whole-model discrete refinement under a frozen global selection rule.

不应写成：

> CEGSP improves PT²。

## 成本记录

- GPU：RTX 4090 24GB。
- PyTorch：2.5.1+cu124。
- 最大 PyTorch allocated memory：约 0.95 GB。
- wall-clock：698.5 s，约 11 分 39 秒。
- 主要成本：全 24 层 Q/K 的逐 group candidate 构造使用了较多 Python/torch 小循环；这影响量化时间，不改变模型推理显存或实验 gate。

后续若进入正式方法实现，应将 candidate construction 向量化或缓存，但优化实现必须保持同一 candidate 定义、layer score 和预算规则，不能改变本轮结论。

## 论文 claim 影响

- `C1: affine index-space legality`：支持。P5-A 与 P5-B 均通过 legality/cardinality audit。
- `C2: whole-model affine compatibility`：支持，限定为 OPT-350M、Q/K、预注册 top-4/top-6、当前 calibration protocol。
- `C3: CEGSP beats PT²/strong ternary PTQ`：不支持；仍需真实 PT² baseline/state-export compatibility experiment。
- `C4: generalization across architectures`：不由本轮更新；应沿用 P2C2/P3B 的独立证据，不能把本轮 OPT 结果外推到 Pythia。

## 后续实验边界

本轮完成后不自动启动新实验。下一阶段只有一个合乎逻辑的主问题：在确认 PT² checkpoint 的 affine state、scale granularity、calibration protocol 和 eval parity 后，运行 `Strong PTQ -> affine-index CEGSP` 的 compatibility test。若 PT² protocol 仍不可复现，则保留本轮 compatibility 结果，不用它包装成 strong-baseline gain。
