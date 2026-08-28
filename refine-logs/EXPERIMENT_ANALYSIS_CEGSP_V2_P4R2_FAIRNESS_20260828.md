# CEGSP-V2-P4R2 Strict Edit-Matching 与 QAT Update Scope 公平性检查

日期：2026-08-28  
Run ID：`CEGSP-V2-P4R2-OPT350M-STRICT-EDITMATCH-SCOPE-OFFSET2`  
远端结果：`/root/tqgsp-runs/CEGSP-V2-P4R2-OPT350M-STRICT-EDITMATCH-SCOPE-OFFSET2/result.json`

## 1. 实验目的

P4-R 已经显示 CEGSP 优于 one-step QAT edit-matched baseline，但当时最接近的 edit-matched point 只有 528 个 changed coordinates，而 CEGSP 是 768 个。本实验只补两个公平性检查：

1. 在 768 附近更密地扫 one-step QAT eta；
2. 比较 one-step QAT 更新 all Q/K layers 与只更新 CEGSP selected layers。

CEGSP 主方法不变。

## 2. 固定设置

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| 数据 | Wikitext-2 + C4 validation |
| offsets | fit=8192, val=8192, C4=16384 |
| CEGSP | canonical support relocation top-6 |
| CEGSP selected layers | [13,17,14,19,23,16] |
| CEGSP changed coords | 768 |
| eta sweep | 3e-5 到 1e-3 |
| scopes | all Q/K layers；CEGSP-selected layers only |
| elapsed | 80.49 s |

## 3. 主结果

Baseline：

| Method | val NLL | W2 untouched NLL | C4 untouched NLL | changed coords |
|---|---:|---:|---:|---:|
| Direct | 8.4695 | 8.4652 | 8.1304 | 0 |
| CEGSP | **8.2894** | **8.2971** | **7.8402** | 768 |

### 3.1 all Q/K layers one-step QAT

| Point | eta | changed coords | val NLL | W2 untouched | C4 untouched |
|---|---:|---:|---:|---:|---:|
| edit-matched | 4e-5 | 697 | 8.4564 | 8.4506 | 8.1058 |
| val-best | 1e-3 | 16,799 | 8.3327 | 8.3245 | 7.8298 |

更密 edit matching 后，one-step QAT 从 528 changes 改进到 697 changes，已经接近 CEGSP 的 768，但仍明显弱于 CEGSP：

- val：8.4564 vs 8.2894；
- W2：8.4506 vs 8.2971；
- C4：8.1058 vs 7.8402。

### 3.2 CEGSP-selected layers only one-step QAT

| Point | eta | changed coords | val NLL | W2 untouched | C4 untouched |
|---|---:|---:|---:|---:|---:|
| edit-matched | 2e-4 | 771 | 8.4382 | 8.4374 | 8.1030 |
| val-best | 1e-3 | 3,855 | 8.3336 | 8.3293 | 8.0062 |

当 one-step QAT 只更新 CEGSP selected layers 时，strict edit-matched point 几乎完美匹配 CEGSP 预算：771 vs 768。但仍明显弱于 CEGSP：

- val：8.4382 vs 8.2894；
- W2：8.4374 vs 8.2971；
- C4：8.1030 vs 7.8402。

## 4. 结论

P4R2 进一步排除了两个公平性质疑：

1. **不是因为 edit-matched QAT 没有精确匹配 768 edits。**  
   selected-layer scope 下，one-step QAT 已经达到 771 changes，仍明显弱于 CEGSP。

2. **不是因为 QAT update scope 不公平。**  
   无论 one-step QAT 更新 all Q/K layers，还是只更新 CEGSP selected layers，strict edit-matched one-step 都弱于 CEGSP。

这加强了 P4-R 的机制结论：

> CEGSP 的优势不是简单来自“一次梯度”或“离散变化数量”，而是来自结构化、梯度选择的三值 support relocation。

## 5. 仍需谨慎

- one-step QAT val-best 在 all Q/K scope 下 C4 略优于 CEGSP，但需要 16,799 changes；
- full-model one-step QAT 尚未测试；
- 本实验仍是 OPT-350M 单模型单 offset 的 fairness check。

## 6. Gate 判定

```text
STRONG_PASS_STRICT_EDIT_MATCH_AND_SCOPE_FAIRNESS
```

P4-R 不需要继续扩展。下一步应进入 P5-0 Strong PTQ protocol audit。

