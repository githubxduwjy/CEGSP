# CEGSP-P6-B：seed/offset replication 实验报告

日期：2026-08-28  
云端：RTX 4090 24GB，`torch 2.5.1+cu124`，`bf16`  
总汇总：[summary.json](../results/remote-runs/cegsp_p6b_replication_opt350m_20260828_42168/summary.json)  
原始 replicate 目录：[P6-B raw results](../results/remote-runs/cegsp_p6b_replication_opt350m_20260828_42168/)

## 1. 目的

P6-A 证明了量化点 CE 一阶分数在全 24 层 centered/affine ternary state 上具有排序信息。P6-B 不改变任何方法参数，只改变预先固定的 seed 和 token offset，检验该机制是否依赖单次 Wikitext slice 或随机状态，并检查 P6-A 中 affine random untouched-W2 偏置是否重复出现。

## 2. 固定协议

| replicate | seed | fit/validation/C4 offset | 其余配置 |
|---|---:|---:|---|
| R0 | 20260829 | 0 | 完全沿用 P6-A |
| R1 | 20260830 | 512 | 完全沿用 P6-A |
| R2 | 20260831 | 1024 | 完全沿用 P6-A |

每组均为 OPT-350M 全 24 层 Q/K，seq_len 128，batch size 2，fit/validation/untouched/C4 为 8/8/16/16 batches，group size 128，centered threshold 0.70，affine threshold 0.75，candidate pool 32，每层 8 个固定 gradient ranks 与 8 个 matched-random，1 个 fit batch 梯度。无 optimizer step、QAT teacher、PT² 或 validation/test 选参。

## 3. 完整性审计

- 三个 replicate 均 `status=complete` 且 return code 为 0；
- 三组 seed 和 offset 与预注册值完全一致；
- 每个 representation 均有 192 个 gradient + 192 个 random 候选，覆盖 layers 0--23；
- 每组 384 行、总计 2304 行候选记录；
- 三个结果 JSON、总 summary 和 screen 日志均已拉回；所有递归数值均 finite，无 nonfinite；
- candidates 在 validation/untouched 评估前生成，未使用 held-out 数据反选。

## 4. 每组主要结果

`Delta_rank = mean(Delta NLL_top20) - mean(Delta NLL_random)`，越负越好。

| Rep | 表示 | rho val | rho W2 | top20 Delta val | random Delta val | Delta_rank | top20 rate / random rate | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R0 | centered | 0.6982 | 0.7678 | -0.005323 | -0.000023 | -0.005299 | 1.000 / 0.479 | PASS |
| R0 | affine | 0.7199 | 0.7388 | -0.005588 | +0.000219 | -0.005807 | 1.000 / 0.286 | PASS |
| R1 | centered | 0.6485 | 0.7128 | -0.005546 | -0.000343 | -0.005203 | 1.000 / 0.802 | PASS |
| R1 | affine | 0.7659 | 0.8284 | -0.006520 | -0.000327 | -0.006193 | 1.000 / 0.750 | PASS |
| R2 | centered | 0.7223 | 0.8092 | -0.006140 | -0.000426 | -0.005715 | 1.000 / 0.839 | PASS |
| R2 | affine | 0.7249 | 0.7327 | -0.006809 | -0.000454 | -0.006356 | 0.947 / 0.818 | PASS |

三组的聚合统计为：

| 表示 | rho val mean +/- std | rho W2 mean +/- std | Delta_rank mean +/- std | top20 Delta val mean +/- std |
|---|---:|---:|---:|---:|
| centered | 0.6897 +/- 0.0307 | 0.7633 +/- 0.0395 | -0.005406 +/- 0.000222 | -0.005670 +/- 0.000345 |
| affine | 0.7369 +/- 0.0206 | 0.7666 +/- 0.0437 | -0.006118 +/- 0.000230 | -0.006306 +/- 0.000521 |

每个 replicate/representation 的预注册 gate（`rho_val > 0`、`Delta_rank < 0`、top-20% improvement rate 高于 random、score-bin 总体有序）均通过。centered 和 affine 都为 3/3 通过，因此总体判定为 `STABLE_SUPPORT_SCORE_VALIDITY`。

## 5. score-bin 与 split-bias 诊断

centered 和 affine 的五个固定 score bins 在六个组合中都呈现“高 score 候选改善更大、低 score 候选改善减弱”的总体趋势。R1 centered 的第 3/4 bin 存在很小的局部非单调，但不改变整体排序方向，也没有触发预注册的稳定性否决。

P6-A 中 affine random 在 untouched W2 上的高改善率没有稳定复现：三组 affine random improvement rate 为 `0.875`、`0.104`、`0.734`。因此不能把 P6-A 的高 random-W2 rate 解释为普遍的 random perturbation regularization；更可能是 slice-dependent variation。重要的是，gradient score 的 rho 和 top-vs-random validation effect 在三组均保持同向。

## 6. 结论与主张边界

### 得到支持的结论

1. 量化点 CE 一阶分数的排序有效性不是单一 seed/offset 的偶然现象；centered 与 affine 均在 3 个固定 replicate 上保持正相关。
2. top-score 合法 support relocation 相对 matched-random 的效应方向在 6/6 个 representation-replicate 组合中一致；`Delta_rank` 的波动很小。
3. affine score-validity 可以作为正式方法机制的一部分，而不只是单层或单次实验的 diagnostic。

### 仍不能声称的结论

- 尚未证明全模型组合 patch 在新的 seed/offset 上一定改善最终 PPL；
- 尚未证明优于 PT² 或其他健康的 strong ternary PTQ；
- 尚未证明 QAT gap closure 或 7B/8B scaling。

## 7. 下一阶段建议

score-validity 机制支线可以结束，不再继续增加相同类型的 seed/offset。下一步应进入固定预算的 whole-model composition consistency：使用预先冻结的全局候选池，比较 high-score、matched-random 和 low-score 三类 patch，测试 `Delta L_high < Delta L_random < Delta L_low` 是否成立。该实验应独立于 P6-B 的 held-out 结果，不重新搜索 threshold、sign、group size 或 layer。

