# P6-B：centered/affine score-validity 的 seed/offset replication

## 目的

P6-A 在 OPT-350M 全 24 层、centered/affine 两种三值表示上发现：量化点 CE 一阶分数能够排序一次合法 support relocation。P6-B 只检验这一机制是否依赖某一个随机 seed 或 Wikitext token slice；不新增 action、不改变阈值、不改变 group size、不选择新 layer，也不进行 whole-model patch。

## 预注册配置

三组配置在启动前固定：

| replicate | seed | Wikitext fit offset | Wikitext validation offset | C4 report offset |
|---|---:|---:|---:|---:|
| R0 | 20260829 | 0 | 0 | 0 |
| R1 | 20260830 | 512 | 512 | 512 |
| R2 | 20260831 | 1024 | 1024 | 1024 |

除 seed/offset 外全部保持 P6-A：OPT-350M、24 个 decoder layers、Q/K、seq_len 128、batch size 2、fit/val/untouched/C4 batches 为 8/8/16/16、group size 128、centered threshold 0.70、affine threshold 0.75、candidate pool 32、每层 8 个固定 gradient ranks 和 8 个 matched-random、1 个 fit batch 梯度、bf16、无 optimizer/QAT/PT²。

每个 replicate 仍分别评估 centered 与 affine。候选生成只使用 fit split 的 deployed ternary CE gradient；validation、untouched W2 和 C4 不参与候选生成、排序或选择。

## 主要指标

每个 representation 报告：

1. `rho_val` 与 `rho_untouched`：score 和 `-Delta NLL` 的 Spearman 相关；
2. `Delta_rank = mean(Delta NLL_top20) - mean(Delta NLL_random)`，越负越好；
3. top-20% 与 random 的 improvement rate；
4. 五个固定 score bins 的 validation Delta NLL；
5. affine random 在 validation 与 untouched W2 的 improvement-rate 差异；
6. rows 数量、24 层覆盖、finite/nonfinite、实际 seed/offset。

## 预注册 gate

单个 replicate/representation 的方向性 gate：

- `rho_val > 0`；
- `Delta_rank < 0`；
- top-20% validation improvement rate > random rate；
- score bins 从高分到低分总体呈改善减弱趋势，允许相邻 bin 有小幅非单调波动。

P6-B 的机制稳定性判定：

- **稳定支持**：centered 和 affine 各至少 2/3 replicate 通过方向性 gate，且两种表示的 `rho_val` 均值为正、`Delta_rank` 均值为负；
- **表示依赖/采样敏感**：只有一种表示满足上述稳定条件，或另一表示出现明显 offset 依赖；
- **当前机制不稳定**：两种表示均未达到稳定支持。

这些 gate 只用于判断 score-validity 的稳健性，不用于选择最优 seed、offset、threshold 或最终模型。

## 运行纪律

- 三个 replicate 顺序运行，避免单张 RTX 4090 上并发争用。
- 每个 replicate 结果单独保存；汇总文件只能在三组结束后生成。
- 不根据中途结果修改后续 replicate；不启动 P6-C、7B/8B 或新的模块实验。
