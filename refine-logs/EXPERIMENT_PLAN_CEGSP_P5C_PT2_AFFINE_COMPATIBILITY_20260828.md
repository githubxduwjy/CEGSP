# CEGSP-P5-C：真实 PT² → affine CEGSP compatibility

日期：2026-08-28

## 研究问题

P5-A/P5-B 已验证 affine ternary index-space 中的 CEGSP 合法性和整体 fixed-rule 行为，但它们的 affine baseline 是从 FP 权重直接构造的，不是 PT² 的实际输出。

P5-C 只回答一个更窄、更关键的问题：

> 当官方 PT² ATQ 先完成全模型 ternary PTQ 后，是否仍能从 PT² 的真实 quantized state 恢复 `mu/alpha/T`，并在同一 PT² 模型上执行 affine-index CEGSP？

## 阶段顺序

### P5-C0：state parity

在 PT² 官方 `TernaryQuantizer.quantize` 调用内部捕获每个 quantized block 的真实 `T`。由实际 `q` 和 `T` 解出 affine `mu/alpha`，并检查：

- `T` 的取值只包含 `-1/0/+1`；
- `q = mu + alpha*T` 的 residual 接近零；
- 最终 PT² Q/K 参数与捕获的 quantized blocks 的 dtype-cast 一致；
- group size 为 128、每个 row-group 一个 `mu/alpha`；
- Q/K mapping、tensor shape 和 block boundaries 一致；
- 本轮使用 `ssr=False`，因此不启用 permutation；
- PT² calibration 使用与 CEGSP 相同的 Wikitext-2 fit tensors；
- evaluator 使用同一套 Wikitext-2 validation/untouched 和 C4 untouched batches。

PT² 官方代码本身返回的 `T` 是 placeholder，因此不能使用该返回值；必须使用本轮 capture 的内部 state。若 P5-C0 失败，不运行 performance comparison，并将状态记为 `BLOCK_STATE_PARITY`。

### P5-C1：performance compatibility

仅在 P5-C0 通过后运行：

1. PT² ATQ：官方 PT² full-model output，作为 strong baseline。
2. PT² + affine CEGSP：只改 PT² Q/K 的 `T`，冻结 PT² 的 `mu/alpha`、其它层和其它 projection；使用预注册整体规则。
3. PT² + matched random affine：同一 selected layer set、同一每层 edit 数和同一 affine feasible space，随机 donor/receiver/sign。
4. P5-B affine CEGSP：作为 diagnostic reference，不作为同一 strong-baseline 比较。

## 固定配置

- 模型：`facebook/opt-350m`。
- PT² commit：`9e943e6`，method=`atq`，`ssr=False`。
- 候选模块：全部 24 层的 Q/K；PT² 本身仍按官方流程量化全部 decoder Linear modules。
- group size / block size：128。
- PT² calibration：Wikitext-2 fit split 的 8 batches × batch size 2，每条输入使用 evaluator 的 128 token input，合计 16 calibration samples；seed 20260828。
- evaluator：Wikitext-2 validation 8 batches、Wikitext-2 untouched 8 batches、C4 untouched 8 batches；sequence length 128、batch size 2、offset 全 0。
- CEGSP gradient：PT² quantized model 上 fit split 的 1 batch CE gradient。
- layer selection：每层 Q/K candidate 的 top-8 score 之和排序，固定选 top-6 layers；不看 validation/untouched。
- relocation：每个 selected layer 64 pairs，共 384 pairs / 768 changed coordinates。
- sign rule：`affine_fp`，使用 PT² group 的 affine-relative FP direction；不试其它 sign rule。
- `mu/alpha`：从 PT² capture 恢复后冻结；不重估。
- 无 QAT teacher、无 latent FP training、无 optimizer、无 budget sweep。

## Gate

### State parity gate

全部以下条件必须满足：

- `T` illegal count = 0；
- max `|q - (mu + alpha*T)|` 小于 `1e-3`（capture float32；适配首层约
  `1e3` 量级值的 FP32 绝对舍入）；
- final model 与 capture q 的 dtype-cast residual 小于 `1e-3`；部署后的
  FP16 tensor 不再额外要求在 FP32 中保持精确三水平仿射，因为三个水平会
  独立发生 FP16 舍入；
- Q/K shape、group boundary、scale granularity 全部一致；
- calibration fingerprint 与 CEGSP fit fingerprint 一致；
- no permutation flag（`ssr=False`）。

### Strong compatibility gate

在 state parity 通过后，PT²+CEGSP 必须满足：

- Wikitext-2 validation NLL < PT² baseline；
- Wikitext-2 untouched NLL < PT² baseline；
- Wikitext-2 untouched NLL < matched random；
- 所有结果 finite，CEGSP relocation 合法且 support cardinality 不改变。

C4 作为 transfer 指标：下降是额外加分，持平或小幅波动不单独否定 W2 compatibility。

## 解释边界

- P5-C0 PASS、P5-C1 PASS：支持 “CEGSP provides complementary task-aware refinement after official PT² affine ternary initialization”。
- P5-C0 PASS、P5-C1 FAIL：说明真实 PT² 状态可接入，但当前 residual support refinement 在 strong PT² 上没有收益；不能把 P5-B 结果外推为 PT² gain。
- P5-C0 FAIL：不比较 NLL；先修复 state export/parity harness，不能通过从最终权重事后拟合状态绕过失败。
