# P9-S1：官方 PT² Llama-2-7B + affine-index CEGSP 兼容性实验

## 状态

`AWAITING_HUMAN_REVIEW`。本文件是提交云端前的预注册方案；尚未启动新任务。

## 研究问题

在官方 PT² 的强三值初始化之后，冻结的 affine-index CEGSP 是否仍能带来独立的任务损失改进？该实验不再比较 Direct+CEGSP 与 PT²，而是直接检验：

\[
\text{PT}^{2}\quad\longrightarrow\quad \text{PT}^{2}+\text{CEGSP}.
\]

## 关键实现约束

P9-S0 保存的 Hugging Face checkpoint 只包含部署后的权重，不包含 PT² 的每组 `T`、`mu`、`alpha` 及 SSR permutation 元数据。因此，P9-S1 必须在同一官方 PT² 配置下重新量化一次，并在量化器调用内部捕获真实 ternary state；不能从最终 FP16 权重事后拟合出状态后直接宣称 parity。

Llama-2-7B 使用 32 个 decoder layers。CEGSP 只作用于 `self_attn.q_proj` 和 `self_attn.k_proj` 的 32×2 个投影，MLP、V/O 和 PT² 已经产生的其它权重保持不变。SSR 的列重排必须保留，候选坐标必须在部署后的 SSR 坐标中定义。

## 冻结协议

### PT² 初始化

- 模型：`/CEGSP/model`（通过已验证的 `Llama-2-7b-hf` 路径别名加载）。
- 官方 PT² commit：`9e943e68bdb27469929a4fe7e5720926b9d952d7`。
- 方法：`atq --ssr`。
- group size：128；scale：PT² 原生的 per-row per-group affine 表示。
- calibration：WikiText-2，`nsamples=128`，`seqlen=2048`。
- `percdamp=0.01`，`num_p=1`，`salient_metric=hessian`，GPTQ/SSR 保持开启。
- CEGSP 运行期间不重新拟合 `mu/alpha`，不改变 PT² 的量化结果和 permutation。

### CEGSP refinement

- 只使用 quantized-point cross-entropy gradient 产生候选排序。
- 候选单位：同一 row、同一 group 内的一次 support relocation；保持该 group 的非零 cardinality 不变。
- 目标：`one nonzero -> zero` 与 `one zero -> receiver sign` 的成对交换。
- sign rule：使用已经冻结的 affine CEGSP 规则；不新增 sign rule。
- layer selection：对全部 32 层 Q/K 候选计算 probe saliency，按固定规则选 top-6 层；不得根据 untouched 结果换层。
- edit budget：每个选中层 64 对 relocation，共最多 384 对、768 个坐标变化；不扫 budget。
- `mu/alpha`：冻结；只改变 ternary index `T`。
- 主方法：`PT2 + affine-index CEGSP`。
- 随机对照：相同 6 个层、相同每层编辑数、相同 cardinality 的 random relocation；随机种子预注册为 `20260831+6`。
- baseline：原始 PT² state，不进行任何 CEGSP 修改。

## 必须先通过的 state-parity gate

在计算任何 PPL/NLL 之前，检查：

1. 32 层 Q/K 共 64 个目标模块均被捕获；
2. 每个 group 的边界和 group size=128 一致；
3. 捕获的 `T` 仅含 `{-1,0,+1}` 且 finite；
4. 捕获状态满足 `Q = mu + alpha*T`，FP32 最大 residual `<1e-3`；
5. 捕获的 `Q` cast 到部署 dtype 后与 PT² 实际权重的最大 residual `<1e-3`；
6. SSR permutation 被记录，CEGSP 编辑坐标属于 SSR 后的实际部署坐标；
7. baseline、CEGSP、random 三个状态均无 cardinality violation、nonfinite 或非法 codebook。

任一项失败：只写出 parity 诊断，性能比较标记为 `NOT_RUN_STATE_PARITY_FAILED`，不调整阈值、不换层、不换预算。

## 评估与预注册 gate

- 评估：官方 PT² Llama evaluator；WikiText-2 和 C4，sequence length=2048；报告 NLL 与 PPL。
- CEGSP 候选选择只看预先定义的 fit/calibration split；untouched W2/C4 只用于最终评估。
- 所有指标必须 finite；记录 GPU 峰值显存、量化时间、CEGSP refinement 时间和实际 changed coordinates。

### 判定

- `STRONG_PASS`：CEGSP 相对于 PT² 在 untouched W2、C4 均降低 NLL，并且优于 matched random。
- `PASS`：W2 明显降低 NLL，C4 不恶化超过预注册数值容差，且优于 matched random。
- `WEAK_PASS`：提升很小但 finite、state-legal、W2 可复现，记录为 residual refinement evidence，不夸大为 SOTA。
- `FAIL_COMPATIBILITY`：state parity 通过但 CEGSP 不优于 PT²；保留为 strong-baseline negative finding，停止扫参。

无论结果正负，都不启动 P7-C，不进行 budget/sign/layer/epsilon 事后搜索。

## 成本与预期

这是一次重新执行官方 PT² 的 7B A100 实验，预计量化阶段约 25–35 分钟，CEGSP 梯度与三组评估约 10–25 分钟，峰值显存预计低于 A100 80GB。P9-S0 的 13.5GB checkpoint 仅作为独立 baseline/恢复参考；P9-S1 的可审计 state 由本次 instrumented run 产生。

## 结果解释边界

P9-S1 若成功，只能支持“CEGSP 在强 affine PTQ 初始化后仍有互补 residual refinement”；不能据此声称超过 QAT、普遍超过所有 ternary PTQ，或证明 CEGSP 在所有模型/层上均有效。若失败，也不否定此前 centered/affine 机制证据，只说明当前 CEGSP 对官方 PT² 的剩余误差没有可测的独立收益。

## 启动前人工审核项

- [ ] 同意重新量化以捕获 PT² ternary state，而不是从 checkpoint 反拟合
- [ ] 同意 32 层、Q/K-only、top-6、每层 64 对、总计最多 384 对的固定规则
- [ ] 同意 state-parity 失败时不运行性能比较
- [ ] 同意不做额外超参扫描、不启动 P7-C

