# CEGSP 论文级验证总方案

日期：2026-08-27

## 0. 先给结论：研究阶段需要从“模块验证”切换到“证据闭环”

CEGSP 已经不再处于“某一层、某一组超参数是否有效”的阶段。当前已有证据覆盖：

- OPT-125M、350M、1.3B、2.7B 的规模迁移；
- Wikitext-2 多 offset、较大 untouched holdout，以及 C4 transfer；
- random control、matched control、support relocation 与 signflip 对照；
- Pythia-1B/GPT-NeoX fused-QKV 的非 OPT 架构迁移；
- RTX 4090 上的运行时间和显存成本。

因此后续不能再按照“发现一个现象—新增一个小模块—单独验证一次”的循环推进。论文真正缺的不是更多局部正结果，而是四个可被审稿人一次性检查的闭环：

1. CEGSP 是否确实弥合 direct ternary PTQ 与 QAT 之间的 gap；
2. 收益是否来自三值的 zero-support 结构，而不是普通低 bit 的梯度编辑；
3. 结果是否能在统一协议、跨模型、跨数据和下游任务上成立；
4. 额外计算是否仍然是 PTQ 级别，而不是隐性 QAT。

后续路线固定为 CEGSP，不再因为单个负结果更换研究问题。允许调整的是证据强度、默认预算、基线实现和适用边界。

## 1. Problem Anchor 与论文主张

### 1.1 固定问题

三值/1.58-bit PTQ 直接把 FP 权重投影到 `{-alpha, 0, +alpha}` 时，往往明显弱于 QAT。问题不是再设计一个更复杂的阈值，而是：

> 在不使用 QAT teacher、QAT checkpoint、latent full-precision weights 或 optimizer update 的条件下，能否利用部署后三值点本身的函数梯度，系统地修复 direct ternary PTQ 的离散失配？

### 1.2 论文只保留两个主张

**C1：量化点函数修复主张。** 在 direct ternary PTQ 得到的部署点上计算 CE gradient，并用它指导少量离散三值编辑，可以在不进行 QAT 的情况下稳定降低 held-out NLL，且能跨模型规模和至少两个架构族迁移。

**C2：三值结构与成本主张。** CEGSP 的主要离散动作是保持非零预算的 zero-support relocation；它相对 nonzero-only signflip 和 random editing 具有额外信号，同时只增加一次量化点梯度和有限的候选评估，成本显著低于 QAT。

这两个主张已经足够形成一篇论文。不要再把跨层联合优化、teacher distillation、路径 barrier、旋转搜索、CVaR 或多目标损失并入核心方法；它们会稀释主线且无法直接回答当前问题。

### 1.3 必须排除的反主张

- **A1：收益只是“梯度编辑一般有效”。** 需要 support relocation、signflip、random 和预算匹配对照。
- **A2：收益只是 validation 过拟合。** 需要固定 validation selection，使用不参与选择的 Wikitext/C4 holdout，并报告按 batch 的 paired bootstrap CI。
- **A3：收益只来自 OPT 的模块命名或数据切分。** 需要 Pythia/GPT-NeoX 以及至少一个新的 calibration offset。
- **A4：CEGSP 实际上是隐性 QAT。** 需要记录反向次数、无 optimizer step、无 latent weight，并与 QAT wall-clock/显存比较。
- **A5：收益来自弱的 direct ternary baseline。** 至少需要一个可审计的强三值 PTQ baseline（优先 PT²；若实现不兼容则明确写“未复现”，不能冒充公平比较）。

## 2. 方法冻结：后续实验中的 CEGSP 是什么

为避免实验过程中不断改变方法，下面定义 canonical CEGSP。所有主实验只能使用这一版本；若要加入新机制，必须先作为附录候选，不得改变主表。

### 2.1 Direct ternary 起点

对每个线性层按输入维度分组，令 FP 权重为 `W`，三值状态为 `S`：

\[
S_{ij}=\begin{cases}
\operatorname{sign}(W_{ij}), & |W_{ij}|>\tau_g,\\
0, & \text{otherwise},
\end{cases}
\qquad
Q(W)_{ij}=\alpha_g S_{ij}.
\]

当前工程默认使用 group size 128、`tau_g = 0.7 mean(|W|_g)`，并以非零权重平均幅值估计 `alpha_g`。这组参数在所有主实验中冻结；它不是研究贡献，而是统一起点。

### 2.2 量化点 CE gradient

将模型中所有被部署的线性权重替换为 `Q(W)` 后，在 fit calibration split 上计算：

\[
G_l=\nabla_{Q(W_l)}\mathcal{L}_{CE}
\bigl(f_{Q(W)};D_{fit}\bigr).
\]

这里的梯度是对“已经量化的模型”求得的局部函数信号，不更新参数。当前实现只对 attention Q/K 产生梯度并生成候选，其他参数保持 direct ternary 状态。

### 2.3 三值离散候选

对一个 group 内的 active index `d` 与 zero index `r`，support relocation 为：

\[
S_d\leftarrow 0,
\qquad
S_r\leftarrow \operatorname{sign}(W_r^{FP}).
\]

它保持该 group 的非零数目不变，因此不是额外剪枝预算，而是在三值可行域内移动零态支撑。候选由一阶 CE score `-<G, Delta Q>` 排序，再在 validation 上做 layer-level top-k 选择。nonzero-only signflip 作为结构对照，只允许：

\[
S_i\in\{-1,+1\}\longrightarrow -S_i,
\]

不使用 zero-support。最终 canonical 方法是 joint support/sign family，但必须保留 support-only 和 signflip-only 作为匹配层预算的对照。

### 2.4 层预算与选择

不再按每个模型手调 k。统一报告相对层数的两档预算：

\[
k_{25}=\max(1,\lceil0.25L\rceil),
\qquad
k_{50}=\max(1,\lceil0.50L\rceil),
\]

其中 `L` 是模型层数。每层候选编辑数固定为 64。若要使用默认配置，主结果使用 `k25`，`k50` 作为预算敏感性和上限检查；不使用 untouched test 选择 k。

### 2.5 融合 QKV 的统一接口

架构 adapter 只负责模块定位，不改变算法：

- OPT：独立 `q_proj/k_proj`；
- GPT-NeoX/Pythia：从 fused `query_key_value` 中暴露连续 Q/K 行切片；
- 前向与反向仍经过原始 fused Linear；只读取和写回 Q/K slice。

这使“跨架构”成为实现接口的可复用性，而不是另一个算法贡献。

## 3. 实验总架构：五个 block，避免碎片化

| Block | 目标 | 主论文位置 | 是否必须 |
|---|---|---|---|
| B0 | 统一 harness、强基线和审计 | 方法/实验设置 | 必须 |
| B1 | 多模型、多数据的主结果 | Main Table 1 | 必须 |
| B2 | PTQ–QAT gap 与成本闭环 | Main Table 2/Figure 3 | 必须 |
| B3 | 三值特异性与简洁性 | Main Table 3/Figure 4 | 必须 |
| B4 | 跨 offset、下游任务和失败边界 | Robustness/Appendix | 必须，但可分批 |
| B5 | 可解释性和更多架构 | Appendix | 可选 |

每个 block 同时回答一组审稿问题；不再为 support、signflip、k、offset 各自开一篇实验。

## 4. B0：统一协议、基线和审计

### 4.1 目的

先冻结一个可复用 benchmark harness，避免后续结果无法比较。B0 不产生方法创新结论，但没有它，B1–B4 的数值不具备论文可信度。

### 4.2 统一数据协议

每个模型使用同一逻辑分割：

- `D_fit`：Wikitext-2 train，用于 CE gradient；
- `D_val`：Wikitext-2 validation 的前一段，用于 layer/k selection；
- `D_W_holdout`：Wikitext-2 validation 的不重叠后段，只用于最终报告；
- `D_C4_holdout`：C4 validation 独立 token 区间，只用于 transfer 报告；
- `D_task`：下游任务，完全不参与任何选择。

每个 split 记录 token offset、batch 数、序列长度、token 数和 dataset source。主实验至少使用 32 个 Wikitext holdout batch 和 32 个 C4 holdout batch；2.7B 若显存或时间受限，必须在启动前预注册 24 batch，不能跑完后临时缩减。

### 4.3 基线层级

`direct ternary` 只能作为诊断下界，不能作为论文竞争性结论的唯一基线。最多保留三类 baseline family，但必须先完成强基线审计：

1. **Direct ternary PTQ**：当前 direct ternary 起点；
2. **Strong ternary PTQ**：第一优先是 PT²-LLM 的完整 ATQ（ITF+AGA，必要时包含 SSR）；第二优先是 PT-BitNet 等 block-output reconstruction 型三值 PTQ。必须记录原始实现、版本、calibration token 数、实际 codebook、scale/shift metadata 和是否使用 outlier/等价变换；
3. **QAT reference**：只作为上界和 gap 参照，不向 CEGSP 提供 teacher、logits、checkpoint 或 latent weights。

PT² 如果由于架构、版本或输入接口无法公平迁移，必须单独记录 `baseline-unavailable`，不能把 CEGSP 对弱 baseline 的提升写成 SOTA。最近还出现了 PTQTP、ScaleQ-1.58 和 TWLA 等不同目标/表示的工作：PTQTP 使用 trit-plane 结构分解，ScaleQ-1.58 强调 reasoning-trace calibration，TWLA 同时改变 activation bit-width；它们需要先做有效 bit、校准假设和部署格式审计，不能未经归一化直接并入 W1.58A16 主表。

### 4.3.1 强基线必须通过的公平性检查

- 同一模型、同一 tokenizer、同一 Wikitext/C4 split 和相同 calibration token 预算；
- 同一目标：weight-only W1.58A16。若方法使用 asymmetric `mu+alpha*S`、outlier branch、低秩残差或 mixed precision，单独列出 metadata 和有效 bpw；
- 同一评价：FP16、direct ternary、strong PTQ、CEGSP 都在 untouched Wikitext/C4 上报告 per-batch NLL；
- 先在至少一个 OPT 模型上复现强基线，再谈 CEGSP 是否超过它。复现误差超过预注册容差时，只能标为 `baseline-reproduction-failed`；
- 对 Pythia 等架构，如果强基线没有 adapter，不能把“强基线缺席”解释成 CEGSP 的跨架构优势。

### 4.3.2 关键的 2×2 组合实验

强基线比较不能只做 `direct vs strong vs CEGSP`，还必须测量 CEGSP 是否对强方法具有独立增益：

| 起点 | 不加 CEGSP | 加 CEGSP |
|---|---|---|
| direct ternary | direct | direct + CEGSP |
| strong ternary PTQ | strong | strong + CEGSP |

其中 `strong + CEGSP` 使用基线实际导出的三值状态作为初始状态；若基线为 `Q=mu+alpha*S`，CEGSP 只编辑 `S`，默认冻结 `mu/alpha`，不重新引入一套独立 quantizer。这样可区分三种结论：

1. CEGSP 超过 strong：可以作为 standalone ternary PTQ 方法主张；
2. CEGSP 不超过 strong，但 strong+CEGSP 继续改善 strong：应将论文定位为“quantized-point function repair layer”，创新仍成立，但不能宣称替代 PT²；
3. strong+CEGSP 也无增益：当前 CEGSP 更像针对 direct PTQ 的修复器，论文价值不足，除非机制分析本身形成独立贡献。

### 4.4 B0 通过标准

- 同一模型、同一数据 split 下 direct ternary 可重复；
- FP16、direct ternary、CEGSP 三种状态的 NLL 均 finite；
- result JSON 记录模型、架构 adapter、split、offset、层数、非零率、耗时、显存和 clean-room flags；
- 每一个方法只使用 fit/val 允许的数据；C4 和 task 不参与选择；
- 自动检查不允许出现 QAT 文件路径、optimizer step 或隐式 FP 权重更新。

## 5. B1：统一主结果矩阵——一次性回答“是否普适”

### 5.1 目的

把当前零散的 OPT scale 和 Pythia cross-architecture 结果合并成论文主表，并先加入强 ternary PTQ 和 `strong+CEGSP` 组合，避免只相对 direct baseline 得出过强结论。

### 5.2 模型矩阵

第一版主矩阵使用三格，控制成本同时覆盖规模和架构：

| Family | Model | 层数/规模角色 |
|---|---|---|
| OPT | OPT-1.3B | 已有 scale 证据，作为中等规模 |
| OPT | OPT-2.7B | 已有 larger-scale 证据 |
| GPT-NeoX | Pythia-1B | 已有跨架构证据 |

如果需要一格更小的 sanity，可把 OPT-350M 放入 appendix，不再把它作为主结论来源。

### 5.3 统一系统和指标

每个模型、每个预注册 offset 运行：

- FP16 reference；
- direct ternary；
- strong ternary PTQ（优先 PT²-LLM ATQ）；
- CEGSP support-only `k25`；
- CEGSP joint `k25`；
- CEGSP joint `k50`；
- strong PTQ + CEGSP joint `k25`；
- signflip-only `k25` 作为结构对照；
- 若 strong PTQ 无法在该架构运行，必须在表中写 `N/A: adapter unavailable`，不能用 direct 结果替代。

首要指标：Wikitext holdout NLL、C4 holdout NLL、相对 direct ternary 的 paired delta 及 95% bootstrap CI。次要指标：PPL、非零率、编辑数量、耗时、峰值显存。

### 5.4 B1 的统一 gate

将每个模型×offset 视为一个 cell，不根据单个最佳模型下结论：

- 至少 4/6 cells 中，CEGSP joint 在 Wikitext 和 C4 同时改善；
- 在至少一个共同支持 strong PTQ 的模型上，`strong+CEGSP` 不劣于 strong PTQ；若要声称 standalone SOTA，则必须额外满足 CEGSP standalone 不劣于 strong PTQ；
- 所有模型的平均 paired delta 为负；
- 不允许出现一个模型在两个 holdout 上同时明显恶化（预设恶化阈值 0.02 NLL）；
- Pythia cell 必须保留 adapter metadata；
- `k25` 和 `k50` 都报告，默认配置只能由 validation 规则预先确定。

如果 B1 只部分通过，结论收窄为“在 OPT/GPT-NeoX decoder LM 上有效”，而不是添加新模块救援。

### 5.5 论文产物

- Main Table 1：三模型、FP16/direct/strong PTQ/CEGSP 的 NLL、PPL、显存和时间；
- Figure 2：direct ternary → CEGSP → FP16 的 gap 图；
- Appendix Table：每个模型的 per-layer selection、编辑类型和非零率。

## 6. B2：PTQ–QAT gap 与成本闭环

### 6.1 为什么必须做

目前 CEGSP 已证明“比 direct ternary 好”，但论文不能自动把这句话升级为“弥合了 QAT gap”。需要在同一模型、同一数据和同一三值码本下测量：

\[
\Delta_{PTQ}=L_{direct}-L_{FP16},
\qquad
\Delta_{QAT}=L_{direct}-L_{QAT},
\]

并报告 CEGSP 的 gap closure：

\[
\operatorname{Closure}(CEGSP)=
\frac{L_{direct}-L_{CEGSP}}
{L_{direct}-L_{QAT}}.
\]

该比例只有在 QAT 确实优于 direct 且分母为正时报告；否则只报告原始 NLL，不强行计算比例。

### 6.2 QAT 的角色

QAT 是独立 reference/upper-bound pipeline：

- 使用同一预训练模型和同一三值权重定义；
- 使用独立的训练过程和固定训练预算；
- 不把 QAT 结果、checkpoint、logits、latent weights 传给 CEGSP；
- CEGSP 的运行脚本不读取任何 QAT artifact。

QAT 不需要覆盖所有大模型。优先在 OPT-350M 与 Pythia-410M/1B 做 gap gauge；大模型用 CEGSP 的 strict-PTQ 结果和已测成本支撑扩展性。

### 6.3 成本指标

必须把成本拆成：

- 模型/数据加载；
- direct ternary apply；
- CE gradient collection；
- candidate generation；
- single-layer validation；
- final patch evaluation；
- 峰值显存。

比较三种 end-to-end pipeline：direct PTQ、CEGSP、QAT。论文中同时报告：

1. 总 wall-clock；
2. 相对 direct PTQ 的额外时间；
3. 相对 QAT 的时间比例；
4. 是否保留 optimizer state 或 FP latent copy。

### 6.4 B2 gate

- CEGSP 在至少一个小模型和一个非 OPT 模型上闭合正的 QAT gap；
- gap closure 为正，且默认目标为至少 25%；如果低于 25%，不能称“显著弥合”，只能称“部分恢复”；
- CEGSP 无 optimizer step；
- CEGSP 额外 wall-clock 不超过 direct PTQ end-to-end pipeline 的 2 倍，且远低于 QAT；若超过，则优先减少评估集合或优化实现，不引入新学习模块。

若 QAT reference 本身没有稳定优于 direct ternary，说明 gap gauge 不成立；这不是 CEGSP 失败，应重新检查 QAT 训练预算和码本一致性。

### 6.5 论文产物

- Main Table 2：FP16/direct/QAT/CEGSP 的 NLL 与 gap closure；
- Figure 3：质量—时间 Pareto 图；
- Appendix：QAT 训练步数、显存、优化器和收敛曲线。

## 7. B3：一次性完成三值特异性与简洁性验证

### 7.1 目的

不再分别验证 support、signflip、random、budget。将它们放入同一套 matched protocol，用一次综合实验回答：收益是否三值原生、是否需要 joint、是否值得保留额外组件。

### 7.2 统一对照

在 OPT-350M 与 Pythia-1B 各选一个固定 offset，运行：

1. direct ternary；
2. CEGSP support-only `k25`；
3. CEGSP signflip-only `k25`；
4. CEGSP joint `k25`；
5. random support `k25`；
6. random signflip/joint `k25`；
7. same-layer support vs same-layer signflip；
8. 可选 binary-like no-zero control，仅放 appendix，不改变主方法。

所有方法使用相同 layer budget、相同 max-edits、相同 fit/val/holdout；不能让某个 control 多编辑或少编辑。

### 7.3 判定逻辑

- joint > random：证明不是任意离散扰动；
- support > matched signflip：支持 zero-support 的三值特异性；
- joint ≥ support 且 signflip 仍有增益：最终方法保留 joint，但不夸大“只有 zero-support 才有效”；
- support 与 signflip 都相当：主张收窄为 quantized-point CE editing，三值特异性只作为辅助观察；
- all-layer 明显劣化：保留 top-k budget 作为稳定性设计，不把它解释为方法失败。

### 7.4 简洁性检查

只做一个 deletion check：

- canonical CEGSP：一次 CE gradient + 一次候选排序 + 一次 top-k patch；
- overbuilt variant：增加第二次 CE gradient 或跨层迭代搜索。

如果 overbuilt variant 不能在 holdout 上稳定超过 canonical，论文明确说明额外迭代不值得；若能超过，也只作为 future work，不能把论文变成大搜索系统。

### 7.5 论文产物

- Main Table 3：support/signflip/joint/random matched controls；
- Figure 4：每个层的 candidate score 与 holdout delta 的关系；
- Appendix：编辑状态转移统计和零态比例。

## 8. B4：泛化、下游和失败边界

### 8.1 Calibration offset robustness

对 Pythia-1B 做至少两个新的 offset（例如 fit/val/C4 各平移到预注册的非重叠区间），复用 B1 的 `k25/k50`，不重新扫 k。OPT-1.3B 或 2.7B 只需选择一个模型复现一个 offset，以控制成本。

判定不是“每个 offset 都赢”，而是：

- 多数 offset 的 Wikitext/C4 paired delta 为负；
- 平均 delta 为负且 CI 不显示严重不稳定；
- 某个 offset 失败只记为 split sensitivity，不触发方向切换。

### 8.2 下游任务

当前 LAMBADA-style hard top-1 出现 floor effect，不能作为主下游证据。下一版统一使用一个能保留连续分数的任务协议：

- LAMBADA sequence NLL/accuracy，报告 NLL 为主；或
- HellaSwag/PIQA 的 multiple-choice log-likelihood，报告 accuracy 和 mean log-likelihood。

任务数据只用于最终评估，不用于 layer/k 选择。优先选择 2 个任务，不要同时铺开十几个 benchmark。

下游 gate：CEGSP 平均任务 log-likelihood 不劣于 direct ternary，且至少一个任务改善；若 NLL 改善但任务无改善，论文只声称 language-model function preservation，不声称通用下游收益。

### 8.3 失败边界

记录以下情形，而不是隐藏：

- all-layer patch 的跨层干扰；
- k50 相对 k25 的边际收益下降；
- 某一架构或 offset 对 CEGSP 不敏感；
- support relocation 与 signflip 差距变小；
- NLL 改善不能转化为 hard accuracy。

这些负结果可以形成 Figure/Appendix，帮助论文呈现方法边界，避免只展示最优数字。

## 9. B5：可选，不得阻塞主论文

以下内容只有在 B1–B4 通过后才做：

- 第二个非 OPT family，如带独立 Q/K 的 GPT-J 类架构；
- 结构化 support relocation（如按 channel/block 约束）；
- 更大模型或更多下游任务；
- 更精细的候选搜索、beam 或跨层联合目标。

它们都不是当前主方法成立所必需的，不能在 B1 之前抢占算力。

## 10. 实际运行顺序与决策门

### M0：Harness 与强 baseline audit

**Run group：** `CEGSP-11A-AUDIT`

内容：统一 split 记录、direct parity、Pythia adapter parity、PT²/strong ternary PTQ baseline 接口确认、`strong+CEGSP` 状态接口和 cost timer。预计 0.5–1 GPU 小时，主要是工程工作。

**决策：** M0 不通过时停止数值主实验，先修 harness；不把诊断故障写成方法负结果。

### M1：统一主矩阵

**Run group：** `CEGSP-11B-MAIN-MATRIX`

内容：OPT-1.3B、OPT-2.7B、Pythia-1B；direct、强 ternary PTQ、CEGSP joint/support/signflip、`strong+CEGSP`；`k25/k50`；Wikitext/C4 holdout。优先复用缓存，预计 3–6 GPU 小时。

**决策：** 通过 B1 才进入论文主结果；若 CEGSP 只赢 direct 而输 strong，则暂不否定方法，转为检查 `strong+CEGSP`；若组合也失败，则收窄为 direct-repair 诊断结果，不增加新模块。

### M2：QAT gap 与成本

**Run group：** `CEGSP-11C-GAP-COST`

内容：OPT-350M + Pythia-410M/1B 的 QAT reference 与 CEGSP 对照；同码本、同数据、固定训练预算；记录完整 cost breakdown。预计 4–12 GPU 小时，取决于 QAT 步数。

**决策：** 若 QAT gap 不能被可靠测出，先修 gap gauge；若 CEGSP closure 正但低于 25%，论文主张改为“部分弥合”。

### M3：综合机制与简洁性

**Run group：** `CEGSP-11D-MECHANISM-COMPACT`

内容：两个模型一次性执行 support/signflip/joint/random/deletion 对照。预计 1–3 GPU 小时。

**决策：** 只决定论文措辞和 canonical default，不开启新的算法分支。

### M4：offset + downstream

**Run group：** `CEGSP-11E-GENERALIZATION`

内容：Pythia 新 offset、一个 OPT 新 offset、两个连续分数下游任务。预计 2–5 GPU 小时。

**决策：** 通过后冻结结果，进入论文写作；不再以局部结果驱动新方向。

## 11. 统计与报告协议

每个 holdout batch 都保存 per-batch loss，而不只保存均值。对 `method - direct` 做 paired bootstrap 10,000 次，报告均值、95% CI 和改善比例。每个模型、offset、方法的 selection history 都要保存。

主表只使用预先定义的 aggregate：

\[
\bar{\delta}=\frac{1}{N}\sum_{m=1}^{N}\delta_m,
\]

不允许从 Wikitext、C4、task 三者中事后选择对自己最有利的指标。若一个指标改善而另一个恶化，报告 mixed result，并根据预注册 gate 判定，不重新定义主指标。

结果标签固定为：

- `diagnostic`：harness 或指标问题；
- `module-positive`：局部模块有信号；
- `robust-positive`：跨 offset/model 后稳定；
- `paper-claim-ready`：主矩阵、gap、三值对照、成本与至少一个下游闭环完成。

## 12. 明确不再做的事情

- 不再逐层枚举新的 projection mask；
- 不再为每个模型事后调 epsilon、threshold factor 或 k；
- 不再加入 QAT teacher、QAT logits、latent weight 或 teacher distillation；
- 不再因为某一个 offset 或某一个下游任务负结果更换 CEGSP；
- 不再把 all-layer 失败解释成整个方法失败；
- 不再把一次 support-positive 写成普适三值理论；
- 不再以增加实验数量代替强 baseline、QAT gap 和下游闭环。

## 13. 止损与转向条件

只有出现以下情况，才允许修改主方法或重新审视研究方向：

1. 在统一协议下，CEGSP joint 在至少 3 个独立 model×offset cell 上同时无法改善两个 holdout，而 support/signflip/random 中无简单且稳定的替代；
2. CEGSP 的额外成本超过 direct PTQ 两倍，并且无法通过评估缓存/候选实现优化；
3. Pythia 与 OPT 都显示 support/signflip/random 无差异，证明收益与三值 zero state 无关；
4. QAT gap 在码本一致、训练充分的条件下存在，但 CEGSP 在多个模型上 closure 接近零；
5. 结果只能在 validation 上改善，任何 untouched Wikitext/C4 都不能改善。

除此之外，负结果只用于调整默认 k、选择策略或论文适用边界，不允许重新发明方法。

## 14. 当前首选实验

下一批不是单个模块实验，而是 `CEGSP-11A → 11B` 的连续 paper-readiness batch：

1. 完成统一 harness/strong baseline audit；
2. 用固定 `k25/k50` 在 OPT-1.3B、OPT-2.7B、Pythia-1B 上重建统一主表；
3. 输出 per-batch bootstrap CI、总成本和 architecture metadata；
4. 只有主表通过后，才做 QAT gap、下游和 deletion check。

这条路线可以让一个实验同时服务于“普适性、稳定性、成本和论文主表”，而不是再花几轮只回答“support 还是 signflip 更好”。
