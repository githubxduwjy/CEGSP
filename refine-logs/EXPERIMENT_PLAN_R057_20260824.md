# R057 实验方案：checkpoint gate 与受控超参数敏感性

**问题**：既往 hard-T 结果不稳定，究竟是核心方法失效，还是单一超参数配置没有覆盖有效区域？  
**方法主张**：局部 hard-T 候选需要在双分布 checkpoint-monotonic gate 下选择；少量、预注册的超参数敏感性可以改善候选质量，但不能使用 test 事后挑参。  
**日期**：2026-08-24

## 1. Claim Map

| Claim | 为什么重要 | 最低可信证据 | 对应实验 |
|---|---|---|---|
| C1：此前失败具有超参数敏感性，而非所有 hard-T 候选必然无效 | 防止从单配置负结果过度外推 | 在 `(10,11)` 上，预注册配置集合中至少一个非 official 候选通过 gate，并在 untouched test 的 W2/C4 mean-NLL、CVaR 全部不退化 | R057A |
| C2：选出的配置不是只适合单一成功层 | 防止把层 11 的特例包装成通用机制 | 将 R057A 冻结配置迁移到 `(30,31)`；checkpoint gate 能安全接受有益候选或回退 official，且 untouched test 不退化 | R057B |
| Anti-claim：收益只是大规模搜索或偷看 test 的结果 | 审稿人最可能的公平性质疑 | 仅 5 个 OFAT 配置；选择只读取 gate；test 仅用于一次最终判定；完整保存所有负结果 | R057A/B 审计 |

R057 不主张全模型最优 PPL，也不主张已得到联合跨层求解器。它只回答“合理但有限的超参数选择能否改变 checkpoint-gated hard-T 的结论”。

## 2. 总体结构

### 阶段 A：R057A 超参数敏感性与冻结选择

- 层窗口：只使用既往成功窗口 `(10,11)`。
- calibration：WikiText2，seed 由配置指定。
- score windows：88--103。
  - gate：88--95。
  - untouched test：96--103。
- 数据分布：WikiText2 与 C4 同时评分。
- 每个配置均比较：`official`、`hard_l10`、`hard_l11`、`hard_l10_l11`。
- 所有配置完成后，机器只根据 gate 选择一个 `(配置, 候选)`；然后只对这个冻结选择解释 test 结果。

### 阶段 B：R057B 冻结配置的深层迁移

- 仅当 R057A 选择了非 official 且通过 untouched test 时运行。
- 层窗口：既往失败窗口 `(30,31)`。
- 使用 R057A 冻结的 `calib_nsamples`、`blocksize`、`max_steps`；不重新选参。
- score windows：104--119。
  - gate：104--111。
  - untouched test：112--119。
- 候选：`official`、`hard_l30`、`hard_l31`、`hard_l30_l31`。
- checkpoint gate 可以接受非 official，也可以安全回退 official；不得因为深层结果调整 R057A 参数。

### 阶段 C：R057C 稳定性确认（条件运行）

仅当 R057A/B 均无 untouched regression 时运行：

- 固定 A 选出的配置与 gate，不再搜索参数。
- calibration seed 增加 1、2；测试序列保持冻结。
- `validation_fraction` 先保持 0.25。只有 seed 复现后，才在附录做 `{0.125, 0.25, 0.5}` 敏感性；这些结果不能反向改变主配置。
- 目标是报告选择稳定性与均值±标准差，而不是再次寻找最好 seed。

## 3. R057A 配置矩阵

采用 one-factor-at-a-time（OFAT），默认配置作为共同锚点：

| ID | calib_nsamples | blocksize | max_steps | validation_fraction | seed | 目的 |
|---|---:|---:|---:|---:|---:|---|
| H0 | 8 | 128 | 4 | 0.25 | 0 | 既往默认配置 |
| H1 | 8 | 128 | 2 | 0.25 | 0 | 检验过度更新 |
| H2 | 8 | 128 | 8 | 0.25 | 0 | 检验更新不足 |
| H3 | 16 | 128 | 4 | 0.25 | 0 | 检验校准样本不足 |
| H4 | 8 | 64 | 4 | 0.25 | 0 | 检验块粒度过粗 |

本轮明确不运行 3×2×2 的 12 组合全因子搜索。若 H1--H4 中某一单因素改变通过 R057A/B，组合效应属于后续 R058，不得在本轮追加。

### 暂时固定的参数

- `validation_fraction=0.25`：与 R042--R054 保持可比；避免同时改变数据划分与更新预算。
- `seed=0`：用于选择阶段；seed 1/2 只做条件复现。
- `mean_epsilon=0`, `cvar_epsilon=0`：禁止事后放宽。
- `seqlen=2048`：保持评估语境不变。
- 投影集合与层集合冻结：禁止 projection mask 或额外层枚举。

注意：`blocksize=64` 会改变 scale/metadata 开销，因此 R057 只把它作为机制敏感性结果；任何最终 1.58-bit/bpw 声明必须单独做 bit-matched 比较。

## 4. 冻结选择规则

对每个配置中的每个候选，先相对该配置的 matched `official` 计算 gate 指标。

### 4.1 checkpoint eligibility

候选必须同时满足：

1. W2/C4 的 layer-10 mean NMSE delta 均 `<=0`；
2. W2/C4 的 layer-11 boundary mean NMSE delta 均 `<=0`；
3. gate `nonfinite_count=0`。

### 4.2 functional eligibility

checkpoint eligible 后，还必须满足 gate 上：

- W2/C4 mean-token-NLL delta 均 `<=0`；
- W2/C4 CVaR10-NLL-increase delta 均 `<=0`。

### 4.3 配置与候选选择

定义

\[
S(h,q)=\max_{d\in\{W2,C4\},m\in\{mean,CVaR\}}
\Delta L_{d,m}(h,q).
\]

- 仅在双重 eligible 的非 official `(h,q)` 中选择 `S` 最小者。
- 若并列，按固定顺序 `H0,H1,H2,H3,H4`，再按 `hard second, hard first, hard both` 选择，避免实现细节导致不稳定。
- 若没有非 official 候选 eligible，则选择 `official/H0`，判为 `INCONCLUSIVE_OVERCONSERVATIVE`，不进入 R057B。
- test 数值不得参与上述选择。

## 5. 机器判定 Gate

### R057A

- `SUPPORT_A`：选择非 official，且冻结选择在 untouched test 上 W2/C4 mean-NLL、CVaR10、nonfinite 全部 delta `<=0`。
- `FAIL_A_GENERALIZATION`：选择非 official，但 test 任一功能指标 delta `>0`。
- `INCONCLUSIVE_OVERCONSERVATIVE`：没有非 official 通过 gate。
- `INVALID`：配置、样本编号、行数、有限性或 matched official 不完整。

### R057B

- `SUPPORT_B_ACCEPT`：冻结参数在 `(30,31)` 选择非 official，且 untouched test 全指标不退化。
- `SUPPORT_B_SAFE_FALLBACK`：gate 选择 official，test 等于 official；这支持安全门但不支持深层收益。
- `FAIL_B_TRANSFER`：选择非 official 后 test 发生任一退化。
- `INVALID`：完整性失败。

### 整体解释

| A | B | 结论 |
|---|---|---|
| SUPPORT_A | SUPPORT_B_ACCEPT | 强支持“超参数敏感 + checkpoint gate 可迁移” |
| SUPPORT_A | SUPPORT_B_SAFE_FALLBACK | 支持安全自适应门；收益仍层位置依赖 |
| SUPPORT_A | FAIL_B_TRANSFER | 超参数能修复成功窗口，但 checkpoint gate 尚不能跨深度泛化 |
| FAIL_A_GENERALIZATION | 不运行 | 单因素调参不足以解释既往失败 |
| INCONCLUSIVE | 不运行 | gate 过度保守，不能宣称方法有效或无效 |

单次负结果只关闭对应命题，不否定三值 PTQ、hard-T 或跨层约束的全部研究方向。

## 6. 完整性审计

R057A 期望原始评分行数：

\[
5\ configs\times4\ candidates\times2\ datasets\times16\ sequences=640.
\]

R057B 单配置期望 128 行。每轮必须检查：

- config 与目录名一致；
- 序列严格为预注册区间；
- 每个 candidate/dataset 行数完整；
- 所有浮点数 finite；
- `nonfinite_count` 完整记录；
- matched official 来自相同超参数配置；
- 选择程序在读取 test 结果前已根据 gate 冻结 selection；
- 所有正负配置都写入 tracker 和 TSV，不删除失败目录。

## 7. 运行顺序与预算

| Milestone | 内容 | 预计 4090 时间 | Stop/Go |
|---|---|---:|---|
| M0 | 远端 py_compile、bash -n、GPU witness | 2--3 分钟 | 任一失败只修 harness |
| M1 | H0 smoke，确认行数和 score windows | 13--18 分钟 | 产物完整后继续 |
| M2 | H1--H4 串行完成 R057A | 55--90 分钟 | 全部完成后一次性机器选择 |
| M3 | R057A 完整性和 gate 判定 | 2--5 分钟 | 仅 SUPPORT_A 进入 B |
| M4 | R057B 冻结迁移 | 13--25 分钟 | 按 B gate 解释 |
| M5 | 条件 seed 复现 | 30--60 分钟 | 仅 A/B 无退化时运行 |

R057A+B 预计约 1.3--2.2 GPU 小时；若包括 seed 复现，总计约 2--3 GPU 小时。`nsamples=16` 可能是最长单项。

## 8. 下午 SSH 提交清单

提交前只需完成：

1. 同步 `r048_distribution_holdout_gate.py`、R057A runner 和 analyzer 到 `/root/PT2-LLM-official/remote-tools/`；
2. 在 `/root/PT2-LLM/venv` 中运行 `py_compile`；
3. 确认 `/root/models/Llama-2-7b-hf` 可读、GPU 空闲；
4. 用独立 screen 启动 R057A，并将 stdout/stderr 写入结果目录；
5. R057A 完成前不启动 R057B，不读取非选中配置的 test 做人工挑选。

建议 screen 名：`r057a_hparam_gate`。  
建议结果目录：`/root/PT2-LLM-official/aris-runs/r057a_hparam_gate_20260824`。

## 9. 有意删去的实验

- 12 组合全因子扫描：成本高且容易变成调参论文。
- epsilon sweep：已看到 test 后再放宽会破坏预注册。
- 新层或 projection mask 搜索：不能回答当前超参数归因问题。
- 直接全模型 PPL：R057 仍是接受门机制验证，尚未达到全模型集成 gate。
- 同时搜索 validation fraction 与 seed：留到冻结配置后的稳定性阶段。
