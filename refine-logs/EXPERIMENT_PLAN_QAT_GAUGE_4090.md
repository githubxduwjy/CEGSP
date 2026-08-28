# RTX 4090 实验计划：三值 QAT–PTQ 离散 Gauge 诊断

**问题**：三值 QAT 是否学习到 direct PTQ 无法恢复、且超越边际稀疏度与显著性的耦合算子结构？  
**方法主张**：先做机制诊断；只有发现 gauge-controlled、可迁移的残余三值交互规律，才设计联合 PTQ 方法。  
**硬件**：单张 RTX 4090 24GB  
**日期**：2026-08-24

## 1. Claim Map

| Claim | 为什么重要 | 最低可信证据 | 对应实验 |
|---|---|---|---|
| C1：短程 ternary QAT 产生 direct PTQ 不具备的残余耦合结构 | 决定“从 QAT 借机制”是否有事实基础 | 在匹配零率、scale 粒度、幅值和 salience 后，QAT 的 Q/K 或 V/O 依赖显著偏离分层 permutation null，并在至少 3/4 个代表层方向一致 | G4090-03/04 |
| C2：残余结构具有函数意义，而非坐标或剪枝伪影 | 决定该结构能否成为论文主张 | 交互指标在 held-out C4/W2 上增量预测 composed-operator distortion；causal support shuffle 在保持边际统计时显著恶化 QAT operator fidelity | G4090-05 |
| Anti-claim：观察仅来自更多训练、共同显著性或任意坐标基底 | 审稿人最强反驳 | 同 checkpoint、matched zero-rate、salience-preserving null、signed-permutation gauge stress 和计算/数据审计全部成立 | G4090-04/05 |

本阶段不主张提出新 PTQ 方法，不运行 Idea 2 的 full-model co-ternarization。

## 2. 4090 资源设计

### 2.1 模型

- sanity：OPT-125M；
- primary：OPT-350M；
- conditional replication：TinyLlama-1.1B，只在 primary gate 通过后运行，且不属于首轮预算。

选择 OPT 的原因是参数规模适合 24GB、Q/K/V/O 结构清晰，并与 PT² 的 OPT 实验家族接近。首轮不使用 7B 模型做 QAT。

### 2.2 QAT 固定配置

| 参数 | 固定值 |
|---|---:|
| 权重前向 | group-wise ternary `{-alpha,0,+alpha}` |
| latent weights | BF16 |
| activations | BF16，不量化 |
| optimizer | AdamW；不依赖 bitsandbytes |
| sequence length | 512 |
| micro-batch | 1 |
| gradient accumulation | 8 |
| effective tokens/step | 4096 |
| steps | 256 |
| 总训练 token | 1,048,576 |
| gradient checkpointing | 开启 |
| KV cache | 关闭 |
| seed | 0；通过 gate 后才增加 1/2 |

短 QAT 的目的不是达到 SOTA perplexity，而是产生可观测的离散适配轨迹。不得因为 loss 尚未收敛而事后追加步数。

### 2.3 显存策略

1. FP、PTQ、QAT 三个分支顺序执行，任意时刻 GPU 只驻留一个模型；
2. QAT 完成后只保存部署三值码、scale 和必要统计，不保存 optimizer state 供后续分析；
3. ternary mask 转成 CPU `uint8` 或 bit-packed 文件；
4. forward hook 每次只启用一个目标层，在线累计协方差和 loss summary；
5. 不保存所有 token 的全层 hidden states；
6. 每 20 step 记录 `torch.cuda.max_memory_allocated()`；
7. 预注册显存 gate：峰值必须小于 21.5 GiB，保留约 2.5 GiB 给 CUDA workspace 和碎片。

若 OOM：

1. 首次只将 sequence length 从 512 降为 256，同时把 accumulation 从 8 调为 16，保持每 step 4096 token；
2. 仍 OOM 则启用 CPU optimizer-state offload；
3. 仍 OOM 判为 harness failure，不改模型、不删指标、不降低 zero-rate controls。

## 3. 数据与严格划分

- QAT-fit：C4 train 的固定不重叠片段，1,048,576 token；
- PTQ calibration：另一段 C4 train，128 条 × 512 token；
- diagnostic validation：再取 32 条 C4 × 512；
- untouched operator test：32 条 C4 validation + 32 条 WikiText-2 test windows，各 512 token；
- 所有 document IDs、token offsets 和 hash 在运行前写入 manifest；
- untouched test 不参与 threshold、zero-rate 或 QAT checkpoint 选择。

QAT 固定使用第 256 step checkpoint；不按 untouched 指标早停。

## 4. 分支与公平性

从同一个 FP checkpoint 导出：

1. `FP`：只提供权重和 operator targets；
2. `PTQ-direct`：强三值初始化器，固定 group size、scale convention；
3. `QAT-short`：使用完全相同的三值前向码本与 group size，更新 latent weights 256 steps；
4. `PTQ-zero-matched`：对 direct PTQ threshold 做预先规定的单调求解，使每层/每 group 的零率匹配 QAT-short，用来排除边际 sparsity 差异。

所有比较报告实际 bpw、group metadata、zero rate、正负比例与 scale 分布。

## 5. Experiment Blocks

### Block B0：4090 harness 与显存验收

- Claim tested：实验可在单张 4090 上诚实运行。
- 模型：OPT-125M。
- 内容：16 step QAT、8 条 PTQ calibration、1 层 Q/K mask 导出、一次 operator metric。
- 指标：finite loss、ternary invariant、峰值显存、step time、导出可重载一致性。
- 成功标准：peak VRAM < 18 GiB；16 step 全 finite；导出模型逐元素只含三值状态；重载 operator 输出一致。
- 失败解释：仅说明 harness/环境问题，不评价研究假设。
- 优先级：MUST-RUN。

### Block B1：同 checkpoint 分支构建

- Claim tested：QAT/PTQ 差异可以在严格匹配设置下测量。
- 模型：OPT-350M。
- 分支：FP、PTQ-direct、QAT-short、PTQ-zero-matched。
- 指标：train loss trajectory、held-out NLL（诊断）、零率、符号比例、scale、weight NMSE、VRAM、wall-clock。
- 成功标准：全部 finite；QAT 三值快照可部署；PTQ-zero-matched 的每层 zero-rate 与 QAT 差异不超过 0.2 percentage point。
- 失败解释：若无法 zero-match，则后续 support 比较不具备公平性。
- 优先级：MUST-RUN。

### Block B2：残余三值依赖与分层 Null

- Claim tested：C1。
- 层：0、7、15、23。
- 算子：Q/K 与 V/O；首轮不分析 MLP/SwiGLU。
- 统计量：
  - matched-index support contingency；
  - conditional mutual information；
  - latent-dimension active-path coverage；
  - sign-conditioned support dependence；
  - Q/K logit NMSE 与 V/O composed-map NMSE。
- null 保留：每矩阵总零率、row/column/head/group support counts、正负比例、weight-magnitude decile、activation/Hessian-salience decile。
- permutation：pilot 200 次；通过后才增加到 1000 次。
- 成功标准：QAT-short 相对 PTQ-zero-matched 在至少 3/4 层出现一致方向，且 salience-preserving null `p<0.01`（FDR 后）或绝对 z-score > 2.58。
- 失败解释：关闭“QAT 学到残余 support grammar”的主张。
- 优先级：MUST-RUN。

### Block B3：Gauge 压力测试与函数增量解释力

- Claim tested：C2 与 anti-claim。
- gauge：head 内 signed permutations、允许被 scale 吸收的对角符号变换；原始与变换结果按 canonical head/dimension alignment 比较。
- 函数指标：

$$
D_{QK}=
\frac{\mathbb E\|XW_QW_K^\top X^\top-X\widehat W_Q\widehat W_K^\top X^\top\|_F^2}
{\mathbb E\|XW_QW_K^\top X^\top\|_F^2+\epsilon}.
$$

对 V/O 使用对应 composed-map NMSE。
- 预测比较：baseline features（weight NMSE、zero-rate、salience）对比 baseline + interaction statistics；采用 leave-one-layer-out，不在同层拟合和评估。
- 成功标准：interaction statistics 的 held-out incremental `R^2 >= 0.05`，且结论在所有离散 gauge stress 下不改变方向。
- 失败解释：若只在原始坐标成立，判为坐标伪影。
- 优先级：MUST-RUN。

### Block B4：Causal support intervention

- Claim tested：残余依赖具有因果函数意义。
- 对每个选择 head 构造：QAT joint support、独立 shuffle、anti-aligned shuffle、pruning-shared support。
- 所有 intervention 保持各矩阵 row/column/group 零率、sign count、scale 和 salience decile。
- 每种 intervention 32 个样本；不重新调 scale。
- 成功标准：QAT joint support 的 worst-domain operator distortion 相对三个 matched controls 中最佳者至少低 5%，并在至少 3/4 层成立。
- 失败解释：若相关性存在但 intervention 无效，只能形成观察性诊断，不能进入方法论文。
- 优先级：MUST-RUN，但只在 B2 通过后运行。

### Block B5：4090 条件复制

- 模型：TinyLlama-1.1B 或第二个 300–500M 家族。
- 配置：先保持 sequence 512/micro-batch 1；若峰值超过 21.5 GiB，使用 256/accumulation 16。
- 触发：B2、B3、B4 全通过。
- 目的：leave-family-out 验证，而不是调优首模型规律。
- 优先级：NICE-TO-HAVE/CONDITIONAL。

## 6. Run Order and Milestones

| Run ID | 目标 | 预计 4090 时间 | VRAM 目标 | Decision Gate |
|---|---|---:|---:|---|
| G4090-00 | 环境/显存 preflight | 5–10 min | < 4 GiB | BF16、CUDA、SDPA、磁盘通过 |
| G4090-01 | OPT-125M 16-step harness | 20–40 min | < 18 GiB | finite + ternary invariant + export parity |
| G4090-02 | OPT-350M PTQ/direct/zero-match | 20–40 min | < 12 GiB | zero-match <=0.2 pp |
| G4090-03 | OPT-350M 256-step short QAT | 1–2.5 h | < 21.5 GiB | finite；固定 step-256 snapshot |
| G4090-04 | 四层 null/gauge analysis | 30–90 min | < 12 GiB | C1 gate |
| G4090-05 | causal intervention | 30–60 min | < 12 GiB | C2 gate |
| G4090-06 | second-family replication | 2–5 h | < 21.5 GiB | 仅全部主 gate 通过后 |

首轮 must-run 总预算约 3–5.5 GPU-hours。若只希望两小时内获得第一轮信号，可在 G4090-03 先运行 OPT-125M 256 steps；该结果只能决定 harness/机制是否值得在 350M 复制，不能形成论文结论。

## 7. 机器判定 Gate

### SUPPORT_DIAGNOSTIC

同时满足：

1. 所有数据、zero-rate 与 finite 审计通过；
2. B2 在至少 3/4 层通过；
3. B3 incremental `R^2 >= 0.05` 且 gauge stress 方向稳定；
4. B4 QAT support 相对最佳 matched control 的 worst-domain distortion 改善至少 5%。

### OBSERVATIONAL_ONLY

B2 通过，但 B3 或 B4 不通过。允许保留诊断结果，不得启动 Idea 2 方法开发。

### REJECT_INTERACTION_GRAMMAR

B2 在 salience-preserving null 后不通过，或结果仅在原始 coordinate basis 成立。停止该分支，不通过增加 QAT steps、permutation 数或事后换层拯救。

### INVALID_HARNESS

OOM、nonfinite、split 泄漏、zero-rate mismatch 或导出不一致。只允许修复明确工程问题后重跑同一预注册配置。

## 8. 实验明确不做

- 不用 LLaMA-2-7B 做首轮 QAT；
- 不缓存完整 teacher logits；
- 不同时加载 FP/PTQ/QAT；
- 不搜索 projection mask、层组合、threshold epsilon；
- 不根据 W2/C4 test 选择 checkpoint；
- 不在诊断通过前实现 full-model joint ternarization；
- 不因一次负结果否定所有 QAT-to-PTQ 迁移，只否定被该 gate 检验的 interaction grammar。

## 9. 最终 Checklist

- [ ] 4090 峰值显存审计已记录
- [ ] 同 checkpoint 与相同 ternary codebook
- [ ] PTQ-zero-matched 对照完成
- [ ] salience-preserving null 完成
- [ ] gauge stress 完成
- [ ] Causal intervention 完成或按 gate 停止
- [ ] 未接触 untouched test 做选择
- [ ] 所有正负结果进入 tracker
