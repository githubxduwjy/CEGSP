# PTQ 1.58-bit：R058 后研究方向复盘与转向建议

**日期**：2026-08-24  
**实验范围**：R014--R058  
**当前状态**：当前 Auto Research 实验循环停止；不得启动 R059、R057B 或其他新实验  
**正式外部 verdict**：`REVIEW_UNAVAILABLE`（三次规定的 Codex reviewer 调用均在 300 秒超时；已按 skill 规则 fail closed 并保存 trace）

## 1. 先给结论

我们确实进入了一个**局部死胡同**，但不是整个“三值 PTQ 的函数保持”方向失效。

死胡同是：继续把局部 Hessian/activation hard-`T` 更新生成的候选，交给越来越复杂的小样本 gate、checkpoint veto、层位置或超参数筛选去寻找偶然赢家。R048、R049、R050、R057A、R058 已经表明，这个选择过程会随数据窗口、校准 seed 和轻微配置改变而翻转。再调 gate 或 epsilon，学术上会变成选择规则工程，而不是新的三值表示方法。

仍成立的研究主线是：**三值 support `T` 的可学习性是真实的，但 `T` 必须由跨层/模型函数误差直接塑造，同时保留单层约束作为 trust region；不能先按局部误差形成 hard 候选，再事后过滤。**

首选转向是“短窗口跨层函数保持的三值 support 联合优化”；备选是“同一函数目标下的 soft-to-hard continuation”。校准数据对齐保留为后续辅助轴，不进入当前主贡献。

## 2. R058 完整性与结果

R058 正常退出（exit code 0），原始产物已拉取。机械核验全部通过：

- 配置与预注册完全一致：H0、固定 `hard_l11`、seed 1、layers (10,11)、8/8/8 calibration/gate/test、blocksize 128、max_steps 4、epsilon 0。
- 4 candidates x 2 datasets x 16 sequences = 128/128 行。
- 序列严格为 120--135；gate 120--127，untouched test 128--135。
- `finite_ok=true`，`nonfinite_total=0`。

`hard_l11 - official` 的配对均值如下，负数表示更好：

| Split | 数据集 | mean-token NLL | CVaR10 NLL increase | mean-NLL wins | CVaR wins |
|---|---|---:|---:|---:|---:|
| Gate | Wikitext2 | +0.018083 | +0.056855 | 2/8 | 4/8 |
| Gate | C4 | -0.091131 | -0.222336 | 8/8 | 6/8 |
| Untouched test | Wikitext2 | -0.026185 | -0.098823 | 8/8 | 7/8 |
| Untouched test | C4 | -0.080279 | -0.226816 | 7/8 | 6/8 |

checkpoint layer-11 NMSE delta 在 W2/C4 上分别为 `+0.000440568/+0.000243718`。但 R058 的预注册决策是 `REJECT_CANDIDATE`，因为候选已先违反 W2 功能 gate。故 R058 **没有复制** R057A 的“功能全通过但被 checkpoint 单独误杀”；它证明的是 gate/test 在 W2 上发生符号翻转。

## 3. 结果到主张：现在能说什么

由于 `result-to-claim` reviewer 超时，下面是确定性证据边界，不冒充独立外部 verdict。

| 主张 | 状态 | 证据边界 |
|---|---|---|
| activation/Hessian-aware 更新三值 `T` 是真实局部机制 | 支持 | R042c 在 56 blocks 的 untouched local test 上 median +5.8289%、mean +12.1401%、win 96.43%，且优于 matched ungated control |
| 局部 reconstruction 改善足以推出模型功能保持 | 否定 | R043/R044/R045：局部与 FP16-rest 指标全面偏好 hard-`T`，模型 W2 仍退化或 NaN |
| 当前 hard-`T` proposal + 小样本 gate 能稳定选择模型级赢家 | 不支持，当前实现应停止 | R048/R050 测试失败，R049 fold 选择翻转，R051 单点成功，R057A 单配置正例，R058 seed/window 下 gate/test 再翻转 |
| strict checkpoint zero-regression gate 普遍过保守 | 不支持 | R057A 是一个干净正例；R058 未复制，因为功能 gate 自身先失败 |
| error cancellation 是跨层失败的主机制 | 否定 | R054 预注册 gate 两个 split 均失败，risk correlation 低于 boundary-only comparator |
| 跨层函数目标一定能解决问题 | 尚未验证 | 它是由 R045/R046 暴露的目标错配推出的高信息量候选，不是已有结果 |

允许写进研究叙事的最强句子是：

> 在所测 LLaMA-2-7B blocks 上，校准验证的离散三值 support 更新能稳定降低局部 activation-weighted reconstruction error；但在完整量化轨迹中，这种局部优势对分布、窗口和校准 seed 敏感，不能由局部或 isolated-layer 指标可靠地转化为语言模型功能保持。

目前不能写“提出了更好的 1.58-bit PTQ 方法”，也不能写“checkpoint gate 已被证明过保守”。

## 4. 思路如何演进，以及哪些分支已经关闭

1. R014--R033：从 Haar/配对/带状 grid 出发，机制增益不足，关闭局部 pairing 主线。
2. R034--R041：activation-sorted Hadamard 发现真实 representation signal，但全层与 SSR 集成不稳定；关闭固定 transform 叠加与 layer/projection mask 枚举。
3. R042--R046：转向离散 `T` 更新，确认局部 support 可优化，同时证明局部与 FP16-rest proxy 不足；研究问题从“找更好坐标”升级为“保持量化模型函数”。
4. R047--R053：引入 quantized-context、双分布和短跨层窗口 gate；出现 R051 正例，但跨窗口稳定性不足。
5. R054：排除 error-cancellation 这一特定解释。
6. R057A--R058：确认超参数会影响结果，但默认正例无法跨 seed/window 稳定复制；当前 hard gate 路线达到止损点。

明确关闭：

- local Haar/activation pairing；
- Hadamard 与 SSR 的直接堆叠；
- layer/projection mask 枚举；
- block NMSE 或 isolated FP16-rest metric 作为充分 acceptance rule；
- 0/1 层窗口继续增加样本量；
- cancellation-constrained solver；
- 围绕 R057A/R058 事后调 epsilon、checkpoint threshold 或更多 OFAT 超参。

尚未关闭：

- activation-aware ternary support 学习本身；
- structured transform 作为初始化/预条件器，而非主贡献；
- 跨层函数保持联合优化；
- soft-to-hard 离散求解；
- 明确标注为 task-adaptive PTQ 的 calibration alignment。

## 5. 四个候选方向的比较

| 方向 | 科学信息量 | 新颖性潜力 | 实现/算力 | 主要风险 | 判断 |
|---|---:|---:|---:|---|---|
| A. 继续 hard-T gate | 低 | 低 | 低到中 | gate engineering、选择偏差、反复符号翻转 | 停止 |
| B. 跨层函数保持 + 单层 trust region | 高 | 中到高 | 中 | 与 sliding reconstruction/CAT-Q 重叠；若仅换损失则创新不足 | **首选** |
| C. soft-to-hard 离散优化 | 中到高 | 中 | 中到高 | 退化成轻量 QAT；soft/hard endpoint gap | 备选 solver/备选方向 |
| D. calibration-data alignment | 中 | 中 | 低到中 | 混淆数据收益与算法收益；改变为 task-adaptive PTQ | 辅助轴，不作主线 |

首选 B 的理由不是“跨层更复杂”，而是它直接回应最强反例：R045 表明 isolated layer 指标看不见伤害，R046 表明真实 quantized context 能看见分布分裂。下一方法应让这种函数信号**参与 `T` 的形成**，而不是只在候选形成后做 gate。

## 6. 首选转向：跨层函数保持的三值 support 联合优化

工作名：Windowed Function-Preserving Ternarization（WFPT）。

对冻结的两层窗口 `l:l+1`，以 PT²/ATQ 初始化 `T^0_j, alpha_j, mu_j`，只更新窗口内 support 与 scale/shift。模型其余层保持量化基线，输入来自真实 quantized prefix。

建议目标为：

\[
\min_{T,\alpha,\mu}\;
\underbrace{\max_{d\in\{W2,C4\}}\left[
\mathrm{KL}(p_{fp}^{(d)}\Vert p_q^{(d)})+
\eta\,\mathrm{CVaR}_{\tau}(\ell_{token}^{(d)})
\right]}_{\text{跨层/模型函数保持}}
+\lambda\sum_{j=l}^{l+1}
\underbrace{\frac{\|H_{j+1}^q-H_{j+1}^{fp}\|_F^2}{\|H_{j+1}^{fp}\|_F^2}}_{\text{单层 trust region}}
+\beta\sum_j\underbrace{d_H(T_j,T_j^0)}_{\text{稀疏 support 变化}}.
\]

其中 `T_j in {-1,0,+1}`，`W_j^q=alpha_j T_j+mu_j`。核心是第一项必须在真实 quantized context 下度量模型函数；第二项保留用户提出的单层约束，防止跨层补偿破坏某一层；第三项控制小校准集上的离散自由度。

第一版不应再用“局部生成完整 hard candidate + 后验 gate”。应采用预算受限的 block-coordinate 更新：用局部 Hessian 只做 proposal ranking，每一小批 flip 的接受由上述联合目标直接决定，再 refit `alpha,mu`。这样 Hessian 是搜索加速器，不是最终优化目标。

与已有 sliding-layer reconstruction 的差异必须落到两个点，否则不够新：

- 联合目标直接更新三值离散 support `T`，而不是只重构连续 scale/weight surrogate；
- 单层 trust region 与跨分布 worst-domain function loss共同约束 support trajectory，而不是单纯扩大 reconstruction window。

如果实现后只是“CAT-Q 的窗口损失加到 PT²”，应立即视为创新不足并止损。

## 7. 备选转向：soft-to-hard continuation

若首选方法在 matched objective 下表现为“hard coordinate search 找不到下降方向”，而不是“找到方向但 test 不泛化”，再切换到 soft-to-hard solver：

\[
\pi_{ijk}=\mathrm{softmax}(z_{ijk}/\tau),\qquad
\tilde T_{ij}=\sum_{k\in\{-1,0,1\}}k\pi_{ijk},
\]

逐步降低温度 `tau`，加入 initializer proximity/KL 与 entropy schedule，最后 hard projection。只有 hard endpoint 在 untouched 数据保留 soft gain 才算成功。

它不能被用来挽救一般化失败：若 soft 与 hard 都在 validation 好、test 坏，问题是数据/目标，不是 solver。若所需反向步数接近微调成本，也应停止并承认方法越过 PTQ 边界。

## 8. 校准数据对齐的定位

当前只使用冻结的 W2/C4 混合与 worst-domain 聚合，避免把算法收益和数据收益混在一起。model-generated reasoning traces 暂不进入核心实验。

只有当通用 PTQ 路线明确失败、且论文愿意重定位为 task-adaptive ternary PTQ 时，才单独比较 generic calibration、model-generated text、task traces，并严格使用相同 token budget。这个方向不是补丁，而是另一套研究问题。

## 9. 下一批最小验证实验（仅方案，不执行）

### E1：目标是否可识别

- 固定 window (10,11)、同一初始化、同一 flip/evaluation budget、冻结 fit/validation/test IDs。
- 对比：official；R042c local hard-T；R047--R058 式 post-hoc gate；WFPT joint objective。
- 主指标：untouched W2/C4 mean NLL、CVaR token loss、teacher-logit KL；辅助指标：两层 local NMSE、flip fraction、时间。
- 通过条件：WFPT 在 validation 被选中且 untouched test 的 W2/C4 mean/CVaR 均不退化；worst-domain function metric 优于 local hard-T 与 post-hoc gate。不得用 test 选 lambda/beta。

### E2：单层约束是否必要

- 只比较 `L_func` 与 `L_func + lambda L_local`；lambda 用量纲归一后的固定 1:1 初值，不做 sweep。
- 通过条件：加入 local term 后，最差单层 drift 明显下降，同时保留至少 50% 的 function gain；否则“保留单层约束”没有证据，需删除而非调参。

### E3：一次冻结复制

- 仅 E1/E2 通过后运行；冻结方法与所有权重，在 early window (0,1) 与新 calibration seed 复制。
- 通过条件：两个域、mean/CVaR、finite 均无退化；方向与 (10,11) 一致。失败则不进入全层/多模型实验。

## 10. 明确止损条件

- E1 在 matched budget 下不能优于 local hard-T 与 post-hoc gate：停止 WFPT，不再扩大窗口或加 loss。
- E2 中 local term 没有稳定性收益或完全压死更新：删除该项；若删除后仍不能泛化，停止“跨层 + 单层约束”主张。
- E3 在新 seed/early window 再次发生 gate/test 或 W2/C4 符号翻转：判定跨层目标仍不可识别，停止该主线。
- 两层窗口量化成本超过 official 对应窗口的 3 倍，且没有清晰 worst-domain function gain：不符合 PTQ 定位。
- soft-to-hard 最终 hard endpoint 丢失超过 20% soft gain，或需要接近 QAT 的数据/步数：停止 soft solver。
- calibration alignment 只有 task traces 有效、generic data 无效：必须改称 task-adaptive PTQ，不能作为通用 PTQ 结果。

## 11. 对“是否钻牛角尖”的最终判断

前半段实验没有白做：它系统排除了坐标配对、固定旋转堆叠、局部 proxy、isolated-layer scoring、单一 cancellation 解释和严格 checkpoint gate。真正钻牛角尖的起点是 R047 之后逐渐把研究重心放在“怎样从 hard-T 候选中选出赢家”，而不是“怎样让三值 support 的形成过程直接服从模型函数”。R058 给出了足够的止损证据。

因此应保留发现、停止当前算法、改变优化层级。下一阶段的论文主张不应是“更聪明的 gate”，而应是：

> 三值 PTQ 的离散 support 需要在短跨层量化轨迹中被直接优化，单层 reconstruction 只作为约束而非主目标。

这是一条值得验证的新主线，但尚未被今天的实验验证。

## 12. 审稿与 skill 可用性说明

- `analyze-results` 已完成原始数据表、paired delta、完整性与结果解释。
- `result-to-claim`、`research-review`、`research-refine` 均按规定调用；三个 Codex reviewer 请求都在 300 秒超时，未返回 thread/verdict。依照 skills 的 fail-closed 规则，正式外部结论记为 `REVIEW_UNAVAILABLE`，本报告中的方向排序是基于已核验实验的 executor synthesis，不冒充独立审稿共识。
- Traces：`.aris/traces/result-to-claim/2026-08-24_run01/`、`.aris/traces/research-review/2026-08-24_run01/`、`.aris/traces/research-refine/2026-08-24_run01/`。

## 主要证据路径

- `results/remote-runs/r058_checkpoint_veto_seed1_20260824/`
- `refine-logs/EXPERIMENT_ANALYSIS_R058_20260824.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `autoresearch/ptq_results.tsv`
- `.aris/evidence_precheck_after_r058.json`
