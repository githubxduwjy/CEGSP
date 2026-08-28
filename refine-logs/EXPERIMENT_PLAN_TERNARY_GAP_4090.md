# RTX 4090 实验方案：三值 QAT–PTQ Gap 的证明、归因与弥合

**问题锚点**：严格 weight-only 三值 PTQ 为什么通常落后于三值 QAT？这个差距是否来自三值特有的零态、符号和路径抵消结构？

**核心原则**：先证明 gap 存在，再证明它是三值相关的，随后只针对被实验证实的 gap 来源设计 PTQ 弥合器。没有通过 gap gate，不实现 full-model 新方法。

**硬件**：单张 RTX 4090 24GB。

**首轮模型**：OPT-125M 做 harness；OPT-350M 做主实验；TinyLlama-1.1B 只在主 gate 全部通过后条件复制。

**状态**：仅完成计划，不启动实验。

## 1. 需要证明的主张

本方案只保留两个主张，避免把“QAT 机制发现”和“PTQ 新优化器”混成一个未经验证的故事。

| 主张 | 最低可信证据 | 对应阶段 |
|---|---|---|
| C1：在相同三值码本、零率、scale 粒度、校准 token 和可比计算预算下，三值 QAT 在 held-out 函数指标上优于 direct PTQ，且 gap 不是只由边际零率或权重显著性造成 | QAT 相对 PTQ 的 composed-operator distortion 在至少 3/4 个代表层降低，bootstrap 95% CI 不跨 0；C4 和 Wikitext-2 方向不冲突 | A |
| C2：gap 的主要来源可以被分解为三值的 support (M)、sign (S)、shared scale α 及其耦合；一个不依赖完整 QAT 的 ternary-aware PTQ 投影器能稳定关闭 gap 的一部分 | 先通过 M/S 反事实归因，再由无 teacher 的 bridge 在至少 3/4 层关闭预注册比例的 gap，并保持端到端 NLL 不退化 | B、C |

### 必须排除的反主张

- gap 只是 QAT 使用了更多优化步数或更大校准集；
- gap 只是 QAT 改善了零率、正负比例或 scale，而不是三值结构本身；
- 所谓三值规律换成 binary 或 4-level 仍然成立；
- bridge 只是把 calibration set 过拟合，或只是增加了计算量；
- 需要一个完整 QAT teacher 才能获得结果，这样方法实际上已接近 ATQ。

## 2. 三值表示与 gap 的可检验定义

每个 group 的部署权重固定写成：

```text
W_hat_g = alpha_g * T_g
T_g = M_g ⊙ S_g
M_g ∈ {0, 1}                 # zero support
S_g ∈ {-1, +1}               # active sign
T_g ∈ {-1, 0, +1}
```

这里的重点不是把三值形式重新命名，而是把三值 PTQ 的自由度拆成三个具有不同函数作用的部分：

1. `M` 决定连接是否存在；
2. `S` 决定活跃连接之间的正负抵消；
3. `alpha` 让所有 active weights 共享幅值，无法像 4-bit 那样用多个幅值补偿错误。

对 Q/K，定义每个输入维度对 attention 双线性算子的有效路径数和符号抵消：

```text
A_QK(i, j) = sum_r M_Q(i, r) * M_K(j, r)

C_QK(i, j) = 1 -
  abs(sum_r M_Q(i,r) * M_K(j,r) * S_Q(i,r) * S_K(j,r))
  / (A_QK(i,j) + eps)
```

对 V/O 使用同样定义。`A` 只由 zero support 决定，`C` 同时依赖 support 和 sign。二值权重没有 `M` 的自由度；4-level 权重还可以使用额外幅值补偿，因此这两个量是三值特异性诊断的核心。

函数指标不使用单层 weight NMSE 作为唯一证据。Q/K 使用：

```text
D_QK = E || X W_Q W_K^T X^T
          - X W_Q_hat W_K_hat^T X^T ||_F^2
       / (E || X W_Q W_K^T X^T ||_F^2 + eps)
```

V/O 使用 `D_VO`，同时报告每个单层输出误差和端到端 held-out NLL。所有公式在本计划中同时给出纯文本版本，便于 VS Code 无数学插件时阅读。

定义固定预算下的 QAT–PTQ gap：

```text
Gap_b = D_PTQ,b - D_QAT,b
NormGap_b = Gap_b / (D_PTQ,b + eps)
```

其中 `b` 表示 binary、ternary 或 4-level。gap 只有在 QAT 和 PTQ 使用相同 group size、同一部署码本、相同零率控制、相同校准 token 数，并在未参与选择的验证集上测量时才有意义。

## 3. 数据、模型和公平性协议

### 3.1 模型与层

- sanity：OPT-125M；只做 16 step QAT、导出和显存检查；
- primary：OPT-350M；固定分析层 0、7、15、23；
- 第一阶段只分析 attention 的 Q/K 与 V/O，不引入 MLP，避免把 gap 归因扩散到太多结构；
- conditional：TinyLlama-1.1B，只在 C1、C2 全部通过后复制。

### 3.2 数据 split

- `fit-A`：C4 train 固定片段，共 1,048,576 tokens；PTQ 和 QAT 使用相同 token 预算；
- `val-B`：不重叠的 C4 片段，32 条 × 512 tokens；只用于早停或 bridge 的内部选择；
- `untouched-C`：C4 validation 32 条 × 512 tokens；
- `untouched-W`：Wikitext-2 32 个 512-token windows；
- `untouched-C/W` 不参与 threshold、zero-rate matching、checkpoint 选择或 bridge 超参数选择。

QAT 固定使用 step-256 checkpoint。step-64 仅用于判断优化轨迹是否仍在改善，不允许根据 untouched 指标挑 checkpoint。若需要 full-QAT 参考，step-512 作为预注册的条件运行，而不是看到结果后追加。

### 3.3 相同设置

| 项目 | 固定值 |
|---|---|
| weight forward | group-wise `{-alpha, 0, +alpha}` |
| latent weight / activation | BF16 / BF16 |
| optimizer | AdamW；不使用 bitsandbytes |
| sequence / micro-batch | 512 / 1 |
| gradient accumulation | 8，effective 4096 tokens/step |
| QAT steps | 256；总计 1,048,576 fit tokens |
| gradient checkpointing | 开启 |
| KV cache | 关闭 |
| seed | 0；主 gate 通过后才增加 seed 1 |
| storage report | theoretical ternary entropy 与实际 2-bit packing 分开报告 |

“1.58-bit”必须明确区分理论三值熵和实际部署存储。若实现使用 2-bit packing，就只能声称实际 2-bit weight code，不能把物理存储直接写成 1.58 bpw。

### 3.4 显存策略

1. FP、PTQ、QAT 分开串行加载，GPU 同时只驻留一个模型；
2. QAT 结束后只保存 ternary code、group scale 和统计摘要，不保存完整 optimizer state 供后续分析；
3. mask 保存为 CPU `uint8` 或 bit-packed 文件；
4. hook 只在线累计 Q/K、V/O 的协方差、路径计数和 operator summary，不缓存全模型 token activation；
5. 每 20 step 记录 `max_memory_allocated`、step time、finite 状态；
6. 峰值显存硬门槛为 21.5 GiB。

若 OOM，只允许预注册的 `sequence 512 -> 256`、`gradient accumulation 8 -> 16`，保持每步 token 数不变。若仍 OOM，只判为 harness/environment failure，不删除指标、不换模型、不改变研究问题。

## 4. 阶段 A：先证明 QAT–PTQ gap 存在

### Block A0：4090 harness

- **比较**：OPT-125M FP、direct ternary PTQ、16-step ternary QAT；
- **检查**：三值不变量、导出重载 parity、finite loss、峰值显存；
- **gate**：peak <18 GiB；所有输出 finite；导出权重仅有 `{-alpha,0,+alpha}`；
- **失败含义**：只说明环境或实现错误，不评价 gap。

### Block A1：同预算 gap reproduction

在 OPT-350M 上从同一 FP checkpoint 生成：

1. `FP`：函数参考；
2. `PTQ-direct`：使用当前最强可复现三值 PTQ 初始化器；
3. `PTQ-zero-matched`：在不改变优化器的前提下，将零率与 QAT 对齐；
4. `QAT-64`：训练轨迹诊断；
5. `QAT-256`：固定预算 QAT reference。

PTQ 使用 `fit-A` 做校准，QAT 使用相同 `fit-A` 做更新；两者的 token 数相同。PTQ 的 wall-clock、forward 次数和显存单独报告，不能把 QAT 的反向计算隐藏在“相同 calibration data”之后。

**主要指标**：`D_QK`、`D_VO`、单层 output NMSE、C4/W2 held-out NLL、weight NMSE、zero rate、sign ratio、scale 分布、wall-clock。

**C1 gap gate**：

- `QAT-256` 在至少 3/4 个代表层上相对 `PTQ-zero-matched` 的 `D_QK` 或 `D_VO` 改善至少 5%；
- layer-wise bootstrap 95% CI 不跨 0；
- C4 与 Wikitext-2 不出现方向相反的系统性翻转；
- QAT zero rate 与 PTQ-zero-matched 相差不超过 0.2 percentage point；
- QAT-64 到 QAT-256 的 operator distortion 仍有下降，排除“QAT 只是随机初始化波动”。

若 C1 不通过，停止“弥合 gap”的方法开发，只报告：在当前模型、码本和固定预算下没有证据表明存在可弥合 gap。若 C1 只在训练集成立、在 val-B 消失，判为 calibration overfit，不进入阶段 B。

### Block A2：binary / 4-level specificity control

只有 A1 gap gate 通过后运行，先选两个代表层以控制 4090 成本：

- binary：`{-alpha, +alpha}`，没有 zero state；
- ternary：`{-alpha, 0, +alpha}`；
- 4-level：固定 2-bit code，例如预注册对称四级码本；
- 每种 bit 使用自己的 direct PTQ 与 budget-matched QAT。

该对照不声称三种设置物理 bpw 相同，而是分别报告理论 alphabet bits、实际 packing、group metadata 和函数 distortion。它只回答：

```text
NormGap_ternary 是否明显大于 NormGap_binary / NormGap_4level？
```

若所有 bit 的 gap 近似相同，研究应改称一般 low-bit PTQ–QAT gap，不再主张三值特异性。若 ternary 的 `NormGap` 在两个层和两个 operator 上显著更大，才进入三值归因。

## 5. 阶段 B：用三值反事实定位 gap 来源

这一阶段使用 A1 的 QAT checkpoint 作为**诊断 oracle**，不是部署方法。目的是回答“QAT 到底改了什么”，而不是偷用 QAT 结果伪装成 PTQ。

### Block B1：M/S/alpha 单因素反事实

在相同 group 上构造以下变体：

| 变体 | support M | sign S | scale alpha | 解释 |
|---|---|---|---|---|
| P0 | PTQ | PTQ | PTQ | direct PTQ 起点 |
| P1 | QAT | PTQ | PTQ | 只替换 zero support |
| P2 | PTQ | QAT | PTQ | 只替换 active sign |
| P3 | PTQ | PTQ | QAT-refit | 只替换共享幅值 |
| P4 | QAT | QAT | QAT | QAT ternary pattern oracle |
| P5 | matched shuffle | matched shuffle | matched | 破坏联合关系的 null |

P1–P4 只用于 gap attribution，不能称为 PTQ 方法。每个变体都匹配 layer/group zero rate、sign ratio、row/column/head counts 和 salience decile。

定义每个组件的可解释 gap closure：

```text
Closure_M = (D_P0 - D_P1) / (D_P0 - D_P4 + eps)
Closure_S = (D_P0 - D_P2) / (D_P0 - D_P4 + eps)
Closure_alpha = (D_P0 - D_P3) / (D_P0 - D_P4 + eps)
Closure_joint = (D_P0 - D_P4) / (D_P0 - D_P4 + eps)
```

不要求这些 closure 之和等于 1，因为 M、S、alpha 存在交互。关键是比较单因素和联合因素。

### Block B2：路径级因果 null

对 Q/K 和 V/O 分别进行：

1. `support-only shuffle`：保持每个矩阵的零率、row/column/head/group 统计；
2. `sign-only shuffle`：只在 active support 内打乱符号；
3. `joint M-S shuffle`：保持 M 与 S 的边际统计，但打乱它们之间的配对；
4. `salience-preserving shuffle`：在相同 activation/Hessian-salience decile 内完成上述 shuffle；
5. `QK/VO path shuffle`：分别破坏两矩阵之间的共同有效路径。

主要指标：`A_QK`、`C_QK`、`A_VO`、`C_VO`、`D_QK`、`D_VO`。pilot 200 次 permutation；只有在预注册信号通过后才增加到 1000 次。

**B gate**：至少一个三值组件的 closure 在 3/4 层方向一致，并且 salience-preserving null 后仍有显著 operator distortion 差异；如果只看到 weight NMSE 差异，不通过。

若 P4 只优于 P0，但 P1–P3 和路径 null 都不能解释，结论是“QAT 需要联合适配，但当前没有可迁移的低维三值规律”，不得强行设计 bridge。

## 6. 阶段 C：只测试一个无完整 teacher 的 PTQ 弥合器

阶段 C 的目标不是复制 QAT，而是在 fixed FP checkpoint + calibration data 上关闭一部分已被阶段 B 证实的 gap。优先选择无 teacher 版本，避免 PTQ 成本接近 ATQ。

### C0：强基线与计算预算

所有 bridge 与 `PTQ-direct` 使用相同 fit-A、相同 group size、相同候选评估次数和相同最终 scale refit 次数。额外报告：

- forward 次数；
- 是否需要 backward；
- wall-clock；
- 峰值显存；
- CPU/GPU 数据搬运；
- 实际 ternary code 和 metadata 大小。

如果 bridge 超过 `PTQ-direct` 3 倍 wall-clock 或需要训练 latent FP weights，自动降级为“QAT-assisted analysis”，不再称为纯 PTQ。

### C1：Ternary Path Projection PTQ（无 teacher）

只在 B gate 说明 Q/K、V/O 路径结构确实是主要来源时运行。其变量仍然严格是部署三值变量：`M`、`S` 和 group `alpha`。

候选目标使用固定的 operator-level calibration loss：

```text
L_path = mean(D_QK_calib, D_VO_calib)
L_local = sum_j layer_output_NMSE_j
```

求解流程固定为三步，不能看到结果后增加搜索空间：

1. 从 `PTQ-direct` 的 `M0, S0, alpha0` 开始；
2. 只允许一个预注册的 local sign flip 或 zero↔nonzero swap proposal；
3. 只接受降低 `L_path` 且满足 `L_local_j <= 1.05 * L_local_j(P0)` 的候选；
4. 固定最多 2 个 passes，冻结 `M/S` 后重拟合 `alpha`；
5. `val-B` 只用于 early stop 和 rollback，untouched-C/W 不参与选择。

这里的 local 约束是安全约束，不是论文创新本身。论文创新只能来自阶段 B 已经证实的 ternary path statistic，以及该 statistic 是否让同样数量的候选搜索更有效。

### C2：teacher-assisted state prior（仅作成本对照）

若 C1 无 teacher 失败，但 B 阶段显示 QAT 的 state margin 明确解释 gap，可运行一个极短的 32-step QAT，仅记录每个 group 的 `-1/0/+1` 状态频率和 state-transition margin，然后停止 QAT，再做离散 PTQ 投影。

该分支必须明确标为 `QAT-assisted PTQ`，不得作为 strict PTQ 主方法。它只回答：gap 是由“信息不可得”造成，还是由“优化器不会利用信息”造成。

不得保存或蒸馏完整 teacher logits，也不得运行完整 256-step QAT 后再称为 PTQ。

### C3：gap closure gate

对任意 bridge 定义：

```text
Closure_bridge =
  (D_PTQ-direct - D_bridge)
  / (D_PTQ-direct - D_QAT-256 + eps)
```

预注册判定：

- `diagnostic success`：至少 3/4 层 `Closure_bridge >= 0.30`，且 bootstrap 95% CI 为正；
- `paper-worthy success`：至少 3/4 层 `Closure_bridge >= 0.50`，C4/W2 端到端 NLL 均不恶化，且 wall-clock 不超过 direct PTQ 的 3 倍；
- 若 bridge 只改善 operator metric 而 NLL 退化，判为局部函数拟合，不进入全模型；
- 若 bridge 只在 calibration fit 上有效，判为过拟合；
- 若无 teacher bridge 失败而 32-step teacher-assisted 成功，只能形成“短 teacher 成本换取部分 gap closure”的次级结果。

## 7. 4090 执行顺序与运行编号

| Run ID | 目的 | 模型 / 数据 | 预计时间 | 状态 |
|---|---|---|---:|---|
| G4090-GAP-00 | CUDA、BF16、三值导出和显存 preflight | OPT-125M | 5–10 min | MUST |
| G4090-GAP-01 | direct PTQ 与 16-step QAT harness | OPT-125M | 20–40 min | MUST |
| G4090-GAP-02 | 固定预算 gap reproduction | OPT-350M；PTQ/QAT-64/256 | 1.5–3 h | MUST |
| G4090-GAP-03 | binary/ternary/4-level specificity control | OPT-350M；2 layers | 1–2 h | CONDITIONAL：A1 通过 |
| G4090-GAP-04 | M/S/alpha oracle attribution | OPT-350M；4 layers | 30–60 min | CONDITIONAL：A1 通过 |
| G4090-GAP-05 | salience-preserving path null | OPT-350M；4 layers | 30–90 min | CONDITIONAL：B1 通过 |
| G4090-GAP-06 | 无 teacher Ternary Path Projection PTQ | OPT-350M；4 layers | 30–90 min | CONDITIONAL：B 通过 |
| G4090-GAP-07 | 32-step teacher-assisted cost control | OPT-350M；4 layers | 30–60 min | CONDITIONAL：C1 无 teacher 失败 |
| G4090-GAP-08 | second-family replication | TinyLlama-1.1B | 2–5 h | NICE：C2 paper-worthy |

首轮必跑预算约 2–4 GPU-hours；完整主线在 C1 无 teacher bridge 运行时约 3–5 GPU-hours。GAP-03、GAP-07 和 GAP-08 都是条件实验，不得事后补跑来挑选有利叙事。

## 8. 机器判定状态机

```text
A1 gap fail
  -> STOP：没有证据证明当前设置存在可弥合的 QAT-PTQ gap

A1 pass, A2 ternary-specific fail
  -> REFRAME：改为 generic low-bit gap，停止三值专属方法

A1+A2 pass, B attribution fail
  -> OBSERVATIONAL ONLY：确认 gap，但没有可迁移的三值来源

B pass, C1 bridge fail
  -> TEST C2 only if state-margin evidence exists

C1 paper-worthy pass
  -> 进入第二模型和多 seed，不自动扩展 full-model

OOM / nonfinite / split leak / export mismatch
  -> INVALID HARNESS：只修一次明确工程故障，重跑同一配置
```

## 9. 明确不做的事情

- 不在 gap 尚未证明前开发新的 full-model ternary PTQ；
- 不把 QAT checkpoint、teacher logits 或 latent FP weights 放进 strict PTQ 方法；
- 不使用 untouched-C/W 选择 threshold、层、checkpoint 或 bridge 超参数；
- 不做 projection mask 枚举；
- 不事后调 epsilon、zero-rate 或 group size；
- 不把 2-bit packing 直接写成 1.58 bpw；
- 不因为一次负结果否定所有方向，只关闭对应的 gap 来源或 bridge 分支；
- 不用 binary/4-level 的正结果替代三值证据。

## 10. 首轮决策

首轮只应执行：

1. `G4090-GAP-00`：确认 harness；
2. `G4090-GAP-01`：确认三值 QAT/PTQ 链路；
3. `G4090-GAP-02`：在 OPT-350M 上严格证明或否定固定预算 gap。

在 `G4090-GAP-02` 完成前，不实现 `Ternary Path Projection PTQ`。这样可以避免再次陷入“先提出一个看似合理的三值优化器，再用单层指标解释全模型”的循环。

## 11. 最终检查清单

- [ ] QAT 与 PTQ 使用相同三值 group、fit token 数和 zero-rate control
- [ ] QAT-64/256 轨迹与固定 checkpoint 已保存
- [ ] Q/K、V/O composed-operator 指标与单层指标分开报告
- [ ] binary 与 4-level control 未被错误表述为相同物理 bpw
- [ ] M、S、alpha 反事实和 path null 完成
- [ ] bridge 只使用 FP checkpoint + calibration（teacher-assisted 单独标记）
- [ ] gap closure 使用 untouched 数据做最终审计，不参与选择
- [ ] 计算成本、显存、数据 split、nonfinite 和所有负结果写入 tracker
