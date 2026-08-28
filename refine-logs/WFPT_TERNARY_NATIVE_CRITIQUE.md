# WFPT 的三值原生性、成本与创新性审查

## 结论

当前版本的“跨层函数保持 + 单层 trust region”（WFPT）不宜直接作为论文核心方法继续实验，理由如下：

1. 当前目标函数和优化框架主要是通用低比特方法，三值性只体现在可行域 `{-1,0,+1}`，没有进入方法的核心机制。
2. 若在离散更新内循环中反复调用完整 FP teacher、完整后缀或全模型评估，成本会逼近重型 reconstruction-based PTQ，甚至在工程形态上接近 calibration training。
3. 跨层输出重构、teacher distillation、Hessian/Fisher 排序以及局部约束均有强近邻工作。它们的组合不足以形成稳固创新点。

因此，建议保留跨层函数保持与单层 trust region 作为安全约束，把核心创新改为：**固定三值活跃预算下的跨层零支撑重分配，以及支撑与极性分离的离散优化**。

## 1. 当前方法是否真正利用了三值特性

### 1.1 当前版本的问题

当前形式大致为：

$$
\min_{\{T_l,\alpha_l,\mu_l\}}
\mathcal{L}_{\mathrm{func}}
\quad
\text{s.t.}\quad
\mathcal{L}_{l,\mathrm{local}} \leq \tau_l,
\qquad
T_l\in\{-1,0,+1\}^{n_l}.
$$

这里的函数损失、跨层重构、trust region、CVaR、minimax 和 Hamming budget 都可直接替换为二值、普通 2-bit 或 4-bit 码本。因此，三值只作为一个约束集合出现，而没有决定优化变量、更新算子和理论解释。

更明显的问题是普通 Hamming 距离：

$$
d_H(T,T')=\sum_i \mathbf{1}[T_i\neq T_i'].
$$

它把以下两种更新都计为一次：

$$
0\rightarrow +1,
\qquad
+1\rightarrow -1.
$$

但两者完全不同。前者改变零/非零支撑；后者在不改变支撑的情况下反转极性，造成的权重跳变量通常约为前者的两倍。普通 Hamming trust region 因而抹掉了三值码本最重要的几何结构。

### 1.2 三值相对 1-bit 和普通 2-bit 的真正特性

三值码可以分解为：

$$
T_l=S_l\odot M_l,
\qquad
S_l\in\{-1,+1\}^{n_l},
\qquad
M_l\in\{0,1\}^{n_l}.
$$

其中：

- `M` 决定零态与非零态，即支撑；
- `S` 决定非零权重的正负极性；
- 零态既是第三个量化状态，又对应跳过乘加的潜在结构稀疏性；
- 同一 group 内非零权重共享缩放参数，改变一个位置的支撑会改变整个 group 的最优尺度，因此不是普通剪枝中的独立 mask。

二值 `{-1,+1}` 没有可优化零支撑；普通四状态 2-bit 量化拥有多个幅值层级，其主要自由度不是单一的“支撑 + 极性”。因此，围绕 `M` 与 `S` 的非对称优化才是对三状态码本的原生利用。该机制也可以推广到含零码本，但这不应被夸大为数学上排他；准确表述应是“为三状态码本原生设计”。

### 1.3 建议的三值原生核心

对于窗口 `W={l,...,l+k-1}`，采用：

$$
\widehat W_j=\mu_j+\alpha_j(S_j\odot M_j),
\qquad j\in\mathcal W.
$$

核心变量首先仅为零支撑 `M`，初期固定 `S`。给定固定活跃预算：

$$
\sum_{j\in\mathcal W}\|M_j\|_0=B_{\mathcal W}.
$$

每一步只允许成对交换：从一个当前非零位置执行 `1 -> 0`，同时在另一个零位置执行 `0 -> 1`。这样不改变窗口的非零总量、理论编码预算和潜在稀疏度。

优化问题为：

$$
\begin{aligned}
\min_{\{M_j,\alpha_j,\mu_j\}_{j\in\mathcal W}}
&\quad
\left\|
F^{\mathrm{fp}}_{\mathcal W}(H_l^q)
-F^q_{\mathcal W}(H_l^q;widehat W_{\mathcal W})
\right\|_2^2 \\
\text{s.t.}
&\quad
\sum_{j\in\mathcal W}\|M_j\|_0=B_{\mathcal W}, \\
&\quad
\mathcal L_{j,\mathrm{local}}(M_j)
\leq
\mathcal L_{j,\mathrm{local}}(M_j^{(0)})+\tau_j,
\quad j\in\mathcal W.
\end{aligned}
$$

极性翻转应使用独立预算：

$$
\sum_{j\in\mathcal W}
\|M_j\odot(S_j-S_j^{(0)})\|_0
\leq B_{\mathrm{sign}},
\qquad
B_{\mathrm{sign}}\ll B_{\mathrm{support}}.
$$

更合适的三值转移距离为：

$$
d_{\mathrm{tri}}(T,T')=
c_m\sum_i\mathbf{1}
\left[(T_i=0)\oplus(T_i'=0)\right]
+c_s\sum_i\mathbf{1}[T_iT_i'=-1],
$$

其中第二项单独惩罚正负反转。这比普通 Hamming 距离更符合三值状态的实际函数跳变。

### 1.4 为什么这不只是剪枝

若只学习 `M` 并固定所有其他量，确实很容易被评价为结构化或非结构化剪枝。必须同时满足以下条件才能形成三值量化方法：

1. `M` 的 birth/death 与共享 `alpha, mu` 联合求解；
2. 支撑交换直接作用于三值编码，而不是在 FP 权重上先剪枝再量化；
3. 固定三值活跃预算，明确计算/存储含义；
4. 用消融证明收益主要来自支撑重分配，而不是通用跨层重构或更多优化步数。

## 2. FP teacher 是否使成本接近 ATQ

### 2.1 需要区分 teacher 的两种使用方式

低成本方式是：对校准集执行一次 FP 前向，缓存窗口边界 hidden states 或窗口输出，随后只优化短窗口。它仍可被合理称为 PTQ。

高成本方式是：每次候选离散更新都重新执行完整 teacher、完整量化后缀或全模型 KL/NLL 计算。若还进行 JVP、Fisher 或多轮反向传播，方法在实际形态上已经接近 calibration training。是否更新原始 FP 权重不是唯一标准；校准 token 数、前后向次数、GPU-hours 和优化器状态同样重要。

### 2.2 当前 WFPT 的成本风险

设窗口数为 `N_w`，每个窗口有 `R` 轮，每轮精确验证 `K` 个候选。若每个候选都运行后缀模型，则主要成本近似为：

$$
C_{\mathrm{WFPT}}
\approx
C_{\mathrm{teacher}}
+N_wR
\left(
C_{\mathrm{rank}}
+K C_{\mathrm{suffix}}
\right).
$$

该成本随窗口、轮次和候选数乘法增长。若保存完整 vocabulary teacher logits，存储量约为：

$$
M_{\mathrm{logits}}
=N_{\mathrm{token}}|V|b.
$$

例如 8 条、每条 2048 token、词表 32k、FP16 logits 已约为 1 GiB；512 条时约为 64 GiB。完整 logits 不适合作为轻量内循环缓存。

### 2.3 建议的成本边界

主方法应删除内循环中的完整 teacher KL。建议约束为：

- FP teacher 对每个校准样本最多一次前向；
- 最多一次全局 backward，用于可选的敏感度近似；
- 内循环只使用缓存的窗口输入/输出，不运行完整 teacher；
- 不存完整词表 logits，可存窗口 hidden targets、top-k logits 或低秩输出统计；
- 总 GPU 时间以 PT² 为基准不超过 2 倍，探索阶段硬上限为 3 倍；
- 必须同时报告 calibration tokens、forward-equivalent passes、backward passes、峰值显存和 wall-clock。

由此，teacher 不必完全删除，但应从“反复参与优化的模型”降为“一次性目标生成器”。完整 KL/NLL 应只用于 untouched validation 和最终诊断。

## 3. 当前创新点是否充分

### 3.1 现有成分的重叠

- 激活和输出重构并非新颖，GPTQ/OBQ 类方法已经从权重范数转向数据加权误差。
- CAT-Q 已进行 soft-to-hard ternarization 和 sliding-layer output reconstruction。
- TernaryLLM 已使用 feature/logit distillation，但属于训练较重的路线。
- AQLM 等工作已经存在 transformer block 或跨层联合码本优化。
- KronQ 等方法已经利用输入与输出/梯度侧敏感度，而非只看单层 `W`。
- teacher KL、CVaR、domain minimax 和 trust region 的组合属于稳健优化包装，不能自然构成三值 PTQ 的核心贡献。

因此，“我们不只看单层权重，而是看函数”不是充分创新表述；该叙述容易被已有工作直接覆盖。

### 3.2 可以成立的创新结构

论文的核心贡献应收敛为一个机制：

> 在固定三值活跃预算下，把离散码分解为零支撑与非零极性，并在相邻层之间执行由函数损失驱动的成对支撑交换；共享尺度在每次交换后重拟合，而单层 trust region 仅限制局部损伤。

这里：

- “支撑/极性分离”回答三值原生性；
- “固定预算的跨层交换”回答为什么需要联合层优化；
- “共享尺度重拟合”把它与普通剪枝区分；
- “单层 trust region”只是稳定器；
- teacher、CVaR 和 minimax 不作为创新点，首版方法中可以全部去掉。

即使如此，创新仍是“有希望但尚未成立”，必须检查与三值剪枝、learned threshold、zero-point/support optimization 和 blockwise code reassignment 的最近邻工作。

## 4. 三个最小判别实验

### E1：三值机制消融

在相同候选数和相同计算预算下比较：

1. 普通 Hamming hard-T 更新；
2. 仅零支撑 birth/death；
3. 仅 polarity flips；
4. 支撑/极性分离并采用独立预算。

若收益无法定位到零支撑更新，三值原生主张不成立。

### E2：跨层活跃预算交换

比较：

1. 每层固定活跃数；
2. 窗口固定总活跃数，允许跨层重分配；
3. 不固定活跃数。

核心判据是方案 2 是否在 untouched Wikitext-2 与 C4 上同时改善，并且不依赖事后 gate。若仅校准集改善或继续出现 W2/C4 分裂，则停止该分支。

### E3：匹配计算量与方法特异性

在完全相同的 forward-equivalent budget 下比较：

1. PT² 基线；
2. CAT-Q 风格窗口重构；
3. 窗口重构 + local trust region；
4. 窗口重构 + ternary support exchange + local trust region。

只有第 4 项稳定超过第 2、3 项，才能说明贡献来自三值机制，而不是更多计算或通用跨层重构。

## 5. 止损条件

以下任一条件成立，应停止该方向：

1. 支撑/极性分离不优于等预算普通 hard-T 更新；
2. 跨层活跃预算交换在 untouched W2/C4 上仍产生稳定的方向分裂；
3. 改善主要由更大 calibration set 或更多优化轮次解释；
4. wall-clock 超过 PT² 三倍，且不能显著优于 CAT-Q 风格匹配成本基线；
5. 新颖性检索发现已有工作已联合完成“固定三值活跃预算 + 跨层支撑交换 + 共享尺度重拟合”。

## 最终判断

用户提出的三个质疑均成立。当前 WFPT 的正确处理不是小幅补充三值术语，而是改变核心优化变量和更新算子。跨层函数保持与单层 trust region 可以保留，但只能作为支撑框架。真正值得验证的核心是三值零态的跨层重分配；若该核心无法在匹配计算量的实验中独立产生收益，则本方向不应继续包装为新方法。
