# WFPT Method Derivation：跨层函数保持与单层 Trust Region 的三值 PTQ

## Target

推导一个可实现、可证伪的三值 PTQ 方法：在短层窗口内联合更新离散三值 support，使更新直接最小化完整量化上下文中的模型函数偏差，同时保留每一层的局部重构约束，避免依赖大误差相消。

工作名为 **Windowed Function-Preserving Ternarization（WFPT）**。

本推导的目标不是证明 WFPT 一定提高 PPL，而是完成四件事：

1. 从模型函数保持这一全局对象推出跨层目标，而不是从局部 NMSE 拼接 loss；
2. 解释为何单层目标缺少跨层 Hessian 的非对角项；
3. 将“保留单层量化约束”写成真正的 trust region；
4. 给出 hard ternary support 的预算受限求解器、闭式 scale/shift refit 和严格的 fit/validation/test 边界。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

方案在以下重构后是连贯的：

- 顶层 invariant object 必须是量化模型相对 FP teacher 的条件预测分布，而不是窗口末端 hidden-state NMSE；
- 单层 reconstruction 是 trust-region constraint，不是与全局函数 loss 地位相同的主目标；
- Hessian/local activation loss 只负责生成廉价的离散候选，不能决定最终 support；
- 跨层函数 loss 必须在真实 quantized prefix 和 frozen quantized remainder 中计算。

尚未被证明或验证的是：该目标能否在小校准集上稳定泛化，以及其计算成本能否保持在 PTQ 范围内。

## Invariant Object

贯穿整个推导的唯一顶层对象是量化模型对 FP teacher 的**预测函数偏差**：

$$
\mathcal R_{\mathrm{func}}(\Theta_q)
=
\mathbb E_{x\sim\mathcal D}
\left[
\frac{1}{|x|}\sum_t
D_{\mathrm{KL}}\!\left(
p_{fp}(\cdot\mid x_{<t})\,\Vert\,
p_q(\cdot\mid x_{<t};\Theta_q)
\right)
\right].
$$

这里 `Theta_q` 是完整量化模型参数。局部重构、window hidden drift、Hessian-weighted weight error 都只是这个对象的 proxy、constraint 或搜索近似，不能替代它。

## Assumptions

1. FP 模型参数冻结，作为 teacher；除三值 support、scale 和 shift 外，不更新预训练参数。
2. 已有一个完整、有限的 PT²/ATQ baseline `Theta_q^0`，WFPT 从该点初始化。
3. 每次只优化连续的短窗口 `W_l={l,...,l+w-1}`；首轮固定 `w=2`。
4. 窗口之前使用 baseline 的 quantized prefix，窗口之后使用 baseline 的 frozen quantized suffix。
5. 校准数据预先分为 fit、validation、untouched test；test 不参与 support、step、乘子或停止点选择。
6. 使用至少两个预先声明的 calibration domains，以暴露 R046/R058 所示的分布符号翻转。
7. 一阶/二阶传播公式只在小 support update 邻域内作为 approximation；最终接受使用 exact forward objective 复核。
8. PTQ 成本受限：不进行全参数训练，不保存 optimizer state for FP weights，不执行无限迭代。

## Notation

- `L`：Transformer 层数；`l`：当前窗口起始层；`w`：窗口长度。
- `f_j(h;W_j)`：第 `j` 个 Transformer block。
- `W_j`：FP 权重；`W_j^q`：三值量化后的有效权重。
- `T_j in {-1,0,+1}^{m_j x n_j}`：离散三值 support。
- `alpha_j, mu_j`：按 row 或 group 定义的 scale 与 shift。
- `W_j^q=alpha_j odot T_j+mu_j odot 1`：广播后的量化权重。
- `T_j^0,alpha_j^0,mu_j^0`：PT²/ATQ initializer。
- `h_j^q`：候选量化模型进入第 `j` 层的 hidden state。
- `bar h_j`：在相同 quantized-prefix 输入下，由 FP window 产生的 local reference state。
- `z_q,z_{fp}`：量化模型与 FP teacher logits。
- `p_q,p_{fp}`：对应的 token conditional distributions。
- `d in {1,...,M}`：预注册 calibration domain。
- `r_{d,t}`：第 `d` 个 domain、第 `t` 个 token 的 teacher KL。
- `B`：允许改变的 ternary entries 总预算；`rho_j`：单层 trust-region 半径。
- `S=XX^T/N`：一层输入激活的经验二阶矩；采用 row-vector 记法时等价转置。

## Derivation Strategy

推导路线为：

1. 从完整模型 teacher KL 定义函数保持；
2. 对两层窗口线性化，显式得到跨层 interaction term；
3. 由 interaction term 说明逐层独立优化为何不充分；
4. 由扰动传播上界导出单层 trust region 的作用；
5. 写成 robust constrained mixed-discrete optimization；
6. 将约束问题转成增广 Lagrangian，仅用于求解；
7. 用 Hessian gain 做候选 shortlist，用 exact function objective 做 support update；
8. 固定 `T` 后闭式 refit `alpha,mu`；
9. 用 validation 早停和 untouched test 隔离一般化证据。

## Derivation Map

1. **Exact definition**：teacher KL 定义完整量化模型的函数偏差。
2. **Approximation A**：短窗口附近的一阶 hidden/logit perturbation。
3. **Identity under A**：两层 global quadratic 展开产生 cross term。
4. **Proposition 1**：非零 cross term 时，逐层局部排序不保证全局排序。
5. **Proposition 2**：局部 trust region 给出窗口末端与 logits 扰动上界。
6. **Exact optimization target**：minimax/CVaR function risk with local and Hamming constraints。
7. **Approximation B**：局部 Hessian gain 仅用于筛选 ternary coordinate candidates。
8. **Exact conditional solution**：固定 `T` 后，activation-weighted `alpha,mu` 为 2x2 normal equation。
9. **Algorithmic safeguard**：每个候选 tranche 用 exact forward objective 复核，修正 Taylor/Hessian surrogate 误差。

## Main Derivation

### Step 1：三值参数化是 mixed-discrete optimization

对第 `j` 层的一个 row/group，定义

$$
W_j^q(T_j,\alpha_j,\mu_j)
=\alpha_j\odot T_j+\mu_j\odot\mathbf 1,
\qquad T_j\in\{-1,0,+1\}^{m_j\times n_j}.
$$

因此 PTQ 的真实变量分为：

$$
\underbrace{T_j}_{\text{离散表示结构}},
\qquad
\underbrace{(\alpha_j,\mu_j)}_{\text{连续 nuisance parameters}}.
$$

固定 `T_j` 后 scale/shift 是低维连续问题；允许 `T_j` 变化后，问题成为组合优化。R042c 已表明固定 support 并非局部最优，但 R043--R058 同时表明局部 support 改善不能直接推出完整模型改善。

### Step 2：定义完整模型函数保持风险

对 domain `d` 的 token `t`，定义非负函数偏差

$$
r_{d,t}(\Theta_q)
=D_{\mathrm{KL}}\!\left(
p_{fp}(\cdot\mid x_{<t})\,\Vert\,
p_q(\cdot\mid x_{<t};\Theta_q)
\right).
$$

为同时约束平均 token 与长尾 token，定义

$$
\mathcal R_d(\Theta_q)
=(1-\eta)\frac{1}{N_d}\sum_{t=1}^{N_d}r_{d,t}
+\eta\operatorname{CVaR}_{\tau}
\left(\{r_{d,t}\}_{t=1}^{N_d}\right),
$$

其中 upper-tail CVaR 的等价定义为

$$
\operatorname{CVaR}_{\tau}(r)
=\min_{\zeta\in\mathbb R}
\left[
\zeta+\frac{1}{\tau N_d}
\sum_{t=1}^{N_d}(r_{d,t}-\zeta)_+
\right].
$$

为避免只改善 C4、伤害 W2 的平均化掩盖，顶层目标采用 domain robust risk：

$$
\mathcal R_{\mathrm{rob}}(\Theta_q)
=\max_{d\in\{1,\ldots,M\}}\mathcal R_d(\Theta_q).
$$

这是 method 的 exact conceptual objective。实际求解可用 epigraph：

$$
\min_{\Theta_q,u}u,
\qquad \text{s.t. }\mathcal R_d(\Theta_q)\le u,\;\forall d,
$$

或用显式标注的 smooth approximation：

$$
\mathcal R_{\mathrm{softmax}}
=\gamma\log\sum_d
\exp\left(\mathcal R_d/\gamma\right),
\quad
\mathcal R_{\mathrm{rob}}
\le \mathcal R_{\mathrm{softmax}}
\le \mathcal R_{\mathrm{rob}}+\gamma\log M.
$$

### Step 3：两层窗口的扰动传播

考虑 `w=2`：

$$
h_{l+1}=f_l(h_l;W_l),
\qquad
h_{l+2}=f_{l+1}(h_{l+1};W_{l+1}).
$$

令量化 support 更新造成权重扰动 `Delta W_l,Delta W_{l+1}`。在 baseline quantized trajectory 附近，一阶近似为

$$
\delta h_{l+1}
\approx B_l\,\mathrm{vec}(\Delta W_l),
$$

$$
\delta h_{l+2}
\approx
J_{l+1}\delta h_{l+1}
+B_{l+1}\,\mathrm{vec}(\Delta W_{l+1}),
$$

其中

$$
J_{l+1}=\frac{\partial f_{l+1}}{\partial h_{l+1}},
\qquad
B_j=\frac{\partial f_j}{\partial\mathrm{vec}(W_j)}.
$$

若 frozen quantized suffix 到 logits 的映射为 `G_{l+2}`，则

$$
\delta z
\approx
P_{l+2}
\left(
J_{l+1}B_l\,\mathrm{vec}(\Delta W_l)
+B_{l+1}\,\mathrm{vec}(\Delta W_{l+1})
\right),
$$

其中 `P_{l+2}=partial G_{l+2}/partial h_{l+2}`。

该近似必须在真实 quantized-prefix state `h_l^q` 上计算；用 FP16 prefix 会改变 `J,B,P` 的取值，正是 R045 无法预测 R046 的原因之一。

### Step 4：跨层 Hessian 的非对角项

对 teacher KL 在 baseline candidate 周围做二阶展开：

$$
\Delta\mathcal R_d
\approx
g_d^T\delta z+rac12\delta z^TF_d\delta z,
$$

其中 `g_d` 是 baseline 相对 teacher 的一阶偏差，`F_d` 可取 logits Fisher/Gauss--Newton curvature。

记

$$
a=P_{l+2}J_{l+1}B_l\,\mathrm{vec}(\Delta W_l),
\qquad
b=P_{l+2}B_{l+1}\,\mathrm{vec}(\Delta W_{l+1}),
$$

则二阶项有精确展开：

$$
\frac12(a+b)^TF_d(a+b)
=\frac12a^TF_da
+\frac12b^TF_db
+\underbrace{a^TF_db}_{\text{cross-layer interaction}}.
$$

逐层独立 reconstruction 只近似前两项，遗漏 `a^T F_d b`。这个非对角项可以为正或负：

- 为正时，两个局部“好”更新可能在全局相长，伤害模型函数；
- 为负时，某个局部略差的更新可能与相邻层形成有益补偿；
- 不同 domain 的 `F_d,J,B,P` 不同，interaction 的符号可以跨 W2/C4 翻转。

#### Proposition 1：局部排序不蕴含函数排序

在上述二阶近似成立且 `a^TF_db != 0` 时，仅由 `a^TF_da` 和 `b^TF_db` 分别最小化得到的更新，不保证最小化 `(a+b)^TF_d(a+b)`。

证明是直接的：global quadratic 比 local separable quadratic 多出一个随联合选择变化的非零项 `2a^TF_db`。只要两个候选的 local 项差小于 cross term 差，global ranking 即可反转。该命题是局部线性化模型下的 proposition，不是对非线性 Transformer 的全局 theorem。

### Step 5：为什么仍然需要单层 trust region

若只最小化 window/global function loss，优化器可能利用巨大的中间层误差相消：

$$
J_{l+1}\delta h_{l+1}
\approx-\delta h_{l+2}^{(W_{l+1})},
$$

使窗口末端暂时较小，但中间状态离开 baseline 邻域，导致：

1. Taylor approximation 失效；
2. 换 domain 后 cancellation 消失；
3. 后续非线性层放大 residual drift；
4. support 过度适配小校准集。

因此，对每层定义使用相同 quantized-prefix input 的 hybrid FP reference：

$$
\bar h_l=h_l^{q,0},
\qquad
\bar h_{j+1}=f_j(\bar h_j;W_j),
\quad j\in\mathcal W_l.
$$

候选局部误差定义为

$$
\mathcal E_{j,d}(T_{l:j})
=
\frac{\|h_{j+1}^{q,d}(T)-\bar h_{j+1}^{d}\|_F^2}
{\|\bar h_{j+1}^{d}\|_F^2+\epsilon_{num}}.
$$

注意 `bar h` 不是 FP model 从 FP prefix 得到的 state；它以相同 quantized-prefix state 为起点，从而隔离当前窗口产生的附加误差。

将单层约束写成 trust region：

$$
\max_d\mathcal E_{j,d}(T_{l:j})
\le
\max_d\mathcal E_{j,d}(T_{l:j}^0)+\rho_j,
\qquad j\in\mathcal W_l.
$$

`rho_j` 是预注册的允许补偿半径，不是根据 test 结果调出的 epsilon。`rho_j=0` 是严格不退化特例，但可能过度保守；正 `rho_j` 允许有限的跨层补偿。

#### Proposition 2：trust region 控制传播半径

假设窗口第二层对输入是 `L_{l+1}`-Lipschitz，frozen suffix 对窗口输出是 `L_s`-Lipschitz，且两层附加扰动分别满足 `||delta h_{l+1}||<=e_l`、`||delta h_{l+2}^{(W)}||<=e_{l+1}`，则一阶邻域内

$$
\|\delta z\|
\le
L_s\left(L_{l+1}e_l+e_{l+1}\right)
+O(e_l^2+e_{l+1}^2).
$$

因此 local trust region 不保证任务 loss 一定改善，但它限制了 function optimizer 可利用的中间层扰动规模，并维持局部近似的可信范围。

### Step 6：最终 constrained objective

对窗口 `W_l`，WFPT 的主问题写为

$$
\begin{aligned}
\min_{\{T_j,\alpha_j,\mu_j\}_{j\in\mathcal W_l}}
&\quad \mathcal R_{\mathrm{rob}}(\Theta_q)\\
\text{s.t.}
&\quad T_j\in\{-1,0,+1\}^{m_j\times n_j},\\
&\quad \max_d\mathcal E_{j,d}(T_{l:j})
\le \max_d\mathcal E_{j,d}(T_{l:j}^0)+\rho_j,
\quad \forall j\in\mathcal W_l,\\
&\quad \sum_{j\in\mathcal W_l}d_H(T_j,T_j^0)\le B.
\end{aligned}
$$

其中 Hamming budget

$$
d_H(T_j,T_j^0)=\sum_{r,c}\mathbf 1[T_{j,rc}\ne T_{j,rc}^0]
$$

直接控制离散自由度，回应小校准集过拟合问题。

该 constrained form 是论文中应优先展示的 method object。为了求解，可构造增广 Lagrangian：

$$
\mathcal J
=\mathcal R_{\mathrm{softmax}}
+\sum_j\lambda_j
\left[
\max_d\mathcal E_{j,d}
-\max_d\mathcal E_{j,d}(T_j^0)-\rho_j
\right]_+^2
+\beta[d_H(T,T^0)-B]_+^2.
$$

这里 `lambda_j,beta` 是 constraint multipliers；它们不是 paper claim 本身。实现上优先使用 feasibility filter 加 exact objective，而不是进行大范围 lambda sweep。

### Step 7：局部 Hessian 只生成候选

对一行 FP weight `w`、当前 quantized row `q=alpha t+mu 1`、激活二阶矩 `S`，局部误差为

$$
L_{loc}(q)=(w-q)^TS(w-q).
$$

将一个 coordinate 从 `t_c` 改为 `s in {-1,0,+1}`，在暂不 refit `alpha,mu` 时

$$
\delta q=\alpha(s-t_c)e_c.
$$

则局部 loss 的 exact change 为

$$
\Delta L_{loc}
=-2\delta q^TS(w-q)+\delta q^TS\delta q.
$$

所以候选 gain 为

$$
G_{loc}(c,s)
=2\delta q^TS(w-q)-\delta q^TS\delta q.
$$

R042c 的 hard-T search 本质上按该量排序并局部接受。WFPT 只用 `G_loc` 从海量 ternary flips 中筛出 top-`K` shortlist：

$$
\mathcal C_K=\operatorname{TopK}_{(j,r,c,s)}G_{loc}(j,r,c,s).
$$

候选的最终排序必须加入 function change：

$$
\widehat{\Delta\mathcal J}(c)
=\widehat{\Delta\mathcal R}_{\mathrm{rob}}(c)
+\widehat{\Delta\mathcal P}_{trust}(c),
$$

其中 function change 可用一次 JVP/Fisher-vector product 近似。只有 exact forward 复核后仍满足 constraints 且降低 `R_rob` 的 candidate tranche 才被接受。

这与旧 gate 的本质区别是：旧方法先按局部目标形成完整 candidate，再问是否接受；WFPT 的跨层函数目标参与每一轮 support 坐标选择。

### Step 8：固定 support 后闭式 refit scale/shift

固定某一行 ternary code `t` 后，定义

$$
\min_{\alpha,\mu}
(w-\alpha t-\mu\mathbf1)^TS
(w-\alpha t-\mu\mathbf1).
$$

令

$$
a=t^TSt,\quad b=t^TS\mathbf1,\quad
d=\mathbf1^TS\mathbf1,
$$

$$
y_1=t^TSw,\qquad y_2=\mathbf1^TSw.
$$

一阶条件给出 normal equation：

$$
\begin{bmatrix}a&b\\b&d\end{bmatrix}
\begin{bmatrix}\alpha\\\mu\end{bmatrix}
=
\begin{bmatrix}y_1\\y_2\end{bmatrix}.
$$

若 `ad-b^2 != 0`，则

$$
\alpha^*=\frac{dy_1-by_2}{ad-b^2},
\qquad
\mu^*=\frac{ay_2-by_1}{ad-b^2}.
$$

这是 exact conditional minimizer，与 R042c 实现一致。当分母过小或结果 nonfinite 时回退到 initializer 的 finite solution。每次 support tranche 改变后 refit `alpha,mu`，再重新计算 exact function/trust objective。

### Step 9：预算受限的 hard block-coordinate solver

对每个窗口，算法如下。

```text
Algorithm 1: WFPT hard-support solver
Input: FP teacher, complete PT² baseline, window W_l, fit/val domains,
       trust radii {rho_j}, Hamming budget B, shortlist K, max rounds R

1. Cache baseline quantized-prefix activations h_l^{q,0}.
2. Cache FP teacher logits and hybrid FP-window local references bar h_j.
3. Initialize T <- T^0 and refit alpha,mu on fit activations.
4. Evaluate exact robust function risk and local constraint values.
5. For round = 1,...,R:
   a. Compute activation/Hessian local gains for all legal one-step flips.
   b. Keep only top-K proposals; Hessian is ranking only.
   c. Estimate each proposal's cross-layer function/trust change.
   d. Form a small non-conflicting candidate tranche with best joint decrease.
   e. Apply tranche provisionally; refit alpha,mu in closed form.
   f. Run exact quantized-context forward pass.
   g. Accept only if robust fit risk decreases, all local constraints are
      feasible, Hamming budget is respected, and all outputs are finite.
   h. Otherwise backtrack tranche size once; if still rejected, stop window.
6. Select the iterate with lowest validation robust risk among feasible iterates.
7. Freeze T; refit alpha,mu on fit+validation only.
8. Return the quantized window. Do not read untouched test.
```

限制 backtrack 一次是为了防止算法退化成未经预注册的搜索。若第一批实验需要频繁 backtrack 或大量 K/R 调参，应判定 hard solver 不适合，而不是继续枚举。

#### 计算复杂度

设每个窗口最多执行 `R` 轮，每轮保留 `K` 个 local shortlist。local gain 可以对 rows、coordinates 和三个 ternary states 向量化计算；昂贵部分不应是 `K` 次完整模型 forward，而是：

$$
C_{window}
\approx C_{prefix-cache}
+R\left(C_{local-rank}+C_{K\text{-surrogate}}+C_{exact-forward}\right).
$$

每轮先用 JVP/Fisher surrogate 排列 `K` 个候选，只对选出的一个 tranche 做一次 exact forward；若 rejected，最多进行一次缩小 tranche 的复核。显存主要由当前两层、prefix cache、teacher logits 或 teacher probabilities 构成，不维护 FP weights 的 optimizer state。首轮实现必须同时报告 `R`、accepted flips、exact forwards、wall time 和 peak VRAM；若两层窗口成本超过对应 official quantization 的预注册上限而没有明确函数收益，应止损。

### Step 10：fit、validation 与 test 的角色

- **Fit**：计算 support proposal、function objective、trust constraints 和 scale/shift refit。
- **Validation**：只选择迭代停止点与 baseline fallback；不得修改 domain、rho、B、K 或 objective form。
- **Untouched test**：只在方法完全冻结后评价 NLL、CVaR、teacher KL、PPL 和 downstream metrics。

WFPT 仍需要 validation，但它不再是旧式 post-hoc gate：validation 不负责从局部形成的多个方法中挑赢家，而只对同一 function-shaped optimization path 做早停。

#### Proposition 3：算法内单调性与可行性

若 exact acceptance step 被严格执行，且 baseline `T^0` 在 constraints 下可行，则 fit 上被接受的 iterate 序列满足

$$
\mathcal R_{rob}^{(k+1)}<\mathcal R_{rob}^{(k)},
$$

并始终满足 local/Hamming constraints。该性质只保证 fit objective 的单调性与算法可行性，不保证 validation/test 改善。

### Step 11：从两层推广到一般窗口

对 `w>2`，一阶 logit perturbation 为

$$
\delta z
\approx
\sum_{j=l}^{l+w-1}
P_{j+1}B_j\,\mathrm{vec}(\Delta W_j),
$$

二阶 function quadratic 包含

$$
\frac12\sum_j a_j^TFa_j
+\sum_{i<j}a_i^TFa_j.
$$

cross terms 数量为 `O(w^2)`，搜索与过拟合风险随窗口扩大。故 `w=2` 不是理论最优，而是最小能识别相邻层 interaction 的实验切片。只有两层结果跨 domain/seed 复制后才有理由扩大窗口。

## Remarks and Interpretation

### 1. 为什么这是“函数塑造 support”，而不是“更聪明的 gate”

核心差别在优化时序：

$$
\text{旧路线：local objective -> 完整 hard candidate -> function gate},
$$

$$
\text{WFPT：local shortlist -> function-aware coordinate choice -> exact verification}.
$$

函数目标改变了 `T` 的形成路径，而不只是候选的最终去留。

### 2. 为什么单层项是 constraint

如果把 function、local、Hamming 三项简单加权，审稿人会合理质疑 lambda/beta sweep。constrained form 先定义可接受集合，再把 multiplier 视为求解工具，科学对象更清楚：在有限局部漂移和有限 support 变化内最小化模型函数偏差。

### 3. 为什么主目标用 teacher KL

teacher KL 直接比较完整条件分布，不依赖少量 hard labels；相比只优化 ground-truth NLL，它保留 dark knowledge，并能对每个 token 定义 tail risk。最终有效性仍必须用 NLL/PPL/任务指标确认。

### 4. 为什么 local reference 使用 quantized-prefix input

若 local target 来自纯 FP trajectory，候选会被迫同时补偿 prefix 已产生的误差，混淆当前窗口的因果作用。hybrid reference 固定相同 prefix，只衡量当前 FP window 与 quantized window 的差异。

### 5. 与 sliding-layer reconstruction 的创新边界

仅扩大重构窗口不构成足够贡献。WFPT 的潜在新意必须同时包括：

1. 直接优化 hard ternary support，而非只优化 continuous surrogate/scale；
2. 完整 quantized context 的 model-function risk，而非只看 window hidden reconstruction；
3. local reconstruction 作为显式 trust-region feasible set；
4. domain-robust/tail-sensitive support selection。

若实现缺少前两点，方法应被视为现有 sliding reconstruction 的组合增强。

## Boundaries and Non-Claims

- 本文档给出 coherent method derivation，不证明 WFPT 会降低 PPL。
- Taylor/Fisher 公式是 candidate-scoring approximation；exact forward 才是接受依据。
- local trust region 只控制扰动，不保证跨 domain 泛化。
- minimax W2/C4 只证明对声明的 calibration domains 鲁棒，不等于所有任务鲁棒。
- 两层窗口是最小实验单元，不支持直接声称全模型联合优化。
- teacher KL 优化可能对校准数据过拟合，必须由 untouched data 判定。
- 未进行 bit accounting、kernel 或 latency 实验前，不能声称严格 1.58 bpw 或推理加速。
- 若优化步数、数据量或 optimizer state 接近 QAT，必须重新定位方法，不能继续称为轻量 PTQ。

## Open Risks

1. **目标成本**：每个 support tranche 的 full quantized-context exact forward 可能过慢；需要测量每个 accepted flip 的 GPU-hour 信息增益。
2. **函数目标泄漏**：若 fit/validation token 太少，teacher KL 仍会过拟合；不能通过增加 test-like calibration 来补救。
3. **constraint 半径**：`rho_j` 必须预注册。严格 0 可能重现 checkpoint gate 的过保守，过大则失去 trust-region 作用。
4. **候选 shortlist 偏差**：若 Hessian ranking 排除了 function-beneficial/local-harmful flips，WFPT 仍可能找不到联合最优；E1 应记录 shortlist recall 的小规模 exact diagnostic。
5. **alpha/mu mismatch**：闭式 refit 最小化 local quadratic，不直接最小化 teacher KL；exact acceptance 可防止恶化，但可能限制搜索效率。
6. **novelty overlap**：需要逐项核对 CAT-Q、滑动层重构、GPTQ/OBQ、TernaryLLM/soft ternary 与 PT² AGA，确认 hard support、quantized-context function loss 和 trust-region feasible set 的组合是否真正未被覆盖。
7. **分布目标冲突**：若 W2/C4 的 Pareto front 没有共同改进点，minimax 可能只返回 baseline；这将是有意义的负结果，而不是继续调权重的理由。

## Minimal Falsification Package

本方法进入实现前只预注册三项：

1. **E1 objective identifiability**：在固定 (10,11) 窗口和 matched budget 下，比较 local hard-T、post-hoc gate 与 WFPT；WFPT 必须在 untouched W2/C4 mean/CVaR 均不退化并改善 worst-domain teacher KL。
2. **E2 trust-region necessity**：比较 function-only 与 constrained WFPT；constraint 必须降低最差单层 drift，同时保留至少 50% function gain。
3. **E3 frozen replication**：冻结全部方法设置，在新 seed 与 (0,1) 窗口复制；若再次出现 domain 或 gate/test 符号翻转，则停止该主线。

在 E1--E3 前不扩大到 all-layer，不加入 soft-to-hard、rotation 或 task traces。
