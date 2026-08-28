# Li et al. (2026) 论文理解与 TDBT 实验修正

来源：`/home/x1shan/Zotero/storage/3BFTDYA2/Li 等 - 2026 - Understanding Quantization-Aware Training Gradients at Quantized Weights Bias to the Low-Loss Basin.pdf`  
论文：*Understanding Quantization-Aware Training: Gradients at Quantized Weights Bias to the Low-Loss Basin*，Li, Ma, Cui，arXiv:2606.09012v1。

## 1. 论文真正证明了什么

论文没有声称 PTQ 普遍弱于 QAT。它给出的是一个局部、充分条件框架：

1. FP 训练轨迹位于低损失 river \(\mathcal M\) 上；river 周围存在一个近似平坦的 anisotropic basin \(\mathcal T\)，离开 basin 后进入陡峭 valley wall。
2. FP checkpoint \(w_{fp}\) 在 basin 内，但一次量化后的部署点 \(q_0=Q(w_{fp})\) 落在 basin 外。此时量化误差尺度与 basin 宽度同量级。
3. Hessian/PTQ 的局部二次代理是在 \(w_{fp}\) 处建立的；basin 内的法向曲率可能很小，因此代理会低估跨越 basin 边界的真实代价，并可能选择一个真实损失高的 codeword。
4. STE-QAT 使用

   \[
   w_{k+1}=w_k-\eta\nabla f(Q(w_k)),\qquad q_k=Q(w_k),
   \]

   梯度在部署的量化点 \(q_k\) 处计算，却更新 latent FP 权重 \(w_k\)。当 \(q_k\) 在 basin 外时，sharp-wall 假设使梯度获得指向 river/basin 的 inward normal component；因此后续 \(q_k\) 可能重新进入 basin。

定理 1 给出的是首次进入 basin 的有限步上界，不是全局最优性定理。它依赖量化误差有界、梯度有界、投影映射正则、river 宽度足够大和步长足够小等假设。两个 corollary 分别说明：近似收敛的 FP checkpoint 上，QAT 的最终量化损失由 basin flatness、量化误差和沿 river 的累计误差控制；尚未收敛的 checkpoint 上，QAT 还可能沿 river 继续下降。

论文的三个可检验预测是：

- 网格更粗、bit 更低时，PTQ 越可能跨出 basin；网格足够细时，PTQ–QAT 差距应消失。
- 成功 QAT 时，部署量化模型的 loss 应先下降，而 latent FP loss 不一定同步下降；这是 quantization compatibility correction，而不只是普通 FP fine-tuning。
- 等预算 FP fine-tuning 后再 PTQ，若主要沿 river 移动而没有减少法向量化误差，则不能复现 QAT 的收益。

## 2. 与我们当前实验的对应关系

### 已经对齐的部分

- `G4090-TDBT-02` 使用了固定的三值网格，并比较了 direct PTQ、固定网格 W-only QAT 和等预算 FP-FT→PTQ；这与论文“同一 quantizer grid 隔离 QAT 的量化特异性”原则一致。
- 125M 和 350M 上 QAT 都优于 FP-FT→PTQ control，说明当前实验确实观察到不能简单归因于普通微调的量化特异性收益。

### 尚未对齐、必须修正的部分

- 我们只记录了初始/最终 NLL，没有记录 \(f(Q(w_k))\) 与 \(f(w_k)\) 的逐步轨迹，因此没有检验论文最关键的“部署 loss 先改善、latent loss 不同步改善”预测。
- 350M 小 Wikitext holdout 上 direct PTQ 略优于 FP，因而不能把该批数据称为稳定 PTQ–QAT gap；需要更大的、预先固定的 holdout 和分层函数指标。
- `G4090-TDBT-01` 的 zero-mediated path 只比较了同一终点的离散 transition barrier，没有 latent weight、basin 投影或 QAT gradient；它是三值状态图的次级机制诊断，不是对 Li et al. 定理的验证。
- “零态介导符号翻转”不是论文定理的必要条件。论文只要求量化点梯度具有 inward component；QAT 的 latent 权重可以连续越过零，并不保证 deployed ternary state 必须经历显式 zero state。

另外，轨迹审计发现旧版 OPT 评估曾把 `y=batch[:,1:]` 作为 `labels` 传给内部还会自动 shift 的 HuggingFace OPT，造成重复 shift。旧版 NLL 结果保留为 legacy harness 记录，不再用于主结论。修正后的显式 next-token CE 结果为：FP holdout `4.0347`，direct ternary PTQ `9.6975`，QAT-256 `5.4503`；等预算 FP-FT 后再 PTQ 为 `9.6225`。因此 QAT 相对该 control 低 `4.1722` NLL，gap closure 为 `0.7500`。这是当前最可靠的 PTQ–QAT 量化特异性证据。

## 3. 下一实验：Ternary Compatibility Trajectory Audit

在启动跨层 beam/path 优化前，先在 4090 上做一个最小但直接对应论文预测的轨迹审计：

- 模型：OPT-125M；使用已经产生 direct PTQ gap 的同一固定三值网格。
- 数据：Wikitext-2 train calibration 与 validation holdout，保持当前 32/16 batches、seq=128；不改变 threshold、group size 或学习率来追指标。
- 对照：FP checkpoint、direct ternary PTQ、固定网格 W-only QAT、同预算 FP-FT→PTQ。
- 记录点：\(k=0,1,8,32,64,128,256\)。每个点保存

  \[
  f(w_k),\quad f(Q(w_k)),\quad \|Q(w_k)-w_k\|,
  \]

  以及 ternary state 的 0→±1、±1→0、+1↔−1 次数和每层统计。

预注册判别条件：

1. 初始 direct PTQ 在 holdout 上明显劣于 FP，且所有 loss finite；
2. QAT 的 \(f(Q(w_k))\) 在前若干步下降，而 \(f(w_k)\) 没有同幅度同步下降；
3. QAT 的最终部署 loss 优于同预算 FP-FT→PTQ；
4. 若 QAT 主要靠显式 `+1→0→−1` 发生，记录为三值特有支持；若主要是支撑/极性其它变化，则关闭“zero 是必要通道”claim，但保留“固定三值网格上的量化兼容性运输”主线。

只有 1–3 同时成立，才进入下一阶段的跨层 Q/K、V/O 组合算子实验；本轮 1–3 已在 OPT-125M 诊断上成立，但仍需更大模型和 untouched split 复现。若后续 2 不成立，则先关闭“用离散路径模拟 QAT basin correction”的叙事，转而分析校准数据或量化网格拟合问题。

## 4. 对研究主线的调整

当前主线不再是“零态本身必然降低 barrier”，而是：

> 在固定三值部署网格下，PTQ 选择的 codeword 是否破坏了 quantization compatibility；能否用低成本、有限校准步的离散/latent 运输，使部署量化点重新进入低损失 basin？

零态只作为三值体系提供的一种候选离散过渡机制。它必须通过 trajectory 和跨层函数指标证明，而不能仅凭同终点的局部 barrier 结果作为核心创新。
