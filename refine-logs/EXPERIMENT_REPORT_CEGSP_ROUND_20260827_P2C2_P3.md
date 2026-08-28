# CEGSP 本轮实验总结报告：P2C2 跨架构验证与 P3 canonical 队列状态

日期：2026-08-27  
本轮目标：继续 CEGSP 实验，但避免继续零散调模块；优先完成跨架构证据整理，并启动固定规则的泛化验证。

## 1. 本轮实际完成了什么

本轮完成了两件事：

1. **完成并整理 P2C2：Pythia-1B 非 OPT 跨架构验证**
   - 远端原始结果：`/root/tqgsp-runs/CEGSP-V2-P2C2-PYTHIA1B-CROSSARCH/result.json`
   - 本地分析报告：`/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_ANALYSIS_CEGSP_V2_P2C2_20260827.md`
   - 本地 tracker 已更新：`/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_TRACKER_CEGSP_PAPER.md`

2. **尝试启动 P3：canonical CEGSP 固定规则验证队列**
   - 远端 screen：`CEGSP_V2_P3_CANONICAL_VALIDATE`
   - 计划顺序：
     1. `CEGSP-V2-P3A-OPT350M-CANONICAL-OFFSET2`
     2. `CEGSP-V2-P3B-PYTHIA1B-CANONICAL-OFFSET1`
   - 但启动后观察到：
     - screen 存在；
     - GPU 当时空闲；
     - `/root/tqgsp-runs/CEGSP-V2-P3-CANONICAL-VALIDATE/console.log` 为 0 字节；
     - 本地随后出现 DNS 解析失败，未能继续读取 screen 当前画面。

因此，P3 目前只能标记为：**submitted-but-not-verified / 疑似未实际运行**。不能把它算作完成实验，也不能使用它产生任何结论。

## 2. P2C2 实验目的

P2C2 要回答一个论文级问题：

> CEGSP 是否只是 OPT-family 上的偶然现象，还是可以通过 architecture adapter 迁移到非 OPT 结构？

此前 OPT-125M/350M 的结果已经说明：

- CE 梯度引导的三值编辑优于 random；
- OPT-350M 上全层编辑会过编辑，小预算 top-k 更稳；
- 固定 top-6 在 Wikitext 和 C4 holdout 上有迁移。

但这些都属于 OPT-family。P2C2 将模型换成 `EleutherAI/pythia-1b`，使用 GPT-NeoX fused-QKV adapter，只编辑 Q/K 对应 row slices，从而验证跨架构普适性。

## 3. P2C2 实验配置

| 项目 | 设置 |
|---|---:|
| 模型 | `EleutherAI/pythia-1b` |
| 架构 adapter | GPT-NeoX fused QKV row-slice |
| 编辑对象 | Q/K |
| 数据 | Wikitext-2 |
| fit / val / untouched batches | 8 / 8 / 32 |
| seq len | 128 |
| k sweep | 4, 8, 16 |
| max edits | 64 |
| dtype | bf16 |
| GPU | RTX 4090 24GB |
| 运行时间 | 54.09 s |
| 峰值显存 | 4.21 GB |

Clean-room 核验：

- 使用量化点 CE gradient：是；
- 使用 optimizer step：否；
- 使用 QAT checkpoint / latent weights / logits / state prior：否；
- 使用 QAT teacher：否。

这说明 P2C2 仍然是纯 post-training / optimizer-free 的 CEGSP 验证。

## 4. P2C2 主要结果

Direct ternary PTQ：

| split | NLL |
|---|---:|
| val | 8.5657 |
| untouched W2 | 9.1132 |

CE-guided support relocation：

| k | layers | val ΔNLL | untouched W2 ΔNLL |
|---:|---|---:|---:|
| 4 | [7,6,5,8] | -0.5364 | -0.5095 |
| 8 | [7,6,5,8,2,1,4,3] | -0.4683 | -0.3995 |
| 16 | all 0-15 | -0.4468 | -0.2674 |

CE-guided signflip：

| k | layers | val ΔNLL | untouched W2 ΔNLL |
|---:|---|---:|---:|
| 4 | [7,9,12,1] | -0.2038 | -0.1728 |
| 8 | [7,9,12,1,6,15,11,8] | -0.2694 | -0.2662 |
| 16 | all 0-15 | -0.3197 | -0.2814 |

Random controls：

| control | k | val ΔNLL | untouched W2 ΔNLL |
|---|---:|---:|---:|
| random joint | 4 | -0.0042 | -0.0023 |
| random joint | 8 | -0.0050 | -0.0028 |
| random joint | 16 | -0.0071 | -0.0045 |
| random candidate on CE layers | 4 | -0.0025 | -0.0025 |
| random candidate on CE layers | 8 | -0.0038 | -0.0022 |
| random candidate on CE layers | 16 | -0.0003 | +0.0006 |

额外 matched control：

| control | k | val ΔNLL | untouched W2 ΔNLL |
|---|---:|---:|---:|
| support on signflip-selected layers | 4 | -0.5775 | -0.5069 |
| support on signflip-selected layers | 8 | -0.6217 | -0.5423 |
| signflip on support-selected layers | 4 | -0.1415 | -0.1853 |
| signflip on support-selected layers | 8 | -0.1755 | -0.1534 |

## 5. P2C2 说明了什么

### 5.1 成立的结论

1. **CEGSP 不是 OPT-only 现象**

在 Pythia-1B 上，CE-guided support top-4 将 untouched W2 NLL 从 9.1132 降到 8.6037，改善 0.5095。这是一个明显的跨架构正结果。

2. **architecture adapter 路线可行**

GPT-NeoX fused QKV row-slice adapter 能工作，说明 CEGSP 可以通过架构适配器扩展到不同模型族，而不是绑定在 OPT 的 q_proj/k_proj 命名结构上。

3. **随机编辑几乎无效，CE 梯度信号是真实的**

random controls 的改善大约只有 0.002–0.005，而 CE-guided support top-4 的 untouched 改善为 0.5095。这个差距说明收益不是“随便动几个三值状态就会变好”，而是量化点 CE gradient 提供了有效选择信号。

4. **当前最强的三值特异性证据是 support relocation**

Pythia-1B 上 support 明显强于 signflip。更重要的是，把 support 动作套在 signflip 选出的 top-8 层上，untouched W2 改善达到 0.5423，比原始 support top-8 更强。

这提示论文主叙事应从“mixed 动作最好”改得更稳：

> CEGSP 的核心不是泛泛地修改低比特权重，而是在三值 `{−α,0,+α}` 状态中，利用 CE gradient 在量化点上识别哪些零态/非零态支撑应当重分配。

### 5.2 不能过度声称的部分

1. **不能声称超过最新 ternary PTQ**

P2C2 只对比了 direct ternary 和 random controls。强 ternary PTQ / PT² 官方可比基线还没有解决，因此不能写成“优于 SOTA”。

2. **不能声称 W2+C4 双域泛化**

P2C2 因离线缓存限制没有 C4，因此只能算 Wikitext 跨架构证据。

3. **不能声称固定 top-k 已经解决**

OPT-125M、OPT-350M、Pythia-1B 的最佳 k 不一致：

- OPT-125M：top-12 仍有增益；
- OPT-350M：top-6 最稳，top-12 后开始退化；
- Pythia-1B：top-4 最稳。

这说明后续必须研究固定预算规则或无测试集 early-stopping，而不是事后挑 k。

## 6. P3 canonical 队列状态

P3 的设计目的不是继续调模块，而是验证固定规则：

- Q/K 编辑；
- support relocation 为主；
- 小预算 top-k；
- 新 offset；
- 不用 untouched test 选参。

计划包含：

1. OPT-350M 新 offset，带 Wikitext + C4；
2. Pythia-1B 新 offset，Wikitext-only。

但当前状态是：

- 已提交 screen：`CEGSP_V2_P3_CANONICAL_VALIDATE`；
- 观察到 screen 存在；
- GPU 当时为 0% / 4MiB，占用很低；
- console.log 为 0 字节；
- 后续远端核验遇到本地 DNS 解析失败。

所以 P3 暂时不能算开始成功。下一次继续时应先做：

1. 重新连接远端；
2. 读取 screen 当前画面；
3. 如果确认卡住，则终止该空 screen；
4. 用更稳的远端 shell 脚本文件方式重启同一 P3 预注册命令；
5. 不修改 P3 的实验内容。

## 7. 对论文路线的影响

本轮不是方向切换，而是把 CEGSP 收窄得更像论文方法：

### 原来较松散的说法

CEGSP 是 support / signflip / mixed 的组合式三值编辑。

### 现在更稳的说法

CEGSP 是一种量化点函数梯度引导的三值支撑重分配方法：

- 三值特性来自 `{−α,0,+α}` 的零态/非零态支撑；
- CE gradient 在量化点上判断哪些支撑位置对 loss 更敏感；
- 方法不做 optimizer step，不引入 QAT teacher；
- signflip / mixed 是辅助消融，不是当前主 claim。

## 8. 下一步建议

下一步不应继续开新 idea，而应做三个“论文闭环必需实验”：

1. **修复并完成 P3 canonical 固定规则验证**
   - 目标：证明规则固定后，在新 offset 上仍然改善；
   - 这是防止“我们只是挑了有利 split/k”的关键。

2. **强 PTQ 基线审计**
   - 目标：解决“只赢 direct ternary，不一定赢最新 ternary PTQ”的风险；
   - 在该 gate 前，论文 claim 应写成 mechanism/complement，而不是 SOTA。

3. **QAT gap 与成本闭环**
   - 目标：测 direct / CEGSP / one-step QAT / small-step QAT 的同场 gap；
   - 报告 wall-clock 和显存，明确 CEGSP 相对 QAT 的成本优势。

## 9. 本轮最终判定

P2C2：`PASS_CROSS_ARCH_WITH_SUPPORT_DOMINANCE`  
P3：`SUBMITTED_BUT_NOT_VERIFIED`

当前 CEGSP 方向没有陷入死胡同，但应严格从“模块探索”转入“固定方法 + 主表验证 + 强基线审计”阶段。
