# CEGSP-V2-P2C2 Pythia-1B 跨架构实验分析

日期：2026-08-27  
Run ID：`CEGSP-V2-P2C2-PYTHIA1B-CROSSARCH`  
远端原始结果：`/root/tqgsp-runs/CEGSP-V2-P2C2-PYTHIA1B-CROSSARCH/result.json`

## 1. 实验目的

本实验用于回答 CEGSP 是否只是 OPT-family 的偶然现象。此前 OPT-125M/350M 已经显示：

- CE 梯度选择的三值编辑显著优于 random；
- OPT-350M 上存在过编辑，固定 top-6 比 top-12/18/24 更稳；
- 固定小预算在 Wikitext 和 C4 holdout 上均能转移。

P2C2 将模型换成 `EleutherAI/pythia-1b`，使用 GPT-NeoX fused-QKV architecture adapter，只编辑 Q/K 对应行切片，验证：

1. architecture adapter 是否能正确工作；
2. CEGSP 是否能跨非 OPT 架构改善 direct ternary PTQ；
3. 三值 support relocation / signflip / joint 的相对作用是否保持；
4. CE-guided edits 是否显著强于 random edits。

## 2. 配置与完整性核验

| 项目 | 结果 |
|---|---:|
| 模型 | `EleutherAI/pythia-1b` |
| adapter family | `gpt_neox` |
| adapter layout | `gpt_neox_fused_qkv_row_slices` |
| target projection | `qk` |
| 数据 | Wikitext-2 |
| fit / val / untouched batches | 8 / 8 / 32 |
| seq len | 128 |
| k sweep | 4, 8, 16 |
| max edits per projection | 64 |
| threshold factor | 0.7 |
| dtype | bf16 |
| GPU | RTX 4090 24GB |
| elapsed | 54.09 s |
| peak CUDA memory | 4.21 GB |
| 状态 | complete |

Clean-room invariants：

- 使用量化点 CE gradient：yes；
- 使用 optimizer steps：no；
- 使用 QAT checkpoint / latent weights / logits / state prior：no；
- 使用 TDBT path barrier：no。

因此该实验仍属于 optimizer-free post-training CEGSP 验证，不是 QAT teacher 或 QAT-lite。

注意：本次为离线环境，C4 迁移被关闭，因此该实验只作为跨架构 Wikitext evidence，不作为 W2+C4 双域 gate。

## 3. 主要结果

Direct ternary PTQ：

| split | NLL |
|---|---:|
| val | 8.5657 |
| untouched W2 | 9.1132 |

CE-guided top-k：

| 方法 | k | layers | val ΔNLL | untouched W2 ΔNLL |
|---|---:|---|---:|---:|
| support | 4 | [7,6,5,8] | -0.5364 | -0.5095 |
| support | 8 | [7,6,5,8,2,1,4,3] | -0.4683 | -0.3995 |
| support | 16 | all 0-15 | -0.4468 | -0.2674 |
| signflip | 4 | [7,9,12,1] | -0.2038 | -0.1728 |
| signflip | 8 | [7,9,12,1,6,15,11,8] | -0.2694 | -0.2662 |
| signflip | 16 | all 0-15 | -0.3197 | -0.2814 |
| joint | 4 | [7,6,5,8] | -0.5364 | -0.5095 |
| joint | 8 | [7,6,5,8,2,1,4,3] | -0.4683 | -0.3995 |
| joint | 16 | all 0-15 | -0.4468 | -0.2674 |

Random controls：

| control | k | val ΔNLL | untouched W2 ΔNLL |
|---|---:|---:|---:|
| random joint | 4 | -0.0042 | -0.0023 |
| random joint | 8 | -0.0050 | -0.0028 |
| random joint | 16 | -0.0071 | -0.0045 |
| random candidate on CE joint layers | 4 | -0.0025 | -0.0025 |
| random candidate on CE joint layers | 8 | -0.0038 | -0.0022 |
| random candidate on CE joint layers | 16 | -0.0003 | +0.0006 |

Matched controls：

| control | k | val ΔNLL | untouched W2 ΔNLL |
|---|---:|---:|---:|
| support on signflip-selected layers | 4 | -0.5775 | -0.5069 |
| support on signflip-selected layers | 8 | -0.6217 | -0.5423 |
| signflip on support-selected layers | 4 | -0.1415 | -0.1853 |
| signflip on support-selected layers | 8 | -0.1755 | -0.1534 |

## 4. 结论

### 4.1 正结果

1. **跨架构成立**  
   在非 OPT-family 的 GPT-NeoX/Pythia-1B 上，CE-guided support relocation 将 untouched W2 NLL 从 9.1132 降到 8.6037，改善 0.5095。这不是 OPT 专属现象。

2. **architecture adapter 有效**  
   fused QKV row-slice adapter 能正确定位 Q/K 投影，且只需约 4.21GB 显存、54秒即可完成本次验证，符合 4090 24GB 低成本验证目标。

3. **random control 基本为零**  
   random joint 的 untouched 改善只有约 0.002–0.005，而 CE-guided support top-4 改善 0.5095，差距超过两个数量级。该结果支持“不是随便改几个三值状态都会变好”，而是 CE gradient 在量化点上提供了有效选择信号。

4. **三值特异性更偏向 support relocation，而不是 signflip**  
   Pythia 上 support 明显强于 signflip。更关键的是，将 support 动作套到 signflip 选出的 top-8 层上，untouched W2 改善达到 -0.5423，强于原始 support top-8。这说明当前最稳的三值 claim 不是“混合动作总是最好”，而是：
   > CE gradient 能识别三值支撑搬移的有效层；零态/非零态的支撑重分配是核心收益来源。

### 4.2 负结果与边界

1. **top-k 规律跨模型不一致**  
   OPT-125M top-12 仍增益；OPT-350M top-6 最稳；Pythia-1B top-4 最稳。说明固定全层编辑不是安全方案，CEGSP 需要小预算 trust-region 或 validation-free stopping rule。

2. **joint 未必优于 support**  
   Pythia 本次 joint 退化为 support-only，因为所有 joint-selected edits 都是 support。这提示论文方法应把 joint 作为可选扩展，而不是核心 claim。

3. **C4 缺失**  
   因远端离线缓存限制，本次没有 C4 transfer。P2C2 只能证明跨架构 Wikitext 有效，不能替代双域泛化实验。

4. **还不能声称优于强 ternary PTQ**  
   本实验只相对 direct ternary PTQ 和 random controls 建立机制正证据；在 strong PTQ baseline 完成前，不能写成超越最新 ternary PTQ。

## 5. 对研究方法的影响

P2C2 不要求改变 CEGSP 主方向，但要求修正叙事优先级：

1. 主方法从“mixed support/signflip always best”改成“CE-guided ternary support relocation with optional signflip”；
2. 核心三值特异性放在 `{0, ±α}` 中的 support relocation：在固定三值码本下，把零态/非零态支撑移动到对量化点 CE loss 更有利的位置；
3. top-k 选择不能事后调参，应在下一阶段冻结成小预算规则，例如每个模型按固定 ratio 或固定 top-k cap，仅用 calibration/validation selection，不碰 untouched；
4. Pythia 的正结果增强了论文价值：CEGSP 不是 OPT adapter trick，而是可被 architecture adapter 扩展到 GPT-NeoX fused-QKV 结构。

## 6. 下一步建议

下一步不应继续无边界扩展模块，而应进入论文级主表前的两个关键补洞：

1. **强基线审计**：确认 direct ternary 与 PT²/其他 ternary PTQ 的可比口径，避免“只赢 direct baseline”的质疑；
2. **固定 canonical CEGSP 规则**：根据当前证据，优先采用 support relocation 为主、signflip 为辅的小预算规则，并在 OPT-350M 与 Pythia-1B 上固定 k/cap 后重新跑一次 W2+C4 或至少 W2 双 offset。

建议 canonical rule 暂定为：

- edit projection：Q/K；
- primary move：support relocation；
- optional move：signflip 只作为 matched ablation，不进入默认主方法；
- budget：small top-k trust region，而不是 all-layer；
- selection：只看 calibration/validation split，不用 untouched test；
- report：NLL 与 PPL 同时报，主分析用 NLL，用户展示可转成 PPL。

本实验判定：`PASS_CROSS_ARCH_WITH_SUPPORT_DOMINANCE`。
