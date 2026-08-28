# RTX 4090 云端实验记录：TDBT 首轮诊断

日期：2026-08-26  
云端：`root@xj-member.bitahub.com:42137`  
GPU：NVIDIA GeForce RTX 4090 24GB  
环境：PyTorch 2.5.1+cu124，Transformers 4.46.3，CUDA 12.4  
数据：Wikitext-2 raw；train 作为 calibration，validation 作为 holdout；seq=128。

## 已完成实验

| Run | 配置 | 结果 |
|---|---|---|
| `G4090-TDBT-00` | OPT-125M，32/16 batches，BS=4，group=128，threshold=0.7，BF16 | FP holdout NLL `9.222656`；direct ternary PTQ `9.832031`；finite，harness 通过。 |
| `G4090-TDBT-02-OPT125M` | 同配置，W-only ternary QAT，256 steps，lr=5e-5 | PTQ `9.832031`；QAT holdout `7.302734`；calibration loss `9.875 → 5.750`。 |
| `G4090-TDBT-02-FPFT-OPT125M` | 同预算 FP W-only fine-tune，再一次性 ternary PTQ | FP-FT holdout `6.478516`；再 PTQ `9.640625`。 |
| `G4090-TDBT-02-OPT350M-FIXEDREPORT2` | OPT-350M，BS=2，其余同上，QAT 256 steps；同配置修复报告字段重跑 | FP `9.417969`；direct PTQ `9.347656`；QAT `8.408203`；`gap_closed_fraction=null`（PTQ 不劣于 FP，因此不计算）。 |
| `G4090-TDBT-02-FPFT-OPT350M` | 同预算 FP W-only fine-tune，再一次性 ternary PTQ | FP-FT `7.384766`；再 PTQ `9.066406`；QAT 比该 control 低 `0.658203` NLL。 |
| `G4090-TDBT-01` | OPT-125M layer-0 q_proj，8 packets×32，BF16 path metric | 8/8 finite；BF16 loss 分辨率把多数 state 差异量化掉，bridge reduction `0/8`。 |
| `G4090-TDBT-01-FP32` | 完全相同候选、packet、层和 threshold，仅改 FP32 loss 观测 | 8/8 finite；3/8 正向 sign-flip packet 降低最大单步 barrier；均值 direct `0.00127617`，bridge `0.00066919`。 |
| `G4090-TDBT-01-OPT350M-FP32` | OPT-350M layer-0 q_proj，完全相同 FP32 path metric | 8/8 finite；7/8 packet 降低最大单步 barrier；均值 direct `2.0511e-5`，bridge `1.0394e-5`。 |

全部原始产物均已从云端拉回：`results/remote-runs/G4090-TDBT-*`。云端 screen 已结束，GPU 当前空闲，无残留实验进程。

## 评估 harness 修正与论文对齐结果

旧版 `tdbt_gap_4090.py` 曾把已经右移后的标签再次作为 HuggingFace OPT 的 `labels` 输入，造成内部二次 shift。旧版结果不删除，但统一标记为 legacy measurement。修正后的显式 next-token CE 结果如下：

| Run | FP | direct PTQ | QAT / FP-FT→PTQ |
|---|---:|---:|---:|
| `G4090-TDBT-02-ALIGNED-OPT125M` | 4.0347 | 9.6975 | QAT-256 `5.4503`，gap closure `0.7500` |
| `G4090-TDBT-02-ALIGNED-FPFT-OPT125M` | 4.0347 | 9.6975 | FP-FT `3.8926`，再 PTQ `9.6225` |

在同一固定三值网格和相同 256-step 预算下，QAT 比 FP-FT→PTQ 低 `4.1722` NLL；这才是当前与 Li et al. 论文机制最直接对齐的证据。

## 当前可支持的结论

1. 固定三值网格上的 QAT 改善不能简单解释为“先 FP 微调、再一次性三值化”：125M 和 350M 上，QAT 都优于对应的 FP-FT→PTQ control。这支持继续研究“离散状态随优化改变”的方向，但不等于已经实现了 PTQ 级别的 gap 弥合。
2. `+α→0→−α` 相比 `+α→−α` 在 FP32 的局部 loss barrier 测量上有可重复的减半趋势，且在 OPT-350M layer-0 q_proj 上出现 7/8 packet。该结果支持零态作为离散过渡通道的机制假设，但只属于局部几何证据；相同终点的路径不能直接声称端点任务性能更好。
3. Wikitext-2 小 holdout 上 OPT-350M 的 direct PTQ NLL 略低于 FP，因此不能从这批 NLL 直接宣称存在稳定的 350M PTQ–QAT gap。后续必须采用预先固定的 C4/W2 分割、更多 holdout token 和 Q/K、V/O 组合算子指标。
4. 尚未运行完整的 TDBT beam/trust-region 算法，也没有得到端到端全模型收益；当前结果只允许把 TDBT 作为“有机制支持、尚未完成算法验证”的研究方向。

## 下一验证门槛

- 在 OPT-350M 的 layers 0/7/15/23 上，测 Q/K 与 V/O 组合算子 distortion；比较 direct endpoint、one-shot quantized-gradient 和 zero-mediated path，保持候选预算相同。
- 使用 fit-A/validation-B/untouched-C/W，固定 trust region 与 beam budget；只有 barrier 改善能在 validation/untouched 上复现，才进入跨层联合优化。
- 若 zero-mediated 仅降低局部 step barrier、但组合算子和 held-out NLL 不改善，则关闭“零态专属”claim，把方法主线收敛为“离散路径约束 + 单层 trust region”；若跨层组合算子也改善，才实现 TDBT 的完整叙事。
