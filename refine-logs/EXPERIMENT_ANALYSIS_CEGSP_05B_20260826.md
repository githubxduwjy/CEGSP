# CEGSP-05B：random-control 对照分析（2026-08-26）

## 1. 研究问题与预注册 gate

本实验检验 CEGSP 的收益是否只是“少量三值编辑 + 同一验证集选层”的偶然收益。随机对照使用完全相同的模型、数据切分、最大编辑预算、层数和 validation top-k 选择流程，仅将 CE-gradient 产生的 support/signflip 候选替换为随机候选；每个模型运行 3 个随机重复。

预注册主 gate：CE joint top-k 必须在两个模型上同时优于 random joint top-k 的均值，且同时体现在 untouched WikiText-2（32 batches）和 untouched C4（32 batches）。

## 2. 完整性审计

| 项目 | OPT-350M | OPT-125M | 判定 |
|---|---:|---:|---|
| status | complete | complete | 通过 |
| 层数记录 | 24 | 12 | 通过 |
| validation batches | 8 | 8 | 通过 |
| untouched WikiText batches | 32 | 32 | 通过 |
| untouched C4 batches | 32 | 32 | 通过 |
| random repeats | 3 | 3 | 通过 |
| patch sets | 30 | 30 | 通过 |
| nonfinite 数量 | 0 | 0 | 通过 |
| QAT checkpoint/logits/latent weights/optimizer steps | 未使用 | 未使用 | 通过 clean-room 约束 |

`result.json` 中 `validation_version.name` 仍沿用了早期 CEGSP-01A 的名称。这是结果元数据未同步的 harness 缺陷，实际配置字段、数据批次数、随机重复和指标均与 CEGSP-05B 方案一致；该问题不改变数值结果，但必须在最终论文材料中修正并保留审计记录。`clean_room_invariants` 中关于 path-barrier/TDBT 和 QAT 相关项为“禁止项”，其值为 false 表示未使用，不应误判为失败。

## 3. 原始结果与随机均值

NLL 越低越好；`W32` 与 `C4_32` 是 untouched holdout，未用于层选择。

### OPT-350M（direct ternary：val=8.694630，W32=8.790496，C4_32=8.124830）

| k | CE joint val | CE joint W32 | CE joint C4_32 | random joint W32（均值±std） | random joint C4_32（均值±std） | W32 差值 | C4 差值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8.422876 | 8.581899 | 7.968617 | 8.790053±0.000657 | 8.124594±0.000477 | -0.208154 | -0.155977 |
| 6 | 8.372046 | 8.534658 | 7.914508 | 8.789693±0.000642 | 8.124365±0.000645 | -0.255035 | -0.209857 |

### OPT-125M（direct ternary：val=9.703077，W32=9.698358，C4_32=9.190435）

| k | CE joint val | CE joint W32 | CE joint C4_32 | random joint W32（均值±std） | random joint C4_32（均值±std） | W32 差值 | C4 差值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9.471242 | 9.442045 | 8.818388 | 9.698277±0.000593 | 9.190639±0.000876 | -0.256232 | -0.372251 |
| 3 | 9.417369 | 9.395145 | 8.780429 | 9.698040±0.000718 | 9.190425±0.000956 | -0.302896 | -0.409996 |

随机控制自身的 W32/C4_32 变化约为 `1e-3` 量级，远小于 CE joint 相对随机均值的 `0.156–0.410` NLL 优势；因此主 gate 不是由随机波动触发。

## 4. Gate 判定

**主 gate：通过。** 两个模型、两档 k、两个 untouched 数据源均满足 CE joint < random joint mean，共 4/4 个模型-k 条件通过。

**机制结论：增强。** 05A 说明 CE 编辑在更大 untouched holdout 上能泛化；05B 进一步排除了“任意随机编辑也能得到同样收益”的主要解释。当前最稳健的表述是：

> 在固定三值表示、编辑预算和 validation top-k 选择流程下，部署后三值权重点的 CE 梯度能够比随机 support/signflip 编辑更可靠地找到有益的局部编辑。

这还不等价于证明每一个 support 编辑都优于 signflip 编辑，也不证明全层编辑是安全的；此前 350M 的 all-layer 退化结果仍然有效。因此默认策略应保持“小预算、CE 排序、top-k 层选择”，而不是扩展到全层编辑。

## 5. 对研究方向的影响

本结果不要求更换 CEGSP 方向，也不支持因为单个模型的 all-layer 行为差异而修改主方法。它将方法的核心证据从“编辑有效”推进到“CE-gradient 对编辑质量有选择作用”。下一阶段若继续验证，应优先做固定方向内的泛化/消融：例如新的 calibration offset 或一个更大模型上的同预算复现；应避免无预注册地增加更多搜索自由度。

## 6. 记录的限制

1. 随机对照控制了候选随机性和 validation top-k 流程，但尚未拆分“CE 的层选择收益”和“CE 的层内候选收益”；该问题属于后续机制消融，不影响本次主 gate。
2. 当前指标是 NLL，不是完整下游任务集；论文主张应限制在函数重构/NLL 保持，除非后续另有任务级验证。
3. `validation_version` 名称需要在 harness 中修正后再生成正式论文表格，不能把现有旧名称当作 CEGSP-05B 的独立版本证明。
