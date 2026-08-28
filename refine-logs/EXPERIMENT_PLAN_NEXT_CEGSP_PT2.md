# 下一阶段实验方案：CEGSP 与 PT² 的独立、公平比较

**状态：** 草案，等待人工审核；未提交云端，未启动任何新实验。

**日期：** 2026-08-27

## 1. 研究定位

CEGSP 继续作为独立的三值 PTQ 方法。PT² 是强三值 PTQ baseline，不是 CEGSP 的组成模块。下一阶段只回答一个核心问题：

> 在相同模型、相同量化范围、相同 dtype、相同校准预算和相同官方评测协议下，CEGSP 的独立性能是否优于 direct ternary 和 PT²？

不运行 `PT² + CEGSP` 组合实验，不使用 QAT teacher，也不因为单个负结果更换研究方向。

## 2. 需要验证的论文主张

### 主张 C1：CEGSP 是有效的独立三值 PTQ 方法

最低证据：在至少一个 OPT 模型和一个非 OPT 模型上，CEGSP 在未参与选择的 Wikitext-2 与 C4 holdout 上优于 direct ternary；同时完成与 PT² 的公平比较。

### 主张 C2：CEGSP 的收益与三值离散结构有关

最低证据：在相同编辑预算下，support relocation 优于 random editing 和 nonzero-only signflip；该结论不能只依赖 direct baseline 的单一结果。

### 必须排除的反主张

- CEGSP 只是因为使用了更多校准数据；
- CEGSP 只是因为 direct baseline 过弱；
- CEGSP 只是对 OPT 架构或某一个 validation split 有效；
- CEGSP 的收益来自隐式 QAT。

## 3. 总体运行顺序

### 阶段 A：官方 PT² 协议复现

只运行 OPT-350M，目的是确认 PT² 官方代码、数据准备和评测链路，不产生 CEGSP 优于 PT² 的结论。

系统：

- FP16 reference；
- PT² ATQ；
- PT² ATQ+SSR。

官方设置：

- model：`facebook/opt-350m`；
- calibration dataset：WikiText-2；
- `nsamples=128`；
- `calib_seqlen=2048`；
- `ppl_seqlen=2048`；
- `blocksize=128`；
- `percdamp=0.01`；
- PT² 使用官方默认 seed；
- 使用与官方仓库一致的数据准备方式和 PPL evaluator。

该阶段只允许进行一次明确的 environment/harness 修复。修复必须记录为兼容性修复，不得改变 ATQ、GPTQ 或 SSR 算法。

### 阶段 B：同协议下的独立公平比较

只有阶段 A 完成并通过审计后，才运行本阶段。

在 OPT-350M 上比较：

1. FP16；
2. direct ternary；
3. PT²-ATQ；
4. PT²-ATQ+SSR；
5. CEGSP。

所有方法必须满足：

- 使用 FP16；
- 量化相同的 decoder Linear 层；
- 不量化或保留 `lm_head`、`project_in`、`project_out` 的策略必须一致并明确报告；
- 使用相同的 WikiText-2 calibration token 预算；
- 使用相同的 Wikitext-2/C4 测试集和 2048-token evaluator；
- 任何层、预算或阈值选择只能读取 fit/validation split；
- untouched test 只在配置冻结后读取。

CEGSP 的起点仍是 direct ternary，不从 PT² 的结果中初始化。这样才能判断 CEGSP 是否是独立方法，而不是 PT² 的后处理。

### 阶段 C：跨架构与规模验证

阶段 B 的 harness 通过后，扩展两个模型：

- OPT-1.3B：规模迁移；
- Pythia-1B：非 OPT 架构迁移。

主表只保留五个系统：FP16、direct ternary、PT²、CEGSP、QAT reference（若 QAT reference 已有可靠结果）。

不把 support-only、signflip-only、random 等全部塞入主表，它们统一放到阶段 D 的机制表中。

### 阶段 D：三值特异性与简洁性消融

只在阶段 C 完成后运行一次完整消融，不再逐个模块开实验：

| Variant | 三值动作 | 编辑预算 |
|---|---|---:|
| direct | 无编辑 | 0 |
| random-support | zero-support relocation 随机选择 | 与 CEGSP 相同 |
| signflip-only | 只改变非零符号 | 与 CEGSP 相同 |
| support-only | 只做 support relocation | 与 CEGSP 相同 |
| CEGSP-joint | support relocation + sign 编辑 | 与 CEGSP 相同 |

默认编辑预算使用预先冻结的 `k25`；`k50` 只作为预算敏感性，不根据 test 选择。每层候选数固定为 64。

## 4. 数据与评测协议

每个模型固定以下数据角色：

- `D_fit`：用于 CE gradient 和候选生成；
- `D_val`：用于层/候选选择；
- `D_W_holdout`：Wikitext-2 未见 holdout，只用于最终报告；
- `D_C4_holdout`：C4 未见 holdout，只用于最终报告；
- `D_task`：下游任务，只用于最终报告。

每次运行必须保存：dataset source、token offset、序列长度、batch 数、token 数、split hash 和 seed。

首要指标：

- Wikitext-2 holdout NLL；
- C4 holdout NLL；
- 相对 direct 和 PT² 的 paired delta；
- 每 batch 的 95% bootstrap confidence interval。

辅助指标：

- PPL；
- 有效非零率和实际 bit accounting；
- 量化耗时；
- 峰值显存；
- 反向次数；
- 编辑数量及 support/sign 比例。

NLL 与 PPL 必须同时报告，但选择规则只使用预注册的 NLL。由于 PPL 是 NLL 的指数变换，二者的排序一致；短序列 compact NLL 不再用于 PT² 主比较。

## 5. 预注册 gate

### 阶段 A gate：官方 baseline 可复现性

- 官方 ATQ、ATQ+SSR 均成功完成；
- Wikitext-2/C4 PPL finite；
- 使用 128×2048 calibration 记录可核验；
- 日志确认执行的是 `atq`、GPTQ 和可选 SSR；
- 不存在 QAT checkpoint、optimizer step 或 latent weight；
- 兼容性 wrapper 不改变 OPT 的数值前向逻辑。

若阶段 A 不能完成，只报告 `baseline-reproduction-failed`，不进行 CEGSP+PT² 组合，也不把 compact 结果写成 PT² 败北。

### 阶段 B gate：公平比较

CEGSP 需要同时满足：

- Wikitext-2 和 C4 holdout 均 finite；
- 相对 direct ternary 的平均 paired NLL delta 小于等于 0；
- 至少一个 holdout 数据集相对 direct 有预注册幅度的改善；
- 不允许一个数据集改善而另一个数据集出现明显恶化；
- CEGSP 与 PT² 的量化范围和有效 bit budget 可审计。

若 CEGSP 优于 direct 但不优于 PT²，结论收窄为“独立的 direct ternary repair 方法”，不称为 SOTA。

若 CEGSP 在至少两个模型上同时优于 PT² 和 direct，则可以支持“CEGSP 是有竞争力的独立三值 PTQ 方法”。

### 阶段 C gate：普适性

- 3 个模型中至少 2 个在 Wikitext-2 与 C4 同时改善；
- Pythia 必须通过 fused-QKV adapter 的接口审计；
- 不使用模型特定的 test 调参；
- 量化耗时和显存保持 PTQ 级别。

### 阶段 D gate：三值特异性

- CEGSP-joint 不劣于 support-only；
- support-only 在多数核心 cell 不弱于 signflip-only；
- CEGSP 优于 random-support；
- 删除某一动作后不应出现需要额外自由度才能恢复的迹象。

## 6. 4090 工程约束

阶段 A 首先只跑 OPT-350M。若 128×2048 校准在 24GB RTX 4090 上显存不足，允许把 calibration microbatch 改为 1，并用梯度/统计累积保持总 token 数不变；不得直接把 `nsamples` 或 `calib_seqlen` 缩小后声称官方复现。

OPT-1.3B 和 Pythia-1B 只有在 OPT-350M 通过后进入云端队列。每个运行目录必须保存完整命令、环境版本、日志和 `result.json`。

## 7. 明确不做的事项

- 不启动 `PT²+CEGSP` 组合实验；
- 不把 CEGSP 降级为 PT² plugin；
- 不使用 QAT 模型作为 CEGSP teacher；
- 不用 untouched test 选择 threshold、k、offset 或 epsilon；
- 不做 projection mask 枚举；
- 不因为单次负结果修改核心研究问题；
- 不在人工批准前同步脚本或启动 screen。

## 8. 云端启动前人工审核清单

在用户明确批准前，状态保持 `DRAFT_WAITING_HUMAN_APPROVAL`。

批准前必须再次提交：

1. 本文件；
2. 实际将运行的命令；
3. 预期运行时间和显存；
4. 输出目录；
5. 当前 git diff / 脚本变更摘要；
6. 失败后的唯一允许修复项。

本方案当前只完成设计，没有提交到云端。
