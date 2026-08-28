# CEGSP-10A: Cross-architecture validation on GPT-NeoX/Pythia-1B

日期：2026-08-27

## 目的

验证 CEGSP 是否能从 OPT-family 迁移到一个非 OPT 架构，同时保持三值 PTQ、量化点 CE 梯度和 Q/K 离散编辑的定义不变。该实验的主要新增内容是 architecture adapter，而不是新算法模块。

## 预注册主张

在 `EleutherAI/pythia-1b`（GPT-NeoX 架构）上，direct ternary PTQ 后的 quantized-point CE-gradient guided Q/K editing，若在 untouched Wikitext 和独立 C4 holdout 上同时降低 NLL，则记为 `PASS_CROSS_ARCH`；否则记为当前设置下的 `PARTIAL_CROSS_ARCH` 或 `FAIL_CROSS_ARCH`，不据此否定 CEGSP 整体方向。

## 固定方法与边界

- strict PTQ：不加载 QAT checkpoint、QAT logits、latent weights，不做 optimizer update。
- 先对模型全部 `torch.nn.Linear` 做 direct ternary PTQ，group size 128，threshold factor 0.7。
- 仅编辑 attention Q/K；GPT-NeoX 的 fused `query_key_value` 由 adapter 暴露为连续 Q/K/V 行切片，前向仍使用原 fused Linear。
- 候选由 deployed ternary weights 上的 CE gradient 生成；保留 support relocation、signflip、joint 三类输出。
- layer selection 使用 validation single-layer ranking；不枚举 projection mask，不事后调整 epsilon。

## 运行配置

- Model: `EleutherAI/pythia-1b`
- Family: GPT-NeoX / Pythia（非 OPT）
- Device: single RTX 4090 24GB
- Layers: all model layers as reported by `config.num_hidden_layers`
- Sequence length: 128; batch size: 2
- Fit/validation: Wikitext-2 train/validation, 8/8 batches
- Untouched: Wikitext-2 validation 24 batches and C4 validation 24 batches
- Calibration offsets: 0 for this first cross-family transfer run
- k sweep: `4,8` (bounded by layer count)
- Per-layer edit budget: 64; support/signflip top-k: 8
- CE gradient batches: 1
- dtype: bf16; seed: 20260826

## 机器判定 gate

1. Adapter gate: model loads, all layers expose Q/K, fused QKV slice shapes are valid, and result records `architecture_adapter=gpt_neox_fused_qkv`.
2. Integrity gate: fit=8, val=8, untouched W=24, C4=24; all reported NLL finite; clean-room invariants remain true.
3. Main gate: at least one `ksweep-joint-top{k}-qk` has both `delta_vs_direct_untouched_w < 0` and `delta_vs_direct_untouched_c4 < 0`.
4. Cost gate: completes on one RTX 4090 without OOM; report runtime and peak memory.

## 解释规则

- `PASS_CROSS_ARCH`: adapter 和方法在非 OPT 架构的两个 untouched 分布上均通过。
- `PARTIAL_CROSS_ARCH`: adapter 完整，但仅 Wikitext 或仅 C4 改善；需要后续 split/offset，不更换方法族。
- `FAIL_CROSS_ARCH`: adapter 完整、数据与指标完整，但两个 untouched 分布均未改善；记录为跨架构边界证据，下一步优先检查层选择和 calibration offset，不自动换题。
- `DIAGNOSTIC_FAILURE`: 仅 harness/环境问题；只修复一次明确故障，不把它当作方法负结果。

## Formal execution note

首轮 `CEGSP-10A-PYTHIA1B-CROSSARCH` 已完成数值运行，但结果 metadata 缺少 adapter 字段，因此只作为诊断记录。随后只补充该字段并以完全相同配置完成正式复跑：`CEGSP-10A-PYTHIA1B-CROSSARCH-RERUN`。两次运行均成功通过 GPT-NeoX fused-QKV 的切片读写和反向路径；正式复跑结果用于 gate 判定。
