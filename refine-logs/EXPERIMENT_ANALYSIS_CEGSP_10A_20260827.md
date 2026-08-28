# CEGSP-10A analysis: cross-architecture validation on Pythia-1B

日期：2026-08-27

## 1. Run identity and implementation

- Formal run ID: `CEGSP-10A-PYTHIA1B-CROSSARCH-RERUN`
- Diagnostic first run: `CEGSP-10A-PYTHIA1B-CROSSARCH`（数值完成，但 metadata 缺少 adapter 字段，不用于正式 gate）
- Remote raw result: `/root/tqgsp-runs/CEGSP-10A-PYTHIA1B-CROSSARCH-RERUN/result.json`
- Remote log: `/root/tqgsp-runs/CEGSP-10A-PYTHIA1B-CROSSARCH-RERUN.log`
- Model: `EleutherAI/pythia-1b`
- Architecture: GPT-NeoX (`model_type=gpt_neox`), 16 layers, hidden size 2048
- Adapter: `gpt_neox_fused_qkv_row_slices`
- Edited projections: Q/K only; Q, K and V are row slices of the fused `query_key_value` Linear
- Device: one NVIDIA RTX 4090, 24 GB

The adapter retains the original fused parameter for the model forward and backward pass. CEGSP reads and writes only the Q/K row slices, and extracts only those slices from the fused CE gradient. Therefore this is a genuine architecture-path test rather than an OPT-only alias.

## 2. Integrity audit

The formal result records:

- `status=complete`
- Wikitext source: `wikitext-2-raw-v1`
- fit batches: 8
- validation batches: 8
- untouched Wikitext batches: 24
- untouched C4 batches: 24
- all reported NLLs finite
- `uses_ce_gradient_at_quantized_weights=true`
- `uses_optimizer_steps=false`
- `uses_qat_checkpoint=false`
- `uses_qat_logits=false`
- `uses_qat_latent_weights=false`
- `uses_qat_state_prior=false`
- `uses_path_barrier_or_tdbt_transport=false`

Quantization covered 65 Linear modules and 908,328,960 weights. Runtime was 64.93 s after the model was cached; peak allocated memory was 4,210,368,512 bytes (about 3.92 GiB).

## 3. Raw NLL results

Direct ternary PTQ baseline:

| Setting | Val NLL | Untouched Wikitext-24 | Untouched C4-24 |
|---|---:|---:|---:|
| direct ternary | 8.716101 | 9.068829 | 8.412687 |

FP16 reference, for context only, was 3.127598 / 3.187117 / 3.325834 on the same three splits.

| Patch set | Val Δ | Wikitext Δ | C4 Δ |
|---|---:|---:|---:|
| support top4 | -0.582148 | -0.590785 | -0.386891 |
| signflip top4 | -0.230726 | -0.258397 | -0.187279 |
| joint top4 | -0.582148 | -0.590785 | -0.386891 |
| support top8 | -0.459059 | -0.426615 | -0.205578 |
| signflip top8 | -0.319692 | -0.339404 | -0.226440 |
| joint top8 | -0.459059 | -0.426615 | -0.205578 |

The corresponding formal joint NLLs are:

- joint top4: val 8.133953, Wikitext 8.478044, C4 8.025797;
- joint top8: val 8.257042, Wikitext 8.642214, C4 8.207110.

Both joint patch sets improve both untouched distributions, so the preregistered gate is `PASS_CROSS_ARCH`.

## 4. Interpretation

This is the first evidence that the current CEGSP claim is not confined to OPT's separate Q/K module layout. The method survives a fused-QKV architecture after exposing Q/K as row-slice editing targets. The strongest setting is the smaller joint top4 set; in this run all joint-selected layers chose support relocation, while the nonzero-only signflip control also improved but by a smaller margin.

The result supports the following bounded claim:

> On OPT and GPT-NeoX/Pythia decoder LMs, a strict-PTQ CE-gradient editing pass at deployed ternary weights can improve direct ternary NLL on held-out Wikitext and C4 without QAT artifacts.

It does not yet prove universal architecture independence. Only one non-OPT family, one seed/offset, and language-model NLL metrics were tested. The large improvement should therefore be treated as cross-family transfer evidence, not as a final SOTA claim.

## 5. What this changes in the paper plan

- Architecture adapter is now a reusable implementation contribution: separate Q/K and fused QKV are handled under one CEGSP interface.
- The core algorithm remains fixed; no new loss, teacher, optimizer, or projection-mask enumeration was introduced.
- The next paper-level work should be a stronger downstream or second non-OPT family evaluation, not another small support/signflip ablation.
- If compute is limited, the most informative follow-up is a repeated calibration offset on Pythia-1B using the already fixed top4/top8 settings; it should be pre-registered before execution.

## 6. Gate decision

**PASS_CROSS_ARCH** — adapter, clean-room, split, finite-metric, dual-untouched-distribution, and RTX 4090 cost gates all pass.
