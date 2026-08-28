# CEGSP-09A analysis: OPT-1.3B scale validation on RTX 4090

日期：2026-08-27

## 1. Run identity

- Run ID: `CEGSP-09A-OPT13B-O0-U32-SCALE`
- Remote raw result: `/root/tqgsp-runs/CEGSP-09A-OPT13B-O0-U32-SCALE/result.json`
- Remote log: `/root/tqgsp-runs/CEGSP-09A-OPT13B-O0-U32-SCALE.log`
- Model: `facebook/opt-1.3b`
- Layers: 0--23, Q/K only
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Selection: CE gradient at deployed ternary weights; no QAT teacher, no QAT logits, no latent weights, no optimizer steps
- Calibration/eval:
  - fit batches: 8
  - validation batches: 8
  - untouched Wikitext batches: 32
  - untouched C4 batches: 32
- Runtime: 130.85 s
- Peak CUDA memory allocated: 4.75 GB on RTX 4090

## 2. Pre-registered gate

Gate: at least one CE joint top-k patch set must improve both untouched Wikitext and untouched C4 NLL over direct ternary PTQ, with acceptable 4090 cost.

Result: **PASS_SCALE**.

Both joint candidates pass:

| Patch set | Val NLL | Δ val | Wikitext-32 NLL | Δ Wikitext-32 | C4-32 NLL | Δ C4-32 |
|---|---:|---:|---:|---:|---:|---:|
| direct ternary | 9.344537 | — | 9.316974 | — | 8.696613 | — |
| `ksweep-joint-top8-qk` | 9.129274 | -0.215263 | 9.123373 | -0.193601 | 8.617256 | -0.079358 |
| `ksweep-joint-top12-qk` | 9.116114 | -0.228423 | 9.065096 | -0.251878 | 8.594036 | -0.102577 |

## 3. Additional controls from the same run

Support relocation remains the strongest ternary-specific primitive at this scale:

| Patch set | Δ Wikitext-32 | Δ C4-32 | Interpretation |
|---|---:|---:|---|
| `ksweep-support-top8-qk` | -0.216061 | -0.115651 | strongest C4 transfer among top-8 variants |
| `ksweep-support-top12-qk` | -0.238097 | -0.115243 | strongest C4 transfer among top-12 variants |
| `ksweep-signflip-top8-qk` | -0.174338 | -0.043810 | improves, but weaker than support relocation |
| `ksweep-signflip-top12-qk` | -0.227775 | -0.066435 | improves, but weaker C4 transfer |
| `cegsp-support-selected-qk` | -0.285474 | -0.102748 | best Wikitext transfer among selected support variants |
| `ce-signflip-selected-qk` | -0.281199 | -0.063137 | similar Wikitext, weaker C4 |

This supports the locked interpretation:

1. The method is not merely "any low-bit sign editing"; the zero-support relocation channel is consistently useful.
2. The final method should still allow joint support/sign decisions, because signflip is sometimes selected and improves NLL, but support relocation is the scale-robust core module.
3. The correct next step is scale/generalization validation, not another tiny 350M ablation.

## 4. Cost and feasibility

The cost is clearly PTQ-like in this harness:

- Total runtime: 130.85 s
- CE gradient collection: 0.67 s
- Direct PTQ application: 0.71 s
- Edit generation and single-layer eval: 20.89 s
- Patch-set NLL eval: 12.54 s
- Peak memory: 4.75 GB

The runtime is dominated by model/tokenizer/data loading rather than the CE-gradient edit mechanism itself. This answers a recurring risk: CEGSP does not currently resemble QAT cost on OPT-1.3B; it remains a lightweight post-training edit pass.

## 5. Claim status after CEGSP-09A

Supported:

- CEGSP transfers from small OPT models to OPT-1.3B under the same strict PTQ clean-room invariants.
- CE-gradient editing at the deployed ternary point improves held-out NLL on both Wikitext and C4.
- Three-valued support relocation is a real component, not just cosmetic terminology.
- 4090-scale validation is feasible.

Not yet supported:

- Downstream task accuracy gains. CEGSP-08A showed LAMBADA hard accuracy floor effects.
- Generality beyond OPT-family decoder models.
- Strong claims on 2.7B+ or 6.7B scale.
- End-to-end compressed inference speedup.

## 6. Decision

Do not pivot. Do not reinterpret this as a new method. The next experiment should enlarge model scale while keeping the method fixed.

Recommended next run: `CEGSP-09B-OPT27B-O0-U32-SCALE`, using OPT-2.7B on the same 4090, Q/K only, direct ternary, k=12/16, Wikitext/C4 untouched holdout. If memory fails, reduce holdout batches first; do not change the algorithmic claim.
