# Experiment Plan: TQGSP-01A 4090 Validation

日期：2026-08-26

## Problem Anchor

TDBT2-02A 否定了当前 path/barrier 版本：`TDBT2-G/F` 与 endpoint greedy 完全一致，低障碍路径没有提供可观增益。但同一次实验显示，量化点梯度选择三值支撑交换在 Q/K operator 上有明显信号。

本实验不继续 TDBT path。它验证新的、更窄的主张：

> 三值 PTQ 的有效改进点可能不是离散路径，而是利用三值零态提供的支撑自由度，在部署三值点上用少量梯度做 support projection。

## Clean-Room Boundary

- 不使用 QAT checkpoint。
- 不使用 QAT logits。
- 不使用 QAT latent weights。
- 不使用 QAT state prior。
- 不启动 R057B、TDBT2-02B 或任何 path/barrier follow-up。
- 旧 R014–R058 和 TDBT2-02A 只用于确定风险与缩小问题，不作为本实验 pass/fail 证据。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence |
|---|---|---|
| C1: TQG-SP has ternary-specific signal | 回答“这是不是普通 low-bit gradient edit” | `TQGSP-support-G` 在 held-out operator NMSE 上稳定优于 `support-random`、`support-forward`、`NZ-signflip-G` |
| C2: operator proxy is not merely cosmetic | 回答“proxy 好看但端到端无用” | patched model 的 Wikitext val/untouched NLL 不明显劣于 direct PTQ；若能下降则是强正证据 |
| C3: cost remains PTQ-like | 回答“是不是接近 QAT” | 输出 wall-clock breakdown；额外成本来自 1 个量化点 backward + small discrete projection |

## Experimental Setup

- Run ID: `TQGSP-01A`
- Script: `remote-tools/tqgsp_support_projection_4090.py`
- Cloud: `root@xj-member.bitahub.com` on RTX 4090 24GB
- Model: `facebook/opt-350m`
- Data: Wikitext-2 raw, train split for calibration, validation split for val/untouched
- Layers: `0,7,15,23`
- Operator: `qk`
- Sequence length: 128
- Batch size: 2
- Calibration / val / untouched batches: 16 / 8 / 8
- Group size: 128
- Threshold factor: 0.7
- Candidate pool: 512
- Max edits: 64
- Gradient batches: 1

## Compared Systems

| Variant | Role |
|---|---|
| `direct-ternary` | direct PTQ baseline |
| `support-random` | same ternary support-swap budget, random locations |
| `support-forward` | support-swap selected by forward/magnitude salience |
| `TQGSP-support-G` | proposed gradient-guided ternary support projection |
| `NZ-signflip-G` | nonzero-only gradient sign-flip control; does not use zero support |

## Metrics

Primary:

- Held-out operator NMSE on `val`.
- Held-out operator NMSE on `untouched_w`.

Secondary:

- End-to-end NLL for direct PTQ and patched variants.
- Wall-clock timing: load/data, hidden collection, proxy validation, direct PTQ apply, NLL eval.
- Peak CUDA memory.

## Decision Gate

Mechanism gate:

- `TQGSP-support-G` should beat all three controls on `untouched_w` operator NMSE in at least 3/4 tested layers.
- It should not reverse direction between `val` and `untouched_w`.

Transfer gate:

- `TQGSP-support-G` patched model should not degrade untouched NLL by more than `+0.02` versus direct PTQ.
- If it improves NLL, that is strong evidence to scale to more layers.
- If operator improves but NLL degrades, the next turn should modify the objective toward CE/NLL-aware projection rather than expand layers.

Cost gate:

- Report measured wall-clock breakdown.
- If the method cost is dominated by candidate search rather than one gradient, reduce candidate pool or use one-shot top-K only.
- If cost approaches QAT-like repeated optimization, stop this formulation.

## Expected Outcomes and Interpretation

Positive:

- `TQGSP-support-G` beats random/forward/signflip controls and does not hurt NLL. Next experiment should expand selected Q/K layers and test C4 transfer.

Mixed:

- Operator improves but NLL is flat. Keep TQG-SP as a proxy mechanism, then test CE-aware layer weighting.

Negative:

- Signflip matches support-swap: ternary-specific claim weakens.
- Random matches support-swap: candidate signal likely unreliable.
- NLL degrades: operator proxy is insufficient; do not scale blindly.

## Run Command

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python /root/tqgsp-work/tqgsp_support_projection_4090.py \
  --run-id TQGSP-01A \
  --model facebook/opt-350m \
  --layers 0,7,15,23 \
  --operators qk \
  --seq-len 128 \
  --batch-size 2 \
  --fit-batches 16 \
  --val-batches 8 \
  --untouched-batches 8 \
  --candidate-pool 512 \
  --max-swaps 64 \
  --grad-batches 1 \
  --nll-sanity \
  --e2e-nll \
  --out-dir /root/tqgsp-runs
```

