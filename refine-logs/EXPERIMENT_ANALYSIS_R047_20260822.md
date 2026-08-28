# R047 Joint Window Screen Analysis

**Run**: `results/remote-runs/r047_joint_window_screen_20260822/metrics.json`  
**Date**: 2026-08-22  
**Status**: completed

## Raw Summary

WikiText2:

| variant | mean NLL | delta vs official | CVaR10 delta | layer0 NMSE delta | layer1 NMSE delta | layer1 cosine drift delta |
|---|---:|---:|---:|---:|---:|---:|
| official | 3.196654 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hard_l0 | 3.244508 | +0.047855 | +0.164419 | -0.017140 | -0.010467 | -0.002042 |
| hard_l1 | 3.185570 | -0.011083 | -0.005353 | 0.000000 | -0.033759 | +0.000207 |
| hard_l0_l1 | 3.224024 | +0.027371 | +0.199473 | -0.017140 | -0.014401 | -0.005174 |

C4:

| variant | mean NLL | delta vs official | CVaR10 delta | layer0 NMSE delta | layer1 NMSE delta | layer1 cosine drift delta |
|---|---:|---:|---:|---:|---:|---:|
| official | 4.185430 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hard_l0 | 4.071085 | -0.114345 | -0.132833 | -0.016100 | -0.008712 | +0.005965 |
| hard_l1 | 4.113750 | -0.071680 | -0.160189 | 0.000000 | -0.010378 | -0.003817 |
| hard_l0_l1 | 4.035415 | -0.150015 | -0.253348 | -0.016100 | +0.004964 | +0.006333 |

Runtime: 680.48 s. Peak GPU memory: 4063 MiB.

## Findings

1. Layer-1 hard-T is the cleanest single-layer candidate.
   - On WikiText2, `hard_l1` improves mean NLL by 0.0111 and slightly improves CVaR10 by 0.0054.
   - On C4, `hard_l1` improves mean NLL by 0.0717 and CVaR10 by 0.1602.
   - This is the first hard-T setting in the recent sequence that improves both distributions at the sequence-scoring level.

2. Layer-0 hard-T remains distribution-split.
   - It reproduces R046: WikiText2 worsens by 0.0479 mean NLL, while C4 improves by 0.1143.
   - The local and two-layer hidden metrics improve on both datasets, so hidden NMSE alone still cannot predict downstream NLL.

3. Applying hard-T to both layer 0 and layer 1 is not a stable fix.
   - It is best on C4: mean NLL improves by 0.1500 and CVaR10 by 0.2533.
   - It still worsens WikiText2 mean NLL by 0.0274 and CVaR10 by 0.1995.
   - Therefore adjacent-layer joint activation improvement is not sufficient; R048 needs a multi-distribution accept/reject gate.

4. Joint metrics are useful diagnostically, but not sufficient as the sole objective.
   - On WikiText2, `hard_l0_l1` improves layer1 NMSE and cosine drift but worsens final NLL.
   - On C4, `hard_l0_l1` worsens layer1 NMSE/cosine drift but strongly improves final NLL.
   - This argues against a simple weighted objective such as `local + lambda * joint_nmse` as the next step.

## Interpretation

R047 supports the broader research direction but narrows the implementation path. The right next move is not to turn on more layers or a larger joint loss. The evidence says:

```text
hard-T update is layer-dependent and distribution-dependent.
short-window trajectory metrics expose interactions,
but final held-out NLL must remain part of the gate.
```

## Recommended R048

Run a conservative accept/reject gate, not a weighted objective:

```text
accept candidate only if:
  local layer metric does not regress beyond epsilon_local
  and short-window hidden metric does not regress beyond epsilon_joint
  and WikiText2 held-out NLL/CVaR does not regress
  and C4 held-out NLL/CVaR does not regress
```

The first R048 candidate should be `hard_l1` as the positive control, then test whether `hard_l0` can be rescued by joint/distribution gating.
