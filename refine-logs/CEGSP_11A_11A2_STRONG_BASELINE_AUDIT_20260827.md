# CEGSP-11A/11A2 Strong Baseline Audit Analysis

Generated on cloud: 2026-08-27T13:52:05

## Result Table

| system | val NLL | W untouched NLL | C4 untouched NLL | dVal vs direct | dW vs direct | dC4 vs direct | quant sec | total sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp | 3.8039 | 3.7937 | 3.5447 |  |  |  |  | 3.7966 |
| direct_ternary | 8.6946 | 8.7905 | 8.1248 |  |  |  |  | 0.6153 |
| pt2_atq_ssr_False;calib=NonexNone | 9.8508 | 9.9961 | 9.5996 | 1.1562 | 1.2057 | 1.4748 | 27.4636 | 29.7624 |
| pt2_atq_ssr_True;calib=NonexNone | 10.0746 | 10.4731 | 9.7975 | 1.3800 | 1.6826 | 1.6727 | 42.6639 | 45.0053 |
| pt2_atq_ssr_True;calib=16x512 | 9.4541 | 9.5179 | 8.7792 | 0.7595 | 0.7274 | 0.6544 | 44.3255 | 46.5805 |

## Audit Judgment

- `CEGSP-11A` and `11A2` confirm that the official PT2 code can run on this 4090 instance after a pure harness compatibility wrapper for OPT `position_embeddings`.
- Under matched compact evaluation, PT2 ATQ and ATQ+SSR are finite but worse than canonical direct ternary. Longer 512-token calibration improves over compact SSR on C4 slightly but remains worse than direct on all reported splits.
- This is not sufficient to claim CEGSP beats PT2. It should be recorded as `baseline-reproduction-protocol-mismatch`: official PT2 likely needs its original full PPL/eval regime and/or larger calibration to be a fair strong baseline.
- Do not start `PT2+CEGSP` as a main claim yet, because the current PT2 point is not a reproduced strong baseline. If we edit it, we would only show CEGSP can repair a degraded baseline.

## Next Responsible Move

1. Preserve these results as a strong-baseline audit, not as a competitive claim.
2. For the paper path, add an explicit baseline-reproduction table: official PT2 full PPL protocol vs matched compact protocol. This separates reproduction failure from method performance.
3. Continue CEGSP experiments only on claims already supported: direct ternary repair, ternary support-relocation specificity, cross-model/cross-architecture transfer, and quantization-cost advantage.
4. If compute allows, run one official-style PT2 command with its native PPL output for OPT-350M to see whether PT2 itself reproduces outside the matched compact harness.

## Artifacts
- `/root/tqgsp-runs/CEGSP-11A-AUDIT-OPT350M/result.json`
- `/root/tqgsp-runs/CEGSP-11A-AUDIT-OPT350M-SSR/result.json`
- `/root/tqgsp-runs/CEGSP-11A2-PT2-LONGCALIB-OPT350M-SSR/result.json`
