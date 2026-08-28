# R058 checkpoint-veto replication analysis

## Preregistered question

With H0 and `hard_l11` frozen, calibration seed changed to 1, and fresh Wikitext2/C4 sequences 120--135, does the functional success observed in R057A replicate while a positive checkpoint-NMSE delta vetoes it?

## Integrity audit

- Process exit code: 0.
- Configuration matches the preregistration: LLaMA-2-7B, calibration/gate/test samples 8/8/8, score start 120, sequence length 2048, block size 128, seed 1, validation fraction 0.25, max steps 4, layers (10,11), and zero mean/CVaR epsilon.
- Candidates are exactly `official`, `hard_l10`, `hard_l11`, and `hard_l10_l11`.
- Score records: 128/128 expected (4 candidates x 2 datasets x 16 sequences).
- Sequence IDs: 120--135; gate 120--127 and untouched test 128--135.
- All metrics are finite; total nonfinite count is 0.

## Raw comparison for the fixed candidate

All values are paired mean deltas, `hard_l11 - official`; negative is better.

| Split | Dataset | Mean-token NLL delta | CVaR10 NLL-increase delta | Mean-NLL sequence wins | CVaR sequence wins |
|---|---:|---:|---:|---:|---:|
| Gate | Wikitext2 | +0.0180829 | +0.0568551 | 2/8 | 4/8 |
| Gate | C4 | -0.0911314 | -0.2223361 | 8/8 | 6/8 |
| Untouched test | Wikitext2 | -0.0261849 | -0.0988231 | 8/8 | 7/8 |
| Untouched test | C4 | -0.0802791 | -0.2268165 | 7/8 | 6/8 |

Checkpoint-NMSE deltas on the gate split were positive at layer 11 on both datasets: Wikitext2 `+0.000440568`, C4 `+0.000243718`; layer-10 deltas were zero because the fixed candidate modifies layer 11 only.

## Machine decision

The preregistered decision is `REJECT_CANDIDATE`:

- `gate_pass=false`, because both frozen Wikitext2 functional deltas are positive.
- `test_pass=true`, because all four untouched-test functional deltas are non-positive.
- `checkpoint_veto=true`, but this is not the decisive cause: the candidate had already failed the functional gate.

Therefore R058 does **not** replicate R057A's `functionally successful but checkpoint-vetoed` pattern. It instead reveals gate/test sign instability on Wikitext2 under a calibration-seed change. The result weakens the case for continuing the current strict hard-T candidate-selection loop; it does not refute activation-aware ternary support optimization as a broader direction.

## Immediate implication

No R059, R057B, or replacement hyperparameter search is authorized. The next action is a one-time direction review across R014--R058, followed by pausing Auto Research.
