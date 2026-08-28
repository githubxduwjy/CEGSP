# Pivot Plan: Activation-Weighted Pairing for 1.58-bit PTQ

Date: 2026-08-21

## Current Status

The previous Haar-pairing hypothesis failed its preregistered mechanism gate.
Similarity pairing reliably reduced Haar high-band energy, but that proxy did not
translate into activation-weighted ternary error reduction:

- Similarity vs random high-band energy reduction: 56/56 blocks, median 5.56%.
- Shared-grid activation-weighted error: similarity was worse than random by median 1.25%.
- Band-grid activation-weighted error: similarity improved over random by only median 0.35%.
- Gate target was 5% median activation-weighted improvement, so the main Haar line stops.

This is a useful negative result, not a mere failed run. It says that weight-space
similarity and high-frequency energy are insufficient objectives for 1.58-bit PTQ.

## New Hypothesis

Pairing should be selected by the quantity the quantizer actually cares about:
activation-weighted output distortion, not weight cosine or high-band energy alone.

For a candidate pair of input columns `(i, j)`, score the fixed Haar high-frequency
component with:

```text
score(i, j) = ||W[:, i] - W[:, j]||_2^2 * E[(X[:, i] - X[:, j])^2]
```

Then greedily match low-score pairs inside each PT2 block. This targets pairs whose
Haar high band is simultaneously small in weight space and cheap under the observed
activation distribution.

## Result

R033b completed on the remote RTX 4090.

- `activation_hf`: median weighted-error improvement vs random = 0.7337%.
- `activation_hf`: block win rate vs random = 67.86%.
- `activation_cov`: median weighted-error improvement vs random = -0.8199%.
- Gate: failed.

Decision: local Haar-pairing remains closed. The next viable direction is not a
new 2-column pairing heuristic, but a richer structured rotation or a calibration
data pivot.

## Proposed Run

Run ID: R033

Purpose: Activation-weighted pairing mechanism test.

Model and scope:

- LLaMA-2-7B-HF local checkpoint.
- Layers 0, 10, 20, 31.
- Projections: q, k, v, o, up, gate, down.
- Two 128-column blocks per projection.
- Calibration: WikiText2, seed 0, 8 samples, 128 deterministic tokens per sample.
- Grid: band grid from R019.

Compared systems:

- identity + ATQ
- random Haar + band grid
- weight-cosine similarity Haar + band grid
- dissimilar Haar + band grid
- activation_hf: greedy pairing by the proposed activation-weighted high-band proxy
- activation_cov: greedy pairing by weight cosine times activation cosine

Primary metric:

- Activation-weighted NMSE, paired block-by-block against random Haar + band grid.

Gate:

- `activation_hf` must improve median activation-weighted NMSE by at least 5% vs random.
- `activation_hf` must win at least 70% of paired blocks.

Interpretation:

- If the gate passes: implement the same pairing objective inside the PT2 quantization
  path and run a TinyLlama or reduced LLaMA PPL screen.
- If the gate fails: retire local Haar-pairing as the main paper idea and pivot to
  orthogonal distribution shaping or calibration-data selection.

## Local Artifacts

New scripts prepared:

- `remote-tools/activation_pairing_diagnostics.py`
- `remote-tools/run_activation_pairing_diagnostics.sh`

Suggested remote command after syncing:

```bash
bash /root/PT2-LLM-official/run_activation_pairing_diagnostics.sh \
  /root/PT2-LLM-official/aris-runs/activation_pairing_r033_20260821 band
```

Suggested retrieval:

```bash
rsync -av -e 'ssh -S /tmp/ptq_bitahub_42066.sock -p 42066' \
  root@xj-member.bitahub.com:/root/PT2-LLM-official/aris-runs/activation_pairing_r033_20260821/ \
  results/remote-runs/activation_pairing_r033_20260821/
```

## Literature-Aware Backup Pivots

If R033 fails, two cleaner directions remain:

1. Rotation-shaping pivot: replace local fixed Haar with structured learned rotations
   that explicitly reshape weights toward ternary-friendly distributions and also
   smooth activation outliers.
2. Calibration pivot: keep PT2/CAT-style ternarization, but change calibration data
   to task/reasoning traces so the ternary grid is aligned to the deployment task
   instead of generic WikiText2 tokens.
