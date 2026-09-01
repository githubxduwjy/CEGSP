# CEGSP: Quantized-Point Gradient Support Refinement

This repository contains the research code and reproducibility metadata for
CEGSP, a post-training ternary refinement method. CEGSP keeps the ternary
codebook fixed during a local edit and uses the cross-entropy gradient at the
deployed quantized point to rank legal same-group support relocations.

The current mechanism claim is deliberately narrow: the first-order score
`-<G, Delta Q>` can prioritize useful ternary support relocations in centered
and affine representations. The repository does not claim that every strong
ternary PTQ baseline is healthy or that CEGSP already dominates PT².

## Repository layout

- `remote-tools/` — CEGSP, ternary support-projection, diagnostics, and fixed
  experiment launchers.
- `reference-code/pt2_official_9e943e6/` — small, pinned PT² reference subset
  used for protocol and state compatibility checks.
- `autoresearch/` — the separate AutoResearch project and PTQ program files.
- `env/` — validated Python/CUDA dependency specification and smoke test.
- `refine-logs/` — experiment plans and analysis notes; large raw results and
  model weights are intentionally excluded from Git.
- `MIGRATION_4090.md` — 4090 migration instructions, reusable entry points,
  and the detached PT2 sidecar contract.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install -r env/requirements-cegsp-cu124.txt
```

A small score-validity run uses the fixed protocol documented in
`refine-logs/EXPERIMENT_PLAN_CEGSP_P6A_SCORE_VALIDITY_20260828.md`. The main
entry points are:

```bash
python remote-tools/cegsp_p6a_score_validity_4090.py --help
python remote-tools/cegsp_p6b_replication_4090.py --help
```

The filenames retain the historical `4090` suffix because these scripts were
first validated there; they are ordinary PyTorch scripts and can run on an
A100 after the environment smoke test passes.

## Reproducibility rules

- Do not commit model weights, Hugging Face caches, credentials, or raw result
  directories.
- Keep calibration/validation/untouched splits explicit in every result JSON.
- Do not use untouched data to select thresholds, budgets, or candidates.
- Record GPU, PyTorch/CUDA versions, model revision, seed, and token offsets.
