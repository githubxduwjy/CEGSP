# TernRefine

This repository contains the core implementation and reproducibility scripts for
the paper:

**TernRefine: Gradient-Guided Fixed-Capacity Refinement of Ternary LLMs**

TernRefine starts from an already deployed ternary model and refines only the
discrete assignment. It keeps the ternary codebook fixed and preserves exact
groupwise side-state cardinality. The method has three parts:

- **CPSR**: capacity-preserving side-state relocation defines legal donor and
  receiver moves inside a quantization group.
- **QGP**: quantized-point gradient projection scores each legal move with
  `-<G, Delta Q>` using one task gradient at the deployed quantized state.
- **APG**: adaptive patch growth evaluates prefixes of the fixed ranking on a
  disjoint validation split and selects the first validated patch extent.

The repository is intentionally trimmed for conference artifact review. It
keeps core components and the scripts needed to reproduce the paper's main
mechanism, strong-PTQ, and downstream evaluations. Historical exploration logs,
large result directories, model checkpoints, caches, and unrelated prototypes
are not tracked.

## Layout

- `ternrefine/` - small framework-independent CPSR/QGP/APG utilities.
- `remote-tools/` - paper experiment runners and compatibility tools.
- `reference-code/pt2_official_9e943e6/` - pinned PT2 reference subset used by
  the state-export and strong-initializer scripts.
- `env/` - dependency specification used for the CUDA/PyTorch runs.
- `docs/` - reproducibility notes and script map.

## Main Scripts

Mechanism and OPT-350M controls:

```bash
python remote-tools/cegsp_e1_quantized_vs_fp_gradient_opt350m.py --help
python remote-tools/cegsp_requant_comparison_4090.py --help
python remote-tools/cegsp_requant_faithful_4090.py --help
```

Ordinary-affine and strong-PTQ TernRefine:

```bash
python remote-tools/cegsp_e1_cross_model_apg.py --help
python remote-tools/cegsp_pt2_hba_replication_a100_fastload.py --help
python remote-tools/cegsp_e2_qwen_pt2_hba_replication_a100.py --help
```

PT2 state export, health checks, and downstream evaluation:

```bash
python remote-tools/cegsp_e2_prepare_llama_pt2_state.py --help
python remote-tools/cegsp_e2_llama_lm_eval_downstream.py --help
python remote-tools/cegsp_e2_qwen_lm_eval_downstream.py --help
python remote-tools/cegsp_e2_qwen_pt2_health_a100.py --help
python remote-tools/cegsp_e2_qwen_pt2_state_forensics_a100.py --help
```

## Environment

The experiments were run with CUDA-capable PyTorch. A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install -r env/requirements-cegsp-cu124.txt
```

Large models, Hugging Face caches, PT2 checkpoints, and experiment outputs are
expected to live outside the Git repository.

## Reproducibility Rules

- Do not commit model weights, checkpoints, dataset caches, raw results, or
  credentials.
- Keep fit, APG-selection, W2/C4 test, and downstream data roles explicit.
- Do not use W2/C4 or downstream task data to select a patch.
- Record model path/revision, tokenizer, seed, split indices, GPU, PyTorch/CUDA
  versions, and patch legality audits in result JSONs.
- Treat Qwen PT2 endpoints with skipped strict parity as conditional unless the
  health gate is explicitly passed.
