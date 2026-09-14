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

- `src/ternrefine/` - small framework-independent CPSR/QGP/APG utilities.
- `remote-tools/` - paper experiment runners and compatibility tools.
- `configs/` - frozen experiment protocols used by the paper scripts.
- `reproduce/` - table-to-command entry points.
- `scripts/` - small user-facing utilities such as result auditing.
- `tests/` - smoke tests for CPSR/APG invariants.
- `reference-code/pt2_official_9e943e6/` - pinned PT2 reference subset used by
  the state-export and strong-initializer scripts.
- `env/` - dependency specification used for the CUDA/PyTorch runs.
- `docs/` - reproducibility notes and script map.

## Paper Reproduction Map

Run scripts from the repository root after installing dependencies and setting
model/cache paths required by the selected experiment.

| Paper result | Recommended entry point |
| --- | --- |
| Table 3: quantized-point vs FP-point gradient | `bash reproduce/table3_eval_point.sh --help` |
| Table 4 / APG: validation-selected patch growth | `bash reproduce/table4_apg.sh --help` |
| Table 5: standardized downstream benchmark | `bash reproduce/table5_downstream.sh --help` |
| Strong PT2 Llama replication | `python remote-tools/cegsp_pt2_hba_replication_a100_fastload.py --help` |
| Strong PT2 Qwen replication | `python remote-tools/cegsp_e2_qwen_pt2_hba_replication_a100.py --help` |
| ReQuant controls | `bash reproduce/requant_controls.sh --help` |

The most important minimal mechanism experiment is Table 3. It keeps the same
`Q0`, legal CPSR candidate pool, task objective, and 384-relocation budget, and
changes only the gradient evaluation point.

## Core Algorithm API

The paper algorithm maps to the following implementation-level steps:

```python
from ternrefine import (
    apg_first_non_improvement,
    apply_relocations,
    enumerate_cpsr_moves,
    qgp_score,
)
```

The large experiment runners adapt these primitives to affine ternary states,
PT2 sidecars, Llama/Qwen model layouts, and lm-evaluation-harness downstream
evaluation.

## Environment

The experiments were run with CUDA-capable PyTorch. A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install -r env/requirements-cegsp-cu124.txt
python -m pip install -e .
```

Large models, Hugging Face caches, PT2 checkpoints, and experiment outputs are
expected to live outside the Git repository.

## Smoke Test

```bash
python -m unittest discover -s tests
python scripts/audit_patch.py --help
```

The smoke tests do not require a GPU or model checkpoint. They verify the
capacity-preserving relocation and APG selection invariants on tiny states.

## Auditing Results

Patch/result JSONs can be checked with:

```bash
python scripts/audit_patch.py --result path/to/result.json
```

The audit utility checks the metadata most relevant to the paper claims:
completion status, finite/legal flags, exact relocation counts, exact changed
coordinate counts, cardinality violations, one-backward usage, and no-rerank
flags when those fields are present.

## Reproducibility Rules

- Do not commit model weights, checkpoints, dataset caches, raw results, or
  credentials.
- Keep fit, APG-selection, W2/C4 test, and downstream data roles explicit.
- Do not use W2/C4 or downstream task data to select a patch.
- Record model path/revision, tokenizer, seed, split indices, GPU, PyTorch/CUDA
  versions, and patch legality audits in result JSONs.
- Treat Qwen PT2 endpoints with skipped strict parity as conditional unless the
  health gate is explicitly passed.
