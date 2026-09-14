# TernRefine Anonymous Artifact

This branch is a compact conference-review artifact for:

**TernRefine: Gradient-Guided Fixed-Capacity Refinement of Ternary LLMs**

It is intentionally smaller than the full research repository. The goal is to make the method easy to inspect and to provide a few representative reproduction examples, not to expose every historical debugging runner.

TernRefine starts from an already deployed ternary model and refines only the discrete assignment. It keeps the ternary codebook fixed and preserves exact groupwise side-state cardinality:

- **CPSR** defines legal capacity-preserving donor/receiver relocations.
- **QGP** ranks legal relocations using one task gradient at the deployed quantized point.
- **APG** chooses the patch extent by first validation non-improvement on a disjoint selection split.

## Artifact Layout

```text
.
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── src/ternrefine/
├── examples/
│   ├── smoke_test.py
│   ├── table3_opt350m.py
│   ├── llama2_pt2_refine.py
│   └── _support/
├── configs/
│   ├── table3_opt350m.yaml
│   └── llama2_pt2.yaml
├── reproduce/
│   ├── table3.sh
│   └── llama2_pt2.sh
├── scripts/audit_patch.py
├── tests/test_core.py
└── docs/
    ├── REPRODUCIBILITY.md
    └── PT2_SETUP.md
```

The artifact deliberately omits raw result directories, model checkpoints, dataset caches, cluster launch history, Qwen parity forensics, ReQuant development controls, and the vendored PT2 source copy.

## Quick Start

CPU-only invariant checks:

```bash
python -m pip install -e .
python examples/smoke_test.py
python -m unittest discover -s tests
```

Patch/result metadata audit:

```bash
python scripts/audit_patch.py --result path/to/result.json
```

## Representative Reproduction Examples

Run from the repository root after installing dependencies and preparing the external model/data assets described in `docs/REPRODUCIBILITY.md`.

| Purpose | Entry point |
| --- | --- |
| Core invariants without GPU/model files | `python examples/smoke_test.py` |
| Table 3 mechanism: quantized-point vs FP-point task gradient | `bash reproduce/table3.sh --help` |
| Large-model strong-PTQ example on Llama-2-7B | `bash reproduce/llama2_pt2.sh --help` |

The Table 3 example keeps the same `Q0`, CPSR action space, loss, and 384-relocation budget; only the gradient evaluation point changes. The Llama example starts from a frozen PT2 ternary deployment and runs TernRefine/APG on the Q/K scope with fixed ranking and no reranking.

Additional experiment-specific runners used during development will be released with the full repository after review.

## PT2 Dependency

PT2 is treated as an external initializer. This artifact does not vendor the PT2 implementation. For strong-initializer runs, provide a compatible PT2 checkout and the frozen PT2 checkpoint/sidecar exported by your environment. See `docs/PT2_SETUP.md`.

## Reproducibility Rules

- Do not commit model weights, checkpoints, dataset caches, raw results, or credentials.
- Keep fitting, APG-selection, W2/C4 test, and downstream data roles explicit.
- Do not use W2/C4 or downstream tasks to select a patch.
- Record model path/revision, tokenizer, seed, split indices, GPU, PyTorch/CUDA versions, and patch legality audits in result JSONs.
