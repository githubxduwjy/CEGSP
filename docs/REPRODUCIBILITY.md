# Reproducibility Notes

This branch is a cleaned anonymous artifact. It provides the core implementation and three representative entry points:

| Component | Command |
| --- | --- |
| CPU smoke/invariant example | `python examples/smoke_test.py` |
| OPT-350M Table 3 mechanism experiment | `bash reproduce/table3.sh --help` |
| Llama-2-7B PT2 + TernRefine example | `bash reproduce/llama2_pt2.sh --help` |

It is not intended to be a complete dump of all development runners. Historical debugging scripts, Qwen conditional-parity forensics, ReQuant controls, raw logs, checkpoints, caches, and cluster-specific launch files are intentionally excluded from this review package.

## Data Boundaries

TernRefine uses separate roles for fitting, APG selection, and final evaluation:

- fitting data computes the single quantized-point task gradient;
- APG selection data chooses the prefix length;
- WikiText2/C4 test and downstream tasks are evaluated only after patch selection;
- downstream tasks are never used for patch selection.

## External Assets

The following assets must be supplied by the user or cluster environment:

- pretrained OPT-350M or Llama-2-7B model directories, or equivalent Hugging Face identifiers when downloads are permitted;
- calibration/evaluation data caches;
- a compatible external PT2 checkout for strong-initializer examples;
- CUDA GPU resources for model-scale examples.

The repository `.gitignore` excludes model weights, checkpoints, caches, raw results, and credentials.

## Expected Scope of Claims

This artifact supports inspection of the algorithm, the key quantized-point-vs-FP-point mechanism example, and one clean large-model PT2 deployment example. It should not be described as containing every appendix ablation or every model-family runner.
