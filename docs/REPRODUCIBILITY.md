# Reproducibility Notes

This artifact is the cleaned code release for TernRefine. It intentionally does
not include raw experiment logs, model checkpoints, Hugging Face caches, or
large result directories.

## Core Paper Claims Covered by Scripts

| Paper component | Script |
| --- | --- |
| Quantized-point vs full-precision-point gradient control | `remote-tools/cegsp_e1_quantized_vs_fp_gradient_opt350m.py` |
| Ordinary-affine cross-model APG | `remote-tools/cegsp_e1_cross_model_apg.py` |
| ReQuant-style and faithful reconstruction baselines | `remote-tools/cegsp_requant_comparison_4090.py`, `remote-tools/cegsp_requant_faithful_4090.py` |
| PT2 detached state export and parity checks | `remote-tools/cegsp_p9s2_detached_pt2_plugin.py`, `remote-tools/cegsp_e2_prepare_llama_pt2_state.py` |
| Llama PT2 + TernRefine replication | `remote-tools/cegsp_pt2_hba_replication_a100_fastload.py` |
| Qwen PT2 health, forensics, and conditional TernRefine | `remote-tools/cegsp_e2_qwen_pt2_health_a100.py`, `remote-tools/cegsp_e2_qwen_pt2_state_forensics_a100.py`, `remote-tools/cegsp_e2_qwen_pt2_hba_replication_a100.py` |
| Standard downstream evaluation | `remote-tools/cegsp_e2_llama_lm_eval_downstream.py`, `remote-tools/cegsp_e2_qwen_lm_eval_downstream.py` |
| Efficiency and APG first-stop audits | `remote-tools/cegsp_qwen_q2_q5_efficiency_apg.py` |

## Data Boundaries

TernRefine uses separate roles for fitting, APG selection, and final evaluation:

- fitting data computes the single quantized-point task gradient;
- APG selection data chooses the prefix length;
- WikiText2/C4 test and downstream tasks are evaluated only after patch
  selection;
- downstream tasks are never used for patch selection.

## External Assets

The following assets must be provided by the user or cluster environment:

- pretrained model directories, for example Llama-2-7B or Qwen3-8B;
- PT2 data caches and official PT2 code when running strong-initializer scripts;
- lm-evaluation-harness for standardized downstream evaluation;
- CUDA GPU resources suitable for the selected model size.

The repository `.gitignore` excludes model weights, checkpoints, caches, raw
results, and credentials.
