# Script Index

The code release keeps only scripts needed for the paper's main experiments and
their direct dependencies.

## Core Dependencies

- `cegsp_ce_gradient_4090.py`: shared CE-gradient and compact NLL helpers.
- `cegsp_v2_p4_gap_cost_4090.py`: shared OPT-350M evaluation utilities.
- `cegsp_p5a_affine_adapter_feasibility_4090.py`: affine ternary state adapter.
- `cegsp_p11_p12_4090.py`: task-vs-reconstruction helper routines.
- `cegsp_p7_a100_scaling.py`: large-model affine APG utilities.
- `cegsp_p9s2_detached_pt2_plugin.py`: PT2 capture, sidecar, parity, and
  evaluator compatibility layer.

## Experiment Entrypoints

- `cegsp_e1_quantized_vs_fp_gradient_opt350m.py`
- `cegsp_e1_cross_model_apg.py`
- `cegsp_requant_comparison_4090.py`
- `cegsp_requant_faithful_4090.py`
- `cegsp_pt2_hba_detached_a100.py`
- `cegsp_pt2_hba_replication_a100.py`
- `cegsp_pt2_hba_replication_a100_fastload.py`
- `cegsp_e2_prepare_llama_pt2_state.py`
- `cegsp_e2_llama_lm_eval_downstream.py`
- `cegsp_e2_qwen_lm_eval_downstream.py`
- `cegsp_e2_qwen_pt2_health_a100.py`
- `cegsp_e2_qwen_pt2_state_forensics_a100.py`
- `cegsp_e2_qwen_pt2_hba_skipgate_a100.py`
- `cegsp_e2_qwen_pt2_hba_replication_a100.py`
- `cegsp_qwen_q2_q5_efficiency_apg.py`

The historical `4090` suffix on a few filenames reflects the first validation
device, not a hard hardware requirement.
