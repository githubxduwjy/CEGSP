# PT2 Setup

Strong-initializer experiments depend on an external PT2 installation and model
checkpoints. This repository includes only a pinned reference subset under
`reference-code/pt2_official_9e943e6/` for protocol inspection and compatibility
checks.

Expected external inputs:

- a PT2 code checkout or installed package matching the experiment runner;
- PT2 calibration caches or datasets;
- Llama-2-7B and/or Qwen3-8B model directories;
- enough GPU memory for the selected model.

The PT2 sidecar/export scripts record:

- ternary codes `T`;
- groupwise codebook parameters `mu` and `alpha`;
- Q/K module mapping;
- SSR/permutation metadata when present;
- state hashes and reload/parity diagnostics.

Llama PT2 endpoints should pass strict state parity. Qwen PT2 endpoints must be
treated as conditional unless the health gate records an explicit strict pass.

