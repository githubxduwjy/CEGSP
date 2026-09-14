# PT2 Setup

Strong-initializer examples depend on an external PT2 installation and model checkpoints. This artifact does not vendor PT2 source code. In our runs, PT2 was pinned to upstream commit `9e943e6`; use the same commit or record the exact commit used in your run manifest.

Expected external inputs:

- a PT2 checkout or installed package compatible with the runner;
- PT2 calibration caches or datasets;
- a Llama-2-7B model directory;
- a frozen PT2 checkpoint and sidecar export when using the fast-load path;
- enough GPU memory for Llama-2-7B.

The PT2 sidecar/export path records:

- ternary codes `T`;
- groupwise codebook parameters `mu` and `alpha`;
- Q/K module mapping;
- SSR/permutation metadata when present;
- state hashes and reload/parity diagnostics.

For the Llama-2-7B strong-initializer example, strict state parity should pass before applying any TernRefine patch. Do not bypass patch-state checks.
