# CEGSP environment

The CEGSP scripts were validated in a single-GPU Linux container with:

- Python 3.11.15
- PyTorch 2.5.1 built for CUDA 12.4
- Transformers 4.46.3
- Datasets 3.0.1
- Accelerate 1.0.1
- NumPy 2.4.6
- pandas 2.2.2
- SciPy 1.17.1
- tokenizers 0.20.3
- safetensors 0.4.5
- huggingface-hub 0.36.2

The same software stack was used for the RTX 4090 CEGSP P6-A/P6-B runs. The
GPU is not part of the Python environment: an A100 with CUDA 12.4 is expected
to use the same stack. The A100 itself must still pass the smoke test before
being used for paper results.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install -r env/requirements-cegsp-cu124.txt
```

For a fresh machine, verify the runtime before downloading a model:

```bash
python - <<'PY'
import torch
import transformers
import datasets
import pandas

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("pandas", pandas.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
PY
```

## Model and data

Model weights and datasets are intentionally not stored in Git. Set persistent
cache locations on the target machine if desired:

```bash
export HF_HOME=/persistent/cache/huggingface
export HF_DATASETS_CACHE=/persistent/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/persistent/cache/huggingface/hub
```

The main CEGSP scripts download models through Hugging Face APIs and use
Wikitext-2/C4 calibration data. For reproducible paper runs, record the model
revision and the exact dataset split in the result JSON.

## Scope of dependencies

`remote-tools/cegsp_*.py` and `remote-tools/tqgsp_*.py` use the environment
above. The older `pt2_*`, `r045_*`, `r046_*`, rotation, and Hessian scripts also
expect the external PT² repository on `PYTHONPATH` (including `pt2_llm` and its
own dependencies); that repository is not vendored here.
