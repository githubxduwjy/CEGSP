#!/usr/bin/env python3
"""Clean FP16 reference for the official PT2 data/evaluation protocol.

This intentionally does not call ``quant_sequential``.  The released
``quantize.py ... fp16`` option still enters its GPTAQ compensation path and
therefore is not a no-op reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from quantize import get_model
from pt2_llm.data import get_loaders
from pt2_llm.eval_ppl import opt_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--out", default="/root/tqgsp-runs/CEGSP-12A-FP16-CLEAN-OPT350M/result.json")
    args = p.parse_args()

    started = time.time()
    model = get_model(args.model, args.seqlen)
    model.eval()
    model.config.use_cache = False
    results = {}
    for dataset in ("wikitext2", "c4"):
        _, testloader = get_loaders(
            dataset,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.model,
        )
        results[dataset] = float(opt_eval(model, testloader, "cuda", dataset, False))

    payload = {
        "run_id": "CEGSP-12A-FP16-CLEAN-OPT350M",
        "config": vars(args),
        "protocol": {
            "official_data_loader": True,
            "official_opt_eval": True,
            "quant_sequential_called": False,
            "dtype": str(next(model.parameters()).dtype),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "ppl": results,
        "elapsed_sec": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
