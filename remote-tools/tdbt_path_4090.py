#!/usr/bin/env python3
"""Mechanism-only test for zero-mediated ternary sign transitions.

The direct and zero-mediated paths have the same endpoint.  Therefore the
test compares the largest *per-transition* loss increase, not endpoint loss:
    +alpha -> -alpha
versus
    +alpha -> 0 -> -alpha.

Candidate packets are selected once by |grad * weight| on the calibration
split.  No mask enumeration or post-hoc epsilon selection is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tdbt_gap_4090 import build_batches, evaluate, apply_direct_ptq


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layer", default="model.decoder.layers.0.self_attn.q_proj")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--packets", type=int, default=8)
    p.add_argument("--packet-size", type=int, default=32)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--out-dir", default="/root/tdbt-runs")
    return p.parse_args()


def get_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    cur = model
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def loss_for_weight(
    model: torch.nn.Module,
    module: torch.nn.Module,
    weight: torch.Tensor,
    batches: List[torch.Tensor],
    device: torch.device,
) -> float:
    with torch.no_grad():
        module.weight.data.copy_(weight)
        return evaluate(model, batches, device)


def main() -> None:
    a = args()
    set_seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    out = Path(a.out_dir) / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.config.use_cache = False
    calib, holdout, source = build_batches(
        tokenizer, a.seq_len, a.batch_size, a.calib_batches, a.eval_batches, a.seed
    )
    apply_direct_ptq(model, a.group_size, a.threshold_factor)
    module = get_module(model, a.layer)
    if not hasattr(module, "weight"):
        raise ValueError(f"target has no weight: {a.layer}")
    base = module.weight.detach().clone()
    base_loss = evaluate(model, holdout, device)

    # One calibration gradient only; this is candidate ranking, not training.
    for p in model.parameters():
        p.requires_grad = False
    module.weight.requires_grad = True
    model.train()
    model.zero_grad(set_to_none=True)
    batch = calib[0]
    x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
    calib_loss = model(input_ids=x, labels=y, use_cache=False).loss.float()
    calib_loss.backward()
    grad = module.weight.grad.detach().float()
    module.weight.requires_grad = False
    model.eval()
    score = (grad * base.float()).abs().flatten()
    eligible = base.flatten().ne(0)
    ranked = torch.argsort(torch.where(eligible, score, torch.full_like(score, -1)), descending=True)
    selected = ranked[eligible[ranked]][: a.packets * a.packet_size]
    if selected.numel() < a.packets * a.packet_size:
        raise RuntimeError("not enough nonzero ternary states for fixed packet budget")
    packets = selected.view(a.packets, a.packet_size).cpu().tolist()
    threshold = 0.05 * base_loss
    rows: List[Dict[str, object]] = []
    for packet_id, indices in enumerate(packets):
        flat_base = base.flatten()
        zero = base.clone().flatten()
        flip = base.clone().flatten()
        idx = torch.tensor(indices, device=base.device, dtype=torch.long)
        zero[idx] = 0
        flip[idx] = -flat_base[idx]
        zero = zero.view_as(base)
        flip = flip.view_as(base)
        zero_loss = loss_for_weight(model, module, zero, holdout, device)
        flip_loss = loss_for_weight(model, module, flip, holdout, device)
        direct_step = flip_loss - base_loss
        bridge_steps = [zero_loss - base_loss, flip_loss - zero_loss]
        bridge_max = max(bridge_steps)
        rows.append(
            {
                "packet": packet_id,
                "packet_size": len(indices),
                "indices_sha256": __import__("hashlib").sha256(
                    np.asarray(indices, dtype=np.int64).tobytes()
                ).hexdigest(),
                "base_loss": base_loss,
                "zero_loss": zero_loss,
                "flip_loss": flip_loss,
                "direct_step_increase": direct_step,
                "bridge_step_increases": bridge_steps,
                "bridge_max_step_increase": bridge_max,
                "direct_safe_5pct": direct_step <= threshold,
                "bridge_safe_5pct": bridge_max <= threshold,
                "bridge_reduces_max_step": bridge_max < direct_step,
            }
        )
        module.weight.data.copy_(base)
    result = {
        "run_id": a.run_id,
        "model": a.model,
        "layer": a.layer,
        "config": vars(a),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "data": {
            "source": source,
            "calib_batches": len(calib),
            "eval_batches": len(holdout),
            "split": "train calibration / validation holdout",
        },
        "metric_definition": {
            "trust_delta": threshold,
            "direct_path": "base -> sign-flipped endpoint",
            "bridge_path": "base -> zero intermediate -> same sign-flipped endpoint",
            "primary": "max per-transition loss increase",
        },
        "rows": rows,
        "summary": {
            "packets": len(rows),
            "direct_safe_count": sum(r["direct_safe_5pct"] for r in rows),
            "bridge_safe_count": sum(r["bridge_safe_5pct"] for r in rows),
            "bridge_reduces_count": sum(r["bridge_reduces_max_step"] for r in rows),
            "mean_direct_step_increase": float(np.mean([r["direct_step_increase"] for r in rows])),
            "mean_bridge_max_step_increase": float(np.mean([r["bridge_max_step_increase"] for r in rows])),
            "finite": all(math.isfinite(float(r["flip_loss"])) for r in rows),
        },
        "status": "complete",
        "elapsed_sec": time.time() - start,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
