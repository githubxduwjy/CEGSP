#!/usr/bin/env python3
"""Audit the Li et al. QAT mechanism on a fixed ternary grid.

This records both the deployed fake-quantized path f(Q(w_k)) and the latent
full-precision path f(w_k) during the same W-only STE-QAT run.  It is a
mechanism audit, not a new optimizer and not a claim of end-to-end TDBT gain.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tdbt_gap_4090 import (
    TernarySTELinear,
    apply_direct_ptq,
    build_batches,
    replace_linears,
    ternary_params,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--qat-steps", type=int, default=256)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--out-dir", default="/root/tdbt-runs")
    return p.parse_args()


def precise_loss(model: nn.Module, batches: List[torch.Tensor], device: torch.device) -> float:
    """Cross entropy in FP32 from the model logits, avoiding HF loss rounding."""
    model.eval()
    values: List[float] = []
    with torch.no_grad():
        for batch in batches:
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            logits = model(input_ids=x, use_cache=False).logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            value = float(loss.cpu())
            if not math.isfinite(value):
                raise RuntimeError("nonfinite precise loss")
            values.append(value)
    return float(np.mean(values))


def set_quantized(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TernarySTELinear):
            module.use_quantized = enabled


def ternary_state(module: TernarySTELinear) -> torch.Tensor:
    w = module.weight.detach().float()
    out, inp = w.shape
    pad = (module.group_size - inp % module.group_size) % module.group_size
    wg = F.pad(w, (0, pad)).view(out, -1, module.group_size)
    state = torch.where(wg.abs() > module.threshold.unsqueeze(-1), wg.sign(), torch.zeros_like(wg))
    return state.detach().cpu()


def snapshot_states(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: ternary_state(module) for name, module in model.named_modules() if isinstance(module, TernarySTELinear)}


def state_transitions(previous: Dict[str, torch.Tensor], current: Dict[str, torch.Tensor]) -> Dict[str, int]:
    z2n = n2z = flip = 0
    for name, cur in current.items():
        prev = previous[name]
        z2n += int(((prev == 0) & (cur != 0)).sum())
        n2z += int(((prev != 0) & (cur == 0)).sum())
        flip += int(((prev * cur) < 0).sum())
    return {"zero_to_nonzero": z2n, "nonzero_to_zero": n2z, "sign_flip": flip}


def quant_distance(model: nn.Module) -> Dict[str, float]:
    total = 0.0
    count = 0
    max_abs = 0.0
    for module in model.modules():
        if isinstance(module, TernarySTELinear):
            w = module.weight.detach().float()
            out, inp = w.shape
            pad = (module.group_size - inp % module.group_size) % module.group_size
            wg = F.pad(w, (0, pad)).view(out, -1, module.group_size)
            state = torch.where(wg.abs() > module.threshold.unsqueeze(-1), wg.sign(), torch.zeros_like(wg))
            q = (module.alpha.unsqueeze(-1) * state).reshape(out, inp + pad)[:, :inp]
            diff = (q - w).abs()
            total += float(diff.sum())
            count += diff.numel()
            max_abs = max(max_abs, float(diff.max()))
    return {"mean_abs_q_minus_w": total / max(count, 1), "max_abs_q_minus_w": max_abs}


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, TernarySTELinear):
            module.weight.requires_grad = True
            params.append(module.weight)
    return params


def record_point(
    model: nn.Module,
    point: int,
    calib: List[torch.Tensor],
    holdout: List[torch.Tensor],
    device: torch.device,
    previous_states: Dict[str, torch.Tensor] | None,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    states = snapshot_states(model)
    transitions = {"zero_to_nonzero": 0, "nonzero_to_zero": 0, "sign_flip": 0}
    if previous_states is not None:
        transitions = state_transitions(previous_states, states)
    set_quantized(model, True)
    q_calib = precise_loss(model, calib, device)
    q_holdout = precise_loss(model, holdout, device)
    set_quantized(model, False)
    latent_calib = precise_loss(model, calib, device)
    latent_holdout = precise_loss(model, holdout, device)
    set_quantized(model, True)
    row = {
        "step": point,
        "quantized_calib_nll": q_calib,
        "quantized_holdout_nll": q_holdout,
        "latent_calib_nll": latent_calib,
        "latent_holdout_nll": latent_holdout,
        "quant_distance": quant_distance(model),
        "state_transitions_since_previous": transitions,
    }
    return row, states


def main() -> None:
    a = parse_args()
    set_seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(a.out_dir) / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    calib, holdout, source = build_batches(
        tokenizer, a.seq_len, a.batch_size, a.calib_batches, a.eval_batches, a.seed
    )
    dtype = torch.bfloat16
    fp = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    fp.config.use_cache = False
    fp_holdout = precise_loss(fp, holdout, device)
    fp_calib = precise_loss(fp, calib, device)
    apply_direct_ptq(fp, a.group_size, a.threshold_factor)
    ptq_holdout = precise_loss(fp, holdout, device)
    ptq_calib = precise_loss(fp, calib, device)
    del fp
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    replaced = replace_linears(model, a.group_size, a.threshold_factor)
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    checkpoints = {0, 1, 8, 32, 64, 128, 256}
    rows: List[Dict[str, object]] = []
    previous: Dict[str, torch.Tensor] | None = None
    row, previous = record_point(model, 0, calib, holdout, device, previous)
    rows.append(row)
    for step in range(1, a.qat_steps + 1):
        model.train()
        batch = calib[(step - 1) % len(calib)]
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
        optimizer.zero_grad(set_to_none=True)
        set_quantized(model, True)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite QAT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step in checkpoints:
            row, previous = record_point(model, step, calib, holdout, device, previous)
            row["training_calib_nll"] = float(loss.detach().cpu())
            rows.append(row)
            print(
                f"step={step} q_holdout={row['quantized_holdout_nll']:.6f} "
                f"latent_holdout={row['latent_holdout_nll']:.6f} "
                f"transitions={row['state_transitions_since_previous']}",
                flush=True,
            )
    result = {
        "run_id": a.run_id,
        "model": a.model,
        "config": vars(a),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "data": {"source": source, "calib_batches": len(calib), "eval_batches": len(holdout), "split": "train calibration / validation holdout"},
        "metric_definition": {
            "loss": "FP32 cross entropy from BF16 logits",
            "quantized_path": "f(Q(w_k)) with fixed ternary alpha/threshold grid",
            "latent_path": "f(w_k) with the same latent weights and quantization bypassed",
            "state_transitions": "counts between recorded checkpoints; zero-mediated sign flip is not assumed",
        },
        "baseline": {"fp_calib_nll": fp_calib, "fp_holdout_nll": fp_holdout, "ptq_calib_nll": ptq_calib, "ptq_holdout_nll": ptq_holdout},
        "qat": {"replaced_linears": replaced, "steps": a.qat_steps, "rows": rows},
        "status": "complete",
        "elapsed_sec": time.time() - start,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out / 'result.json'} elapsed={result['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
