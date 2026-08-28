#!/usr/bin/env python3
"""4090-sized first-pass experiments for ternary PTQ/QAT gap diagnosis.

This script intentionally keeps the first experiment small and auditable:
  * FP model on a fixed calibration/holdout split;
  * groupwise ternary PTQ with a frozen threshold/scale;
  * optional W-only QAT using an STE around the same ternary projection.

It is not the final TDBT implementation.  The output is used to decide
whether the discrete basin-transport hypothesis is worth implementing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=["smoke", "gap", "fpft"], default="smoke")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--qat-steps", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--out-dir", default="/root/tdbt-runs")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def build_batches(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    calib_batches: int,
    eval_batches: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], str]:
    """Build deterministic train/holdout token batches.

    Wikitext-2 is small and public, which makes the first 4090 experiment
    reproducible.  A deterministic text fallback keeps harness failures
    separate from algorithm failures if the dataset endpoint is unavailable.
    """
    source = "wikitext-2-raw-v1"
    try:
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        valid = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = "\n".join(x["text"] for x in train if x["text"].strip())
        valid_text = "\n".join(x["text"] for x in valid if x["text"].strip())
    except Exception as exc:
        source = f"deterministic-fallback:{type(exc).__name__}"
        text = (
            "The discrete ternary state space contains negative, zero, and positive states. "
            "This controlled calibration stream is used only to test the experiment harness. "
        )
        train_text = text * 2000
        valid_text = (text + " Holdout examples must remain disjoint from calibration examples. ") * 600

    def make(text: str, n_batches: int, offset: int) -> List[torch.Tensor]:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        needed = n_batches * batch_size * (seq_len + 1)
        if ids.numel() < needed:
            reps = (needed + ids.numel() - 1) // ids.numel()
            ids = ids.repeat(reps)
        ids = ids[offset : offset + needed].view(n_batches, batch_size, seq_len + 1)
        return [x.clone() for x in ids]

    calib = make(train_text, calib_batches, 0)
    holdout = make(valid_text, eval_batches, 0)
    return calib, holdout, source


def ternary_params(
    weight: torch.Tensor,
    group_size: int,
    threshold_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed ternary scale, threshold, and state for a 2-D weight."""
    out, inp = weight.shape
    pad = (group_size - inp % group_size) % group_size
    w = F.pad(weight.float(), (0, pad)).view(out, -1, group_size)
    mean_abs = w.abs().mean(dim=-1, keepdim=True)
    threshold = threshold_factor * mean_abs
    mask = w.abs() > threshold
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    alpha = (w.abs() * mask).sum(dim=-1, keepdim=True) / denom
    state = torch.where(mask, w.sign(), torch.zeros_like(w))
    return alpha.squeeze(-1), threshold.squeeze(-1), state.squeeze(-1)


def ternary_project(
    weight: torch.Tensor,
    alpha: torch.Tensor,
    threshold: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    out, inp = weight.shape
    pad = (group_size - inp % group_size) % group_size
    w = F.pad(weight.float(), (0, pad)).view(out, -1, group_size)
    a = alpha.unsqueeze(-1)
    t = threshold.unsqueeze(-1)
    state = torch.where(w.abs() > t, w.sign(), torch.zeros_like(w))
    q = a * state
    return q.reshape(out, inp + pad)[:, :inp].to(weight.dtype)


def iter_linear_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


@torch.no_grad()
def evaluate(model: nn.Module, batches: List[torch.Tensor], device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for batch in batches:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        # Explicit next-token CE.  Passing y as `labels` here would make
        # OPT shift once more internally and silently score t+2 against t.
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("nonfinite evaluation loss")
        losses.append(value)
    return float(np.mean(losses))


def apply_direct_ptq(model: nn.Module, group_size: int, threshold_factor: float) -> Dict[str, int]:
    counts = {"linear_modules": 0, "weights": 0, "nonzero_states": 0}
    with torch.no_grad():
        for _, module in iter_linear_modules(model):
            alpha, threshold, state = ternary_params(module.weight.data, group_size, threshold_factor)
            q = ternary_project(module.weight.data, alpha, threshold, group_size)
            module.weight.data.copy_(q)
            counts["linear_modules"] += 1
            counts["weights"] += module.weight.numel()
            counts["nonzero_states"] += int(state.ne(0).sum().item())
    return counts


class TernarySTELinear(nn.Module):
    def __init__(self, original: nn.Linear, group_size: int, threshold_factor: float):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.group_size = group_size
        self.use_quantized = True
        self.weight = nn.Parameter(original.weight.detach().clone())
        if original.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        alpha, threshold, _ = ternary_params(self.weight.data, group_size, threshold_factor)
        self.register_buffer("alpha", alpha)
        self.register_buffer("threshold", threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quantized:
            return F.linear(x, self.weight, self.bias)
        out, inp = self.weight.shape
        pad = (self.group_size - inp % self.group_size) % self.group_size
        w = F.pad(self.weight.float(), (0, pad)).view(out, -1, self.group_size)
        a = self.alpha.unsqueeze(-1)
        t = self.threshold.unsqueeze(-1)
        state = torch.where(w.abs() > t, w.sign(), torch.zeros_like(w))
        q = (a * state).reshape(out, inp + pad)[:, :inp].to(self.weight.dtype)
        q_ste = self.weight + (q - self.weight).detach()
        return F.linear(x, q_ste, self.bias)


def replace_linears(module: nn.Module, group_size: int, threshold_factor: float) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, TernarySTELinear(child, group_size, threshold_factor))
            replaced += 1
        else:
            replaced += replace_linears(child, group_size, threshold_factor)
    return replaced


def train_qat(
    model: nn.Module,
    batches: List[torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
) -> List[float]:
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    history: List[float] = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite QAT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % max(1, steps // 8) == 0:
            log(f"qat step {step + 1}/{steps} loss={history[-1]:.6f}")
    return history


def train_fp_ft(
    model: nn.Module,
    batches: List[torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
) -> List[float]:
    """Same-budget full-precision W-only fine-tuning control."""
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    trainable: List[nn.Parameter] = []
    for _, module in iter_linear_modules(model):
        module.weight.requires_grad = True
        trainable.append(module.weight)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    history: List[float] = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite FP-FT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % max(1, steps // 8) == 0:
            log(f"fpft step {step + 1}/{steps} loss={history[-1]:.6f}")
    return history


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, object] = {
        "run_id": args.run_id,
        "model": args.model,
        "mode": args.mode,
        "config": vars(args),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "status": "running",
    }
    started = time.time()
    log(f"loading {args.model} on {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    calib, holdout, data_source = build_batches(
        tokenizer, args.seq_len, args.batch_size, args.calib_batches, args.eval_batches, args.seed
    )
    result["data"] = {
        "source": data_source,
        "calib_batches": len(calib),
        "eval_batches": len(holdout),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "split": "train calibration / validation holdout",
    }
    log(f"data source={data_source} calib={len(calib)} holdout={len(holdout)}")
    fp_loss = evaluate(model, holdout, device)
    result["metrics"] = {"fp_holdout_nll": fp_loss}
    log(f"fp holdout nll={fp_loss:.6f} ppl={math.exp(fp_loss):.4f}")

    ptq_model = model
    ptq_counts = apply_direct_ptq(ptq_model, args.group_size, args.threshold_factor)
    ptq_loss = evaluate(ptq_model, holdout, device)
    result["ptq"] = {"counts": ptq_counts, "holdout_nll": ptq_loss, "holdout_ppl": math.exp(ptq_loss)}
    log(f"ptq holdout nll={ptq_loss:.6f} ppl={math.exp(ptq_loss):.4f} nonzero={ptq_counts['nonzero_states']}")

    if args.mode == "gap" and args.qat_steps > 0:
        del ptq_model
        del model
        torch.cuda.empty_cache()
        qat = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        qat.config.use_cache = False
        replaced = replace_linears(qat, args.group_size, args.threshold_factor)
        for name, p in qat.named_parameters():
            if "weight" not in name or "ternary" not in name.lower():
                # TernarySTELinear parameters are the only trainable weights;
                # embeddings, norms, and biases remain frozen for W-only QAT.
                p.requires_grad = False
        for module in qat.modules():
            if isinstance(module, TernarySTELinear):
                module.weight.requires_grad = True
        before = evaluate(qat, holdout, device)
        history = train_qat(qat, calib, device, args.qat_steps, args.lr)
        after = evaluate(qat, holdout, device)
        gap = ptq_loss - fp_loss
        result["qat"] = {
            "replaced_linears": replaced,
            "steps": args.qat_steps,
            "lr": args.lr,
            "before_holdout_nll": before,
            "after_holdout_nll": after,
            "after_holdout_ppl": math.exp(after),
            "calib_first_nll": history[0] if history else None,
            "calib_last_nll": history[-1] if history else None,
            "ptq_minus_fp_nll": gap,
            "gap_closed_fraction": (ptq_loss - after) / gap if gap > 1e-6 else None,
            "gap_status": "measurable" if gap > 1e-6 else "not_measurable_on_this_holdout",
        }
        gap_display = result["qat"]["gap_closed_fraction"]
        log(f"qat holdout nll={after:.6f} ppl={math.exp(after):.4f} gap_closed={gap_display}")

    if args.mode == "fpft" and args.qat_steps > 0:
        del ptq_model
        del model
        torch.cuda.empty_cache()
        ft = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        ft.config.use_cache = False
        before = evaluate(ft, holdout, device)
        history = train_fp_ft(ft, calib, device, args.qat_steps, args.lr)
        after_fp = evaluate(ft, holdout, device)
        counts = apply_direct_ptq(ft, args.group_size, args.threshold_factor)
        after_ptq = evaluate(ft, holdout, device)
        result["fpft"] = {
            "steps": args.qat_steps,
            "lr": args.lr,
            "before_holdout_nll": before,
            "after_fp_holdout_nll": after_fp,
            "after_ptq_holdout_nll": after_ptq,
            "after_ptq_holdout_ppl": math.exp(after_ptq),
            "calib_first_nll": history[0] if history else None,
            "calib_last_nll": history[-1] if history else None,
            "counts": counts,
        }
        log(f"fpft fp_nll={after_fp:.6f} post_ptq_nll={after_ptq:.6f} ppl={math.exp(after_ptq):.4f}")

    result["status"] = "complete"
    result["elapsed_sec"] = time.time() - started
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
