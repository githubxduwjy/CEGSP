#!/usr/bin/env python3
"""P5-A: affine-ternary CEGSP adapter feasibility.

This experiment is intentionally small and protocol-focused.  It does not run
PT² itself and it does not claim strong-baseline performance.  It checks whether
CEGSP's support relocation can be defined inside a PT²-style affine ternary
codebook,

    Q = mu + alpha * T,  T in {-1, 0, +1},

with frozen mu/alpha.  The only optimized variable is the ternary index T.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_ce_gradient_4090 import (
    collect_ce_qk_grads,
    projection_weight,
    set_projection_weight,
    target_modules,
)
from cegsp_v2_p4_gap_cost_4090 import build_c4_untouched_batches
from tqgsp_support_projection_4090 import (
    build_wikitext_splits,
    evaluate_nll,
    log,
    parse_csv_ints,
)


@dataclass
class AffineCode:
    mu: torch.Tensor
    alpha: torch.Tensor
    T: torch.Tensor
    valid: torch.Tensor
    original_shape: Tuple[int, int]
    group_size: int
    fp_padded: torch.Tensor


@dataclass(frozen=True)
class AffineEdit:
    layer: int
    key: str
    row: int
    block: int
    donor: int
    receiver: int
    donor_sign: int
    receiver_sign: int
    score_formula: float
    score_exact: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="13")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=8)
    p.add_argument("--c4-untouched-batches", type=int, default=8)
    p.add_argument("--fit-token-offset", type=int, default=0)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--max-edits", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def safe_ppl(nll: float) -> float:
    try:
        return float(math.exp(float(nll)))
    except OverflowError:
        return float("inf")


def with_ppl(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        key: {"nll": float(value), "ppl": safe_ppl(float(value))}
        for key, value in metrics.items()
    }


def _pad_columns(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocks = (cols + group_size - 1) // group_size
    padded_cols = blocks * group_size
    padded = torch.zeros((rows, padded_cols), dtype=torch.float32)
    padded[:, :cols] = weight.detach().float().cpu()
    valid = torch.zeros((rows, padded_cols), dtype=torch.bool)
    valid[:, :cols] = True
    return padded.view(rows, blocks, group_size), valid.view(rows, blocks, group_size)


def make_affine_code(
    weight: torch.Tensor,
    group_size: int,
    threshold_factor: float,
) -> AffineCode:
    padded, valid = _pad_columns(weight, group_size)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1).float()
    mu = (padded * valid.float()).sum(dim=-1, keepdim=True) / count
    centered = (padded - mu) * valid.float()
    threshold = threshold_factor * centered.abs().sum(dim=-1, keepdim=True) / count
    T = torch.zeros_like(padded, dtype=torch.int8)
    T[(centered > threshold) & valid] = 1
    T[(centered < -threshold) & valid] = -1
    active = T.abs().float()
    denom = active.sum(dim=-1, keepdim=True)
    alpha_raw = (centered * T.float()).sum(dim=-1, keepdim=True)
    alpha = torch.where(denom > 0, alpha_raw / denom.clamp_min(1.0), torch.zeros_like(alpha_raw))
    return AffineCode(
        mu=mu,
        alpha=alpha,
        T=T,
        valid=valid,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        fp_padded=padded,
    )


def affine_weight(code: AffineCode, T: torch.Tensor | None = None) -> torch.Tensor:
    state = code.T if T is None else T
    q = code.mu + code.alpha * state.float()
    rows, cols = code.original_shape
    return q.view(rows, -1)[:, :cols].contiguous()


def codebook_audit(code: AffineCode, T: torch.Tensor | None = None) -> Dict[str, float]:
    state = code.T if T is None else T
    q = code.mu + code.alpha * state.float()
    residual = torch.minimum(
        torch.minimum((q - (code.mu - code.alpha)).abs(), (q - code.mu).abs()),
        (q - (code.mu + code.alpha)).abs(),
    )
    illegal = ((state < -1) | (state > 1) | (~code.valid & (state != 0))).sum()
    return {
        "num_illegal_states": int(illegal.item()),
        "max_codebook_residual": float(residual[code.valid].max().item()),
        "active_support": int((state.abs()[code.valid] > 0).sum().item()),
    }


def snapshot_qk(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    rows: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        refs = target_modules(model, layer)
        rows[layer] = {
            key: projection_weight(refs[key]).detach().float().cpu().clone()
            for key in ("q", "k")
        }
    return rows


def restore_qk(model: torch.nn.Module, weights: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer, layer_weights in weights.items():
        refs = target_modules(model, layer)
        for key, weight in layer_weights.items():
            set_projection_weight(refs[key], weight)


def apply_affine_patch(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]] | None = None,
) -> None:
    for layer, layer_codes in codes.items():
        refs = target_modules(model, layer)
        for key, code in layer_codes.items():
            T = None if states is None else states[layer][key]
            set_projection_weight(refs[key], affine_weight(code, T))


def _receiver_sign_affine_fp(code: AffineCode, row: int, block: int, receiver: int) -> int:
    centered = float(code.fp_padded[row, block, receiver] - code.mu[row, block, 0])
    return 1 if centered >= 0 else -1


def _receiver_sign_grad_best(grad: torch.Tensor, row: int, block: int, receiver: int) -> int:
    g = float(grad[row, block, receiver])
    return -1 if g >= 0 else 1


def build_group_candidates(
    layer: int,
    key: str,
    code: AffineCode,
    grad_2d: torch.Tensor,
    sign_rule: str,
) -> List[AffineEdit]:
    rows, blocks, group = code.T.shape
    grad = torch.zeros((rows, blocks, group), dtype=torch.float32)
    flat = grad.view(rows, -1)
    gcols = grad_2d.shape[1]
    flat[:, :gcols] = grad_2d.detach().float().cpu()
    candidates: List[AffineEdit] = []
    for row in range(rows):
        for block in range(blocks):
            valid = code.valid[row, block]
            active = (code.T[row, block] != 0) & valid
            inactive = (code.T[row, block] == 0) & valid
            if not bool(active.any()) or not bool(inactive.any()):
                continue
            alpha = float(code.alpha[row, block, 0])
            if alpha == 0.0:
                continue
            donor_values = alpha * grad[row, block] * code.T[row, block].float()
            donor_values = donor_values.masked_fill(~active, -float("inf"))
            donor = int(torch.argmax(donor_values).item())
            if sign_rule == "affine_fp":
                recv_score_vec = torch.empty(group, dtype=torch.float32).fill_(-float("inf"))
                inactive_idx = torch.where(inactive)[0]
                for idx in inactive_idx.tolist():
                    sr = _receiver_sign_affine_fp(code, row, block, idx)
                    recv_score_vec[idx] = -alpha * grad[row, block, idx] * sr
            elif sign_rule == "grad_best":
                recv_score_vec = alpha * grad[row, block].abs()
                recv_score_vec = recv_score_vec.masked_fill(~inactive, -float("inf"))
            else:
                raise ValueError(f"unknown sign_rule={sign_rule}")
            receiver = int(torch.argmax(recv_score_vec).item())
            sr = (
                _receiver_sign_affine_fp(code, row, block, receiver)
                if sign_rule == "affine_fp"
                else _receiver_sign_grad_best(grad, row, block, receiver)
            )
            sd = int(code.T[row, block, donor].item())
            score_formula = alpha * (
                float(grad[row, block, donor]) * sd
                - float(grad[row, block, receiver]) * sr
            )
            d_q_donor = -alpha * sd
            d_q_receiver = alpha * sr
            score_exact = -(
                float(grad[row, block, donor]) * d_q_donor
                + float(grad[row, block, receiver]) * d_q_receiver
            )
            candidates.append(
                AffineEdit(
                    layer=layer,
                    key=key,
                    row=row,
                    block=block,
                    donor=donor,
                    receiver=receiver,
                    donor_sign=sd,
                    receiver_sign=sr,
                    score_formula=float(score_formula),
                    score_exact=float(score_exact),
                )
            )
    candidates.sort(key=lambda item: item.score_formula, reverse=True)
    return candidates


def apply_edits(
    codes: Dict[int, Dict[str, AffineCode]],
    edits: List[AffineEdit],
) -> Dict[int, Dict[str, torch.Tensor]]:
    states = {
        layer: {key: code.T.clone() for key, code in layer_codes.items()}
        for layer, layer_codes in codes.items()
    }
    for edit in edits:
        state = states[edit.layer][edit.key]
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = int(edit.receiver_sign)
    return states


def select_unique_edits(candidates: List[AffineEdit], max_edits: int) -> List[AffineEdit]:
    used = set()
    selected: List[AffineEdit] = []
    for edit in candidates:
        donor_key = (edit.layer, edit.key, edit.row, edit.block, edit.donor)
        receiver_key = (edit.layer, edit.key, edit.row, edit.block, edit.receiver)
        if donor_key in used or receiver_key in used:
            continue
        selected.append(edit)
        used.add(donor_key)
        used.add(receiver_key)
        if len(selected) >= max_edits:
            break
    return selected


def build_random_edits(
    codes: Dict[int, Dict[str, AffineCode]],
    n_edits: int,
    seed: int,
) -> List[AffineEdit]:
    rng = random.Random(seed)
    groups = []
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            rows, blocks, _ = code.T.shape
            for row in range(rows):
                for block in range(blocks):
                    valid = code.valid[row, block]
                    donors = torch.where((code.T[row, block] != 0) & valid)[0].tolist()
                    receivers = torch.where((code.T[row, block] == 0) & valid)[0].tolist()
                    if donors and receivers and float(code.alpha[row, block, 0]) != 0.0:
                        groups.append((layer, key, row, block, donors, receivers))
    rng.shuffle(groups)
    selected: List[AffineEdit] = []
    for layer, key, row, block, donors, receivers in groups:
        code = codes[layer][key]
        donor = rng.choice(donors)
        receiver = rng.choice(receivers)
        sd = int(code.T[row, block, donor].item())
        sr = rng.choice([-1, 1])
        selected.append(
            AffineEdit(
                layer=layer,
                key=key,
                row=row,
                block=block,
                donor=donor,
                receiver=receiver,
                donor_sign=sd,
                receiver_sign=sr,
                score_formula=float("nan"),
                score_exact=float("nan"),
            )
        )
        if len(selected) >= n_edits:
            break
    return selected


def cardinality_violations(
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]],
) -> int:
    violations = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            before = code.T.abs().sum(dim=-1)
            after = states[layer][key].abs().sum(dim=-1)
            violations += int((before != after).sum().item())
    return violations


def audit_all(
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]] | None = None,
) -> Dict[str, object]:
    entries: Dict[str, object] = {}
    total_illegal = 0
    max_residual = 0.0
    total_active = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            T = None if states is None else states[layer][key]
            audit = codebook_audit(code, T)
            entries[f"L{layer}.{key}"] = audit
            total_illegal += int(audit["num_illegal_states"])
            max_residual = max(max_residual, float(audit["max_codebook_residual"]))
            total_active += int(audit["active_support"])
    return {
        "entries": entries,
        "total_illegal_states": total_illegal,
        "max_codebook_residual": max_residual,
        "total_active_support": total_active,
    }


def eval_metrics(
    model: torch.nn.Module,
    device: torch.device,
    val_batches: List[torch.Tensor],
    w2_batches: List[torch.Tensor],
    c4_batches: List[torch.Tensor],
) -> Dict[str, float]:
    metrics = {
        "val": evaluate_nll(model, val_batches, device),
        "wikitext2_untouched": evaluate_nll(model, w2_batches, device),
    }
    if c4_batches:
        metrics["c4_untouched"] = evaluate_nll(model, c4_batches, device)
    return metrics


def main() -> None:
    args = parse_args()
    start_time = time.time()
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    log(f"Loading {args.model} dtype={dtype} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    layers = parse_csv_ints(args.layers)
    fit_batches, val_batches, w2_batches, wikitext_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.fit_token_offset,
        args.val_token_offset,
    )
    c4_batches = build_c4_untouched_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )

    fp_qk = snapshot_qk(model, layers)
    codes: Dict[int, Dict[str, AffineCode]] = {
        layer: {
            key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor)
            for key in ("q", "k")
        }
        for layer in layers
    }

    baseline_audit = audit_all(codes)
    log(f"Baseline affine audit: illegal={baseline_audit['total_illegal_states']} residual={baseline_audit['max_codebook_residual']}")

    fp_metrics = eval_metrics(model, device, val_batches, w2_batches, c4_batches)
    apply_affine_patch(model, codes)
    affine_metrics = eval_metrics(model, device, val_batches, w2_batches, c4_batches)

    grads = collect_ce_qk_grads(model, fit_batches, layers, device, args.grad_batches)

    variants: Dict[str, Dict[str, object]] = {}
    for sign_rule in ("affine_fp", "grad_best"):
        candidates: List[AffineEdit] = []
        for layer in layers:
            for key in ("q", "k"):
                candidates.extend(
                    build_group_candidates(
                        layer,
                        key,
                        codes[layer][key],
                        grads[layer][key],
                        sign_rule,
                    )
                )
        candidates.sort(key=lambda item: item.score_formula, reverse=True)
        edits = select_unique_edits(candidates, args.max_edits)
        states = apply_edits(codes, edits)
        apply_affine_patch(model, codes, states)
        metrics = eval_metrics(model, device, val_batches, w2_batches, c4_batches)
        audit = audit_all(codes, states)
        variants[sign_rule] = {
            "num_candidate_groups": len(candidates),
            "num_edits": len(edits),
            "changed_coordinates": 2 * len(edits),
            "score_identity_max_abs_error": (
                max(abs(e.score_formula - e.score_exact) for e in edits) if edits else 0.0
            ),
            "cardinality_violations": cardinality_violations(codes, states),
            "audit": audit,
            "metrics": with_ppl(metrics),
            "delta_vs_affine_nll": {
                key: float(metrics[key] - affine_metrics[key]) for key in metrics
            },
            "top_scores": [float(e.score_formula) for e in edits[:10]],
        }

    random_edits = build_random_edits(codes, args.max_edits, args.seed + 17)
    random_states = apply_edits(codes, random_edits)
    apply_affine_patch(model, codes, random_states)
    random_metrics = eval_metrics(model, device, val_batches, w2_batches, c4_batches)
    random_audit = audit_all(codes, random_states)

    restore_qk(model, fp_qk)
    elapsed = time.time() - start_time
    max_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    result = {
        "run_id": args.run_id,
        "experiment": "CEGSP-P5-A affine ternary adapter feasibility",
        "status": "complete",
        "config": vars(args),
        "wikitext_source": wikitext_source,
        "interpretation_scope": (
            "Protocol feasibility only: PT2-style affine codebook is constructed "
            "from FP weights; no PT2 checkpoint/result compatibility claim is made."
        ),
        "fp_metrics": with_ppl(fp_metrics),
        "affine_baseline_metrics": with_ppl(affine_metrics),
        "affine_baseline_audit": baseline_audit,
        "variants": variants,
        "random_relocation": {
            "num_edits": len(random_edits),
            "changed_coordinates": 2 * len(random_edits),
            "cardinality_violations": cardinality_violations(codes, random_states),
            "audit": random_audit,
            "metrics": with_ppl(random_metrics),
            "delta_vs_affine_nll": {
                key: float(random_metrics[key] - affine_metrics[key]) for key in random_metrics
            },
        },
        "gate": {
            "legality_pass": (
                baseline_audit["total_illegal_states"] == 0
                and all(
                    variant["audit"]["total_illegal_states"] == 0
                    and variant["cardinality_violations"] == 0
                    for variant in variants.values()
                )
                and random_audit["total_illegal_states"] == 0
            ),
            "gradient_signal_pass_w2": (
                variants["affine_fp"]["metrics"]["wikitext2_untouched"]["nll"]
                < random_metrics["wikitext2_untouched"]
            ),
            "strong_init_improvement_pass_val_w2": (
                variants["affine_fp"]["metrics"]["val"]["nll"] < affine_metrics["val"]
                and variants["affine_fp"]["metrics"]["wikitext2_untouched"]["nll"]
                < affine_metrics["wikitext2_untouched"]
            ),
        },
        "elapsed_sec": elapsed,
        "max_memory_gb": max_memory_gb,
    }
    out_path = out_dir / "p5a_affine_adapter_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"Wrote {out_path}")
    log(json.dumps(result["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
