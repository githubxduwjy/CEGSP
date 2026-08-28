#!/usr/bin/env python3
"""P6-A: full-layer score-validity for centered and affine ternary states.

The experiment is mechanism-only.  It evaluates a fixed set of one-step legal
support relocations after collecting fit-split CE gradients.  It does not use
validation to choose a layer, budget, sign rule, or candidate, and it does not
run PT2 or QAT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_ce_gradient_4090 import (
    collect_ce_qk_grads,
    projection_weight,
    set_projection_weight,
    target_modules,
)
from cegsp_p5a_affine_adapter_feasibility_4090 import (
    AffineCode,
    AffineEdit,
    affine_weight,
    build_group_candidates,
    build_random_edits,
    make_affine_code,
)
from cegsp_v2_p4_gap_cost_4090 import build_c4_untouched_batches
from tqgsp_support_projection_4090 import (
    Candidate,
    apply_candidate,
    apply_direct_ptq_local,
    build_wikitext_splits,
    evaluate_nll,
    gradient_projection_candidates_unique,
    make_code,
    random_support_candidates,
    sign_from_fp,
    weight_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--cegsp-root", default="/root/tqgsp-work")
    p.add_argument("--layers", default=",".join(str(i) for i in range(24)))
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=16)
    p.add_argument("--c4-batches", type=int, default=16)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--centered-threshold", type=float, default=0.70)
    p.add_argument("--affine-threshold", type=float, default=0.75)
    p.add_argument("--candidate-pool", type=int, default=32)
    p.add_argument("--evaluated-per-layer", type=int, default=8)
    p.add_argument("--random-per-layer", type=int, default=8)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--fit-token-offset", type=int, default=0)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return p.parse_args()


def safe_exp(x: float) -> float | None:
    try:
        value = math.exp(float(x))
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_layers(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def eval_pack(model: torch.nn.Module, val: List[torch.Tensor], untouched: List[torch.Tensor], c4: List[torch.Tensor] | None, device: torch.device) -> Dict[str, float]:
    out = {
        "val": float(evaluate_nll(model, val, device)),
        "untouched_w": float(evaluate_nll(model, untouched, device)),
    }
    if c4:
        out["untouched_c4"] = float(evaluate_nll(model, c4, device))
    return out


def snapshot_qk(model: torch.nn.Module, layers: Iterable[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        int(layer): {
            key: projection_weight(target_modules(model, int(layer))[key]).detach().float().cpu().clone()
            for key in ("q", "k")
        }
        for layer in layers
    }


def set_qk_weight(model: torch.nn.Module, layer: int, key: str, weight: torch.Tensor) -> None:
    set_projection_weight(target_modules(model, layer)[key], weight)


def apply_full_affine(model: torch.nn.Module, group_size: int, threshold: float) -> Dict[str, AffineCode]:
    codes: Dict[str, AffineCode] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            code = make_affine_code(module.weight.data, group_size, threshold)
            codes[name] = code
            module.weight.data.copy_(affine_weight(code).to(device=module.weight.device, dtype=module.weight.dtype))
    return codes


def direct_score(code: Any, grad: torch.Tensor, candidate: Candidate) -> float:
    padded = F.pad(grad.float(), (0, code.pad)).view_as(code.state)
    q = code.alpha.unsqueeze(-1) * code.state
    receiver = code.alpha.unsqueeze(-1) * torch.where(
        code.fp_padded >= 0, torch.ones_like(code.fp_padded), -torch.ones_like(code.fp_padded)
    )
    d = int(candidate.donor)
    r = int(candidate.receiver)
    flat_grad = padded.flatten()
    flat_q = q.flatten()
    flat_receiver = receiver.flatten()
    return float(-(flat_grad[d] * (-flat_q[d]) + flat_grad[r] * flat_receiver[r]).item())


def affine_score(code: AffineCode, grad: torch.Tensor, edit: AffineEdit) -> float:
    rows, blocks, group = code.T.shape
    padded = torch.zeros((rows, blocks, group), dtype=torch.float32)
    padded.view(rows, -1)[:, : grad.shape[1]] = grad.detach().float().cpu()
    alpha = float(code.alpha[edit.row, edit.block, 0])
    sd = int(code.T[edit.row, edit.block, edit.donor].item())
    sr = int(edit.receiver_sign)
    return float(alpha * (padded[edit.row, edit.block, edit.donor].item() * sd - padded[edit.row, edit.block, edit.receiver].item() * sr))


def select_fixed_positions(items: List[Any], count: int) -> List[Any]:
    if not items:
        return []
    if len(items) <= count:
        return list(items)
    fixed = [0, 1, 2, 4, 8, 16, 24, 31]
    indices = [i for i in fixed if i < len(items)]
    if len(indices) < count:
        indices = np.linspace(0, len(items) - 1, count, dtype=int).tolist()
    seen: set[int] = set()
    out: List[Any] = []
    for idx in indices:
        if idx not in seen:
            out.append(items[idx])
            seen.add(idx)
        if len(out) >= count:
            break
    return out


def describe_candidate(rep: str, kind: str, layer: int, candidate: Any, score: float) -> Dict[str, Any]:
    if rep == "centered":
        payload = {
            "matrix": candidate.matrix_key,
            "donor": int(candidate.donor),
            "receiver": int(candidate.receiver),
        }
    else:
        payload = {
            "matrix": candidate.key,
            "row": int(candidate.row),
            "block": int(candidate.block),
            "donor": int(candidate.donor),
            "receiver": int(candidate.receiver),
            "receiver_sign": int(candidate.receiver_sign),
        }
    return {"representation": rep, "kind": kind, "layer": int(layer), "score": float(score), **payload}


def set_baseline_qk(model: torch.nn.Module, rep: str, layers: List[int], codes: Dict[int, Dict[str, Any]], states: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer in layers:
        for key in ("q", "k"):
            if rep == "centered":
                weight = weight_from_state(codes[layer][key], states[layer][key], refit=False)
            else:
                weight = affine_weight(codes[layer][key], states[layer][key])
            set_qk_weight(model, layer, key, weight)


def set_one_candidate(
    model: torch.nn.Module,
    rep: str,
    layer: int,
    candidate: Any,
    codes: Dict[int, Dict[str, Any]],
    base_states: Dict[int, Dict[str, torch.Tensor]],
) -> None:
    if rep == "centered":
        state = apply_candidate(base_states[layer], codes[layer], candidate, endpoint=True)
        key = candidate.matrix_key
        weight = weight_from_state(codes[layer][key], state[key], refit=False)
    else:
        layer_codes = {layer: codes[layer]}
        from cegsp_p5a_affine_adapter_feasibility_4090 import apply_edits

        states = apply_edits(layer_codes, [candidate])
        key = candidate.key
        weight = affine_weight(codes[layer][key], states[layer][key])
    set_qk_weight(model, layer, key, weight)


def score_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grad = [r for r in rows if r["kind"] == "gradient"]
    rand = [r for r in rows if r["kind"] == "random"]
    ranked = sorted(grad, key=lambda r: float(r["score"]), reverse=True)
    top_n = max(1, len(ranked) // 5)
    top = ranked[:top_n]

    def rho(xs: List[float], ys: List[float]) -> float | None:
        if len(xs) < 3:
            return None
        rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64)))
        ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64)))
        if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
            return None
        value = float(np.corrcoef(rx, ry)[0, 1])
        return value if math.isfinite(value) else None

    def mean(items: List[Dict[str, Any]], key: str) -> float | None:
        values = [float(x[key]) for x in items]
        return float(np.mean(values)) if values else None

    def rate(items: List[Dict[str, Any]]) -> float | None:
        return float(np.mean([bool(x["improved"]) for x in items])) if items else None

    bins: List[Dict[str, Any]] = []
    if ranked:
        for idx, group in enumerate(np.array_split(np.asarray(ranked, dtype=object), 5)):
            group_rows = list(group)
            bins.append({
                "bin": idx + 1,
                "n": len(group_rows),
                "score_min": float(min(float(x["score"]) for x in group_rows)),
                "score_max": float(max(float(x["score"]) for x in group_rows)),
                "mean_delta_val_nll": mean(group_rows, "delta_val_nll"),
                "val_improvement_rate": rate(group_rows),
                "mean_delta_untouched_w_nll": mean(group_rows, "delta_untouched_w_nll"),
                "untouched_improvement_rate": float(np.mean([bool(x["improved_untouched"]) for x in group_rows])),
            })
    return {
        "n_gradient": len(grad),
        "n_random": len(rand),
        "spearman_score_vs_negative_delta_val_nll": rho(
            [float(x["score"]) for x in grad], [-float(x["delta_val_nll"]) for x in grad]
        ),
        "spearman_score_vs_negative_delta_untouched_w_nll": rho(
            [float(x["score"]) for x in grad], [-float(x["delta_untouched_w_nll"]) for x in grad]
        ),
        "top20pct_n": len(top),
        "mean_top20pct_delta_val_nll": mean(top, "delta_val_nll"),
        "mean_all_gradient_delta_val_nll": mean(grad, "delta_val_nll"),
        "mean_random_delta_val_nll": mean(rand, "delta_val_nll"),
        "top20pct_val_improvement_rate": rate(top),
        "all_gradient_val_improvement_rate": rate(grad),
        "random_val_improvement_rate": rate(rand),
        "mean_top20pct_delta_untouched_w_nll": mean(top, "delta_untouched_w_nll"),
        "mean_all_gradient_delta_untouched_w_nll": mean(grad, "delta_untouched_w_nll"),
        "mean_random_delta_untouched_w_nll": mean(rand, "delta_untouched_w_nll"),
        "top20pct_untouched_improvement_rate": float(np.mean([bool(x["improved_untouched"]) for x in top])) if top else None,
        "random_untouched_improvement_rate": float(np.mean([bool(x["improved_untouched"]) for x in rand])) if rand else None,
        "fixed_score_bins": bins,
        "gate": {
            "rho_val_positive": bool(rho([float(x["score"]) for x in grad], [-float(x["delta_val_nll"]) for x in grad]) is not None and rho([float(x["score"]) for x in grad], [-float(x["delta_val_nll"]) for x in grad]) > 0),
            "top20_mean_better_than_all_gradient": bool(top and grad and mean(top, "delta_val_nll") < mean(grad, "delta_val_nll")),
            "top20_rate_above_random": bool(top and rand and rate(top) > rate(rand)),
        },
        "rows": rows,
    }


def run_representation(
    model: torch.nn.Module,
    rep: str,
    layers: List[int],
    fit: List[torch.Tensor],
    val: List[torch.Tensor],
    untouched: List[torch.Tensor],
    codes: Dict[int, Dict[str, Any]],
    base_states: Dict[int, Dict[str, torch.Tensor]],
    baseline: Dict[str, float],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    fixed_candidates: List[Dict[str, Any]] = []
    for layer in layers:
        if rep == "centered":
            pool = gradient_projection_candidates_unique(codes[layer], grads[layer], "qk", args.candidate_pool)
            selected = select_fixed_positions(pool, args.evaluated_per_layer)
            for candidate in selected:
                fixed_candidates.append(describe_candidate(rep, "gradient", layer, candidate, direct_score(codes[layer][candidate.matrix_key], grads[layer][candidate.matrix_key], candidate)))
            random_candidates = random_support_candidates(codes[layer], "qk", args.random_per_layer, args.seed + 1009 * (layer + 1))
            for candidate in random_candidates:
                fixed_candidates.append(describe_candidate(rep, "random", layer, candidate, direct_score(codes[layer][candidate.matrix_key], grads[layer][candidate.matrix_key], candidate)))
        else:
            pool: List[AffineEdit] = []
            for key in ("q", "k"):
                pool.extend(build_group_candidates(layer, key, codes[layer][key], grads[layer][key], "affine_fp"))
            pool.sort(key=lambda x: (-float(x.score_formula), x.key, x.row, x.block, x.donor, x.receiver))
            selected = select_fixed_positions(pool[: args.candidate_pool], args.evaluated_per_layer)
            for candidate in selected:
                fixed_candidates.append(describe_candidate(rep, "gradient", layer, candidate, affine_score(codes[layer][candidate.key], grads[layer][candidate.key], candidate)))
            random_edits = build_random_edits({layer: codes[layer]}, args.random_per_layer, args.seed + 1009 * (layer + 1))
            for candidate in random_edits:
                fixed_candidates.append(describe_candidate(rep, "random", layer, candidate, affine_score(codes[layer][candidate.key], grads[layer][candidate.key], candidate)))

    rows: List[Dict[str, Any]] = []
    for item in fixed_candidates:
        layer = int(item["layer"])
        if rep == "centered":
            key = item["matrix"]
            candidate = Candidate(key, int(item["donor"]), int(item["receiver"]), float(item["score"]))
        else:
            candidate = AffineEdit(
                layer=layer,
                key=item["matrix"],
                row=int(item["row"]),
                block=int(item["block"]),
                donor=int(item["donor"]),
                receiver=int(item["receiver"]),
                donor_sign=int(codes[layer][item["matrix"]].T[int(item["row"]), int(item["block"]), int(item["donor"])].item()),
                receiver_sign=int(item["receiver_sign"]),
                score_formula=float(item["score"]),
                score_exact=float(item["score"]),
            )
        set_one_candidate(model, rep, layer, candidate, codes, base_states)
        value_val = float(evaluate_nll(model, val, device))
        value_untouched = float(evaluate_nll(model, untouched, device))
        set_baseline_qk(model, rep, layers, codes, base_states)
        rows.append({
            **item,
            "delta_val_nll": value_val - baseline["val"],
            "delta_untouched_w_nll": value_untouched - baseline["untouched_w"],
            "improved": value_val < baseline["val"],
            "improved_untouched": value_untouched < baseline["untouched_w"],
        })
    summary = score_summary(rows)
    summary["candidate_generation"] = {
        "layers": layers,
        "candidate_pool": args.candidate_pool,
        "evaluated_per_layer": args.evaluated_per_layer,
        "random_per_layer": args.random_per_layer,
        "selection_ranks": [0, 1, 2, 4, 8, 16, 24, 31],
        "validation_used_for_selection": False,
        "untouched_used_for_selection": False,
    }
    summary["baseline"] = baseline
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("P6-A requires CUDA")
    device = torch.device("cuda")
    layers = parse_layers(args.layers)
    if layers != list(range(24)):
        raise ValueError("P6-A requires all OPT-350M layers 0--23")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    running = out_dir / "result.running.json"
    running.write_text(json.dumps({"run_id": args.run_id, "status": "running", "config": vars(args)}, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.fit_token_offset,
        args.val_token_offset,
    )
    c4 = build_c4_untouched_batches(
        tokenizer, args.seq_len, args.batch_size, args.c4_batches, args.c4_token_offset
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.eval()
    fp_qk = snapshot_qk(model, layers)
    fp_metrics = eval_pack(model, val, untouched, c4, device)

    log("running full-model centered ternary baseline")
    apply_direct_ptq_local(model, args.group_size, args.centered_threshold)
    centered_codes = {layer: {key: make_code(fp_qk[layer][key], args.group_size, args.centered_threshold) for key in ("q", "k")} for layer in layers}
    centered_states = {layer: {key: centered_codes[layer][key].state.clone() for key in ("q", "k")} for layer in layers}
    centered_metrics = eval_pack(model, val, untouched, c4, device)
    centered = run_representation(model, "centered", layers, fit, val, untouched, centered_codes, centered_states, centered_metrics, device, args)
    del model
    torch.cuda.empty_cache()

    log("running full-model affine ternary baseline")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.eval()
    apply_full_affine(model, args.group_size, args.affine_threshold)
    affine_codes = {layer: {key: make_affine_code(fp_qk[layer][key], args.group_size, args.affine_threshold) for key in ("q", "k")} for layer in layers}
    affine_states = {layer: {key: affine_codes[layer][key].T.clone() for key in ("q", "k")} for layer in layers}
    affine_metrics = eval_pack(model, val, untouched, c4, device)
    affine = run_representation(model, "affine", layers, fit, val, untouched, affine_codes, affine_states, affine_metrics, device, args)

    def with_ppl(metrics: Dict[str, float]) -> Dict[str, Dict[str, float | None]]:
        return {key: {"nll": value, "ppl": safe_exp(value)} for key, value in metrics.items()}

    result = {
        "run_id": args.run_id,
        "status": "complete",
        "experiment": "CEGSP-P6-A full-layer centered/affine score-validity",
        "config": vars(args),
        "protocol": {
            "cegsp_module_search": False,
            "pt2_called": False,
            "qat_teacher_or_checkpoint": False,
            "optimizer_steps": False,
            "candidate_score": "quantized-point CE gradient first-order -<G,Delta Q>",
            "candidate_action": "one legal same-group support exchange with frozen codebook parameters",
            "centered_full_model": {"threshold": args.centered_threshold, "group_size": args.group_size},
            "affine_full_model": {"threshold": args.affine_threshold, "group_size": args.group_size, "mu_alpha_frozen": True},
            "candidate_selection": {"pool": args.candidate_pool, "fixed_ranks": [0, 1, 2, 4, 8, 16, 24, 31], "random_matched": True},
        },
        "data": {
            "source": source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched),
            "untouched_c4_batches": len(c4) if c4 else 0,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "fit_token_offset": args.fit_token_offset,
            "val_token_offset": args.val_token_offset,
            "c4_token_offset": args.c4_token_offset,
            "split": "Wikitext-2 train fit / validation val + disjoint validation untouched; C4 validation report-only baseline",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "fp_reference": {"nll": fp_metrics, "with_ppl": with_ppl(fp_metrics)},
        "systems": {
            "centered": {
                "baseline": {"nll": centered_metrics, "with_ppl": with_ppl(centered_metrics)},
                "score_validity": centered,
            },
            "affine": {
                "baseline": {"nll": affine_metrics, "with_ppl": with_ppl(affine_metrics)},
                "score_validity": affine,
            },
        },
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if running.exists():
        running.unlink()
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
