#!/usr/bin/env python3
"""CEGSP v2 P0: quantized-point CE gradient, CEGSP, and one-step QAT.

This is a mechanism-boundary experiment for the v2 CEGSP framing.  It keeps
CEGSP as a post-training, optimizer-free discrete edit, while adding matched
One-Step and short Multi-Step QAT-style controls that use the same
quantized-point CE gradient source but update latent FP weights.
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

from cegsp_ce_gradient_4090 import (
    collect_ce_qk_grads,
    projection_weight,
    set_projection_weight,
    target_modules,
)
from tqgsp_support_projection_4090 import (
    apply_direct_ptq_local,
    build_wikitext_splits,
    direct_ternary_weight,
    evaluate_nll,
    gradient_projection_candidates_unique,
    log,
    make_code,
    parse_csv_ints,
    run_one_shot,
    weight_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=16)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--max-edits", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--layer-topk", type=int, default=3)
    p.add_argument("--score-layers", default="0,6,11")
    p.add_argument("--score-candidates", type=int, default=32)
    p.add_argument("--qat-etas", default="0.0,0.01,0.03,0.1,0.3,1.0")
    p.add_argument("--qat-steps", default="1,4")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def snapshot_qk(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    rows: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        refs = target_modules(model, layer)
        rows[layer] = {
            key: projection_weight(refs[key]).detach().float().cpu().clone()
            for key in ("q", "k")
        }
    return rows


def restore_qk(model: torch.nn.Module, layers: List[int], weights: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer in layers:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            set_projection_weight(refs[key], weights[layer][key])


def apply_qk_patch(
    model: torch.nn.Module,
    layers: List[int],
    direct_qk: Dict[int, Dict[str, torch.Tensor]],
    patch_qk: Dict[int, Dict[str, torch.Tensor]],
    selected_layers: List[int],
) -> None:
    restore_qk(model, layers, direct_qk)
    for layer in selected_layers:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            set_projection_weight(refs[key], patch_qk[layer][key])


def tensor_spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    sx = rx.std()
    sy = ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_cegsp_layer_patches(
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    grads: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
    max_edits: int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, object]]]:
    patches: Dict[int, Dict[str, torch.Tensor]] = {}
    traces: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        codes = {key: make_code(fp_qk[layer][key], group_size, threshold_factor) for key in ("q", "k")}
        candidates = gradient_projection_candidates_unique(codes, grads[layer], "qk", max_edits)
        base_states = {key: code.state.clone() for key, code in codes.items()}
        states, trace = run_one_shot(base_states, codes, candidates, max_edits)
        patches[layer] = {
            key: weight_from_state(codes[key], states[key], refit=False)
            for key in ("q", "k")
        }
        traces[layer] = {
            "accepted_edits": len(trace),
            "top_scores": [float(c.score) for c in candidates[: min(5, len(candidates))]],
        }
    return patches, traces


def score_validity(
    model: torch.nn.Module,
    val_batches: List[torch.Tensor],
    device: torch.device,
    direct_qk: Dict[int, Dict[str, torch.Tensor]],
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    grads: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
    n_candidates: int,
    direct_val: float,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for layer in layers:
        codes = {key: make_code(fp_qk[layer][key], group_size, threshold_factor) for key in ("q", "k")}
        candidates = gradient_projection_candidates_unique(codes, grads[layer], "qk", n_candidates)
        base_states = {key: code.state.clone() for key, code in codes.items()}
        for rank, candidate in enumerate(candidates[:n_candidates]):
            states, trace = run_one_shot(base_states, codes, [candidate], 1)
            if not trace:
                continue
            patch = {
                key: weight_from_state(codes[key], states[key], refit=False)
                for key in ("q", "k")
            }
            refs = target_modules(model, layer)
            for key in ("q", "k"):
                set_projection_weight(refs[key], patch[key])
            nll = evaluate_nll(model, val_batches, device)
            for key in ("q", "k"):
                set_projection_weight(refs[key], direct_qk[layer][key])
            rows.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "matrix": candidate.matrix_key,
                    "score": float(candidate.score),
                    "actual_delta_val_nll": float(nll - direct_val),
                    "actual_improved": bool(nll < direct_val),
                }
            )
    scores = [float(r["score"]) for r in rows]
    improvements = [-float(r["actual_delta_val_nll"]) for r in rows]
    top_k = max(1, len(rows) // 10)
    ranked = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    return {
        "n_candidates_evaluated": len(rows),
        "spearman_score_vs_actual_improvement": tensor_spearman(scores, improvements),
        "top10pct_true_improvement_rate": float(np.mean([r["actual_improved"] for r in ranked[:top_k]])) if rows else float("nan"),
        "all_candidate_true_improvement_rate": float(np.mean([r["actual_improved"] for r in rows])) if rows else float("nan"),
        "mean_top10pct_delta_val_nll": float(np.mean([r["actual_delta_val_nll"] for r in ranked[:top_k]])) if rows else float("nan"),
        "mean_all_delta_val_nll": float(np.mean([r["actual_delta_val_nll"] for r in rows])) if rows else float("nan"),
        "rows": rows,
    }


def quantize_latent_qk(
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    latent_qk: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        out[layer] = {}
        for key in ("q", "k"):
            q, _ = direct_ternary_weight(latent_qk[layer][key], group_size, threshold_factor)
            out[layer][key] = q.float().cpu()
    return out


def normalized_latent_step(
    latent_qk: Dict[int, Dict[str, torch.Tensor]],
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    grads: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    eta: float,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        out[layer] = {}
        for key in ("q", "k"):
            fp = fp_qk[layer][key]
            grad = grads[layer][key]
            scale = fp.float().std().clamp_min(1e-8) / grad.float().std().clamp_min(1e-12)
            out[layer][key] = latent_qk[layer][key] - float(eta) * scale * grad
    return out


def run_qat_controls(
    model: torch.nn.Module,
    fit: List[torch.Tensor],
    val: List[torch.Tensor],
    untouched: List[torch.Tensor],
    device: torch.device,
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    direct_qk: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
    etas: List[float],
    step_counts: List[int],
    dtype_label: str,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for steps in step_counts:
        for eta in etas:
            latent = {
                layer: {key: fp_qk[layer][key].clone() for key in ("q", "k")}
                for layer in layers
            }
            for _ in range(steps):
                quant = quantize_latent_qk(fp_qk, latent, layers, group_size, threshold_factor)
                apply_qk_patch(model, layers, direct_qk, quant, layers)
                grads = collect_ce_qk_grads(model, fit, layers, device, grad_batches=1)
                latent = normalized_latent_step(latent, fp_qk, grads, layers, eta)
            quant = quantize_latent_qk(fp_qk, latent, layers, group_size, threshold_factor)
            apply_qk_patch(model, layers, direct_qk, quant, layers)
            val_nll = evaluate_nll(model, val, device)
            untouched_nll = evaluate_nll(model, untouched, device)
            rows.append(
                {
                    "name": f"{steps}-step-qateval-eta-{eta:g}",
                    "steps": int(steps),
                    "eta": float(eta),
                    "val_nll": float(val_nll),
                    "untouched_w_nll": float(untouched_nll),
                    "dtype": dtype_label,
                }
            )
    restore_qk(model, layers, direct_qk)
    best_by_steps: Dict[str, object] = {}
    for steps in step_counts:
        candidates = [r for r in rows if int(r["steps"]) == int(steps)]
        best_by_steps[str(steps)] = min(candidates, key=lambda r: float(r["val_nll"]))
    return {"rows": rows, "best_by_steps": best_by_steps}


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_csv_ints(args.layers)
    score_layers = [x for x in parse_csv_ints(args.score_layers) if x in set(layers)]
    etas = parse_float_list(args.qat_etas)
    step_counts = parse_csv_ints(args.qat_steps)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    timing: Dict[str, float] = {}

    log(f"loading {args.model} run_id={args.run_id} dtype={args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, data_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
    )
    timing["load_tokenizer_and_data_sec"] = time.time() - t0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    timing["load_model_sec"] = time.time() - t0

    t0 = time.time()
    fp_qk = snapshot_qk(model, layers)
    fp_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    timing["snapshot_fp_and_eval_sec"] = time.time() - t0

    t0 = time.time()
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
    direct_qk = snapshot_qk(model, layers)
    direct_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    timing["direct_ptq_and_eval_sec"] = time.time() - t0

    t0 = time.time()
    ce_grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    timing["ce_gradient_collection_sec"] = time.time() - t0

    t0 = time.time()
    cegsp_patches, cegsp_traces = build_cegsp_layer_patches(
        fp_qk,
        ce_grads,
        layers,
        args.group_size,
        args.threshold_factor,
        args.max_edits,
    )
    per_layer = []
    for layer in layers:
        apply_qk_patch(model, layers, direct_qk, cegsp_patches, [layer])
        val_nll = evaluate_nll(model, val, device)
        per_layer.append(
            {
                "layer": layer,
                "single_layer_val_nll": float(val_nll),
                "single_layer_delta_val_nll": float(val_nll - direct_nll["val"]),
                **cegsp_traces[layer],
            }
        )
    ranked = sorted(per_layer, key=lambda r: float(r["single_layer_delta_val_nll"]))
    selected_layers = [int(r["layer"]) for r in ranked[: max(0, min(args.layer_topk, len(ranked)))]]
    apply_qk_patch(model, layers, direct_qk, cegsp_patches, selected_layers)
    cegsp_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    restore_qk(model, layers, direct_qk)
    timing["cegsp_edit_select_eval_sec"] = time.time() - t0

    t0 = time.time()
    score_report = score_validity(
        model,
        val,
        device,
        direct_qk,
        fp_qk,
        ce_grads,
        score_layers,
        args.group_size,
        args.threshold_factor,
        args.score_candidates,
        direct_nll["val"],
    )
    timing["score_validity_sec"] = time.time() - t0

    t0 = time.time()
    qat = run_qat_controls(
        model,
        fit,
        val,
        untouched,
        device,
        fp_qk,
        direct_qk,
        layers,
        args.group_size,
        args.threshold_factor,
        etas,
        step_counts,
        args.dtype,
    )
    timing["qat_controls_sec"] = time.time() - t0

    best_qat_step = {
        steps: row
        for steps, row in qat["best_by_steps"].items()
    }
    one_step = best_qat_step.get("1")
    multi_steps = [row for steps, row in best_qat_step.items() if int(steps) != 1]
    best_multi = min(multi_steps, key=lambda r: float(r["val_nll"])) if multi_steps else None
    denom = float(direct_nll["untouched_w"] - best_multi["untouched_w_nll"]) if best_multi else float("nan")
    gap_ratio = (
        float((direct_nll["untouched_w"] - cegsp_nll["untouched_w"]) / denom)
        if best_multi and abs(denom) > 1e-12
        else float("nan")
    )

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "config": vars(args),
        "validation_version": {
            "name": "CEGSP-v2-P0-gap-and-score-validity",
            "primary_question": "Under the same quantized-point CE-gradient signal, does one-shot ternary support exchange have independent value relative to one-step latent QAT?",
            "gates": {
                "score_validity": "positive Spearman and top-score candidates improve more often than all sampled candidates",
                "cegsp_vs_direct": "CEGSP selected edits should reduce validation NLL and preferably untouched NLL",
                "cegsp_vs_one_step_qat": "CEGSP >= one-step QAT supports discrete move value; otherwise move space needs refinement",
                "gap_ratio": "report only; do not claim gap closure without stable multi-step QAT superiority over direct",
            },
        },
        "clean_room_invariants": {
            "uses_qat_teacher": False,
            "uses_qat_checkpoint": False,
            "uses_qat_logits": False,
            "cegsp_uses_optimizer_steps": False,
            "cegsp_uses_latent_fp_update": False,
            "qat_controls_use_latent_fp_update": True,
            "alpha_frozen_for_cegsp": True,
        },
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched),
            "split": "Wikitext-2 train fit / validation val and later validation untouched",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "quant_counts": quant_counts,
        "nll": {
            "fp": fp_nll,
            "direct_ternary": direct_nll,
            "cegsp_topk": {
                "selected_layers": selected_layers,
                "nll": cegsp_nll,
                "delta_vs_direct_val": float(cegsp_nll["val"] - direct_nll["val"]),
                "delta_vs_direct_untouched_w": float(cegsp_nll["untouched_w"] - direct_nll["untouched_w"]),
            },
            "qat_controls": qat,
        },
        "gap_closure_ratio_untouched_vs_best_multistep_qat": gap_ratio,
        "per_layer_cegsp": per_layer,
        "score_validity": score_report,
        "timing": timing,
        "status": "complete",
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
