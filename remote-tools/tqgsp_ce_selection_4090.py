#!/usr/bin/env python3
"""TQGSP-02A: CE-aware layer selection for ternary support projection.

This experiment is strict PTQ.  It uses no QAT checkpoint, logits, latent
weights, or state priors.  It asks whether cheap Q/K operator gains from
TQG-SP predict end-to-end CE/NLL deltas, and whether a small CE validation gate
can select layers that transfer to an untouched split.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tqgsp_support_projection_4090 import (
    MATRIX_KEYS,
    apply_direct_ptq_local,
    build_wikitext_splits,
    collect_hidden_states,
    compose_weights,
    evaluate_nll,
    evaluate_state,
    gradient_for_operator,
    gradient_projection_candidates_unique,
    log,
    make_code,
    parse_csv_ints,
    precompute_targets,
    run_one_shot,
    target_modules,
    weight_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=8)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--max-swaps", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--operator-topk", type=int, default=6)
    p.add_argument("--ce-topk", type=int, default=6)
    p.add_argument("--min-operator-gain-pct", type=float, default=1.0)
    p.add_argument("--max-select-delta", type=float, default=0.0)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def set_layer_qk_weights(
    model: torch.nn.Module,
    layer: int,
    weights: Dict[str, torch.Tensor],
) -> None:
    modules = target_modules(model, layer)
    with torch.no_grad():
        for key in ("q", "k"):
            module = modules[key]
            module.weight.data.copy_(weights[key].to(device=module.weight.device, dtype=module.weight.dtype))


def snapshot_direct_qk(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    rows: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        modules = target_modules(model, layer)
        rows[layer] = {key: modules[key].weight.detach().float().cpu().clone() for key in ("q", "k")}
    return rows


def patch_set(
    model: torch.nn.Module,
    layers: List[int],
    direct_qk: Dict[int, Dict[str, torch.Tensor]],
    tqgsp_qk: Dict[int, Dict[str, torch.Tensor]],
    selected: List[int],
) -> None:
    for layer in layers:
        set_layer_qk_weights(model, layer, direct_qk[layer])
    for layer in selected:
        set_layer_qk_weights(model, layer, tqgsp_qk[layer])


def safe_corr(xs: List[float], ys: List[float]) -> Dict[str, float | None]:
    if len(xs) < 2:
        return {"pearson": None, "spearman": None}
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def main() -> None:
    args = parse_args()
    started = time.time()
    timing: Dict[str, float] = {}
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_csv_ints(args.layers)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    log(f"loading {args.model} dtype={args.dtype} layers={len(layers)} gpu={torch.cuda.get_device_name(0)}")
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
    fp_weights_by_layer = {
        layer: {key: module.weight.detach().float().cpu() for key, module in target_modules(model, layer).items()}
        for layer in layers
    }
    hidden = {
        "fit": collect_hidden_states(model, fit, layers, device),
        "val": collect_hidden_states(model, val, layers, device),
        "untouched_w": collect_hidden_states(model, untouched, layers, device),
    }
    timing["collect_fp_hidden_sec"] = time.time() - t0

    t0 = time.time()
    fp_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    timing["fp_nll_eval_sec"] = time.time() - t0

    t0 = time.time()
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
    direct_qk = snapshot_direct_qk(model, layers)
    timing["direct_ptq_apply_sec"] = time.time() - t0

    t0 = time.time()
    direct_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    timing["direct_ptq_nll_eval_sec"] = time.time() - t0

    per_layer: List[Dict[str, object]] = []
    tqgsp_qk: Dict[int, Dict[str, torch.Tensor]] = {}
    proxy_t0 = time.time()
    for layer in layers:
        log(f"layer={layer} TQG-SP proxy and single-layer CE")
        fp_weights = fp_weights_by_layer[layer]
        codes = {key: make_code(fp_weights[key], args.group_size, args.threshold_factor) for key in MATRIX_KEYS}
        base_states = {key: code.state.clone() for key, code in codes.items()}
        xs_by_split = {
            "fit": hidden["fit"][layer],
            "val": hidden["val"][layer],
            "untouched_w": hidden["untouched_w"][layer],
        }
        targets_by_split = {
            split: precompute_targets(xs, fp_weights, "qk", device)
            for split, xs in xs_by_split.items()
        }
        base_metrics = {
            split: evaluate_state(base_states, codes, xs, targets_by_split[split], "qk", device, refit=True)
            for split, xs in xs_by_split.items()
        }
        base_weights = compose_weights(codes, base_states, refit=False)
        grads = gradient_for_operator(
            xs_by_split["fit"],
            targets_by_split["fit"],
            base_weights,
            "qk",
            device,
            args.grad_batches,
        )
        candidates = gradient_projection_candidates_unique(codes, grads, "qk", args.max_swaps)
        tq_states, trace = run_one_shot(base_states, codes, candidates, args.max_swaps)
        tq_metrics = {
            split: evaluate_state(tq_states, codes, xs, targets_by_split[split], "qk", device, refit=True)
            for split, xs in xs_by_split.items()
        }
        tqgsp_qk[layer] = {
            key: weight_from_state(codes[key], tq_states[key], refit=True)
            for key in ("q", "k")
        }

        set_layer_qk_weights(model, layer, tqgsp_qk[layer])
        nll_val = evaluate_nll(model, val, device)
        set_layer_qk_weights(model, layer, direct_qk[layer])

        op_gain_val = (base_metrics["val"] - tq_metrics["val"]) / max(base_metrics["val"], 1e-12) * 100.0
        op_gain_untouched = (base_metrics["untouched_w"] - tq_metrics["untouched_w"]) / max(base_metrics["untouched_w"], 1e-12) * 100.0
        per_layer.append(
            {
                "layer": layer,
                "accepted_edits": len(trace),
                "base_operator": base_metrics,
                "tqgsp_operator": tq_metrics,
                "operator_gain_val_pct": op_gain_val,
                "operator_gain_untouched_pct": op_gain_untouched,
                "single_patch_val_nll": nll_val,
                "single_patch_val_delta": nll_val - direct_nll["val"],
                "selected_by_gate": bool(op_gain_val >= args.min_operator_gain_pct and nll_val - direct_nll["val"] <= args.max_select_delta),
            }
        )
    timing["proxy_and_single_layer_ce_sec"] = time.time() - proxy_t0

    selected_ce = [int(row["layer"]) for row in per_layer if row["selected_by_gate"]]
    operator_top = [
        int(row["layer"])
        for row in sorted(per_layer, key=lambda r: float(r["operator_gain_val_pct"]), reverse=True)[: args.operator_topk]
    ]
    ce_top = [
        int(row["layer"])
        for row in sorted(per_layer, key=lambda r: float(r["single_patch_val_delta"]))[: args.ce_topk]
    ]
    all_layers = list(layers)

    patch_sets = {
        "all-tqgsp-qk": all_layers,
        "operator-topk-qk": operator_top,
        "ce-selected-qk": selected_ce,
        "ce-topk-qk": ce_top,
    }

    patch_results: Dict[str, object] = {}
    patch_t0 = time.time()
    for name, selected in patch_sets.items():
        log(f"evaluating patch set {name} layers={selected}")
        patch_set(model, layers, direct_qk, tqgsp_qk, selected)
        scores = {
            "val": evaluate_nll(model, val, device),
            "untouched_w": evaluate_nll(model, untouched, device),
        }
        patch_results[name] = {
            "layers": selected,
            "n_layers": len(selected),
            "nll": scores,
            "delta_vs_direct_val": scores["val"] - direct_nll["val"],
            "delta_vs_direct_untouched_w": scores["untouched_w"] - direct_nll["untouched_w"],
        }
    patch_set(model, layers, direct_qk, tqgsp_qk, [])
    timing["patch_set_nll_eval_sec"] = time.time() - patch_t0

    op_gains = [float(row["operator_gain_val_pct"]) for row in per_layer]
    nll_deltas = [float(row["single_patch_val_delta"]) for row in per_layer]
    corr = safe_corr(op_gains, nll_deltas)

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "config": vars(args),
        "validation_version": {
            "name": "TQGSP-02A-CE-aware-layer-selection",
            "primary_question": "Does TQG-SP operator gain predict end-to-end CE/NLL, and can a small CE gate select transferable Q/K layers?",
            "gate": {
                "proxy_reliability": "operator gain should be negatively correlated with single-layer val NLL delta",
                "selection_transfer": "ce-selected-qk should not degrade untouched NLL versus direct ternary",
            },
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "clean_room_invariants": {
            "uses_qat_checkpoint": False,
            "uses_qat_logits": False,
            "uses_qat_latent_weights": False,
            "uses_qat_state_prior": False,
            "uses_path_barrier_or_tdbt_transport": False,
            "uses_quantized_point_operator_gradient": True,
            "uses_ce_selection_gate": True,
        },
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched),
            "split": "Wikitext-2 train fit / validation val selection and later validation untouched",
        },
        "nll": {
            "fp": fp_nll,
            "direct_ternary": direct_nll,
            "patch_sets": patch_results,
        },
        "quant_counts": quant_counts,
        "per_layer": per_layer,
        "correlation_operator_gain_vs_val_nll_delta": corr,
        "timing": timing,
        "status": "complete",
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()

